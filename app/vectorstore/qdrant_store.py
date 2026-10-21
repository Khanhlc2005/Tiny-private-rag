import logging
import uuid
from collections import Counter
from collections.abc import Callable
from functools import wraps
from typing import Concatenate, ParamSpec, Protocol, TypeVar

from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.core.config import settings

logger = logging.getLogger(__name__)

DENSE_NAME = "dense"
SPARSE_NAME = "sparse"
DENSE_SIZE = 1024
UPSERT_BATCH_SIZE = 64

# Namespace cố định không đổi tránh các point id đổi theo
_POINT_NAMESPACE = uuid.UUID("6f1c2d3e-8a4b-4c5d-9e6f-0a1b2c3d4e5f")

P = ParamSpec("P")
R = TypeVar("R")


class _HasUrl(Protocol):
    url: str

S = TypeVar("S", bound=_HasUrl)


class ChunkRecord(BaseModel):
    """
    chunk_id quy ước: f"{source_file}:{page}:{chunk_index}" - phải xác định
    (cùng input -> cùng chuỗi) để ingest lại cùng file thì ghi đè.
    """
    source_file: str
    page: int
    chunk_id: str
    text: str
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]

def make_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))

def _wrap_errors(func: Callable[Concatenate[S, P], R]) -> Callable[Concatenate[S, P], R]:
    """Đổi lỗi kết nối của Qdrant thành lỗi RuntimeError bằng tiếng Việt"""

    @wraps(func)
    def wrapper(self: S, *args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(self, *args, **kwargs)
        except ResponseHandlingException as exc:
            raise RuntimeError(
                f"Không thể kết nối tới Qdrant tại: {self.url}. "
                "Hãy kiểm tra container Qdrant"
            ) from exc
        except UnexpectedResponse as exc:
            raise RuntimeError(
                f"Qdrant trả về HTTP: {exc.status_code}: {exc.content!r}"
            ) from exc

    return wrapper


class QdrantStore:
    # Test client giả
    def __init__(
        self,
        url: str | None = None,
        collection: str | None = None,
        client: QdrantClient | None = None,
    ):
        self.url = url or settings.qdrant_url
        self.collection = collection or settings.qdrant_collection
        self.client = client or QdrantClient(url=self.url, timeout=10)

    @_wrap_errors
    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            self._check_config()
        else:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    DENSE_NAME: models.VectorParams(
                        size=DENSE_SIZE, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={SPARSE_NAME: models.SparseVectorParams()}
            )
            logger.info("Đã tạo collection Qdrant %s", self.collection)

        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="source_file",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    def _check_config(self) -> None:
        """Collection tồn tại nhưng khác cấu hình thì báo lỗi"""

        params = self.client.get_collection(self.collection).config.params
        vectors = params.vectors
        dense = vectors.get(DENSE_NAME) if isinstance(vectors, dict) else None
        sparse = params.sparse_vectors or {}
        if (
            dense is None
            or dense.size != DENSE_SIZE
            or dense.distance != models.Distance.COSINE
            or SPARSE_NAME not in sparse
        ):
            raise RuntimeError(
                f"Collection Qdrant {self.collection} đã tồn tại nhưng khác cấu hình"
                f"Không khớp ( cần dense: {DENSE_SIZE} cosine + sparse)"
                "Hãy xóa collection cũ hoặc đổi Qdrant collection"
            )

    @_wrap_errors
    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        for i in range(0, len(chunks), UPSERT_BATCH_SIZE):
            batch = chunks[i: i + UPSERT_BATCH_SIZE]
            points = [
                models.PointStruct(
                    id = make_point_id(c.chunk_id),
                    vector = {
                        DENSE_NAME: c.dense,
                        SPARSE_NAME: models.SparseVector(
                            indices=c.sparse_indices,
                            values=c.sparse_values
                        ),
                    },
                    payload = {
                        "source_file": c.source_file,
                        "page": c.page,
                        "chunk_id": c.chunk_id,
                        "text": c.text
                    },
                )
                for c in batch
            ]
            self.client.upsert(self.collection, points=points, wait=True)

    @staticmethod
    def _source_filter(source_file: str) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="source_file",
                    match=models.MatchValue(value=source_file)
                )
            ]
        )

    @_wrap_errors
    def delete_by_source_file(self, source_file: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=self._source_filter(source_file)
            ),
            wait=True,
        )

    @_wrap_errors
    def list_sources(self) -> dict[str, int]:
        """Trả về {source_file: số chunk}"""

        counts: Counter[str] = Counter()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=["source_file"],
                with_vectors=False
            )
            for p in points:
                counts[(p.payload or {})["source_file"]] += 1
            if offset is None:
                break
        return dict(counts)

    @_wrap_errors
    def count(self, source_file: str | None = None) -> int:
        flt = self._source_filter(source_file) if source_file else None
        return self.client.count(self.collection, count_filter=flt, exact=True).count
