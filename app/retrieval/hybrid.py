"""
Hybrid retrieval: dense + sparse (BGE-M3) -> RRF, qua Query API của Qdrant.
"""
import logging
from typing import Protocol

from pydantic import BaseModel
from qdrant_client.http import models

from app.core.config import settings
from app.embedding.bge_m3 import BGEM3Embedder, EmbeddedText
from app.vectorstore.qdrant_store import (
    DENSE_NAME,
    SPARSE_NAME,
    QdrantStore,
    _wrap_errors,
)

logger = logging.getLogger(__name__)


class RetrievedChunk(BaseModel):
    text: str
    source_file: str
    page: int
    chunk_id: str
    score: float


class QueryEmbedder(Protocol):
    """Chỉ khai báo method retriever dùng"""

    def embed_query(self, text: str) -> EmbeddedText: ...


class HybridRetriever:
    def __init__(
        self,
        store: QdrantStore | None = None,
        embedder: QueryEmbedder | None = None,
    ):
        self.store = store or QdrantStore()
        self.embedder = embedder or BGEM3Embedder()
        # _wrap_errors cần self.url để tạo thông báo lỗi tiếng Việt
        self.url = self.store.url

    @_wrap_errors
    def search(
        self,
        query: str,
        top_k: int | None = None,
        source_file: str | None = None,
        candidates: int | None = None,
    ) -> list[RetrievedChunk]:
        """Hybrid: 2 nhánh prefetch (dense, sparse) rồi fuse bằng RRF."""
        top_k = top_k or settings.retrieval_top_k
        candidates = candidates or settings.retrieval_candidates

        if not query.strip() or not self._collection_ready():
            return []

        q = self.embedder.embed_query(query)
        flt = QdrantStore._source_filter(source_file) if source_file else None

        response = self.store.client.query_points(
            collection_name=self.store.collection,
            prefetch=[
                models.Prefetch(
                    query=q["dense"], using=DENSE_NAME, limit=candidates, filter=flt
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=q["sparse_indices"], values=q["sparse_values"]
                    ),
                    using=SPARSE_NAME,
                    limit=candidates,
                    filter=flt,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return [self._to_chunk(p) for p in response.points]

    @_wrap_errors
    def search_dense_only(
        self,
        query: str,
        top_k: int | None = None,
        source_file: str | None = None,
    ) -> list[RetrievedChunk]:
        """Chỉ dùng cho so sánh thử  với score là cosine"""
        top_k = top_k or settings.retrieval_top_k
        if not query.strip() or not self._collection_ready():
            return []

        q = self.embedder.embed_query(query)
        flt = QdrantStore._source_filter(source_file) if source_file else None
        response = self.store.client.query_points(
            collection_name=self.store.collection,
            query=q["dense"],
            using=DENSE_NAME,
            query_filter=flt,
            limit=top_k,
            with_payload=True,
        )
        return [self._to_chunk(p) for p in response.points]

    def _collection_ready(self) -> bool:
        # Collection chưa tồn tại -> Qdrant báo lỗi
        return self.store.client.collection_exists(self.store.collection)

    @staticmethod
    def _to_chunk(point: models.ScoredPoint) -> RetrievedChunk:
        payload = point.payload or {}
        return RetrievedChunk(
            text=payload["text"],
            source_file=payload["source_file"],
            page=payload["page"],
            chunk_id=payload["chunk_id"],
            score=point.score,
        )