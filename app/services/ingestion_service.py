"""Ingestion pipeline: PDF -> chunk -> embedding -> Qdrant.
Nếu parse/embed lỗi thì Qdrant chưa bị thay đổi
"""
import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from app.embedding.bge_m3 import BGEM3Embedder, EmbeddedText
from app.ingestion.chunker import chunk_pages
from app.ingestion.pdf_parser import parse_pdf
from app.vectorstore.qdrant_store import ChunkRecord, QdrantStore

logger = logging.getLogger(__name__)


class IngestError(RuntimeError):
    """Lỗi trong quá trình ingest"""


class IngestResult(BaseModel):
    source_file: str
    pages_with_text: int
    chunk_count: int


# Protocol = interface chỉ khai báo method mà service dùng để   vậy test có thể truyền fake mà không cần kế thừa class thật.
class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[EmbeddedText]: ...


class Store(Protocol):
    def ensure_collection(self) -> None: ...
    def delete_by_source_file(self, source_file: str) -> None: ...
    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None: ...


@contextmanager
def _timed(step: str, source_file: str) -> Generator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.info("%s: %s xong trong %.2fs", source_file, step, time.perf_counter() - start)


def ingest_pdf(
    path: str | Path,
    *,
    store: Store | None = None,
    embedder: Embedder | None = None,
) -> IngestResult:
    """Ingest một file PDF vào Qdrant. Chạy lại nhiều lần vẫn an toàn."""
    store = store or QdrantStore()
    embedder = embedder or BGEM3Embedder()
    name = Path(path).name

    # 1. Parse
    with _timed("parse", name):
        pages = parse_pdf(path)
    if not pages:
        raise IngestError(
            f"{name}: không có trang nào trích xuất được text. "
            "File có thể là bản scan, hãy chạy OCR trước rồi ingest lại."
        )

    # 2. Chunk
    with _timed("chunk", name):
        chunks = chunk_pages(pages)
    if not chunks:
        raise IngestError(f"{name}: Không tạo được chunk nào từ nội dung PDF.")

    # 3. Embed (chia batch theo settings.embedding_batch_size)
    with _timed("embed", name):
        embeddings = embedder.embed_documents([c.text for c in chunks])
    if len(embeddings) != len(chunks):
        raise IngestError(
            f"{name}: số embedding ({len(embeddings)}) khác số chunk ({len(chunks)})."
        )

    records = [
        ChunkRecord(
            source_file=c.source_file,
            page=c.page,
            chunk_id=c.chunk_id,
            text=c.text,
            dense=e["dense"],
            sparse_indices=e["sparse_indices"],
            sparse_values=e["sparse_values"],
        )
        for c, e in zip(chunks, embeddings, strict=True)
    ]

    # 4. Ghi vào Qdrant (mọi thứ đã sẵn trong RAM)
    with _timed("upsert", name):
        store.ensure_collection()
        # Xóa chunk cũ để file ngắn hơn không để lại chunk dở
        store.delete_by_source_file(records[0].source_file)
        store.upsert_chunks(records)

    return IngestResult(
        source_file=records[0].source_file,
        pages_with_text=len(pages),
        chunk_count=len(records),
    )