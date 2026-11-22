"""Integration test cho HybridRetriever với Qdrant thật (issue #17).

Phần lớn test dùng embedder giả với vector xác định nên không cần tải BGE-M3.
Chỉ test cuối dùng model thật (đánh dấu `slow`).
Unit test với Qdrant in-memory nằm ở tests/test_hybrid.py (issue #16).
"""
import uuid

import pytest

from app.retrieval.hybrid import HybridRetriever
from app.vectorstore.qdrant_store import ChunkRecord, QdrantStore


@pytest.fixture
def qdrant_store(qdrant_available):
    """Collection tạm tên ngẫu nhiên, luôn xóa khi xong; không đụng collection thật."""
    name = f"test_{uuid.uuid4().hex[:12]}"
    store = QdrantStore(collection=name)
    store.ensure_collection()
    try:
        yield store
    finally:
        store.client.delete_collection(name)


class FakeEmbedder:
    def __init__(self, table: dict[str, dict]):
        self.table = table

    def embed_query(self, text: str) -> dict:
        return self.table[text]


def _dense(axis: int) -> list[float]:
    v = [0.0] * 1024
    v[axis] = 1.0
    return v


def _chunk(source: str, idx: int, axis: int, token: int, text: str, page: int = 1) -> ChunkRecord:
    return ChunkRecord(
        source_file=source,
        page=page,
        chunk_id=f"{source}:{page}:{idx}",
        text=text,
        dense=_dense(axis),
        sparse_indices=[token],
        sparse_values=[1.0],
    )


@pytest.fixture
def embedder():
    return FakeEmbedder(
        {
            "q": {"dense": _dense(1), "sparse_indices": [3], "sparse_values": [1.0]},
        }
    )


@pytest.mark.integration
def test_hybrid_search_returns_full_payload(qdrant_store, embedder):
    qdrant_store.upsert_chunks([_chunk("a.pdf", 0, axis=1, token=3, text="nội dung A", page=4)])
    results = HybridRetriever(qdrant_store, embedder).search("q", top_k=1)

    assert len(results) == 1
    r = results[0]
    assert (r.text, r.source_file, r.page, r.chunk_id) == ("nội dung A", "a.pdf", 4, "a.pdf:4:0")
    assert isinstance(r.score, float)


@pytest.mark.integration
def test_hybrid_search_filters_by_source_file(qdrant_store, embedder):
    qdrant_store.upsert_chunks(
        [
            _chunk("a.pdf", 0, axis=1, token=3, text="A"),
            _chunk("b.pdf", 0, axis=1, token=3, text="B"),
        ]
    )
    results = HybridRetriever(qdrant_store, embedder).search("q", top_k=5, source_file="b.pdf")

    assert results and {r.source_file for r in results} == {"b.pdf"}


@pytest.mark.integration
def test_hybrid_search_empty_collection_returns_empty_list(qdrant_store, embedder):
    assert HybridRetriever(qdrant_store, embedder).search("q") == []


@pytest.mark.integration
@pytest.mark.slow
def test_real_model_finds_document_code(qdrant_store):
    """Dùng BGE-M3 thật: câu hỏi chứa số hiệu phải trả về chunk chứa số hiệu."""
    pytest.importorskip("FlagEmbedding")
    from app.embedding.bge_m3 import BGEM3Embedder

    embedder = BGEM3Embedder()
    texts = [
        "Quyết định số 1015/QĐ-ĐHCN ngày 8 tháng 9 năm 2015 của Hiệu trưởng ban hành quy chế đào tạo.",
        "Một tín chỉ được quy định bằng 15 tiết học lý thuyết.",
        "Sinh viên phải nộp học phí đúng hạn vào đầu mỗi học kỳ.",
    ]
    embedded = embedder.embed_documents(texts)
    qdrant_store.upsert_chunks(
        [
            ChunkRecord(
                source_file="demo.pdf", page=1, chunk_id=f"demo.pdf:1:{i}", text=t,
                dense=e["dense"], sparse_indices=e["sparse_indices"], sparse_values=e["sparse_values"],
            )
            for i, (t, e) in enumerate(zip(texts, embedded, strict=True))
        ]
    )

    results = HybridRetriever(qdrant_store, embedder).search("1015/QĐ-ĐHCN", top_k=1)
    assert results[0].chunk_id == "demo.pdf:1:0"