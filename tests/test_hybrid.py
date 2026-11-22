"""Unit test cho HybridRetriever (issue #16).

Dùng Qdrant in-memory và embedder giả nên không cần Docker hay BGE-M3.

Ghi chú: warning "Payload indexes have no effect in the local Qdrant" là bình
thường, vì Qdrant local không hỗ trợ payload index.
"""
import uuid
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from app.core.config import settings
from app.retrieval.hybrid import HybridRetriever
from app.vectorstore.qdrant_store import ChunkRecord, QdrantStore


class FakeEmbedder:
    """Trả vector cố định theo bảng tra, không cần load BGE-M3."""

    def __init__(self, table: dict[str, dict]):
        self.table = table

    def embed_query(self, text: str) -> dict:
        return self.table[text]


def _dense(axis: int) -> list[float]:
    v = [0.0] * 1024
    v[axis] = 1.0
    return v


def _chunk(source: str, idx: int, axis: int, token: int, text: str) -> ChunkRecord:
    return ChunkRecord(
        source_file=source,
        page=1,
        chunk_id=f"{source}:1:{idx}",
        text=text,
        dense=_dense(axis),
        sparse_indices=[token],
        sparse_values=[1.0],
    )


@pytest.fixture
def store():
    s = QdrantStore(
        collection=f"test_{uuid.uuid4().hex[:8]}",
        client=QdrantClient(":memory:"),
    )
    s.ensure_collection()
    return s


@pytest.fixture
def embedder():
    return FakeEmbedder(
        {
            # dense xa mọi chunk, chỉ sparse (token 7) khớp
            "so hieu": {"dense": _dense(900), "sparse_indices": [7], "sparse_values": [1.0]},
            "rong": {"dense": _dense(1), "sparse_indices": [3], "sparse_values": [1.0]},
        }
    )


# --- Acceptance #1: từ khóa / số hiệu có trong tài liệu thì chunk đó nằm trong top-k ---
def test_term_match_returns_chunk_in_top_k(store, embedder):
    store.upsert_chunks(
        [
            _chunk("a.pdf", 0, axis=1, token=1, text="đoạn A"),
            _chunk("a.pdf", 1, axis=2, token=2, text="đoạn B"),
            _chunk("a.pdf", 2, axis=3, token=7, text="Quyết định 1015/QĐ-ĐHCN"),
        ]
    )
    results = HybridRetriever(store, embedder).search("so hieu", top_k=3)
    assert "a.pdf:1:2" in [r.chunk_id for r in results]


# --- Acceptance #2: collection rỗng (hoặc chưa tồn tại) trả [] không lỗi ---
def test_empty_collection_returns_empty(store, embedder):
    assert HybridRetriever(store, embedder).search("rong") == []


def test_missing_collection_returns_empty(embedder):
    s = QdrantStore(collection="khong_ton_tai", client=QdrantClient(":memory:"))
    assert HybridRetriever(s, embedder).search("rong") == []


def test_blank_query_returns_empty(store, embedder):
    assert HybridRetriever(store, embedder).search("   ") == []


# --- Acceptance #3: kết quả đủ metadata để trích dẫn (task 3) ---
def test_results_have_citation_metadata(store, embedder):
    store.upsert_chunks([_chunk("a.pdf", 0, axis=1, token=3, text="nội dung")])
    r = HybridRetriever(store, embedder).search("rong", top_k=1)[0]
    assert (r.source_file, r.page, r.chunk_id, r.text) == ("a.pdf", 1, "a.pdf:1:0", "nội dung")
    assert isinstance(r.score, float)


# --- Acceptance #4: retriever không gọi LLM ---
def test_retriever_does_not_import_llm():
    import app.retrieval.hybrid as mod

    imports = [
        line
        for line in Path(mod.__file__).read_text(encoding="utf-8").splitlines()
        if line.startswith(("import ", "from "))
    ]
    assert not any("app.llm" in line or "ollama" in line for line in imports)


# --- Task 4: lọc theo source file ---
def test_source_filter_only_returns_that_file(store, embedder):
    store.upsert_chunks(
        [
            _chunk("a.pdf", 0, axis=1, token=3, text="A"),
            _chunk("b.pdf", 0, axis=1, token=3, text="B"),
        ]
    )
    results = HybridRetriever(store, embedder).search("rong", top_k=5, source_file="b.pdf")
    assert {r.source_file for r in results} == {"b.pdf"}


# --- Task 5: top_k đọc từ Settings, có thể ghi đè bằng tham số ---
def test_top_k_argument_is_applied(store, embedder):
    store.upsert_chunks([_chunk("a.pdf", i, axis=1 + i, token=3, text=str(i)) for i in range(5)])
    assert len(HybridRetriever(store, embedder).search("rong", top_k=2)) == 2


def test_top_k_defaults_to_settings(store, embedder, monkeypatch):
    monkeypatch.setattr(settings, "retrieval_top_k", 3)
    store.upsert_chunks([_chunk("a.pdf", i, axis=1 + i, token=3, text=str(i)) for i in range(5)])
    assert len(HybridRetriever(store, embedder).search("rong")) == 3