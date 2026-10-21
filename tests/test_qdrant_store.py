import uuid
from pathlib import Path

import pytest

from app.vectorstore.qdrant_store import ChunkRecord, QdrantStore


def make_chunk(source_file: str, index: int) -> ChunkRecord:
    return ChunkRecord(
        source_file=source_file,
        page=1,
        chunk_id=f"{source_file}:1:{index}",
        text="nội dung thử",
        dense=[0.1] * 1024,
        sparse_indices=[1, 5],
        sparse_values=[0.3, 0.7],
    )


@pytest.fixture
def store():
    s = QdrantStore(collection=f"test_{uuid.uuid4().hex[:8]}")
    s.ensure_collection()
    yield s
    s.client.delete_collection(s.collection)


def test_ensure_collection_idempotent(store):
    store.ensure_collection()  # lần 2 không lỗi


def test_upsert_same_chunk_twice(store):
    store.upsert_chunks([make_chunk("a.pdf", 0)])
    store.upsert_chunks([make_chunk("a.pdf", 0)])
    assert store.count() == 1


def test_delete_only_target_file(store):
    store.upsert_chunks(
        [make_chunk("a.pdf", 0), make_chunk("a.pdf", 1), make_chunk("b.pdf", 0)]
    )
    store.delete_by_source_file("a.pdf")
    assert store.list_sources() == {"b.pdf": 1}


def test_qdrant_down_raises_vietnamese_error():
    s = QdrantStore(url="http://localhost:6399", collection="x")
    with pytest.raises(RuntimeError, match="Không thể kết nối tới Qdrant"):
        s.ensure_collection()


def test_store_does_not_import_embedding_or_llm():
    import app.vectorstore.qdrant_store as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    for banned in ("sentence_transformers", "fastembed", "ollama", "app.llm"):
        assert banned not in src