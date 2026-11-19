import pymupdf
import pytest

from app.ingestion.pdf_parser import PDFParseError
from app.services.ingestion_service import IngestError, ingest_pdf
from app.vectorstore.qdrant_store import ChunkRecord, make_point_id


class FakeEmbedder:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def embed_documents(self, texts):
        if self.fail:
            raise RuntimeError("embed lỗi")
        return [
            {"dense": [0.0] * 1024, "sparse_indices": [1, 2], "sparse_values": [0.5, 0.5]}
            for _ in texts
        ]


class FakeStore:
    """Store trong bộ nhớ, mô phỏng hành vi upsert đè theo point id."""

    def __init__(self):
        self.points: dict[str, ChunkRecord] = {}

    def ensure_collection(self):
        pass

    def delete_by_source_file(self, source_file):
        self.points = {k: v for k, v in self.points.items() if v.source_file != source_file}

    def upsert_chunks(self, chunks):
        for c in chunks:
            self.points[make_point_id(c.chunk_id)] = c

    def count(self, source_file=None):
        if source_file is None:
            return len(self.points)
        return sum(1 for p in self.points.values() if p.source_file == source_file)


def make_pdf(path, pages: list[str]):
    """Tạo PDF test. Dùng ASCII vì font mặc định của pymupdf không vẽ được dấu tiếng Việt."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=11)
    doc.save(path)
    doc.close()


LONG_TEXT = " ".join(f"Sentence number {i} about retrieval augmented generation." for i in range(40))


def test_ingest_basic_fields(tmp_path):
    pdf = tmp_path / "a.pdf"
    make_pdf(pdf, [LONG_TEXT, LONG_TEXT])
    store = FakeStore()

    result = ingest_pdf(pdf, store=store, embedder=FakeEmbedder())

    assert result.source_file == "a.pdf"
    assert result.pages_with_text == 2
    assert result.chunk_count == store.count("a.pdf") > 0
    for p in store.points.values():
        assert p.source_file and p.page >= 1 and p.chunk_id and p.text


def test_ingest_twice_keeps_point_count(tmp_path):
    pdf = tmp_path / "a.pdf"
    make_pdf(pdf, [LONG_TEXT])
    store = FakeStore()

    first = ingest_pdf(pdf, store=store, embedder=FakeEmbedder())
    second = ingest_pdf(pdf, store=store, embedder=FakeEmbedder())

    assert first.chunk_count == second.chunk_count == store.count()


def test_reingest_shorter_file_leaves_no_orphans(tmp_path):
    pdf = tmp_path / "a.pdf"
    store = FakeStore()

    make_pdf(pdf, [LONG_TEXT, LONG_TEXT, LONG_TEXT])
    long_result = ingest_pdf(pdf, store=store, embedder=FakeEmbedder())
    make_pdf(pdf, [LONG_TEXT])
    short_result = ingest_pdf(pdf, store=store, embedder=FakeEmbedder())

    assert short_result.chunk_count < long_result.chunk_count
    assert store.count("a.pdf") == short_result.chunk_count


def test_other_files_untouched(tmp_path):
    store = FakeStore()
    for name in ("a.pdf", "b.pdf"):
        make_pdf(tmp_path / name, [LONG_TEXT])
        ingest_pdf(tmp_path / name, store=store, embedder=FakeEmbedder())
    before_b = store.count("b.pdf")

    ingest_pdf(tmp_path / "a.pdf", store=store, embedder=FakeEmbedder())

    assert store.count("b.pdf") == before_b


def test_embed_failure_leaves_store_untouched(tmp_path):
    pdf = tmp_path / "a.pdf"
    make_pdf(pdf, [LONG_TEXT])
    store = FakeStore()
    ingest_pdf(pdf, store=store, embedder=FakeEmbedder())
    before = dict(store.points)

    with pytest.raises(RuntimeError):
        ingest_pdf(pdf, store=store, embedder=FakeEmbedder(fail=True))

    assert store.points == before  # dữ liệu cũ còn nguyên


def test_pdf_without_text_raises_ingest_error(tmp_path):
    pdf = tmp_path / "blank.pdf"
    make_pdf(pdf, [""])
    store = FakeStore()

    with pytest.raises(IngestError, match="OCR"):
        ingest_pdf(pdf, store=store, embedder=FakeEmbedder())
    assert store.count() == 0


def test_missing_file_raises_parse_error(tmp_path):
    with pytest.raises(PDFParseError):
        ingest_pdf(tmp_path / "none.pdf", store=FakeStore(), embedder=FakeEmbedder())