import unicodedata

from app.ingestion.chunker import chunk_pages
from app.ingestion.pdf_parser import PageContent


def make_page(text, page=1, source="a.pdf"):
    return PageContent(source_file=source, page=page, text=text)


LONG = " ".join(f"từ{i}" for i in range(300))


def test_chunk_has_all_fields_and_id_format():
    chunks = chunk_pages([make_page("Xin chào", page=3)])
    assert len(chunks) == 1
    c = chunks[0]
    assert (c.source_file, c.page, c.chunk_id, c.text) == (
        "a.pdf", 3, "a.pdf::p3::c1", "Xin chào")


def test_no_chunk_exceeds_chunk_size():
    chunks = chunk_pages([make_page(LONG)], chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)


def test_single_long_word_is_allowed_to_exceed():
    chunks = chunk_pages([make_page("x" * 250)], chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert len(chunks[0].text) == 250


def test_chunk_id_unique_and_stable():
    pages = [make_page(LONG, page=1), make_page(LONG, page=2)]
    a = chunk_pages(pages, chunk_size=100, chunk_overlap=20)
    b = chunk_pages(pages, chunk_size=100, chunk_overlap=20)
    ids = [c.chunk_id for c in a]
    assert len(ids) == len(set(ids))
    assert ids == [c.chunk_id for c in b]


def test_never_merges_across_pages():
    pages = [make_page("AAA", page=1), make_page("BBB", page=2)]
    chunks = chunk_pages(pages)
    assert [(c.page, c.text) for c in chunks] == [(1, "AAA"), (2, "BBB")]


def test_output_is_nfc():
    nfd = unicodedata.normalize("NFD", "Việt Nam")
    chunks = chunk_pages([make_page(nfd)])
    assert chunks[0].text == "Việt Nam"


def test_whitespace_only_page_gives_no_chunk():
    assert chunk_pages([make_page("   \n\n  ")]) == []


def test_short_page_gives_one_chunk():
    assert len(chunk_pages([make_page("ngắn")])) == 1