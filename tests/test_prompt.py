"""Unit test cho Prompt Builder (issue #30). Không cần Qdrant, Ollama hay BGE-M3.

Nội dung chunk dưới đây là dữ liệu GIẢ. Hãy thay bằng đoạn chép từ file1, file2.
"""
import pytest

from app.core.config import settings
from app.rag.prompt import (
    CHUNK_SEPARATOR,
    NO_CONTEXT,
    SYSTEM_PROMPT,
    build_context,
    build_messages,
    format_chunk,
    select_chunks,
)
from app.retrieval.hybrid import RetrievedChunk


def _chunk(text: str, source: str = "file1.pdf", page: int = 1, score: float = 0.5):
    return RetrievedChunk(
        text=text,
        source_file=source,
        page=page,
        chunk_id=f"{source}::p{page}::c1",
        score=score,
    )


@pytest.fixture
def chunks():
    # score cố ý không giảm dần để kiểm tra builder không tự sort lại
    return [
        _chunk("Đoạn thử A: dưới 90% thì không đạt.", "file1.pdf", 3, 0.1),
        _chunk("Đoạn thử B.", "file2.pdf", 7, 0.9),
        _chunk("Đoạn thử C.", "file1.pdf", 5, 0.5),
    ]


def test_system_first_user_last_with_question(chunks):
    messages = build_messages("Khi nào không đạt?", chunks, 5000)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[-1]["content"].endswith("CÂU HỎI: Khi nào không đạt?")


def test_numbers_follow_input_order(chunks):
    content = build_messages("q", chunks, 5000)[-1]["content"]
    labels = [
        "[1] (Nguồn: file1.pdf, trang 3)\nĐoạn thử A",
        "[2] (Nguồn: file2.pdf, trang 7)\nĐoạn thử B",
        "[3] (Nguồn: file1.pdf, trang 5)\nĐoạn thử C",
    ]
    positions = [content.index(label) for label in labels]
    assert positions == sorted(positions)


def test_exact_limit_keeps_all_chunks(chunks):
    limit = len(build_context(chunks))
    assert select_chunks(chunks, limit) == chunks


def test_one_char_short_drops_whole_last_chunk(chunks):
    limit = len(build_context(chunks)) - 1
    assert select_chunks(chunks, limit) == chunks[:2]
    content = build_messages("q", chunks, limit)[-1]["content"]
    assert "[3]" not in content
    assert "Đoạn thử C" not in content


@pytest.mark.parametrize("limit", [0, 1, 50, 80, 120, 200, 5000])
def test_context_never_exceeds_limit(chunks, limit):
    assert len(build_context(select_chunks(chunks, limit))) <= limit


def test_separator_is_counted(chunks):
    without_separator = sum(len(format_chunk(n, c)) for n, c in enumerate(chunks[:2], 1))
    assert len(select_chunks(chunks, without_separator)) == 1
    assert len(select_chunks(chunks, without_separator + len(CHUNK_SEPARATOR))) == 2


def test_stops_at_first_chunk_that_does_not_fit():
    items = [_chunk("ngắn 1"), _chunk("x" * 500), _chunk("ngắn 3")]
    assert select_chunks(items, 200) == items[:1]


@pytest.mark.parametrize("items", [[], [_chunk("x" * 500)]])
def test_empty_or_oversized_context_still_asks_question(items):
    content = build_messages("Câu hỏi?", items, 100)[-1]["content"]
    assert NO_CONTEXT in content
    assert "x" * 20 not in content
    assert content.endswith("CÂU HỎI: Câu hỏi?")


def test_default_limit_comes_from_settings(chunks, monkeypatch):
    monkeypatch.setattr(settings, "rag_max_context_chars", len(format_chunk(1, chunks[0])))
    content = build_messages("q", chunks)[-1]["content"]
    assert "[1]" in content and "[2] (" not in content


def test_explicit_zero_is_not_replaced_by_default(chunks):
    assert NO_CONTEXT in build_messages("q", chunks, 0)[-1]["content"]


def test_input_list_is_not_modified(chunks):
    before = list(chunks)
    build_messages("q", chunks, 60)
    assert chunks == before