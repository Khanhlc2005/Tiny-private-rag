import httpx
import pytest

from app.core.config import settings
from app.llm.ollama_client import OllamaClient

SYSTEM_PROMPT = (
    "Bạn là trợ lý RAG. Chỉ trả lời dựa trên ngữ cảnh được cung cấp, "
    "không mở rộng ngoài yêu cầu."
)
CONSTRAINED_MESSAGES = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": (
            "Ngữ cảnh: Nhân viên được nghỉ phép 12 ngày mỗi năm.\n"
            "Câu hỏi: Nhân viên được nghỉ phép bao nhiêu ngày mỗi năm?"
        ),
    },
]
UNCONSTRAINED_MESSAGES = [
    {
        "role": "user",
        "content": "Hãy viết một bài luận thật dài về lịch sử trí tuệ nhân tạo.",
    },
]


@pytest.fixture(scope="module")
def ollama_available():
    """Bỏ qua integration test nếu Ollama server không chạy."""
    try:
        httpx.get(f"{settings.ollama_base_url}/api/version", timeout=2.0)
    except httpx.HTTPError:
        pytest.skip("Ollama server không chạy — bỏ qua integration test")


@pytest.fixture
def client():
    return OllamaClient()


# ---------- Unit test (không cần Ollama) ----------

def test_chat_raises_runtime_error_when_server_down(client):
    client.base_url = "http://localhost:1"
    with pytest.raises(RuntimeError, match="Không thể kết nối"):
        client.chat(CONSTRAINED_MESSAGES)


def test_stream_chat_raises_runtime_error_when_server_down(client):
    client.base_url = "http://localhost:1"
    with pytest.raises(RuntimeError, match="Không thể kết nối"):
        list(client.stream_chat(CONSTRAINED_MESSAGES))


# ---------- Integration test (cần Ollama đang chạy) ----------

@pytest.mark.integration
def test_chat_returns_non_empty_string(ollama_available, client):
    answer = client.chat(CONSTRAINED_MESSAGES)

    assert isinstance(answer, str)
    assert answer.strip() != ""


@pytest.mark.integration
def test_stream_chat_yields_multiple_chunks(ollama_available, client):
    chunks = list(client.stream_chat(CONSTRAINED_MESSAGES))

    assert len(chunks) > 1
    assert all(isinstance(c, str) and c != "" for c in chunks)
    assert "".join(chunks).strip() != ""


@pytest.mark.integration
def test_unconstrained_answer_is_capped_by_num_predict(ollama_available, client):
    """Câu hỏi không ràng buộc dễ khiến Gemma viết dài — num_predict phải cắt được."""
    answer = client.chat(UNCONSTRAINED_MESSAGES)

    word_count = len(answer.split())
    assert word_count <= settings.ollama_num_predict * 1.5


@pytest.mark.integration
def test_think_false_returns_no_thinking_via_api(ollama_available):
    """Xác nhận think=False có hiệu lực qua HTTP API (không chỉ ở CLI)."""
    payload = {
        "model": settings.ollama_model,
        "messages": CONSTRAINED_MESSAGES,
        "think": False,
        "stream": False,
        "options": {"num_predict": settings.ollama_num_predict},
    }
    response = httpx.post(
        f"{settings.ollama_base_url}/api/chat", json=payload, timeout=120.0
    )
    response.raise_for_status()
    message = response.json().get("message", {})

    assert not message.get("thinking")