"""Fixture dùng chung cho các test cần Qdrant thật."""
import pytest
from qdrant_client import QdrantClient

from app.core.config import settings


@pytest.fixture(scope="session")
def qdrant_available():
    """Bỏ qua integration test nếu Qdrant không chạy (mô phỏng ollama_available)."""
    try:
        QdrantClient(url=settings.qdrant_url, timeout=2).get_collections()
    except Exception:
        pytest.skip("Qdrant không chạy, bỏ qua integration test")