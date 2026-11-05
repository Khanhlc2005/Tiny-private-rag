import subprocess
import sys
from pathlib import Path

import pytest

from app.embedding.bge_m3 import BGEM3Embedder

REPO_ROOT = Path(__file__).resolve().parents[1]


# Test không cần model (chạy nhanh, dùng cho CI)

def test_import_app_khong_keo_torch():
    """import app (và module embedding) không được nạp torch."""
    code = (
        "import sys\n"
        "import app\n"
        "from app.embedding.bge_m3 import BGEM3Embedder\n"
        "BGEM3Embedder()\n"
        "print('torch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_embed_documents_rong_tra_ve_danh_sach_rong():
    # Danh sách rỗng thì không được nạp model
    embedder = BGEM3Embedder()
    assert embedder.embed_documents([]) == []
    assert embedder._model is None


def test_chuyen_doi_sparse_va_dense_bang_model_gia():
    np = pytest.importorskip("numpy")

    class FakeModel:
        def encode(self, texts, **kwargs):
            return {
                "dense_vecs": np.ones((len(texts), 1024), dtype=np.float32),
                # khóa là chuỗi, giá trị là numpy float giống thư viện thật
                "lexical_weights": [
                    {"12": np.float16(0.5), "7": np.float16(0.25)} for _ in texts
                ],
            }

    embedder = BGEM3Embedder()
    embedder._model = FakeModel()

    results = embedder.embed_documents(["a", "b"])

    assert len(results) == 2
    first = results[0]
    assert len(first["dense"]) == 1024
    assert all(type(x) is float for x in first["dense"])
    assert first["sparse_indices"] == [12, 7]
    assert all(type(i) is int for i in first["sparse_indices"])
    assert first["sparse_values"] == [0.5, 0.25]
    assert all(type(v) is float for v in first["sparse_values"])


def test_embed_query_tra_ve_mot_dict_khong_phai_danh_sach():
    np = pytest.importorskip("numpy")

    class FakeModel:
        def encode(self, texts, **kwargs):
            return {
                "dense_vecs": np.zeros((len(texts), 1024), dtype=np.float32),
                "lexical_weights": [{"1": np.float16(0.1)} for _ in texts],
            }

    embedder = BGEM3Embedder()
    embedder._model = FakeModel()

    result = embedder.embed_query("xin chào")

    assert isinstance(result, dict)
    assert set(result) == {"dense", "sparse_indices", "sparse_values"}


# Test cần tải model thật (chạy bằng: pytest -m integration)

@pytest.fixture(scope="module")
def embedder():
    # scope="module": nạp model một lần cho cả file test
    return BGEM3Embedder()


def cosine(a: list[float], b: list[float]) -> float:
    import numpy as np

    va, vb = np.asarray(a), np.asarray(b)
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))


@pytest.mark.integration
def test_dense_dung_1024_chieu(embedder):
    result = embedder.embed_query("Thủ tục làm hộ chiếu cần giấy tờ gì?")
    assert len(result["dense"]) == 1024


@pytest.mark.integration
def test_sparse_khong_rong_va_cung_do_dai(embedder):
    result = embedder.embed_query("Thủ tục làm hộ chiếu cần giấy tờ gì?")
    assert len(result["sparse_indices"]) > 0
    assert len(result["sparse_indices"]) == len(result["sparse_values"])


@pytest.mark.integration
def test_embed_documents_tra_ve_dung_so_luong(embedder):
    texts = ["Hộ chiếu phổ thông", "Đăng ký kinh doanh hộ cá thể", "Đổi căn cước công dân"]
    results = embedder.embed_documents(texts)
    assert len(results) == len(texts)
    assert all(len(r["dense"]) == 1024 for r in results)


@pytest.mark.integration
def test_cau_lien_quan_co_cosine_cao_hon_cau_khong_lien_quan(embedder):
    query = embedder.embed_query("Thủ tục làm hộ chiếu cần giấy tờ gì?")
    related, unrelated = embedder.embed_documents([
        "Hồ sơ xin cấp hộ chiếu gồm tờ khai, ảnh chân dung và căn cước công dân.",
        "Công thức nấu món phở bò gồm xương bò, hành nướng và quế hồi.",
    ])
    assert cosine(query["dense"], related["dense"]) > cosine(
        query["dense"], unrelated["dense"]
    )


@pytest.mark.integration
def test_tieng_viet_co_dau_embed_binh_thuong(embedder):
    text = "Nguyễn Thị Hương nộp đơn đề nghị cấp đổi hộ chiếu tại Hà Nội."
    result = embedder.embed_query(text)
    assert len(result["dense"]) == 1024
    assert len(result["sparse_indices"]) > 0