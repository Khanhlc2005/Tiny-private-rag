# Tiny-Private RAG

Hệ thống RAG (Retrieval-Augmented Generation) chạy **hoàn toàn local/offline** — không cần Internet khi vận hành (chỉ cần Internet một lần để tải model embedding), phù hợp để hỏi đáp trên tài liệu nội bộ, bí mật mà không gửi dữ liệu ra dịch vụ bên ngoài.

## Overview

Bài toán: nhiều tổ chức có tài liệu nội bộ không thể đưa lên các dịch vụ LLM đám mây. Dự án này xây dựng một pipeline RAG nhỏ gọn chạy trên máy cá nhân: đọc tài liệu PDF, lập chỉ mục vào vector database, truy xuất đoạn liên quan và để LLM local sinh câu trả lời.

Trạng thái hiện tại (sau Week 2):

- **Đã chạy được**: nạp PDF (parse → chunk → embed → Qdrant) và tìm đoạn liên quan bằng hybrid search (dense + sparse, RRF).
- **Chưa có**: bước ghép kết quả truy xuất vào prompt để LLM trả lời, API FastAPI và giao diện. `OllamaClient` đã có nhưng chưa được nối vào pipeline.

Thành phần chính:

- Đầu vào: **PDF**, được đọc bằng **PyMuPDF**.
- Embedding: **BAAI/bge-m3** (dense + sparse).
- Vector DB: **Qdrant** chạy bằng Docker.
- LLM: `gemma4:e2b-it-qat` chạy qua **Ollama** (tắt chế độ thinking).

## Tech Stack

| Thành phần    | Công nghệ                                              | Trạng thái                          |
| ------------- | ------------------------------------------------------ | ----------------------------------- |
| LLM           | `gemma4:e2b-it-qat` qua Ollama (`--think=false`)       | Đã có client (chưa nối vào pipeline) |
| PDF parser    | PyMuPDF                                                | Đã có                               |
| Chunking      | `langchain-text-splitters`                             | Đã có                               |
| Embedding     | BAAI/bge-m3 (dense + sparse)                           | Đã có                               |
| Vector DB     | Qdrant v1.19.1 (Docker)                                | Đã có                               |
| Retrieval     | Hybrid search: dense + sparse, fusion RRF              | Đã có                               |
| Backend       | FastAPI + Pydantic + SQLModel                          | Dự kiến (chưa có endpoint)          |
| UI            | Streamlit                                              | Dự kiến                             |
| Evaluation    | RAGAS                                                  | Dự kiến                             |
| Observability | Langfuse (self-hosted)                                 | Dự kiến                             |

## Requirements

- Python 3.12 (đã phát triển và kiểm thử trên 3.12)
- [Docker](https://docs.docker.com/get-docker/) và Docker Compose
- [Ollama](https://ollama.com/download)
- Git

## Installation

```bash
git clone https://github.com/Khanhlc2005/Tiny-private-rag.git
cd Tiny-private-rag

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-ml.txt   # cần để embed (FlagEmbedding, kéo theo torch, dung lượng lớn)

cp .env.example .env             # chỉnh nếu cần
```

Biến môi trường trong `.env` (tên không phân biệt hoa/thường; biến nào bỏ trống thì dùng giá trị mặc định):

| Biến                   | Mặc định                 | Ý nghĩa                                          |
| ---------------------- | ------------------------ | ------------------------------------------------ |
| `OLLAMA_BASE_URL`      | `http://localhost:11434` | Địa chỉ Ollama server                            |
| `OLLAMA_MODEL`         | `gemma4:e2b-it-qat`      | Model dùng để sinh câu trả lời                   |
| `OLLAMA_NUM_PREDICT`   | `512`                    | Số token tối đa mỗi lần sinh                     |
| `QDRANT_URL`           | `http://localhost:6333`  | Địa chỉ Qdrant                                   |
| `QDRANT_COLLECTION`    | `tiny_rag_chunks`        | Tên collection chứa chunk                        |
| `CHUNK_SIZE`           | `800`                    | Kích thước tối đa mỗi chunk (ký tự)              |
| `CHUNK_OVERLAP`        | `100`                    | Số ký tự chồng lấn giữa hai chunk liền kề        |
| `EMBEDDING_MODEL`      | `BAAI/bge-m3`            | Model embedding                                  |
| `EMBEDDING_DEVICE`     | `cpu`                    | Thiết bị chạy embedding                          |
| `EMBEDDING_BATCH_SIZE` | `16`                     | Batch size khi embed                             |
| `RETRIEVAL_TOP_K`      | `5`                      | Số chunk trả về mỗi lần tìm                      |
| `RETRIEVAL_CANDIDATES` | `20`                     | Số ứng viên mỗi nhánh (dense, sparse) trước RRF  |

## Running Qdrant

```bash
docker compose up -d
docker compose ps                # container tinyrag_qdrant phải ở trạng thái running
```

Kiểm tra Qdrant:

- Dashboard: <http://localhost:6333/dashboard>
- REST API: `curl http://localhost:6333/collections`
- gRPC: cổng `6334`

Dữ liệu được lưu ở `./qdrant_storage`. Dừng dịch vụ: `docker compose down`.

## Running Ollama

```bash
ollama serve                     # bỏ qua nếu Ollama đã chạy như một service
ollama pull gemma4:e2b-it-qat
ollama list                      # xác nhận model đã được tải
```

Thử nhanh qua client của dự án:

```bash
python -c "from app.llm.ollama_client import OllamaClient; print(OllamaClient().chat([{'role': 'user', 'content': 'Xin chào'}]))"
```

## Embedding (BGE-M3)

`app/embedding/bge_m3.py` → `BGEM3Embedder` sinh **dense (1024 chiều)** và **sparse (`sparse_indices`, `sparse_values`)** từ một lần chạy model, qua `FlagEmbedding`.

```python
from app.embedding.bge_m3 import BGEM3Embedder

emb = BGEM3Embedder()
emb.embed_documents(["đoạn 1", "đoạn 2"])   # list[EmbeddedText]
emb.embed_query("câu hỏi")                  # EmbeddedText
```

- Model được load lười ở lần gọi đầu, nên `import app` không kéo `torch`.
- **Mặc định chạy CPU**: `gemma4:e2b-it-qat` đã dùng ~2.77GB trên card 4GB, BGE-M3 fp16 cần thêm hơn 1GB nên không đủ VRAM. Chỉ đặt `EMBEDDING_DEVICE=cuda` nếu máy còn dư VRAM khi Ollama đang giữ model.
- **Internet chỉ cần một lần**: lần chạy embedding đầu tiên tải model BGE-M3 từ Hugging Face (~2GB). Sau khi đã tải, đặt `HF_HUB_OFFLINE=1` để chạy hoàn toàn offline.

## Ingestion: nạp tài liệu

Đặt file PDF vào thư mục `data/` (nội dung thư mục này không được commit), bảo đảm Qdrant đang chạy, rồi chạy **từ thư mục gốc của repo**:

```bash
python -m scripts.ingest data/ten_file.pdf
```

Lưu ý dùng `python -m scripts.ingest`, không dùng `python scripts/ingest.py` (lệnh sau lỗi `ModuleNotFoundError: No module named 'app'`).

Luồng xử lý: `parse_pdf` → chunk → embed theo batch → ghi vào Qdrant. Log INFO cho biết thời gian từng bước (parse, chunk, embed, upsert).

- Nạp lại cùng một file thì chunk cũ của file đó bị xóa trước khi ghi, số point không đổi và không còn chunk mồ côi.
- `source_file` lưu **tên file** (không phải đường dẫn), nên hai file khác thư mục nhưng trùng tên sẽ ghi đè nhau.
- Nếu parse hoặc embed lỗi thì Qdrant không bị thay đổi.
- PDF dạng scan (không có text) sẽ báo lỗi; hiện chưa hỗ trợ OCR.

Mỗi point trong collection có payload: `source_file`, `page`, `chunk_id` (dạng `<source_file>:<page>:<chunk_index>`), `text`. Có thể xem payload trên Qdrant Dashboard.

## Retrieval: tìm thử

```bash
python -m scripts.search "Câu hỏi cần tìm"
python -m scripts.search "Câu hỏi cần tìm" --top-k 3
python -m scripts.search "Câu hỏi cần tìm" --source ten_file.pdf   # chỉ tìm trong một file
python -m scripts.search "Câu hỏi cần tìm" --dense-only            # bỏ nhánh sparse để so sánh
```

Kết quả in ra thứ hạng, điểm, tên file, trang, `chunk_id` và đoạn text xem trước.

Cách hoạt động (`app/retrieval/hybrid.py`, `HybridRetriever`): dùng Query API của Qdrant với hai nhánh `prefetch` (dense và sparse, mỗi nhánh lấy `RETRIEVAL_CANDIDATES` ứng viên), sau đó gộp bằng **RRF** và trả `RETRIEVAL_TOP_K` chunk. Collection rỗng trả về danh sách rỗng. Retriever không gọi LLM.

Lưu ý: nhánh sparse ở đây là *lexical weights* mà BGE-M3 học được (learned sparse), **không phải BM25 thuần**. Điểm của chế độ hybrid là điểm RRF, không so sánh trực tiếp được với điểm cosine của `--dense-only`.

## Running Tests

```bash
pytest                           # chạy toàn bộ test
pytest -m "not integration"      # chỉ test không cần dịch vụ ngoài (không cần Qdrant, Ollama, model)
pytest -m integration            # test cần Qdrant / Ollama đang chạy
```

Test đánh dấu `integration` tự bỏ qua nếu dịch vụ tương ứng không chạy. Test dùng collection tạm có tên ngẫu nhiên và dọn sau khi chạy. Test đánh dấu `slow` dùng model BGE-M3 thật nên cần đã cài `requirements-ml.txt` và tải model.

## Project Structure

```
.
├── app/
│   ├── core/config.py                 # Settings đọc từ .env
│   ├── ingestion/
│   │   ├── pdf_parser.py              # parse_pdf: PDF -> list[PageContent]
│   │   ├── chunker.py                 # chia từng trang thành chunk
│   │   └── text_utils.py              # chuẩn hóa text (NFC)
│   ├── embedding/bge_m3.py            # BGEM3Embedder (dense + sparse)
│   ├── vectorstore/qdrant_store.py    # QdrantStore: collection, upsert, xóa theo file
│   ├── retrieval/hybrid.py            # HybridRetriever (dense + sparse, RRF)
│   ├── services/ingestion_service.py  # ingest_pdf: nối parse -> chunk -> embed -> Qdrant
│   ├── llm/ollama_client.py           # OllamaClient
│   └── routers/                       # (chưa có endpoint)
├── scripts/
│   ├── ingest.py                      # CLI nạp một PDF
│   └── search.py                      # CLI tìm thử
├── tests/                             # pytest
├── data/                              # chứa PDF đầu vào (không commit)
├── docker-compose.yml                 # Qdrant
├── pytest.ini
├── requirements.txt
├── requirements-ml.txt                # FlagEmbedding (torch)
└── .env.example
```
