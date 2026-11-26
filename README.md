# Tiny-Private RAG

Hệ thống RAG (Retrieval-Augmented Generation) chạy **local/offline** — không cần Internet khi vận hành (chỉ cần Internet một lần để tải model embedding), phù hợp để hỏi đáp trên tài liệu nội bộ mà không gửi dữ liệu ra ngoài.

## Overview

Dự án xây dựng một pipeline RAG nhỏ gọn trên máy cá nhân: đọc PDF, lập chỉ mục vào vector database, truy xuất đoạn liên quan và để LLM local sinh câu trả lời.

Trạng thái hiện tại (sau Week 2):

- **Đã có**: nạp PDF (parse → chunk → embed → Qdrant) và tìm đoạn liên quan bằng hybrid search (dense + sparse, RRF).
- **Chưa có**: ghép kết quả truy xuất vào prompt để LLM trả lời, API FastAPI, giao diện. `OllamaClient` đã có nhưng chưa nối vào pipeline.

## Tech Stack

| Thành phần    | Công nghệ                                        | Trạng thái                           |
| ------------- | ------------------------------------------------ | ------------------------------------ |
| LLM           | `gemma4:e2b-it-qat` qua Ollama (`--think=false`) | Đã có client (chưa nối vào pipeline) |
| PDF parser    | PyMuPDF                                          | Đã có                                |
| Chunking      | `langchain-text-splitters`                       | Đã có                                |
| Embedding     | BAAI/bge-m3 (dense + sparse)                     | Đã có                                |
| Vector DB     | Qdrant v1.19.1 (Docker)                          | Đã có                                |
| Retrieval     | Hybrid search: dense + sparse, fusion RRF        | Đã có                                |
| Backend       | FastAPI + Pydantic + SQLModel                    | Dự kiến (chưa có endpoint)           |
| UI            | Streamlit                                        | Dự kiến                              |
| Evaluation    | RAGAS                                            | Dự kiến                              |
| Observability | Langfuse (self-hosted)                           | Dự kiến                              |

## Requirements

- Python 3.12
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

Biến môi trường trong `.env` (bỏ trống thì dùng giá trị mặc định):

| Biến                    | Mặc định                 | Ý nghĩa                                         |
| ----------------------- | ------------------------ | ----------------------------------------------- |
| `OLLAMA_BASE_URL`       | `http://localhost:11434` | Địa chỉ Ollama server                           |
| `OLLAMA_MODEL`          | `gemma4:e2b-it-qat`      | Model sinh câu trả lời                          |
| `OLLAMA_NUM_PREDICT`    | `512`                    | Số token tối đa mỗi lần sinh                    |
| `QDRANT_URL`            | `http://localhost:6333`  | Địa chỉ Qdrant                                  |
| `QDRANT_COLLECTION`     | `tiny_rag_chunks`        | Tên collection chứa chunk                       |
| `CHUNK_SIZE`            | `800`                    | Kích thước tối đa mỗi chunk (ký tự)             |
| `CHUNK_OVERLAP`         | `100`                    | Số ký tự chồng lấn giữa hai chunk liền kề       |
| `EMBEDDING_MODEL`       | `BAAI/bge-m3`            | Model embedding                                 |
| `EMBEDDING_DEVICE`      | `cpu`                    | Thiết bị chạy embedding                         |
| `EMBEDDING_BATCH_SIZE`  | `16`                     | Batch size khi embed                            |
| `RETRIEVAL_TOP_K`       | `5`                      | Số chunk trả về mỗi lần tìm                     |
| `RETRIEVAL_CANDIDATES`  | `20`                     | Số ứng viên mỗi nhánh (dense, sparse) trước RRF |
| `RAG_MAX_CONTEXT_CHARS` | `5000`                   | Số ký tự tối đa của context đưa vào prompt      |

## Running Qdrant

```bash
docker compose up -d
docker compose ps                # container tinyrag_qdrant phải ở trạng thái running
```

- Dashboard: <http://localhost:6333/dashboard>
- REST API: `curl http://localhost:6333/collections`
- gRPC: cổng `6334`

Dữ liệu lưu ở `./qdrant_storage`. Dừng dịch vụ: `docker compose down`.

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

`BGEM3Embedder` (`app/embedding/bge_m3.py`) sinh **dense (1024 chiều)** và **sparse** từ một lần chạy model, qua `FlagEmbedding`.

```python
from app.embedding.bge_m3 import BGEM3Embedder

emb = BGEM3Embedder()
emb.embed_documents(["đoạn 1", "đoạn 2"])   # list[EmbeddedText]
emb.embed_query("câu hỏi")                  # EmbeddedText
```

- Model được load ở lần gọi đầu, nên `import app` không kéo `torch`.
- **Mặc định chạy CPU**: `gemma4:e2b-it-qat` đã dùng ~2.77GB trên card 4GB, BGE-M3 fp16 cần thêm hơn 1GB nên không đủ VRAM. Chỉ đặt `EMBEDDING_DEVICE=cuda` nếu còn dư VRAM khi Ollama đang giữ model.
- **Internet chỉ cần một lần**: lần chạy đầu tải BGE-M3 từ Hugging Face (~2GB). Sau đó đặt `HF_HUB_OFFLINE=1` để chạy offline hoàn toàn.

## Ingestion: nạp tài liệu

Đặt PDF vào `data/` (nội dung thư mục này không được commit) và bảo đảm Qdrant đang chạy.

Luồng xử lý: `parse_pdf` → chunk → embed theo batch → ghi vào Qdrant. Log INFO cho biết thời gian từng bước.

- Nạp lại cùng một file thì chunk cũ bị xóa trước khi ghi: số point không đổi, không còn chunk mồ côi.
- `source_file` lưu **tên file**, nên hai file trùng tên (khác thư mục) sẽ ghi đè nhau.
- Parse hoặc embed lỗi thì Qdrant không bị thay đổi.
- PDF dạng scan (không có text) báo lỗi; chưa hỗ trợ OCR.

Payload mỗi point: `source_file`, `page`, `chunk_id` (`<source_file>:<page>:<chunk_index>`), `text`. Có thể xem trên Qdrant Dashboard.

## Retrieval: tìm thử

`HybridRetriever` (`app/retrieval/hybrid.py`) dùng Query API của Qdrant: hai nhánh `prefetch` (dense, sparse; mỗi nhánh lấy `RETRIEVAL_CANDIDATES` ứng viên), gộp bằng **RRF**, trả `RETRIEVAL_TOP_K` chunk. Collection rỗng trả về danh sách rỗng. Retriever không gọi LLM.

Nhánh sparse là _lexical weights_ do BGE-M3 học (learned sparse), **không phải BM25 thuần**. Điểm hybrid là điểm RRF.

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
│   ├── services/ingestion_service.py  # ingest_pdf: parse -> chunk -> embed -> Qdrant
│   ├── llm/ollama_client.py           # OllamaClient
│   └── routers/                       # (chưa có endpoint)
├── tests/                             # pytest
├── data/                              # PDF đầu vào (không commit)
├── docker-compose.yml                 # Qdrant
├── pytest.ini
├── requirements.txt
├── requirements-ml.txt                # FlagEmbedding (torch)
└── .env.example
```
