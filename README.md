# Tiny-Private RAG

Hệ thống RAG (Retrieval-Augmented Generation) chạy **hoàn toàn local/offline** — không cần Internet khi vận hành, phù hợp để hỏi đáp trên tài liệu nội bộ, bí mật mà không gửi dữ liệu ra dịch vụ bên ngoài.

## Overview

Bài toán: nhiều tổ chức có tài liệu nội bộ không thể đưa lên các dịch vụ LLM đám mây. Dự án này xây dựng một pipeline RAG nhỏ gọn chạy trên máy cá nhân: đọc tài liệu PDF, lập chỉ mục vào vector database, truy xuất đoạn liên quan và để LLM local sinh câu trả lời.

- Đầu vào hiện tại: **PDF**, được đọc bằng **PyMuPDF**.
- LLM: `gemma4:e2b-it-qat` chạy qua **Ollama** (tắt chế độ thinking).
- Vector DB: **Qdrant** chạy bằng Docker.

## Tech Stack

| Thành phần | Công nghệ | Trạng thái |
|---|---|---|
| LLM | `gemma4:e2b-it-qat` qua Ollama (`--think=false`) | Đã có client |
| PDF parser | PyMuPDF | Đã có |
| Vector DB | Qdrant (Docker) | Chạy được, chưa tích hợp |
| Embedding | BAAI/bge-m3 (dense + sparse, RRF fusion) | Dự kiến |
| Backend | FastAPI + Pydantic + SQLModel | Dự kiến (mới có skeleton) |
| UI | Streamlit | Dự kiến |
| Evaluation | RAGAS | Dự kiến |
| Observability | Langfuse (self-hosted) | Dự kiến |

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

cp .env.example .env             # chỉnh nếu cần
```

Biến môi trường trong `.env`:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Địa chỉ Ollama server |
| `OLLAMA_MODEL` | `gemma4:e2b-it-qat` | Model dùng để sinh câu trả lời |
| `OLLAMA_NUM_PREDICT` | `512` | Số token tối đa mỗi lần sinh |

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

## Running Tests

```bash
pytest                           # chạy toàn bộ test
pytest -m "not integration"      # bỏ qua test cần Ollama đang chạy
```

Test đánh dấu `integration` sẽ tự bỏ qua nếu Ollama server không chạy.

## Project Structure

```
.
├── app/
│   ├── core/config.py           # Settings đọc từ .env
│   ├── ingestion/
│   │   ├── pdf_parser.py        # parse_pdf: PDF -> list[PageContent]
│   │   └── text_utils.py        # chuẩn hóa text (NFC)
│   ├── llm/ollama_client.py     # OllamaClient.chat
│   ├── routers/                 # (chưa có endpoint)
│   ├── services/                # (chưa có)
│   └── main.py                  # FastAPI entrypoint (skeleton)
├── tests/                       # pytest
├── ui/                          # Streamlit (chưa triển khai)
├── docs/
├── docker-compose.yml           # Qdrant
├── requirements.txt
└── .env.example
```
