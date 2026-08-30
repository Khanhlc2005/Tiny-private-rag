# Tiny-Private RAG

Hệ thống RAG offline, chạy local hoàn toàn — không cần Internet, phù hợp tài liệu nội bộ bí mật.

## Tech Stack

- LLM: gemma4:e2b-it-qat qua Ollama (--think=false)
- Vector DB: Qdrant (Docker)
- Embedding: BAAI/bge-m3 (dense + sparse, RRF fusion)
- Backend: FastAPI + Pydantic + SQLModel
- UI: Streamlit
- Evaluation: RAGAS
- Observability: Langfuse (self-hosted)=
