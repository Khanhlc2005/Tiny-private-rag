from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:e2b-it-qat"
    ollama_num_predict: int = 512

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "tiny_rag_chunks"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()