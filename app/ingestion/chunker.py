from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.ingestion.pdf_parser import PageContent
from app.ingestion.text_utils import normalize_text

SEPARATOR = ["\n\n", "\n", ". ", "? ", "! ", " "]

class Chunk(BaseModel):
    source_file: str
    page: int
    chunk_id: str
    text: str


def chunk_pages(
    pages: list[PageContent],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None
) -> list[Chunk]:
    """Cắt riêng từng trang"""
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=SEPARATOR,
    )

    chunks: list[Chunk] = []
    for page in pages:
        pieces = splitter.split_text(normalize_text(page.text))
        pieces = [p for p in pieces if p.strip()]

        for order, text in enumerate(pieces, start=1):
            chunks.append(
                Chunk(
                    source_file=page.source_file,
                    page=page.page,
                    chunk_id=f"{page.source_file}::p{page.page}::c{order}",
                    text=text,
                )
            )
    return chunks