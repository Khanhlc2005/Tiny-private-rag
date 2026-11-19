"""CLI ingest một PDF.

Chạy từ thư mục gốc của repo:
    python -m scripts.ingest data/abc.pdf

Không chạy `python scripts/ingest.py`: khi đó thư mục gốc không nằm trong
sys.path nên `from app...` sẽ lỗi ModuleNotFoundError.
"""
import argparse
import logging
import sys

from app.ingestion.pdf_parser import PDFParseError
from app.services.ingestion_service import IngestError, ingest_pdf


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest một file PDF vào Qdrant")
    parser.add_argument("pdf", help="Đường dẫn tới file PDF")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        result = ingest_pdf(args.pdf)
    except (PDFParseError, IngestError, RuntimeError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1

    print(
        f"Đã ingest {result.source_file}: "
        f"{result.pages_with_text} trang có text, {result.chunk_count} chunk"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())