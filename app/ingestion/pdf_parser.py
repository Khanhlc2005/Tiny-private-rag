import logging
import re
from pathlib import Path
from typing import cast

import pymupdf
from pydantic import BaseModel

from app.ingestion.text_utils import normalize_text

logger = logging.getLogger(__name__)

_TRAILING_WS = re.compile(r"[ \t]+\n")
_BLANK_LINES = re.compile(r"\n{3,}")

class PageContent(BaseModel):
    source_file: str
    page: int
    text: str


class PDFParseError(ValueError):
    """Lỗi khi đọc PDF"""

def clean_text(text: str) -> str:
    """ NFC -> thống nhất kiểu xuống dòng -> xóa khoảng trắng -> gộp dòng thừa trống -> cắt đầu cuối"""
    text = normalize_text(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_WS.sub("\n", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()

def parse_pdf(file_path: str | Path) -> list[PageContent]:
    """Đọc PDF, trả về danh sách PageContent (mỗi trang mỗi một phần tư)
    
    source_file chỉ là tên file (không gồm thư mục). Hai file trùng tên ở
    thư mục khác nhau sẽ có cùng source_file; việc phân biệt được xử lý ở
    quy ước lưu file upload (issue API tài liệu).
    """
    path = Path(file_path)

    if not path.is_file():
        raise PDFParseError(f"Không tìm thấy file: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise PDFParseError(f"File không phải PDF: {file_path}")

    try:
        doc = pymupdf.open(path)
    except pymupdf.FileDataError as e:
        raise PDFParseError(f"Không đọc được file PDF: {file_path}") from e

    with doc:
        if doc.needs_pass:
            raise PDFParseError(f"File PDF cần mật khẩu: {file_path}")

        page_contents = []
        skipped: list[int] = []
        for page_num in range(len(doc)):
            raw_text = cast(str, doc[page_num].get_text("text", sort=True))
            text = clean_text(raw_text)
            if not text:
                skipped.append(page_num + 1)
                continue
            page_contents.append(
                PageContent(source_file=path.name, page=page_num + 1, text=text)
            )

    if skipped:
        logger.warning(
            "%s: Bỏ qua %d/%d trang không có text (có thể là cần OCR): %s",
            path.name, len(skipped), len(skipped) + len(page_contents), skipped
        )
    
    return page_contents