import logging
import unicodedata

import pymupdf
import pytest

from app.ingestion.pdf_parser import PDFParseError, clean_text, parse_pdf
from app.ingestion.text_utils import normalize_text


def make_pdf(path, pages: list[str]):
    """Tạo PDF thật; chuỗi rỗng = trang trống."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()
    return path


# ---------- normalize_text ----------

def test_normalize_nfd_to_nfc():
    nfc = "ệ"
    nfd = unicodedata.normalize("NFD", nfc)

    assert nfc != nfd
    assert normalize_text(nfd) == nfc


def test_nfd_document_matches_nfc_query():
    doc = unicodedata.normalize("NFD", "Hệ thống tìm kiếm")
    assert "hệ thống" not in doc.lower()
    assert "hệ thống" in normalize_text(doc).lower()


# ---------- clean_text ----------

def test_clean_text_collapses_blank_lines_and_strips():
    assert clean_text("  a  \n\n\n\nb\n") == "a\n\nb"


def test_clean_text_keeps_single_newlines():
    # không được nối dòng: ranh giới đoạn là thông tin cho chunking
    assert clean_text("dòng 1\ndòng 2\n\nđoạn 2") == "dòng 1\ndòng 2\n\nđoạn 2"


def test_clean_text_removes_trailing_whitespace_without_joining_lines():
    assert clean_text("a \t\nb") == "a\nb"


def test_clean_text_unifies_line_endings():
    assert clean_text("a\r\nb\rc") == "a\nb\nc"


def test_clean_text_applies_nfc():
    nfd = unicodedata.normalize("NFD", "Việt Nam")
    assert clean_text(nfd) == "Việt Nam"


def test_clean_text_whitespace_only_becomes_empty():
    assert clean_text(" \n\t\n  ") == ""


# ---------- parse_pdf: thành công ----------

def test_parse_pdf_returns_page_per_page(tmp_path):
    pdf = make_pdf(tmp_path / "ok.pdf", ["Trang mot", "Trang hai"])

    result = parse_pdf(pdf)

    assert [p.page for p in result] == [1, 2]
    assert "Trang mot" in result[0].text
    assert "Trang hai" in result[1].text
    assert all(p.source_file == pdf.name for p in result)


def test_parse_pdf_accepts_str_path(tmp_path):
    pdf = make_pdf(tmp_path / "ok.pdf", ["Noi dung"])
    assert len(parse_pdf(str(pdf))) == 1

def test_parse_pdf_same_filename_in_different_dirs_gives_same_source_file(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    pdf_a = make_pdf(tmp_path / "a" / "doc.pdf", ["Noi dung A"])
    pdf_b = make_pdf(tmp_path / "b" / "doc.pdf", ["Noi dung B"])

    assert parse_pdf(pdf_a)[0].source_file == "doc.pdf"
    assert parse_pdf(pdf_b)[0].source_file == "doc.pdf"

def test_parse_pdf_skips_empty_pages_keeps_original_numbers(tmp_path, caplog):
    pdf = make_pdf(tmp_path / "gap.pdf", ["Trang mot", "", "Trang ba"])

    with caplog.at_level(logging.WARNING, logger="app.ingestion.pdf_parser"):
        result = parse_pdf(pdf)

    assert [p.page for p in result] == [1, 3]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()  # getMessage() sẽ lỗi nếu format sai số đối số
    assert "1/3" in msg
    assert "[2]" in msg


def test_parse_pdf_no_warning_when_all_pages_have_text(tmp_path, caplog):
    pdf = make_pdf(tmp_path / "ok.pdf", ["A", "B"])

    with caplog.at_level(logging.WARNING, logger="app.ingestion.pdf_parser"):
        parse_pdf(pdf)

    assert not caplog.records


def test_parse_pdf_all_pages_empty_returns_empty_list(tmp_path, caplog):
    pdf = make_pdf(tmp_path / "blank.pdf", ["", ""])

    with caplog.at_level(logging.WARNING, logger="app.ingestion.pdf_parser"):
        assert parse_pdf(pdf) == []

    assert "2/2" in caplog.records[0].getMessage()


# ---------- parse_pdf: lỗi ----------

def test_parse_pdf_missing_file(tmp_path):
    with pytest.raises(PDFParseError, match="Không tìm thấy"):
        parse_pdf(tmp_path / "khong_ton_tai.pdf")


def test_parse_pdf_directory_is_not_a_file(tmp_path):
    with pytest.raises(PDFParseError):
        parse_pdf(tmp_path)


def test_parse_pdf_wrong_extension(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hello")

    with pytest.raises(PDFParseError, match="không phải PDF"):
        parse_pdf(f)


def test_parse_pdf_corrupt_file(tmp_path):
    f = tmp_path / "hong.pdf"
    f.write_bytes(b"day khong phai pdf")

    with pytest.raises(PDFParseError):
        parse_pdf(f)


def test_parse_pdf_empty_file(tmp_path):
    f = tmp_path / "rong.pdf"
    f.write_bytes(b"")

    with pytest.raises(PDFParseError):
        parse_pdf(f)


def test_parse_pdf_password_protected(tmp_path):
    src = make_pdf(tmp_path / "src.pdf", ["bi mat"])
    locked = tmp_path / "locked.pdf"
    doc = pymupdf.open(src)
    doc.save(
        locked,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw="123",
        owner_pw="123",
    )
    doc.close()

    with pytest.raises(PDFParseError, match="mật khẩu"):
        parse_pdf(locked)
