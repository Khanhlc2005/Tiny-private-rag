import unicodedata


def normalize_text(text: str) -> str:
    """Chuẩn hóa về NFC cho tiếng Việt"""
    return unicodedata.normalize("NFC", text)