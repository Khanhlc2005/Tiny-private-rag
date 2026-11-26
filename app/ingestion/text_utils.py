import unicodedata

_FIXES = str.maketrans({"\u01a3": "ư", "\u01a2": "Ư"})

def normalize_text(text: str) -> str:
    """Chuẩn hóa về NFC cho tiếng Việt"""
    return unicodedata.normalize("NFC", text.translate(_FIXES))