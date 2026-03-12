# =========================
# 🧼 Normalization Utilities
# =========================

def normalize_name(name: str) -> str:
    """
    Lowercase and strip punctuation for consistent name matching.
    """
    return name.lower().strip().replace(".", "").replace("-", " ")
