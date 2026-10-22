"""Orus-OCR vocabulary (Latin + Bengali block + typographic marks) and label normalisation; no torch needed."""
import unicodedata

# ---------------- vocab: blank = 0, class i+1 = CHARS[i] ----------------
_EXTRA = "‘’“”–—•…°×÷±©®£€₹¢।॥"


def _bn_chars():
    out = []
    for c in range(0x0980, 0x0A00):
        ch = chr(c)
        if unicodedata.category(ch) not in ("Cn",) and not unicodedata.category(ch).startswith(("C", "Z")):
            out.append(ch)
    return out


CHARS = [chr(c) for c in range(0x20, 0x7F)] + _bn_chars() + list(_EXTRA)
IDX = {c: i + 1 for i, c in enumerate(CHARS)}
_MAP = str.maketrans({"‌": "", "‍": "", " ": " ", "\t": " ", "’": "’", "`": "'",
                      "৷": "।", "|": "|", "୧": "", "−": "-", "‐": "-", "‑": "-"})


def canon(t):
    t = unicodedata.normalize("NFC", t).translate(_MAP)
    # Bengali: decompose the 3 nukta letters that NFC keeps split anyway; keep NFC output as is
    return " ".join(t.split())


def encode(t):
    """Label -> ids, or None when a char is out of vocab (line dropped, never altered)."""
    t = canon(t)
    ids = [IDX.get(c) for c in t]
    return None if not t or None in ids else ids


def decode(ids):
    """Greedy CTC collapse of a per-step id sequence."""
    out, prev = [], 0
    for i in ids:
        if i != prev and i != 0:
            out.append(CHARS[i - 1])
        prev = i
    return "".join(out)
