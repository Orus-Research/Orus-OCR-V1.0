"""Training data, step 2: synthetic recognizer lines.

  python ocr_synth.py words                 -> data/ocr/words/   real handwritten word crops (bank for stitching)
  ORUS_H=48 python ocr_synth.py lines <kind> <n> <split> [shard]
        kind = font    : text rendered with print + handwriting-style fonts (BN, EN, code-mixed, lists, Rx)
               stitch  : real handwritten word crops (BN-HTRd, BanglaWriting, IIIT, GNHK, IMGUR5K, Rx) stitched
                         into BN / EN / mixed lines
               num     : numbers, dates, prices, phone numbers, arithmetic (BN + EN digits)
               receipt : receipt rows (items, qty, prices, totals, dot leaders; thermal / dot-matrix fonts)
               rx      : printed prescription lines
               book    : book / newspaper lines (justified, small sizes, scan noise, bleed-through)
               hdigit  : handwritten digits / arithmetic from MNIST digits
        8-12% of each kind is heavily degraded.
        -> data/ocr/rec48/synth_<kind>_<split>_<shard>.{blob,meta,lab}.npy   (ink map 3*ORUS_H x W, text, hand)
  python ocr_synth.py demo                  -> self-check + sample sheet
Each sample is 3 stacked lines (context above / target / below) cut with Indra's crop_line(window=3),
the same rule real crops use. Splits: train/val use train fonts + train-split words; test uses held-out ones.
"""
import json
import os
import pickle
import random
import re
import sys
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import render_printed as RP
from vocab import canon, encode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root: data/, runs/, models/
OUT = f"{ROOT}/data/ocr"
BN_DIG = "০১২৩৪৫৬৭৮৯"
IS_BN = re.compile(r"[ঀ-৿]")

# ---------------- text ----------------
_WORDS = {}


def corpus(lang):
    return RP.text(lang)


_PUNCT = re.compile(r" ([,.;:!?%)\]'’]|n't|'s)")


def words_of(lang, r):
    t = r.choice(corpus(lang))
    if lang == "en":                                   # wikitext tokenization: "word , word" -> "word, word"
        t = _PUNCT.sub(r"\1", t).replace("( ", "(").replace(" @-@ ", "-")
    return t.split()


def bn_digits(s):
    return s.translate(str.maketrans("0123456789", BN_DIG))


MEDS = ["Napa", "Ace", "Seclo", "Losectil", "Fexo", "Monas", "Azithrocin", "Ciprocin", "Maxpro", "Pantonix",
        "Amodis", "Flagyl", "Histacin", "Tusca", "Ceevit", "Calbo-D", "Neuro-B", "Omeprazole", "Paracetamol",
        "Metformin", "Amlodipine", "Montelukast", "Esomeprazole", "Cefixime", "Domperidone", "Atorvastatin"]


def special_text(r):
    """Numbers, dates, prices, phone numbers, prescriptions: the strings the corpora rarely contain."""
    d = lambda a, b: str(r.randint(a, b))
    k = r.randint(0, 6)
    if k == 0:
        t = f"{r.choice(['Tab.', 'Cap.', 'Syp.', 'Inj.', 'Tab', 'T.'])} {r.choice(MEDS)} {r.choice(['5', '10', '20', '40', '250', '500'])}{r.choice(['mg', ' mg', 'mg', 'ml'])} " \
            f"{r.choice(['1+0+1', '1+1+1', '0+0+1', '1-0-1', '1+0+0', '½+0+½'])}"
        if r.random() < 0.5:
            t += " " + r.choice([f"({bn_digits(d(3, 30))} দিন)", f"x {d(3, 30)} days", "খাবার পরে", "খাবার আগে", "after meal", "--- চলবে"])
    elif k == 1:
        t = f"{d(1, 28)}/{d(1, 12)}/{d(1990, 2026)}"
    elif k == 2:
        t = r.choice(["৳ ", "Tk ", "Tk. ", "৳", "$"]) + d(5, 99999) + r.choice(["", "/-", ".00", " টাকা"])
    elif k == 3:
        t = f"01{d(3, 9)}{d(10, 99)}-{d(100000, 999999)}"
    elif k == 4:
        t = f"{r.choice(['BP', 'Pulse', 'Wt', 'Temp', 'Age'])}: {d(60, 180)}{r.choice(['/' + d(50, 110), ' kg', '°F', ' yrs', ''])}"
    elif k == 5:
        t = " ".join(r.choice(["Rx", "Dx:", "C/C:", "O/E:", "Adv:", "Investigation:", "CBC", "RBS", "X-ray", "ECG"]) for _ in range(r.randint(1, 4)))
    else:
        t = f"{d(1, 999)} {r.choice(['x', '+', '-', '=', '÷', '×'])} {d(1, 99)} = {d(1, 9999)}"
    return bn_digits(t) if r.random() < 0.3 else t


def mixed(r, base):
    """Code-mixed line: a BN fragment with EN words swapped in, or EN with BN words."""
    other = "en" if base == "bn" else "bn"
    w = words_of(base, r)
    ow = words_of(other, r)
    p = r.uniform(0.15, 0.45)
    out = [r.choice(ow) if r.random() < p else x for x in w]
    if not any(bool(IS_BN.search(x)) != (base == "bn") for x in out):
        out[r.randrange(len(out))] = r.choice(ow)
    return out


LIST = ["{}.", "{})", "({})", "{}।", "{}:"]


def list_prefix(r):
    i = r.randint(1, 20)
    k = r.random()
    if k < 0.35:
        n = str(i)
    elif k < 0.6:
        n = bn_digits(str(i))
    elif k < 0.75:
        n = "কখগঘঙচছজঝঞটঠ"[i % 12]
    elif k < 0.88:
        n = "abcdefghij"[i % 10]
    else:
        n = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii"][i % 8]
    return r.choice(["•", "-", "*", "➤"]) if r.random() < 0.15 else r.choice(LIST).format(n)


def gen_text(r):
    """One target line of 1..12 tokens; language mix ~ bn 35%, en 30%, mixed 35%."""
    k = r.random()
    if k < 0.1:
        toks = special_text(r).split()
    else:
        lang = r.choices(["bn", "en", "mixbn", "mixen"], [0.35, 0.3, 0.22, 0.13])[0]
        toks = mixed(r, lang[3:]) if lang.startswith("mix") else words_of(lang, r)
        n = r.choice([1, 1, 2, 3]) if r.random() < 0.2 else r.randint(3, 12)
        s = r.randint(0, max(0, len(toks) - n))
        toks = toks[s:s + n]
    if r.random() < 0.15:
        toks = [list_prefix(r)] + toks
    t = canon(" ".join(toks))
    return t if t and encode(t) is not None and len(t) <= 120 else gen_text(r)


# ---------------- fonts ----------------
_FONTS = None


def font_pool():
    """All fonts (RP print/hand/bn + bn_fonts_extra + Google Fonts) with cmap reduced to our vocab;
    cached in data/ocr/font_pool.pkl (rebuilt when the gfonts dir changes)."""
    global _FONTS
    if _FONTS is None:
        import glob, hashlib
        from vocab import CHARS
        cache = f"{OUT}/font_pool.pkl"
        n_gf = len(glob.glob(f"{ROOT}/data/raw/gfonts/*"))
        try:
            c = pickle.load(open(cache, "rb"))
            if c["n_gf"] == n_gf:
                _FONTS = c["fonts"]
                return _FONTS
        except Exception:
            pass
        vocab = {ord(c) for c in CHARS}
        fl = RP.fonts()
        seen = {f["path"] for f in fl}
        from fontTools.ttLib import TTFont
        for p in sorted(glob.glob(f"{ROOT}/data/raw/bn_fonts_extra/**/*.[tTo][tTt][fF]", recursive=True)):
            if p in seen:
                continue
            try:
                cmap = TTFont(p, fontNumber=0, lazy=True).getBestCmap()
            except Exception:
                continue
            if sum(c in cmap for c in RP.BN_RANGE) <= 60:
                continue                                   # legacy (Bijoy) fonts map Bangla to ASCII codes
            has_en = all(ord(c) in cmap for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            split = "test" if int(hashlib.md5(os.path.basename(p).encode()).hexdigest(), 16) % 100 < 15 else "train"
            hand = bool(re.search(r"hand|likhan|script|cursive|brush|(?<!o)pen", os.path.basename(p), re.I))  # not "OpenType"
            fl.append(dict(path=p, kind="hand" if hand else "bn", bn=True, en=has_en, split=split, cmap=set(cmap) & vocab))
        fl += gfont_list(fl)
        out = []
        for f in fl:
            if "cmap" not in f:
                try:
                    f["cmap"] = set(TTFont(f["path"], fontNumber=0, lazy=True).getBestCmap()) & vocab
                except Exception:
                    continue
            if f.get("cat"):                               # Google Fonts: coverage from the cmap
                f["en"] = all(ord(c) in f["cmap"] for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,")
                f["bn"] = f["bn"] and sum(c in f["cmap"] for c in RP.BN_RANGE) > 60
                if not (f["en"] or f["bn"]):
                    continue
            out.append(f)
        _FONTS = out
        tmp = f"{cache}.{os.getpid()}"
        pickle.dump(dict(n_gf=n_gf, fonts=out), open(tmp, "wb"))
        os.replace(tmp, cache)
    return _FONTS


_NOGF = re.compile(r"barcode|redacted|flow|blank|emoji|icons|symbols|wavefont|linefont|material|notomusic|"
                   r"notosanssymbols|kaisei|zcool|sedgwick|bahianita|libre(?:barcode)", re.I)
DOTS = re.compile(r"dotgothic|doto|vt323|silkscreen|pixelify|sharetechmono|majormono|jersey|tiny5|micro5|bitcount|"
                  r"workbench|sixtyfour|courier|cutivemono|syne ?mono|xanhmono|novamono|fragmentmono|spacemono", re.I)


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", re.split(r"[-\[_]", name)[0].lower())


def gfont_list(known):
    """Google Fonts (data/raw/gfonts/<family>__<file>.ttf + <family>__METADATA.pb) -> font dicts.
    kind: HANDWRITING -> hand, else print (bn when the family has the bengali subset); cat = METADATA category.
    Split by family (md5 % 100 < 15 -> test); a family we already had keeps the split of its old files (a family
    with any held-out file stays held out), so the held-out (unseen) fonts stay the same across versions."""
    import glob, hashlib
    d = f"{ROOT}/data/raw/gfonts"
    old = {}
    for f in known:
        k = _norm(os.path.basename(f["path"]))
        old[k] = "test" if f["split"] == "test" or old.get(k) == "test" else "train"
    seen = {os.path.basename(f["path"]) for f in known}
    out = []
    for meta in sorted(glob.glob(f"{d}/*__METADATA.pb")):
        fam = os.path.basename(meta).split("__")[0]
        txt = open(meta, encoding="utf-8", errors="ignore").read()
        cat = (re.search(r'category: "(\w+)"', txt) or [None, "SANS_SERIF"])[1]
        if _NOGF.search(fam) or "SYMBOLS" in cat:
            continue
        split = old.get(fam) or ("test" if int(hashlib.md5(fam.encode()).hexdigest(), 16) % 100 < 15 else "train")
        bn = 'subsets: "bengali"' in txt
        for p in sorted(glob.glob(f"{d}/{glob.escape(fam)}__*.ttf")):
            if p.split("__", 1)[1] in seen:
                continue
            kind = "hand" if cat == "HANDWRITING" else "bn" if bn else "print"
            out.append(dict(path=p, kind=kind, bn=bn, en=True, split=split, cat=cat, fam=fam,
                            dots=bool(DOTS.search(fam)) or cat == "MONOSPACE", w=0.3 if cat == "DISPLAY" else 1.0))
    return out


def _fonts(split):
    return [f for f in font_pool() if f["split"] == split]


def wchoice(r, fl):
    return r.choices(fl, [f.get("w", 1.0) for f in fl])[0]


def pick_fonts(r, split, en_filter=None, unseen_only=False, print_only=False):
    fl = _fonts(split)
    if unseen_only:                                   # unseen-font test: readable categories only (no DISPLAY)
        fl = [f for f in fl if f.get("cat") != "DISPLAY"]
    if print_only:                                    # printed-text sets: no handwriting-style fonts
        fl = [f for f in fl if f["kind"] != "hand"]
    bn = wchoice(r, [f for f in fl if f["bn"]])
    en_pool = [f for f in fl if f["en"]]
    if en_filter:
        en_pool = [f for f in en_pool if en_filter(f)] or en_pool
    elif r.random() < 0.5:
        en_pool = [f for f in en_pool if f["kind"] == "hand"] or en_pool
    en = bn if bn["en"] and not en_filter and r.random() < 0.5 else wchoice(r, en_pool)
    return bn, en


def runs(t):
    """Split text into script runs: [(is_bn, str)]; spaces/punct stick to the run before them."""
    out = []
    for ch in t:
        b = bool(IS_BN.search(ch)) if ch.isalpha() or IS_BN.search(ch) else (out[-1][0] if out else False)
        if out and out[-1][0] == b:
            out[-1][1] += ch
        else:
            out.append([b, ch])
    return out


_TT = {}


def tt(path, size, bn):
    k = (path, size, bn)
    if k not in _TT:
        if len(_TT) > 300:
            _TT.clear()
        _TT[k] = ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM if bn else ImageFont.Layout.BASIC)
    return _TT[k]


def draw_text(dr, x, y, t, fbn, fen, size, space):
    """Draw mixed-script text word by word, run by run (explicit space width, so a label space is always
    visible). Falls back to the BN font when the EN font lacks a glyph. Returns (end x, fonts used)."""
    used = set()
    for wd in t.split(" "):
        for b, s in runs(wd):
            f = fbn if b else fen
            if not b and not all(ord(c) in f["cmap"] for c in s):
                f = fbn
            font = tt(f["path"], size, True)
            dr.text((x, y), s, font=font, fill=255)
            x += font.getlength(s)
            used.add(f["kind"])
        x += space
    return x, used


def text_ok(t, fbn, fen):
    for b, s in runs(t):
        f = fbn if b else fen
        if not all(ord(c) in f["cmap"] or ord(c) in fbn["cmap"] or c == " " for c in s):
            return False
    return True


# ---------------- composition ----------------
def warp(r, a, strength):
    """Elastic + shear distortion of an alpha layer (handwriting-like wobble for font text)."""
    H, W = a.shape
    sh = r.uniform(-0.3, 0.3) * strength
    M = np.float32([[1, sh, -sh * H / 2], [0, 1, 0]])
    a = cv2.warpAffine(a, M, (W, H), borderValue=0)
    if strength > 0.3:
        n = np.random.default_rng(r.randint(0, 1 << 30))
        s = max(4, int(H / 8))
        dx = cv2.resize(n.normal(0, 1, (H // s + 2, W // s + 2)).astype(np.float32), (W, H)) * strength * 2.0
        dy = cv2.resize(n.normal(0, 1, (H // s + 2, W // s + 2)).astype(np.float32), (W, H)) * strength * 2.0
        xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        a = cv2.remap(a, xx + dx, yy + dy, cv2.INTER_LINEAR, borderValue=0)
    return a


def compose(r, alpha, box, heavy=False, look=None):
    """Alpha layer (0..1 ink) -> degraded gray canvas (uint8). heavy: strong noise (noisy-photo / bad-scan set).
    look: None | "thermal" (faded receipt print) | "scan" (book / newspaper page, bleed-through)."""
    H, W = alpha.shape
    th = max(4, box[3] - box[1])
    bg = RP.paper(r, W, H, max(8, int(th * r.uniform(1.4, 2.2)))) if r.random() < 0.6 and look is None else \
        np.full((H, W), r.randint(215, 255), np.float32)
    ink = r.randint(0, 60) if r.random() < 0.8 else r.randint(60, 140)
    if look == "thermal":
        ink = r.randint(40, 150)
        bg = np.full((H, W), r.randint(205, 250), np.float32)
        alpha = alpha * np.clip(np.linspace(r.uniform(0.6, 1), r.uniform(0.6, 1), W, dtype=np.float32)[None] *
                                (1 - 0.25 * (np.random.default_rng(r.randint(0, 1 << 30)).random((H, 1)) < 0.06)), 0, 1)
    elif look == "scan":
        bg = np.full((H, W), r.randint(185, 245), np.float32)          # yellowed / newsprint paper (gray)
        if r.random() < 0.5:                                           # bleed-through of the other side
            ghost = cv2.GaussianBlur(alpha[:, ::-1], (0, 0), r.uniform(0.8, 2.0))
            ghost = np.roll(ghost, r.randint(-th, th), 0)
            bg = bg - ghost * r.uniform(15, 50)
        if r.random() < 0.5:                                           # paper grain
            bg = bg + cv2.GaussianBlur(np.random.default_rng(r.randint(0, 1 << 30)).normal(0, 1, (H, W)).astype(np.float32),
                                       (0, 0), 1.5) * r.uniform(4, 12)
    if r.random() < 0.3:                                   # uneven light
        g = np.linspace(r.uniform(0.75, 1.0), r.uniform(0.75, 1.0), W, dtype=np.float32)[None]
        bg = bg * g
    img = bg * (1 - alpha) + ink * alpha
    if r.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), r.uniform(0.3, 1.2))
    img = img + np.random.default_rng(r.randint(0, 1 << 30)).normal(0, r.uniform(0, 8), img.shape)
    if heavy:
        img = heavy_noise(r, img, th)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if r.random() < 0.5:
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, r.randint(30, 95)])[1], 0)
    if r.random() < 0.1 and not heavy:
        img = np.where(img < r.randint(100, 170), 0, 255).astype(np.uint8)
    return img


def heavy_noise(r, img, th):
    """2-4 strong degradations: sensor noise, salt & pepper, motion / defocus blur, low contrast, shadow,
    resolution loss, strike lines and stains."""
    H, W = img.shape
    n = np.random.default_rng(r.randint(0, 1 << 30))
    for k in r.sample(range(8), r.randint(2, 3)):
        if k == 0:
            img = img + n.normal(0, r.uniform(12, 28), img.shape)
        elif k == 1:
            m = n.random(img.shape)
            p = r.uniform(0.003, 0.02)
            img = np.where(m < p / 2, 0, np.where(m > 1 - p / 2, 255, img))
        elif k == 2:
            L = r.randint(3, max(4, int(th / 6)))
            kern = np.zeros((L, L), np.float32)
            kern[L // 2] = 1 / L
            kern = cv2.warpAffine(kern, cv2.getRotationMatrix2D((L / 2, L / 2), r.uniform(0, 180), 1), (L, L))
            img = cv2.filter2D(img.astype(np.float32), -1, kern / max(kern.sum(), 1e-6))
        elif k == 3:
            img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), r.uniform(0.04, 0.09) * th)
        elif k == 4:
            lo, hi = r.uniform(60, 140), r.uniform(190, 240)
            img = lo + (img / 255.0) * (hi - lo)
        elif k == 5:
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            a = r.uniform(0, np.pi)
            edge = (np.cos(a) * xx + np.sin(a) * yy) > r.uniform(0.2, 0.8) * (abs(np.cos(a)) * W + abs(np.sin(a)) * H)
            img = img * np.where(edge, r.uniform(0.55, 0.8), 1.0)
        elif k == 6:
            s = r.uniform(0.35, 0.6)
            small = cv2.resize(np.clip(img, 0, 255).astype(np.uint8), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        else:
            img = img.astype(np.float32).copy()
            for _ in range(r.randint(1, 3)):                           # pen strokes / folds / stains over the text
                if r.random() < 0.6:
                    y0, y1 = r.randint(0, H - 1), r.randint(0, H - 1)
                    cv2.line(img, (0, y0), (W - 1, y1), r.randint(20, 160), r.randint(1, max(1, th // 10)))
                else:                                                      # translucent stain
                    st = img.copy()
                    cv2.circle(st, (r.randint(0, W - 1), r.randint(0, H - 1)), r.randint(th // 2, 2 * th), r.randint(120, 200), -1)
                    a = r.uniform(0.25, 0.5)
                    img = cv2.GaussianBlur(img * (1 - a) + st * a, (0, 0), 1.0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, r.randint(15, 45)])[1], 0).astype(np.float32)


def crop(img, box):
    from indra_ocr.rec_preproc import crop_line
    from ocr_data import LINE_H
    return crop_line(img, box, height=LINE_H, window=3, max_width=2048)


def tight(a, thr=0.25):
    ys, xs = np.where(a > thr)
    return None if not len(xs) else [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


# ---------------- text of the special kinds ----------------
UNITS = ["kg", "g", "mg", "km", "m", "cm", "mm", "ml", "L", "%", "°C", "°F", "km/h", "Tk", "টাকা", "কেজি", "গ্রাম", "লিটার", "মিটার"]


def num_text(r, mode=None):
    """Numbers and arithmetic: integers / grouped thousands / decimals / fractions / units, expressions with
    + - × ÷ = ( ), BN or EN digits. mode: None (mixed), "arith" (expressions only), "digits" (no expressions)."""
    d = lambda a, b: r.randint(a, b)
    def num():
        k = r.random()
        if k < 0.35:
            return str(d(0, 999))
        if k < 0.55:
            return f"{d(1, 99999):,}" if r.random() < 0.5 else str(d(1000, 9999999))
        if k < 0.75:
            return f"{r.uniform(0, 1000):.{r.randint(1, 3)}f}"
        if k < 0.88:
            return f"{d(1, 19)}/{d(2, 20)}"
        return f"{d(1, 999)}{r.choice(['', ' '])}{r.choice(UNITS)}"
    k = {None: r.random(), "arith": 0.0, "digits": r.uniform(0.45, 1.0)}[mode]
    if k < 0.45:                                            # expression
        n = r.randint(2, 4)
        ops = [r.choice(["+", "-", "×", "÷", "+", "-", "x", "*", "/"]) for _ in range(n - 1)]
        t = num()
        for o in ops:
            t += r.choice([" ", ""]) + o + r.choice([" ", ""]) + num()
        if r.random() < 0.25:
            t = "(" + t + ")" + r.choice([" × ", " ÷ ", " + ", " - "]) + num()
        if r.random() < 0.7:
            t += r.choice([" = ", "=", " = "]) + num()
        if r.random() < 0.15:
            t = r.choice(["Ans: ", "Q.", "(i) ", "উত্তর: ", "সমাধান: ", "a) ", f"{d(1, 20)}. "]) + t
    elif k < 0.7:                                           # number runs (tables, lists of values)
        t = r.choice([" ", "  ", ", ", " | ", "; "]).join(num() for _ in range(r.randint(1, 6)))
    elif k < 0.85:
        t = special_text(r) if mode is None else num()
    else:                                                   # phone, dates, ids, money
        t = r.choice([f"+880 1{d(3, 9)}{d(10, 99)} {d(100000, 999999)}", f"{d(1, 31):02d}.{d(1, 12):02d}.{d(1950, 2030)}",
                      f"{d(1, 31)}-{d(1, 12)}-{d(1950, 2030)}", f"ID: {d(10 ** 9, 10 ** 10 - 1)}", f"৳{d(1, 99999):,}.{d(0, 99):02d}",
                      f"${r.uniform(0, 5000):.2f}", f"{d(0, 23):02d}:{d(0, 59):02d}", f"{d(1, 999)}.{d(0, 99)}%",
                      f"Roll {d(1, 200)}, Reg {d(100000, 999999)}"])
    return bn_digits(t) if r.random() < 0.35 else t


ITEMS_EN = ["Rice", "Chicken Burger", "Coke 250ml", "Mineral Water", "Coffee Latte", "Ice Tea", "French Fries", "Bread",
            "Milk 1L", "Egg (12 pcs)", "Sugar 1kg", "Soybean Oil 5L", "Toothpaste", "Shampoo", "Biscuit", "Noodles",
            "Tea Bag", "Chips", "Paneer", "Fried Rice", "Soup", "Naan", "Kacchi Biryani", "Beef Tehari", "Lassi",
            "Cheese Cake", "Choco Bun", "Hot Dog", "Pepsi", "Lemon Juice", "Salad", "Pizza Large", "Service Charge"]
ITEMS_BN = ["চাল", "ডাল", "আলু", "পেঁয়াজ", "রসুন", "চিনি", "লবণ", "সয়াবিন তেল", "ডিম", "দুধ", "মুরগি", "গরুর মাংস",
            "চা", "বিস্কুট", "সাবান", "আটা", "মসুর ডাল", "কাঁচা মরিচ", "আদা", "পানি", "মিষ্টি", "দই", "পরোটা", "ভাত"]
RC_HEAD = ["TOTAL", "Total", "SUBTOTAL", "Sub Total", "VAT 15%", "VAT (5%)", "Discount", "CASH", "Cash", "CHANGE",
           "Change", "Grand Total", "Net Amount", "Paid", "Due", "TAX", "Rounding", "মোট", "সর্বমোট", "ভ্যাট", "ছাড়",
           "নগদ", "ফেরত", "বাকি"]


def receipt_text(r):
    """One receipt row: item [qty x price] ... amount, totals, header / footer lines; dot leaders sometimes."""
    bn = r.random() < 0.35
    money = lambda: (f"{r.randint(1, 9999)}.{r.randint(0, 99):02d}" if r.random() < 0.6 else f"{r.randint(1, 99999):,}")
    k = r.random()
    if k < 0.5:
        item = r.choice(ITEMS_BN if bn else ITEMS_EN)
        q = r.randint(1, 12)
        parts = [item]
        if r.random() < 0.6:
            parts.append(r.choice([f"{q} x {money()}", f"{q}x", f"x{q}", f"{q} @ {money()}", f"{q} pcs", f"{q}"]))
        parts.append(money())
        t = parts
    elif k < 0.75:
        t = [r.choice(RC_HEAD), r.choice(["", ":", " :"]) + money() if r.random() < 0.3 else money()]
        if r.random() < 0.3:
            t = [t[0] + ":", r.choice(["Tk", "RM", "$", "৳", "BDT"]) + r.choice(["", " "]) + t[1].lstrip(":").strip()]
    else:
        d = lambda a, b: r.randint(a, b)
        t = [r.choice([f"Date: {d(1, 31):02d}/{d(1, 12):02d}/{d(2015, 2026)}", f"Time: {d(0, 23):02d}:{d(0, 59):02d}:{d(0, 59):02d}",
                       f"Invoice No: {d(10000, 999999)}", f"Table {d(1, 40)}", f"Cashier: {r.choice(['Rahim', 'Karim', 'Anna', 'Lee'])}",
                       f"Tel: 0{d(1000000000, 1999999999)}", "Thank you! Please come again", "*** CUSTOMER COPY ***",
                       f"GST Reg No: {d(100000000, 999999999)}", f"বিল নং: {bn_digits(str(d(100, 99999)))}", "ধন্যবাদ, আবার আসবেন",
                       f"Item(s): {d(1, 30)}  Qty: {d(1, 60)}", f"Card **** {d(1000, 9999)}"])]
    if bn and r.random() < 0.6:
        t = [bn_digits(x) for x in t]
    return t                                                # list of fields: left ... right


RX_HEAD = ["Dr. {} {}", "MBBS, FCPS (Medicine)", "MBBS, MD (Cardiology)", "BMDC Reg. No: A-{}", "Chamber: {} Medical Centre",
           "Patient: {} {}", "Age: {} yrs  Sex: {}", "Date: {}/{}/{}", "রোগীর নাম: {}", "বয়স: {} বছর", "তারিখ: {}/{}/{}",
           "ডাঃ {} {}", "এমবিবিএস, এফসিপিএস (মেডিসিন)", "চেম্বার: {} মেডিকেল সেন্টার", "Rx", "Adv:", "C/C:", "O/E:",
           "Follow up after {} days", "{} দিন পর দেখা করবেন", "Investigations: CBC, RBS, S. Creatinine"]
NAMES = ["Rahman", "Karim", "Hossain", "Ahmed", "Islam", "Akter", "Begum", "Chowdhury", "Smith", "Das", "Roy", "Khan"]
NAMES_BN = ["রহিম", "করিম", "হোসেন", "আহমেদ", "ইসলাম", "আক্তার", "বেগম", "চৌধুরী", "দাস", "রায়", "খান"]


def rx_text(r):
    """Printed prescription line: header fields or a drug line (special_text kind 0)."""
    if r.random() < 0.55:
        return _rx_drug(r)
    h = r.choice(RX_HEAD)
    numeric = h.startswith(("Age", "Date", "BMDC", "তারিখ", "বয়স", "Follow", "{}"))
    vals = [str(r.randint(1, 30)) if numeric else r.choice(NAMES_BN if IS_BN.search(h) else NAMES) for _ in range(h.count("{}"))]
    t = h.format(*vals)
    return bn_digits(t) if IS_BN.search(t) and r.random() < 0.7 else t


def _rx_drug(r):
    t = f"{r.choice(['Tab.', 'Cap.', 'Syp.', 'Inj.', 'Tab', 'Susp.', 'Drop'])} {r.choice(MEDS)} {r.choice(['5', '10', '20', '40', '250', '500', '625'])}" \
        f"{r.choice(['mg', ' mg', 'mg', 'ml'])} {r.choice(['1+0+1', '1+1+1', '0+0+1', '1-0-1', '1+0+0', '½+0+½', '0+1+0'])}"
    if r.random() < 0.6:
        t += " " + r.choice([f"({bn_digits(str(r.randint(3, 30)))} দিন)", f"x {r.randint(3, 30)} days", "খাবার পরে",
                             "খাবার আগে", "after meal", "before meal", "--- চলবে", f"{r.randint(1, 3)} months", "continue"])
    if r.random() < 0.2:
        t = f"{r.randint(1, 8)}. " + t
    return t


def news_text(lang, r):
    """Book / newspaper line: 5-14 consecutive words of a BN newspaper article or EN wiki paragraph."""
    c = news_corpus() if lang == "bn" and news_corpus() else None
    toks = r.choice(c).split() if c else words_of(lang, r)
    n = r.randint(5, 14)
    s = r.randint(0, max(0, len(toks) - n))
    t = canon(" ".join(toks[s:s + n]))
    return t if t and encode(t) is not None and len(t) <= 110 else news_text(lang, r)


_NEWS = None


def news_corpus():
    """Paragraphs of data/raw/text_bn_news (zabir-nabil/bangla_newspaper_dataset) -> data/text/bn_news.txt."""
    global _NEWS
    if _NEWS is None:
        out = f"{ROOT}/data/text/bn_news.txt"
        if not os.path.exists(out):
            import glob
            import pyarrow.parquet as pq
            fs = sorted(glob.glob(f"{ROOT}/data/raw/text_bn_news/data/train-*.parquet"))
            if not fs:
                return []
            pf = pq.ParquetFile(fs[0])
            col = [c for c in pf.schema_arrow.names if c in ("content", "text", "article", "body")][0]
            paras = []
            for i in range(pf.num_row_groups):
                for t in pf.read_row_group(i, columns=[col]).column(col).to_pylist():
                    for p in (t or "").split("\n"):
                        p = canon(p)
                        if 60 < len(p) < 1500:
                            paras.append(p)
                if len(paras) > 80000:
                    break
            open(out + ".tmp", "w").write("\n".join(paras))
            os.replace(out + ".tmp", out)
        _NEWS = open(out).read().split("\n")
    return _NEWS


# ---------------- font-rendered lines ----------------
def font_line(r, split, kind="font", heavy=None, unseen=False, print_only=False):
    """kind: font (general BN/EN/mixed text), num (numbers / arithmetic), receipt (rows, dot-matrix / mono /
    thermal look), rx (printed prescription), book (book / newspaper lines, small dense print, scan look).
    heavy: strong degradation (default: 12% of training lines). unseen: test fonts only from readable categories.
    Returns (gray canvas, target box, text, hand)."""
    look, align = None, False
    if kind == "receipt":
        fbn, fen = pick_fonts(r, split, en_filter=(lambda f: f.get("dots")) if r.random() < 0.6 else (lambda f: f["kind"] != "hand"),
                              unseen_only=unseen)
        fields = receipt_text(r)
        lead = r.random() < 0.25 and len(fields) > 1
        t = None
        look = "thermal" if r.random() < 0.6 else None
    elif kind == "book":
        fbn, fen = pick_fonts(r, split, en_filter=lambda f: f["kind"] != "hand" and f.get("cat") != "DISPLAY", unseen_only=unseen)
        lang = "bn" if r.random() < 0.55 else "en"
        if lang == "bn":
            fen = fbn
        t = news_text(lang, r)
        look = "scan"
    else:
        en_filter = (lambda f: f["kind"] != "hand") if kind == "rx" else None
        fbn, fen = pick_fonts(r, split, en_filter=en_filter, unseen_only=unseen, print_only=print_only)
        t = {"font": gen_text, "num": num_text, "rx": rx_text, "digits": lambda r: num_text(r, "digits"),
             "arith": lambda r: num_text(r, "arith")}[kind](r)
    if kind == "receipt":
        t = canon(" ".join(fields))
        if lead:
            t = canon(fields[0] + " " + "." * r.randint(4, 14) + " " + " ".join(fields[1:]))
        if not t or encode(t) is None:
            return None
    if not text_ok(t, fbn, fen):
        fen = fbn
        if not text_ok(t, fbn, fen):
            return None
    size = r.randint(14, 30) if kind == "book" else r.randint(18, 48)
    pitch = int(size * (r.uniform(1.12, 1.45) if kind in ("book", "receipt") else r.uniform(1.3, 2.2)))
    ctx_fn = {"font": gen_text, "num": num_text, "rx": rx_text, "digits": num_text, "arith": num_text, "receipt": lambda r: canon(" ".join(receipt_text(r))),
              "book": lambda r: news_text("bn" if fen is fbn else "en", r)}[kind]
    p_ctx = 0.95 if kind in ("book", "receipt") else 0.75
    ctx = [ctx_fn(r) if r.random() < p_ctx else "" for _ in range(2)]
    W = int(max(tt(fbn["path"], size, True).getlength(s) for s in [t] + ctx) * 1.25 + size * 4)
    H = pitch * 3 + size * 2
    space = size * r.uniform(0.28, 0.5)
    if kind == "receipt":                                   # item ... price: wide gap before the last field
        space = size * r.uniform(0.3, 0.6)
    W = int(W + space * 14 + (size * 12 if kind == "receipt" else 0))
    layers = []
    hand = False
    for i, s in enumerate([ctx[0], t, ctx[1]]):
        im = Image.new("L", (W, H), 0)
        if s and (i == 1 or text_ok(s, fbn, fen)):
            x0 = size * 2 + r.randint(-size, size) * (i != 1 and kind not in ("book",))
            if kind == "receipt" and " " in s and r.random() < 0.7:
                head, _, last = s.rpartition(" ")
                dr = ImageDraw.Draw(im)
                xe, used = draw_text(dr, x0, size // 2 + i * pitch, head, fbn, fen, size, space)
                draw_text(dr, xe + space * r.uniform(2, 10), size // 2 + i * pitch, last, fbn, fen, size, space)
            else:
                _, used = draw_text(ImageDraw.Draw(im), x0, size // 2 + i * pitch, s, fbn, fen, size,
                                    space * (r.uniform(1.0, 1.6) if kind == "book" and r.random() < 0.4 else 1.0))
            hand = hand or (i == 1 and "hand" in used)
        layers.append(np.asarray(im, np.float32) / 255)
    strength = r.uniform(0.2, 1.0) if hand or (kind == "font" and r.random() < 0.3) else 0.0
    if kind == "receipt" and look == "thermal" and r.random() < 0.3:
        layers = [cv2.dilate(a, np.ones((2, 1), np.uint8)) * (np.indices(a.shape)[0] % 2 == 0) for a in layers]  # dot-matrix rows
    if r.random() < 0.5:
        ang = r.uniform(-2, 2)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1)
        layers = [cv2.warpAffine(a, M, (W, H)) for a in layers]
    layers = [warp(random.Random(r.random()), a, strength) for a in layers]   # same distortion seed per layer
    box = tight(layers[1])
    if box is None:
        return None
    alpha = np.clip(sum(layers), 0, 1)
    heavy = (r.random() < 0.12) if heavy is None else heavy
    return compose(r, alpha, box, heavy=heavy, look=look), box, t, int(hand)


# ---------------- word bank + stitching ----------------
def build_words(cap=60000):
    """Real handwritten word crops from page manifests whose line text splits 1:1 onto word boxes."""
    import ocr_data as D
    bank = {"bn": {"train": [], "test": []}, "en": {"train": [], "test": []}}
    srcs = ["bnhtrd", "banglawriting", "gnhk", "imgur5k", "prescription", "iiit"]
    for src in srcs:
        p = f"{OUT}/pages/{src}.jsonl"
        if not os.path.exists(p):
            continue
        n = 0
        for l in open(p):
            pg = json.loads(l)
            sp = "test" if pg["split"] == "test" else "train" if pg["split"] == "train" else None
            if sp is None:
                continue
            gray = None
            for ln in pg["lines"]:
                ws = canon(ln["text"]).split()
                boxes = ln.get("words") or ([ln["box"]] if len(ws) == 1 else [])
                if not ws or len(ws) != len(boxes):
                    continue
                if gray is None:
                    gray = cv2.imread(f"{ROOT}/{pg['image']}", cv2.IMREAD_GRAYSCALE)
                    if gray is None:
                        break
                if boxes[0] is None:
                    boxes = [[0, 0, gray.shape[1], gray.shape[0]]]
                order = sorted(range(len(boxes)), key=lambda i: boxes[i][0]) if len(boxes[0]) == 4 else [0]
                for w, i in zip(ws, order):
                    b = boxes[i]
                    if encode(w) is None:
                        continue
                    if len(b) == 5:
                        g, bb = D.rot_crop(gray, *b)
                        g = g[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]
                    else:
                        x0, y0, x1, y1 = (int(round(v)) for v in b)
                        g = gray[max(0, y0):y1, max(0, x0):x1]
                    if g.size == 0 or g.shape[0] < 8 or g.shape[1] < 4:
                        continue
                    s = 48 / g.shape[0]
                    g = cv2.resize(g, (max(4, int(g.shape[1] * s)), 48), interpolation=cv2.INTER_AREA)
                    lang = "bn" if IS_BN.search(w) else "en"
                    bank[lang][sp].append((cv2.imencode(".png", g)[1].tobytes(), w))
                    n += 1
            if n >= cap:
                break
        print(src, n, flush=True)
    from ocr_data import save_store
    os.makedirs(f"{OUT}/words", exist_ok=True)
    for lang, d in bank.items():
        for sp, items in d.items():
            print(lang, sp, save_store(((p, w, 1) for p, w in items), f"{OUT}/words/{lang}_{sp}"))


_BANK = None


def bank():
    """{lang: {split: Store}} of word crops, memory-mapped (shared across Pool workers)."""
    global _BANK
    if _BANK is None:
        from ocr_data import Store
        _BANK = {l: {s: Store(f"{OUT}/words/{l}_{s}") for s in ("train", "test")} for l in ("bn", "en")}
    return _BANK


def word_alpha(png):
    g = cv2.imdecode(np.frombuffer(png, np.uint8), 0).astype(np.float32)
    bg, fg = np.percentile(g, 92), np.percentile(g, 3)
    a = np.clip((bg - g) / max(20.0, bg - fg), 0, 1)
    a[a < 0.15] = 0
    return a


def stitch_row(r, split, h, mode, n):
    """Stitch n words into one alpha row of height ~3h. Returns (alpha, text)."""
    B = bank()
    toks, imgs = [], []
    lead = "bn" if mode in ("bn", "mixbn") else "en"
    for i in range(n):
        lang = lead
        if mode.startswith("mix") and r.random() < 0.35:
            lang = "en" if lead == "bn" else "bn"
        pool = B[lang][split]
        k = r.randrange(len(pool))
        png, w = pool[k][0], pool.text(k)
        a = word_alpha(png)
        s = h * r.uniform(0.85, 1.15) / a.shape[0]
        imgs.append(cv2.resize(a, (max(2, int(a.shape[1] * s)), max(2, int(a.shape[0] * s)))))
        toks.append(w)
    if mode.startswith("mix") and all(bool(IS_BN.search(t)) == (lead == "bn") for t in toks) and n > 1:
        return stitch_row(r, split, h, mode, n)
    gap = [int(h * r.uniform(0.25, 0.7)) for _ in imgs]
    W = sum(i.shape[1] for i in imgs) + sum(gap) + 4
    H = int(h * 2)
    row = np.zeros((H, W), np.float32)
    x = 2
    for a, g in zip(imgs, gap):
        y = int((H - a.shape[0]) / 2 + r.uniform(-0.08, 0.08) * h)
        y = min(max(0, y), H - a.shape[0])
        row[y:y + a.shape[0], x:x + a.shape[1]] = np.maximum(row[y:y + a.shape[0], x:x + a.shape[1]], a)
        x += a.shape[1] + g
    return row, " ".join(toks)


def stitch_line(r, split, heavy=None):
    h = r.randint(28, 56)
    mode = r.choices(["bn", "en", "mixbn", "mixen"], [0.3, 0.25, 0.3, 0.15])[0]
    n = r.choice([1, 2]) if r.random() < 0.15 else r.randint(3, 10)
    rows = []
    for i in range(3):
        if i != 1 and r.random() < 0.25:
            rows.append((np.zeros((int(h * 2), 10), np.float32), ""))
        else:
            rows.append(stitch_row(r, split, h, mode, n if i == 1 else r.randint(2, 10)))
    pitch = int(h * r.uniform(1.0, 1.6))
    W = max(a.shape[1] for a, _ in rows) + 2 * h
    H = pitch * 2 + int(h * 2) + 2 * h
    alpha = np.zeros((H, W), np.float32)
    tgt = None
    for i, (a, t) in enumerate(rows):
        x, y = h + (r.randint(-h // 2, h // 2) if i != 1 else 0), h + i * pitch
        x = max(0, min(x, W - a.shape[1]))
        alpha[y:y + a.shape[0], x:x + a.shape[1]] = np.maximum(alpha[y:y + a.shape[0], x:x + a.shape[1]], a)
        if i == 1:
            layer = np.zeros_like(alpha)
            layer[y:y + a.shape[0], x:x + a.shape[1]] = a
            tgt = tight(layer)
    t = canon(rows[1][1])
    if tgt is None or encode(t) is None:
        return None
    return compose(r, alpha, tgt, heavy=r.random() < 0.08 if heavy is None else heavy), tgt, t, 1


# ---------------- handwritten digits / arithmetic (MNIST digits + handwriting-font operators) ----------------
_MNIST = {}


def mnist(split):
    """{digit: [28x28 uint8 ink]} from data/raw/mnist (train for train/val, test for test), cached as npz."""
    sp = "test" if split == "test" else "train"
    if sp not in _MNIST:
        f = f"{OUT}/mnist_{sp}.npz"
        if not os.path.exists(f):
            import glob
            import pyarrow.parquet as pq
            t = pq.read_table(glob.glob(f"{ROOT}/data/raw/mnist/mnist/{sp}-*.parquet")[0]).to_pylist()
            x = np.stack([cv2.imdecode(np.frombuffer(d["image"]["bytes"], np.uint8), 0) for d in t])
            np.savez(f + ".tmp.npz", x=x, y=np.array([d["label"] for d in t]))
            os.replace(f + ".tmp.npz", f)
        z = np.load(f)
        _MNIST[sp] = {k: z["x"][z["y"] == k] for k in range(10)}
    return _MNIST[sp]


def hdigit_line(r, split, mode=None, heavy=None):
    """Handwritten numbers / arithmetic: EN digits from MNIST (writer-disjoint test split), operators and
    other characters drawn with a handwriting font (BN digits always from BN handwriting fonts)."""
    t = num_text(r, mode)
    if IS_BN.search(t) and r.random() < 0.6:
        t = t.translate(str.maketrans(BN_DIG, "0123456789"))
    t = canon(re.sub(r"[^\d০-৯+\-×÷=().,/%: x*]", "", t))
    if not t or encode(t) is None:
        return None
    M = mnist(split)
    fl = [f for f in _fonts("test" if split == "test" else "train") if f["kind"] == "hand"]
    fen = wchoice(r, [f for f in fl if f["en"]])
    fbn = wchoice(r, [f for f in fl if f["bn"]] or [f for f in _fonts("train") if f["bn"]])
    h = r.randint(28, 56)
    H = 4 * h
    row = np.zeros((H, int(h * 1.2 * len(t)) + 4 * h), np.float32)
    x = 2 * h
    for c in t:
        if c == " ":
            x += int(h * r.uniform(0.3, 0.7))
            continue
        if c.isdigit() and c in "0123456789":
            g = M[int(c)][r.randrange(len(M[int(c)]))].astype(np.float32) / 255
            g = g[(g > 0.1).any(1)][:, (g > 0.1).any(0)]
            s = h * r.uniform(0.85, 1.1) / max(1, g.shape[0])
            g = cv2.resize(g, (max(2, int(g.shape[1] * s)), max(2, int(g.shape[0] * s))))
            y = int(1.5 * h + r.uniform(-0.06, 0.06) * h)
        else:
            f = fbn if IS_BN.search(c) else fen
            if ord(c) not in f["cmap"]:
                return None
            size = int(h * r.uniform(1.0, 1.4))
            font = tt(f["path"], size, True)
            im = Image.new("L", (size * 2, size * 2), 0)
            ImageDraw.Draw(im).text((size // 2, size // 2), c, font=font, fill=255)
            g = np.asarray(im, np.float32) / 255
            if g.max() < 0.2:
                return None
            ys, xs = np.where(g > 0.2)
            top = ys.min()
            g = g[top:ys.max() + 1, xs.min():xs.max() + 1]
            y = int(2.5 * h - g.shape[0]) if c in ".," else int(2 * h - g.shape[0] / 2)
        y = min(max(0, y), H - g.shape[0])
        if x + g.shape[1] >= row.shape[1]:
            return None
        row[y:y + g.shape[0], x:x + g.shape[1]] = np.maximum(row[y:y + g.shape[0], x:x + g.shape[1]], g)
        x += g.shape[1] + int(h * r.uniform(0.02, 0.2))
    row = row[:, :x + 2 * h]
    row = warp(r, row, r.uniform(0.0, 0.5))
    box = tight(row)
    if box is None:
        return None
    return compose(r, row, box, heavy=r.random() < 0.08 if heavy is None else heavy), box, t, 1


# ---------------- shards ----------------
KINDS = ("font", "stitch", "num", "receipt", "rx", "book", "hdigit")


def sample(kind, seed, split, heavy=None, unseen=False):
    """One synthetic line before cropping: (gray canvas, box, text, hand) or None. kind 'noisy' = text of any font
    kind with heavy degradation. heavy: None = the kind's training rate, False = clean (test sets), True = always.
    unseen: held-out fonts from readable categories only (font test sets)."""
    r = random.Random(f"{kind}{split}{seed}")
    sp = "test" if split == "test" else "train"
    try:
        if kind == "stitch":
            return stitch_line(r, sp, heavy)
        if kind in ("hdigit", "hdigits", "harith"):
            return hdigit_line(r, sp, {"hdigits": "digits", "harith": "arith"}.get(kind), heavy)
        if kind == "print":
            return font_line(r, sp, "font", heavy=heavy, unseen=unseen, print_only=True)
        if kind == "noisy":
            return font_line(r, sp, r.choice(["font", "font", "num", "receipt", "rx", "book"]), heavy=True, unseen=unseen)
        return font_line(r, sp, kind, heavy=heavy, unseen=unseen)
    except Exception:
        return None


def _one(args):
    kind, seed, split = args
    out = sample(kind, seed, split)
    if out is None:
        return None
    img, box, t, hand = out
    ink = crop(img, box)
    from ocr_data import ctc_ok, ink_png
    if not ctc_ok(t, ink.shape[1] // 4):
        return None
    return ink_png(ink), t, hand


def build_lines(kind, n, split, shard=0):
    from ocr_data import REC
    os.makedirs(REC, exist_ok=True)
    off = shard * 10 ** 7
    font_pool()                                            # once, before the fork
    with Pool(int(os.environ.get("ORUS_PROCS", 6))) as pool:
        items = [x for x in pool.imap(_one, ((kind, off + i, split) for i in range(n)), chunksize=64) if x]
    from ocr_data import save_store
    print(kind, split, shard, save_store(items, f"{REC}/synth_{kind}_{split}_{shard}"), "of", n, flush=True)


def demo(out=None):
    out = out or f"{OUT}/synth_demo.png"
    rows = []
    for kind in KINDS + ("noisy",):
        if kind == "stitch" and not os.path.exists(f"{OUT}/words/bn_train.meta.npy"):
            continue
        k = 0
        for i in range(40):
            x = _one((kind, i, "train"))
            if x is None:
                continue
            png, t, hand = x
            im = 255 - cv2.imdecode(np.frombuffer(png, np.uint8), 0)
            assert im.shape[0] % 3 == 0 and encode(t) is not None
            im = im[:, :1200]
            rows.append(cv2.copyMakeBorder(im, 0, 3, 0, 1200 - im.shape[1], cv2.BORDER_CONSTANT, value=128))
            print(kind, hand, t)
            k += 1
            if k == 6:
                break
        assert k >= 4, f"{kind}: too many failed samples"
    cv2.imwrite(out, np.vstack(rows))
    print("wrote", out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "words":
        build_words()
    elif cmd == "lines":
        build_lines(sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 0)
    else:
        demo(*sys.argv[2:])
