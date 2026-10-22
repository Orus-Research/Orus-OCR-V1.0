"""Font pool, text corpora and printed-page rendering helpers used by ocr_synth.py: EN + BN text, print and
handwriting-style fonts, rendered as paper photo / screenshot / photo-of-screen / scan."""
import glob
import hashlib
import os
import random

import cv2
import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root: data/, runs/, models/
D = os.path.join(ROOT, "data")
BN_RANGE = range(0x0980, 0x0A00)


def prep_text(n=60000):
    """Dump paragraphs from the downloaded corpora to data/text/{en,bn}.txt once."""
    import pyarrow.parquet as pq
    os.makedirs(f"{D}/text", exist_ok=True)
    for lang, pat in [("en", "text_en/**/*.parquet"), ("bn", "text_bn/**/*.parquet")]:
        out = f"{D}/text/{lang}.txt"
        if os.path.exists(out):
            continue
        col = pq.read_table(glob.glob(f"{D}/raw/{pat}", recursive=True)[0], columns=["text"]).column("text").to_pylist()
        paras = []
        for t in col:
            for p in t.split("\n"):
                p = " ".join(p.split())
                if 80 < len(p) < 1500 and not p.startswith("="):
                    paras.append(p.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", "."))
            if len(paras) >= n:
                break
        open(out, "w").write("\n".join(paras))


def _font_list():
    fonts = []
    for kind, pat in [("print", "fonts/print/*"), ("hand", "fonts/hand/*"), ("bn", "fonts/bn/*")]:
        fonts += [(kind, f) for f in glob.glob(f"{D}/{pat}")]
    fonts += [("print", f) for f in glob.glob("/usr/share/fonts/truetype/freefont/*.ttf")]
    fonts += [("bn", f) for f in glob.glob("/usr/share/fonts/truetype/noto/Noto*Bengali*.ttf")]
    out = []
    for kind, f in fonts:
        try:
            cmap = TTFont(f, fontNumber=0, lazy=True).getBestCmap()
        except Exception:
            continue
        has_bn = sum(c in cmap for c in BN_RANGE) > 60
        has_en = all(ord(c) in cmap for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,")
        name = os.path.basename(f)
        # deterministic font hold-out: ~15% of fonts never seen in training
        split = "test" if int(hashlib.md5(name.encode()).hexdigest(), 16) % 100 < 15 else "train"
        out.append(dict(path=f, kind=kind, bn=has_bn, en=has_en, split=split))
    return out


_FONTS = None
_TEXT = {}


def fonts():
    global _FONTS
    if _FONTS is None:
        _FONTS = _font_list()
    return _FONTS


def text(lang):
    if lang not in _TEXT:
        prep_text()
        _TEXT[lang] = open(f"{D}/text/{lang}.txt").read().split("\n")
    return _TEXT[lang]


def math_line(r):
    a, b, c = r.randint(1, 99), r.randint(1, 99), r.randint(1, 9)
    return r.choice([f"{c}x + {a} = {b}", f"Q{r.randint(1, 20)}. Solve: {a} x {b} = {a * b}",
                     f"({a} + {b}) / {c} = {round((a + b) / c, 2)}", f"Ans: {a * c} - {b} = {a * c - b}",
                     f"{r.randint(1, 9)}. ({chr(97 + r.randint(0, 5))}) {a}% of {b * 10} = {a * b / 10:g}"])


def paper(r, W, H, line_gap):
    bg = np.full((H, W), r.randint(200, 252), np.float32)
    if r.random() < 0.4:                                   # ruled notebook
        off, col = r.randint(0, line_gap), r.randint(140, 215)
        for y in range(off, H, line_gap):
            bg[y:y + r.choice([1, 1, 2])] = col
        if r.random() < 0.5:
            x = r.randint(30, 120)
            bg[:, x:x + 2] = col - 20
    elif r.random() < 0.15:                                # grid / graph paper
        g = r.randint(18, 40)
        col = r.randint(170, 225)
        bg[::g] = col
        bg[:, ::g] = col
    return bg


def layout(r, W, H, lang, font_path, size, draw_fn, math=True):
    """Fill the page with wrapped paragraphs; draw_fn(x, y, text, font)."""
    eng = ImageFont.Layout.RAQM if lang == "bn" else ImageFont.Layout.BASIC
    font = ImageFont.truetype(font_path, size, layout_engine=eng)
    big = ImageFont.truetype(font_path, int(size * r.uniform(1.2, 1.8)), layout_engine=eng)
    mx, y = r.randint(10, 120), r.randint(10, 100)
    gap = int(size * r.uniform(1.15, 2.0))
    corpus = text(lang)
    while y < H - gap:
        if r.random() < 0.1:
            draw_fn(mx, y, r.choice(corpus)[:40], big)
            y += int(gap * 1.6)
            continue
        para = math_line(r) if math and r.random() < 0.12 else r.choice(corpus)
        line = ""
        for w in para.split(" "):
            cand = (line + " " + w).strip()
            if font.getlength(cand) > W - 2 * mx and line:
                draw_fn(mx, y, line, font)
                y += gap
                line = w
                if y >= H - gap:
                    return
            else:
                line = cand
        draw_fn(mx, y, line, font)
        y += int(gap * r.uniform(1.0, 1.8))


def degrade(r, img, mode):
    H, W = img.shape
    if mode in ("photo", "screenphoto"):
        d = r.uniform(0.0, 0.08)
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        dst = src + np.float32([[r.uniform(-d, d) * W, r.uniform(-d, d) * H] for _ in range(4)])
        img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (W, H),
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=r.randint(20, 120))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        light = 1 - r.uniform(0, 0.45) * ((xx / W) * r.uniform(-1, 1) + (yy / H) * r.uniform(-1, 1)) ** 2
        img = img * light
        if mode == "screenphoto":                          # moire + glow of a monitor photo
            f = r.uniform(0.3, 1.2)
            img = img * (1 - r.uniform(0.03, 0.12) * (np.sin(xx * f + yy * f * r.uniform(0.2, 1.0)) + 1) / 2)
        img = cv2.GaussianBlur(img, (0, 0), r.uniform(0.3, 1.8))
        img = img + np.random.default_rng(r.randint(0, 1 << 30)).normal(0, r.uniform(1, 8), img.shape)
    elif mode == "scan":
        M = cv2.getRotationMatrix2D((W / 2, H / 2), r.uniform(-2, 2), 1)
        img = cv2.warpAffine(img, M, (W, H), borderValue=255)
        img = cv2.GaussianBlur(img, (0, 0), r.uniform(0.2, 1.0))
        img = img + np.random.default_rng(r.randint(0, 1 << 30)).normal(0, r.uniform(0, 6), img.shape)
        if r.random() < 0.3:                               # fax-like binarized scan (RVL style)
            img = np.where(img < r.randint(110, 170), 0, 255).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if r.random() < 0.3:                                   # resolution loss
        s = r.uniform(0.4, 0.8)
        img = cv2.resize(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), (W, H))
    q = r.randint(35, 95)
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 0)


_PHOTOS = {}


def photo_list(split):
    """Imagenette photos: train folder for train/val pages, val folder for test pages (no overlap)."""
    sub = "val" if split == "test" else "train"
    if sub not in _PHOTOS:
        _PHOTOS[sub] = sorted(glob.glob(f"{D}/raw/imagenette/imagenette2-320/{sub}/*/*.JPEG"))
    return _PHOTOS[sub]


def chart(r, w, h):
    """Printed-looking figure: bar / line / pie chart with axes."""
    c = np.full((h, w), r.randint(235, 255), np.uint8)
    kind = r.choice(["bar", "line", "pie"])
    if kind == "pie":
        ctr, rad, a0 = (w // 2, h // 2), min(w, h) // 2 - 5, 0
        for _ in range(r.randint(3, 6)):
            a1 = a0 + r.randint(30, 140)
            cv2.ellipse(c, ctr, (rad, rad), 0, a0, min(a1, 360), r.randint(40, 220), -1)
            a0 = a1
    else:
        cv2.line(c, (20, h - 20), (w - 5, h - 20), 0, 2); cv2.line(c, (20, 5), (20, h - 20), 0, 2)
        n = r.randint(4, 12)
        ys = [r.randint(10, h - 30) for _ in range(n)]
        for i, y in enumerate(ys):
            x = 25 + i * (w - 30) // n
            if kind == "bar":
                cv2.rectangle(c, (x, y), (x + (w - 30) // n - 4, h - 21), r.randint(30, 200), -1)
            elif i:
                cv2.line(c, (25 + (i - 1) * (w - 30) // n, ys[i - 1]), (x, y), r.randint(0, 120), 2)
    return c


def paste_figures(r, bg, split):
    """0-3 photo / chart blocks, as in web pages, slides, textbooks."""
    H, W = bg.shape
    for _ in range(r.choices([0, 1, 2, 3], [0.35, 0.35, 0.2, 0.1])[0]):
        w = r.randint(W // 5, int(W * 0.7)); h = r.randint(H // 8, H // 2)
        if r.random() < 0.7:
            ph = cv2.imread(r.choice(photo_list(split)), 0)
            blk = cv2.resize(ph, (w, h), interpolation=cv2.INTER_AREA)
        else:
            blk = chart(r, w, h)
        y, x = r.randint(0, H - h), r.randint(0, W - w)
        bg[y:y + h, x:x + w] = blk
    return bg


def make_page(seed, split="train"):
    r = random.Random(seed)
    lang = "bn" if r.random() < 0.45 else "en"
    pool = [f for f in fonts() if f["split"] == split and f[lang]]
    if lang == "en" and r.random() < 0.6:                  # oversample handwriting-style fonts: hardest negatives
        pool = [f for f in pool if f["kind"] == "hand"] or pool
    font = r.choice(pool)
    mode = r.choices(["photo", "screen", "screenphoto", "scan"], [0.4, 0.25, 0.1, 0.25])[0]
    size = r.randint(14, 44)
    if mode == "screen":
        W, H = r.randint(700, 1900), r.randint(500, 1200)
    else:
        W = r.randint(700, 1300)
        H = int(W * r.uniform(1.1, 1.5))
    gap = int(size * r.uniform(1.2, 2.0))
    if mode in ("screen", "screenphoto"):
        bg = np.full((H, W), r.choice([255, 255, 250, 245, 235]), np.float32)
        if r.random() < 0.3:                               # UI chrome bars
            bg[: r.randint(30, 80)] = r.randint(200, 240)
    elif mode == "scan":
        bg = np.full((H, W), r.randint(235, 255), np.float32)
    else:
        bg = paper(r, W, H, gap)
    im = Image.fromarray(bg.astype(np.uint8))
    dr = ImageDraw.Draw(im)
    figs = r.random() < (0.6 if mode in ("screen", "screenphoto") else 0.3)
    ink = r.randint(0, 70) if r.random() < 0.8 else r.randint(70, 130)   # black or colored ink in gray
    layout(r, W, H, lang, font["path"], size, lambda x, y, t, f: dr.text((x, y), t, font=f, fill=ink), math=font["en"])
    page = np.asarray(im, np.float32).copy()
    if figs:                                               # figures over the text, like a layout with images
        page = paste_figures(r, page, split)
    return degrade(r, page, mode), dict(lang=lang, font=os.path.basename(font["path"]),
                                                               kind=font["kind"], mode=mode)


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/render_demo"
    os.makedirs(out, exist_ok=True)
    fl = fonts()
    print(len(fl), "fonts;", sum(f["bn"] for f in fl), "bangla;", sum(f["kind"] == "hand" for f in fl), "hand;",
          sum(f["split"] == "test" for f in fl), "held-out")
    for i in range(8):
        img, meta = make_page(i)
        assert img.dtype == np.uint8 and img.ndim == 2
        cv2.imwrite(f"{out}/{i}_{meta['lang']}_{meta['mode']}_{meta['kind']}.jpg", img)
        print(i, meta)
