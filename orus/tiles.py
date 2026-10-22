"""Shared page preprocessing: polarity fix, text-height normalisation, ink-tile extraction.
Used identically by build_data.py (training) and predict.py (inference)."""
import cv2
import numpy as np

TILE = 128          # tile side (px) fed to the model
TARGET_H = 24       # median connected-component height after normalisation (px)
MAX_SIDE = 1600     # analysis resolution for height estimation


def to_gray(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.median(img) < 110:          # dark-mode screenshot / chalkboard -> dark text on light
        img = 255 - img
    return img


def binarize(gray):
    block = max(15, (max(gray.shape) // 40) | 1)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 12)


def ink_mask(gray):
    """Binary ink with ruled/grid lines removed (they are paper, not writing)."""
    b = binarize(gray)
    H, W = b.shape
    lines = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(W // 25, 15), 1)))
    lines |= cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(H // 25, 15))))
    return cv2.subtract(b, cv2.dilate(lines, np.ones((3, 3), np.uint8)))


def text_height(gray):
    """Median connected-component height (px) of text-like blobs, or None."""
    f = min(1.0, MAX_SIDE / max(gray.shape))
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else gray
    n, _, st, _ = cv2.connectedComponentsWithStats(ink_mask(small), connectivity=8)
    H, W = small.shape
    w, h, a = st[1:, 2], st[1:, 3], st[1:, 4]
    keep = (h >= 4) & (h < 0.2 * H) & (w < 0.5 * W) & (a >= 12) & (w < 15 * h)
    if keep.sum() < 8:
        return None
    h, a = h[keep], a[keep].astype(float)   # area-weighted median: specks/line fragments barely count
    o = np.argsort(h)
    return float(h[o][np.searchsorted(np.cumsum(a[o]), a.sum() / 2)]) / f


def normalize(img, jitter=1.0):
    """Gray + rescale so median text height ~= TARGET_H (times jitter)."""
    gray = to_gray(img)
    th = text_height(gray)
    s = (TARGET_H / th) if th else 1024 / max(gray.shape)
    s = float(np.clip(s * jitter, 0.05, 4.0))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(gray, None, fx=s, fy=s, interpolation=interp), s


def tiles(gray, stride=TILE, min_ink=0.02, max_ink=0.5):
    """Grid tiles that contain ink. Returns (N,TILE,TILE) uint8 and (N,2) top-left (y,x)."""
    if min(gray.shape) < TILE:   # pad tiny inputs (single word screenshot etc.)
        ph, pw = max(0, TILE - gray.shape[0]), max(0, TILE - gray.shape[1])
        gray = cv2.copyMakeBorder(gray, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
    ink = ink_mask(gray) > 0
    H, W = gray.shape
    ys = list(range(0, H - TILE + 1, stride)) or [0]
    xs = list(range(0, W - TILE + 1, stride)) or [0]
    out, pos = [], []
    for y in ys:
        for x in xs:
            r = ink[y:y + TILE, x:x + TILE].mean()
            if min_ink <= r <= max_ink:
                out.append(gray[y:y + TILE, x:x + TILE])
                pos.append((y, x))
    if not out:
        return np.zeros((0, TILE, TILE), np.uint8), np.zeros((0, 2), int)
    return np.stack(out), np.array(pos)


if __name__ == "__main__":
    # self-check: synthetic text page -> height estimate near the drawn size, tiles found
    img = np.full((1200, 900), 235, np.uint8)
    for i in range(20):
        cv2.putText(img, "the quick brown fox jumps", (40, 60 + i * 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 20, 2)
    th = text_height(img)
    assert 15 < th < 45, th
    g, s = normalize(img)
    t, p = tiles(g)
    assert len(t) > 5 and t.shape[1:] == (TILE, TILE)
    assert len(tiles(np.full((50, 50), 255, np.uint8))[0]) == 0
    print("ok", th, s, len(t))
