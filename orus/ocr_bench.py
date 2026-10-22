"""Line-level OCR benchmark for Orus-OCR.

  python ocr_bench.py make [set ...]        -> data/ocr/bench/<set>.pkl  (seeded sample of each TEST split)
      item = (gray png at source resolution incl. context, line box in it, text)
  python ocr_bench.py run <system> [arg]    -> runs/bench_<system>.json  CER / WER per set + ms per line
      systems: orus <ckpt.pt|.onnx>, indra; BENCH_SETS="a b" limits the sets, BENCH_N the lines;
      BENCH_TTA=0.8 turns on the confidence-gated re-read, BENCH_LEX=0 turns the word lexicon off
Real sets (PAGE_SETS): BN handwriting bnhtrd, banglawriting, iiit (words), rupom (unseen source); EN handwriting
gnhk, iam, imgur5k (words); prescription (doctor words); printed BN bnprinted (multi-line images removed), swapnil
(book pages -> lines), bench1000 (OCR-Bench1000-Bengali); printed EN sroie, cord_en (deskewed rows, dot leaders
ignored), funsd (forms); textocr_num (scene numbers), crohme (handwritten plain arithmetic).
Synthetic sets (SYN_SETS, held-out font families / test word bank / MNIST test digits): print_bn/en (unseen print
fonts), font_bn/en (unseen fonts incl. handwriting-style),
digits_bn/en, arith_bn/en, hdigits/harith (handwritten), receipt_bn/en, rx_print, book_bn/en, noisy.
0.1-alpha sets: mixed (stitched BN+EN words), font. CRITERIA groups the sets into the reported criteria.
"""
import json
import os
import pickle
import random
import re
import sys
import time

import cv2
import numpy as np
from rapidfuzz.distance import Levenshtein

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root: data/, runs/, models/
B = f"{ROOT}/data/ocr/bench"
# set -> page manifest (data/ocr/pages/<src>.jsonl, test split)
PAGE_SETS = dict(bnhtrd="bnhtrd", banglawriting="banglawriting", iiit="iiit", gnhk="gnhk", iam="iam", imgur5k="imgur5k",
                 prescription="prescription", rupom="rupom", bnprinted="bnprinted", swapnil="swapnil", sroie="sroie",
                 funsd="funsd", cord_en="cord", bench1000="bench1000", textocr_num="textocr", crohme="crohme")
# set -> (ocr_synth kind, script filter): held-out fonts / test word bank / MNIST test digits
SYN_SETS = dict(print_bn=("print", "bn"), print_en=("print", "en"), font_bn=("font", "bn"), font_en=("font", "en"), digits_bn=("digits", "bn"), digits_en=("digits", "en"),
                arith_bn=("arith", "bn"), arith_en=("arith", "en"), hdigits=("hdigits", None), harith=("harith", None),
                receipt_bn=("receipt", "bn"), receipt_en=("receipt", "en"), rx_print=("rx", None), book_bn=("book", "bn"),
                book_en=("book", "en"), noisy=("noisy", None))
SETS = list(PAGE_SETS) + ["mixed", "font"] + list(SYN_SETS)
# criteria -> sets (report groups; a set may serve several criteria)
CRITERIA = {
    "printed BN": ["bnprinted", "swapnil", "bench1000", "print_bn"],
    "printed EN": ["print_en", "sroie", "cord_en", "funsd"],
    "handwritten BN": ["bnhtrd", "banglawriting", "iiit", "rupom"],
    "handwritten EN": ["iam", "gnhk", "imgur5k"],
    "unseen fonts": ["print_bn", "print_en", "font_bn", "font_en"],
    "receipts": ["sroie", "cord_en", "receipt_bn", "receipt_en"],
    "prescriptions": ["prescription", "rx_print"],
    "digits": ["digits_bn", "digits_en", "hdigits", "textocr_num"],
    "arithmetic": ["arith_bn", "arith_en", "harith", "crohme"],
    "noisy": ["noisy", "gnhk", "imgur5k", "textocr_num"],
    "books / newspapers": ["swapnil", "book_bn", "book_en"],
    "mixed BN+EN": ["mixed"],
}
N = 300
DOT_LEADER = re.compile(r"\s*\.{3,}\s*")


def norm(s, t):
    """Per-set scoring normalisation: CORD labels drop dot leaders, so leaders are removed from both sides;
    CROHME labels carry no spacing, so spaces are removed from both sides."""
    if s == "cord_en":
        return DOT_LEADER.sub(" ", t).strip()
    return t.replace(" ", "") if s == "crohme" else t


def page_set(s, src):
    import ocr_data as D
    from vocab import canon, encode
    p = f"{ROOT}/data/ocr/pages/{src}.jsonl"
    if not os.path.exists(p):
        print("skip", s, "(no pages)")
        return None
    pages = [pg for pg in map(json.loads, open(p)) if pg["split"] == "test"]
    cand = [(pg, j) for pg in pages for j in range(len(pg["lines"]))]
    random.Random(0).shuffle(cand)
    out = []
    for pg, j in cand:
        l = pg["lines"][j]
        t = canon(l["text"])
        if encode(t) is None or (s == "cord_en" and len(t) < 2):
            continue
        g = cv2.imread(f"{ROOT}/{pg['image']}", cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        if l.get("words"):
            g, box = D.rot_crop(g, *D.line_rect(l["words"]))
        elif l["box"] is None or s == "rupom":          # whole-image lines
            g, box = whole_line(g)
        elif len(l["box"]) == 5:
            g, box = D.rot_crop(g, *l["box"])
        else:
            x0, y0, x1, y1 = l["box"]
            h = y1 - y0
            if h < 4 or x1 - x0 < 2:
                continue
            X0, Y0 = int(max(0, x0 - h)), int(max(0, y0 - 2 * h))
            g = g[Y0:int(y1 + 2 * h), X0:int(x1 + h)]
            box = [x0 - X0, y0 - Y0, x1 - X0, y1 - Y0]
        out.append((cv2.imencode(".png", g)[1].tobytes(), [float(v) for v in box], t))
        if len(out) == N:
            break
    return out


def syn_set(kind, script):
    """Synthetic test lines from ocr_synth.sample(split='test', unseen fonts): gray canvas + target box + text."""
    import ocr_synth as S
    BN, LAT = re.compile(r"[ঀ-৿]"), re.compile(r"[A-Za-z]")
    out = []
    for seed in range(50 * N):
        x = S.sample(kind, seed, "test", heavy=None if kind == "noisy" else False, unseen=not kind.startswith("h"))
        if x is None:
            continue
        g, box, t, _ = x
        if script == "bn" and not (BN.search(t) and not LAT.search(t)) or script == "en" and BN.search(t):
            continue
        out.append((cv2.imencode(".png", g)[1].tobytes(), [float(v) for v in box], t))
        if len(out) == N:
            break
    return out


def make(names=None):
    """Bench pickles for the named sets (default: all); a set whose source is missing is skipped."""
    import ocr_data as D
    os.makedirs(B, exist_ok=True)
    for s in names or SETS:
        if s in PAGE_SETS:
            out = page_set(s, PAGE_SETS[s])
        elif s in SYN_SETS:
            out = syn_set(*SYN_SETS[s])
        else:
            out = legacy_set(s, D)
        if out:
            pickle.dump(out, open(f"{B}/{s}.pkl", "wb"))
            print(s, len(out), flush=True)


def legacy_set(s, D):
    """0.1-alpha sets from the 32 px synthetic test stores (kept for comparison with the 0.1 reports)."""
    f = dict(mixed="synth_stitch_test_0", font="synth_font_test_0")[s]
    p = f"{ROOT}/data/ocr/rec/{f}"
    if not os.path.exists(p + ".meta.npy"):
        return None
    st = D.Store(p)
    items = [(st[i][0], st.text(i), 1) for i in range(len(st))]
    if s == "mixed":       # keep lines with both scripts
        items = [x for x in items if re.search(r"[ঀ-৿]", x[1]) and re.search(r"[A-Za-z]", x[1])]
    out = []
    for png, t, _ in items[:N]:
        ink = cv2.imdecode(np.frombuffer(png, np.uint8), 0)
        g = 255 - ink
        h = 32 / 1.4
        out.append((cv2.imencode(".png", g)[1].tobytes(), [0.3 * h, 32 + 0.2 * h, g.shape[1] - 0.3 * h, 64 - 0.2 * h], t))
    return out


def whole_line(g):
    """A whole-image line (rupom: scans rotated with black fill) -> upright crop of the paper rectangle."""
    import ocr_data as D
    n, lab, st, _ = cv2.connectedComponentsWithStats((g > 40).astype(np.uint8))
    if n < 2:
        return g, [0, 0, g.shape[1], g.shape[0]]
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    (cx, cy), (w, h), ang = cv2.minAreaRect(cv2.findNonZero((lab == k).astype(np.uint8)))
    if w < h:
        w, h, ang = h, w, ang - 90
    ang = (ang + 45) % 90 - 45                        # rotations are small; never flip the line
    out, box = D.rot_crop(g, cx, cy, w - 6, h - 6, -ang)
    x0, y0, x1, y1 = map(int, box)
    out[:y0], out[y1:], out[:, :x0], out[:, x1:] = 255, 255, 255, 255   # blank everything outside the paper
    return out, box


def load_sets():
    only = os.environ.get("BENCH_SETS", "").split()
    return {s: pickle.load(open(f"{B}/{s}.pkl", "rb")) for s in SETS
            if os.path.exists(f"{B}/{s}.pkl") and (not only or s in only)}


def dec(png):
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)


def line_only(g, box, pad=0.25):
    x0, y0, x1, y1 = box
    h = y1 - y0
    return g[int(max(0, y0 - pad * h)):int(y1 + pad * h), int(max(0, x0 - pad * h)):int(x1 + pad * h)]


# ---------------- systems: fn(list of (gray, box)) -> list of str ----------------
def sys_orus(path):
    """BENCH_LEX=0 turns the word lexicon off (greedy + gated char LM only)."""
    import ocr
    w = os.path.dirname(os.path.abspath(path))
    m = ocr.Recognizer(path, lm=f"{w}/orus_lm.npz", lex=f"{w}/orus_lex.txt.gz" if os.environ.get("BENCH_LEX", "1") == "1" else None)
    tta = float(os.environ.get("BENCH_TTA", "0")) or None              # BENCH_TTA=0.8: confidence-gated re-read
    return lambda items: m.read_boxes([(g, b) for g, b in items], tta=tta)


def sys_indra(_=None):
    from indra_ocr import OrtOCR, model_dir
    o = OrtOCR(model_dir(None), threads=4)
    return lambda items: [o.recognize(g, np.array([b], np.float32))[0][0] for g, b in items]


def run(system, arg=None, limit=N):
    from vocab import canon
    fn = globals()[f"sys_{system}"](arg)
    res = {}
    for s, items in load_sets().items():
        items = items[:limit]
        inp = [(dec(p), b) for p, b, _ in items]
        t0 = time.perf_counter()
        pred = fn(inp)
        ms = 1000 * (time.perf_counter() - t0) / max(1, len(items))
        pred = [norm(s, canon(p)) for p in pred]
        gt = [norm(s, t) for _, _, t in items]
        ed = sum(Levenshtein.distance(p, t) for p, t in zip(pred, gt))
        n = sum(len(t) for t in gt)
        wed = sum(Levenshtein.distance(p.split(), t.split()) for p, t in zip(pred, gt))
        wn = sum(len(t.split()) for t in gt)
        res[s] = dict(cer=round(ed / n, 4), wer=round(wed / wn, 4), ms_per_line=round(ms, 1), n=len(items),
                      sample=[(p, t) for p, (_, _, t) in list(zip(pred, items))[:3]])
        print(system, s, res[s]["cer"], res[s]["wer"], f"{ms:.1f} ms", flush=True)
    tag = system if not arg else f"{system}_{os.path.basename(str(arg)).split('.')[0]}"
    tag += os.environ.get("BENCH_TAG", "")
    json.dump(res, open(f"{ROOT}/runs/bench_{tag}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(sys.argv[2:])
    else:
        run(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None,
            int(os.environ.get("BENCH_N", N)))
