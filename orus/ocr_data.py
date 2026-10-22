"""Training data, step 1: unify every OCR source into page manifests, then cut recognizer line crops.
Raw datasets are expected under data/raw/<dir> and data/bench_raw/<dir> (see docs/DATASETS.md for sources).

  python ocr_data.py pages [src ...]   -> data/ocr/pages/<src>.jsonl
      {"image": path, "lines": [{"box": [x0,y0,x1,y1] | [cx,cy,w,h,deg], "text": str, "hand": 0|1}],
       "split": train|val|test, "src": str, "license": str}
      A crop-level source (word / line images) is one "page" per crop with box = whole image.
  ORUS_H=48 python ocr_data.py crops [src ...]   -> data/ocr/rec48/<src>_<split>.{blob,meta,lab}.npy (mmap store)
      Each line is cut with Indra's crop_line rule (ORUS_H px x tall window 3: 144 px ink map at 48 px).
Splits are writer/page disjoint (hsplit on a writer or page key); official test splits are kept.
"""
import csv
import glob
import hashlib
import json
import os
import re
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import pyarrow.parquet as pq

from vocab import canon, encode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root: data/, runs/, models/
RAW, BENCH = f"{ROOT}/data/raw", f"{ROOT}/data/bench_raw"
OUT = f"{ROOT}/data/ocr"
LINE_H = int(os.environ.get("ORUS_H", "32"))        # recognizer line height in px (ink map = 3 * LINE_H tall)
REC = f"{OUT}/rec" if LINE_H == 32 else f"{OUT}/rec{LINE_H}"
SPECIAL = re.compile(r"%\w+%|\*")      # GNHK %math% %SC% %NA%, BanglaWriting struck-out '*'
MAX_W = 2048


def hsplit(name, val=10, test=15):
    """Deterministic split by an md5 hash of a writer / page key (test %, val %)."""
    h = int(hashlib.md5(name.encode()).hexdigest(), 16) % 100
    return "test" if h < test else "val" if h < test + val else "train"


def rel(p):
    return os.path.relpath(p, ROOT)


def page(img, lines, split, src, lic):
    return dict(image=rel(img), lines=lines, split=split, src=src, license=lic)


def dump_parquet_images(src, rows):
    """Parquet image bytes -> data/ocr/img/<src>/<i>.png (once)."""
    d = f"{OUT}/img/{src}"
    os.makedirs(d, exist_ok=True)
    paths = []
    for i, b in enumerate(rows):
        p = f"{d}/{i}.png"
        if not os.path.exists(p):
            im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(p, im)
        paths.append(p)
    return paths


def full(p):
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return [0, 0, im.shape[1], im.shape[0]] if im is not None else None


def group_lines(words):
    """Word boxes [(x0,y0,x1,y1,text)] -> lines, by vertical overlap with the running line."""
    words = sorted(words, key=lambda w: (w[1] + w[3]) / 2)
    lines = []
    for w in words:
        cy, h = (w[1] + w[3]) / 2, w[3] - w[1]
        for L in lines:
            ly0, ly1 = np.median([v[1] for v in L]), np.median([v[3] for v in L])
            if ly0 - 0.1 * h < cy < ly1 + 0.1 * h:
                L.append(w)
                break
        else:
            lines.append([w])
    out = []
    for L in lines:
        L.sort(key=lambda w: w[0])
        out.append(dict(box=[min(v[0] for v in L), min(v[1] for v in L), max(v[2] for v in L), max(v[3] for v in L)],
                        text=" ".join(v[4] for v in L), hand=1, words=[list(v[:4]) for v in L]))
    return sorted(out, key=lambda l: l["box"][1])


# ---------------- sources: yield page dicts ----------------
def src_gnhk():
    for sd in ("train", "test"):
        for f in sorted(glob.glob(f"{RAW}/gnhk/{sd}/*.jpg")):
            name = os.path.basename(f)[:-4]
            split = "test" if sd == "test" else hsplit(name, 10, 0)
            by = {}
            for w in json.load(open(f[:-4] + ".json")):
                by.setdefault(w["line_idx"], []).append(w)
            lines = []
            for ws in by.values():
                if any(SPECIAL.search(w["text"]) for w in ws):
                    continue
                pts = [(w["polygon"][f"x{i}"], w["polygon"][f"y{i}"]) for w in ws for i in range(4)]
                xs, ys = zip(*pts)
                ws.sort(key=lambda w: min(w["polygon"][f"x{i}"] for i in range(4)))
                wb = [[min(w["polygon"][f"x{i}"] for i in range(4)), min(w["polygon"][f"y{i}"] for i in range(4)),
                       max(w["polygon"][f"x{i}"] for i in range(4)), max(w["polygon"][f"y{i}"] for i in range(4))] for w in ws]
                lines.append(dict(box=[min(xs), min(ys), max(xs), max(ys)], text=" ".join(w["text"] for w in ws), hand=1,
                                  words=wb))
            yield page(f, lines, split, "gnhk", "CC BY 4.0")


def src_banglawriting():
    for f in sorted(glob.glob(f"{RAW}/banglawriting/converted/*.jpg")):
        name = os.path.basename(f)[:-4]
        ws = [(*s["points"][0], *s["points"][1], s["label"]) for s in json.load(open(f[:-4] + ".json"))["shapes"]]
        ws = [(min(a, c), min(b, d), max(a, c), max(b, d), t.strip()) for a, b, c, d, t in ws]
        lines = [l for l in group_lines(ws) if not SPECIAL.search(l["text"])]
        yield page(f, lines, hsplit(name), "banglawriting", "CC BY-NC 3.0")


def xlsx_rows(path):
    """Rows of every sheet as {col letter: value}; plain zip+XML (some BN-HTRd files break openpyxl styles)."""
    import xml.etree.ElementTree as ET
    import zipfile
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    z = zipfile.ZipFile(path)
    ss = []
    if "xl/sharedStrings.xml" in z.namelist():
        ss = ["".join(t.text or "" for t in si.iter(ns + "t")) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(ns + "si")]
    for n in sorted(x for x in z.namelist() if x.startswith("xl/worksheets/sheet")):
        for row in ET.fromstring(z.read(n)).iter(ns + "row"):
            r = {}
            for c in row.iter(ns + "c"):
                v = c.find(ns + "v")
                t = c.find(f"{ns}is/{ns}t")
                val = ss[int(v.text)] if c.get("t") == "s" and v is not None else (t.text if t is not None else (v.text if v is not None else None))
                r[re.sub(r"\d", "", c.get("r"))] = val
            yield r


def src_bnhtrd():
    base = f"{RAW}/bnhtrd_full/BN-HTR_Dataset"
    for wdir in sorted(glob.glob(f"{base}/Recognition_Ground_Truth_Texts/*"), key=lambda p: int(os.path.basename(p))):
        w = os.path.basename(wdir)
        words = {}
        for r in (r for x in glob.glob(f"{wdir}/*.xlsx") for r in xlsx_rows(x)):
            m = re.fullmatch(r"\s*\d+_(\d+)_(\d+)_(\d+)\s*", str(r.get("A")))
            if m and r.get("B") is not None:                     # id = writer_page_line_word
                v = r["B"]
                v = v[:-2] if re.fullmatch(r"\d+\.0", v) else v
                words.setdefault((m[1], int(m[2])), []).append((int(m[3]), v.strip()))
        split = hsplit(f"bnhtrd_w{w}")
        for xml in sorted(glob.glob(f"{base}/Segmentation_Images/Lines/{w}/{w}_*.xml")):
            p = os.path.basename(xml)[:-4].split("_")[1]
            img = next((xml[:-4] + e for e in (".jpg", ".JPG", ".jpeg", ".png") if os.path.exists(xml[:-4] + e)), None)
            if img is None:
                continue
            bb = re.findall(r"<xmin>(\d+)</xmin>\s*<ymin>(\d+)</ymin>\s*<xmax>(\d+)</xmax>\s*<ymax>(\d+)</ymax>", open(xml).read())
            lines = []
            for i, b in enumerate(bb):
                t = " ".join(x for _, x in sorted(words.get((p, i + 1), [])) if x)
                b = [int(v) for v in b]
                yo = f"{base}/Segmentation_Images/Words/{w}/{w}_{p}/{w}_{p}_{i + 1}.txt"
                wb = []
                if os.path.exists(yo):                            # YOLO word boxes, relative to the line crop
                    bw, bh = b[2] - b[0], b[3] - b[1]
                    for r in open(yo):
                        v = r.split()
                        if len(v) == 5:
                            xc, yc, ww, hh = (float(z) for z in v[1:])
                            wb.append([b[0] + (xc - ww / 2) * bw, b[1] + (yc - hh / 2) * bh,
                                       b[0] + (xc + ww / 2) * bw, b[1] + (yc + hh / 2) * bh])
                lines.append(dict(box=b, text=t, hand=1, words=wb))
            yield page(img, lines, split, "bnhtrd", "CC BY 4.0")


def src_iam():
    for sp in ("train", "validation", "test"):
        t = pq.read_table(f"{RAW}/iam_line/data/{sp}.parquet").to_pylist()
        paths = dump_parquet_images(f"iam_{sp}", [d["image"]["bytes"] for d in t])
        for p, d in zip(paths, t):
            yield page(p, [dict(box=full(p), text=d["text"], hand=1)], "val" if sp == "validation" else sp, "iam",
                       "IAM non-commercial")


def src_prescription():
    base = f"{RAW}/prescription_bd/Dataset"
    for sd, sp in (("Training", "train"), ("Validation", "val"), ("Testing", "test")):
        for r in csv.DictReader(open(f"{base}/{sd}/{sd.lower()}_labels.csv")):
            p = f"{base}/{sd}/{sd.lower()}_words/{r['IMAGE']}"
            yield page(p, [dict(box=full(p), text=r["MEDICINE_NAME"], hand=1)], sp, "prescription", "CC BY 4.0")


def src_bnsynthwords():
    t = pq.read_table(glob.glob(f"{RAW}/bn_synth_words/data/*.parquet")[0]).to_pylist()
    for i, (p, d) in enumerate(zip(dump_parquet_images("bn_synth_words", [d["image"]["bytes"] for d in t]), t)):
        yield page(p, [dict(box=full(p), text=d["text"], hand=0)], hsplit(f"bsw{i}", 3, 3), "bnsynthwords", "CC BY 4.0")


def indra_det():
    import ocr
    from indra_ocr import model_dir
    return ocr.Detector(os.path.join(str(model_dir(None)), "det.onnx"), threads=2, pic=None)


def n_lines(det, gray):
    """Text rows in a line image by Indra det (padded with white so a tight crop still detects): boxes that
    overlap vertically are one row; slivers (< half the tallest box, e.g. a cut neighbour line) are ignored."""
    h = gray.shape[0]
    b = det(cv2.copyMakeBorder(gray, h, h, h, h, cv2.BORDER_CONSTANT, value=int(np.percentile(gray, 95))))
    if not len(b):
        return 0
    hs = b[:, 3] - b[:, 1]
    b = b[hs >= 0.5 * hs.max()]
    b = b[np.argsort((b[:, 1] + b[:, 3]) / 2)]
    rows, y1 = 1, b[0, 3]
    for u in b[1:]:
        if min(y1, u[3]) - u[1] < 0.5 * (u[3] - u[1]):
            rows += 1
        y1 = max(y1, u[3]) if min(y1, u[3]) - u[1] >= 0.5 * (u[3] - u[1]) else u[3]
    return rows


def src_bnprinted():
    """Real printed BN line images; images holding 2+ text rows under a one-line label are dropped (Indra det)."""
    det = indra_det()
    t = pq.read_table(glob.glob(f"{RAW}/bn_printed_val/data/*.parquet")[0]).to_pylist()
    for i, (p, d) in enumerate(zip(dump_parquet_images("bn_printed_val", [d["image"]["bytes"] for d in t]), t)):
        if d["type"] == "line" and n_lines(det, cv2.imread(p, cv2.IMREAD_GRAYSCALE)) <= 1:
            yield page(p, [dict(box=full(p), text=d["text"], hand=0)], hsplit(f"bnp{i}"), "bnprinted", "unstated")


def ordered_rows(b):
    """Det boxes -> top-to-bottom order, or None when two boxes share a row (columns: order unknown)."""
    b = b[np.argsort((b[:, 1] + b[:, 3]) / 2)]
    for u, v in zip(b, b[1:]):
        ov = min(u[3], v[3]) - max(u[1], v[1])
        if ov > 0.5 * min(u[3] - u[1], v[3] - v[1]):
            return None
    return b


def src_swapnil():
    """Printed BN pages (bn_ocr_swapnil, 1051 pages with page text): Indra det lines paired with GT lines in
    reading order when the counts match. Rows 0-99 are test (ocr_eval page set = rows 0-29)."""
    det = indra_det()
    pf = pq.ParquetFile(f"{BENCH}/bn_ocr_swapnil/data/train-00000-of-00001.parquet")
    d = f"{OUT}/img/swapnil"
    os.makedirs(d, exist_ok=True)
    i = n_ok = 0
    for rg in range(pf.num_row_groups):
        for row in pf.read_row_group(rg).to_pylist():
            p = f"{d}/{i}.png"
            g = cv2.imdecode(np.frombuffer(row["image"]["bytes"], np.uint8), cv2.IMREAD_GRAYSCALE)
            split = "test" if i < 100 else hsplit(f"swp{i}", 5, 0)
            i += 1
            gt = [x for x in (canon(l) for l in row["text"].split("\n")) if x]
            b = det(g)
            b = ordered_rows(b) if len(b) == len(gt) and len(gt) else None
            if b is None:
                continue
            if not os.path.exists(p):
                cv2.imwrite(p, g)
            n_ok += 1
            yield page(p, [dict(box=[float(v) for v in bb[:4]], text=t, hand=0) for bb, t in zip(b, gt)], split,
                       "swapnil", "unstated")
    print("swapnil pages paired", n_ok, "of", i, flush=True)


def det_pairs(det, g, text):
    """Page + page text -> Indra det line boxes paired with GT lines (top to bottom), or None if the counts
    differ or the layout has side-by-side boxes."""
    gt = [x for x in (canon(l) for l in text.split("\n")) if x]
    b = det(g)
    b = ordered_rows(b) if len(b) == len(gt) and len(gt) else None
    return None if b is None else [dict(box=[float(v) for v in bb[:4]], text=t, hand=0) for bb, t in zip(b, gt)]


def src_sahil():
    """Printed BN book pages (sahil74/bangla-ocr-1000: 700 page images + text): det-paired like Swapnil. Pages whose
    text also occurs in the Swapnil test pages are dropped; 10% of books (by title) are test."""
    det = indra_det()
    sw = pq.read_table(f"{BENCH}/bn_ocr_swapnil/data/train-00000-of-00001.parquet", columns=["text"]).column("text").to_pylist()
    sw_test = {canon(t)[:80] for t in sw[:100]}
    d = f"{OUT}/img/sahil"
    os.makedirs(d, exist_ok=True)
    for i, row in enumerate(pq.read_table(f"{RAW}/bn_ocr_sahil/data/train-00000-of-00001.parquet").to_pylist()):
        if row["image"] is None or canon(row["text"])[:80] in sw_test:
            continue
        g = cv2.imdecode(np.frombuffer(row["image"]["bytes"], np.uint8), cv2.IMREAD_GRAYSCALE)
        lines = det_pairs(det, g, row["text"])
        if lines:
            p = f"{d}/{i}.png"
            if not os.path.exists(p):
                cv2.imwrite(p, g)
            yield page(p, lines, hsplit(row["image"]["path"].rsplit("_Page", 1)[0], 5, 10), "sahil", "unstated")


def src_bench1000():
    """OCR-Bench1000-Bengali: 1000 synthetic printed BN line images (external benchmark; test only)."""
    for l in open(f"{RAW}/bn_ocr_bench1000/metadata.jsonl"):
        d = json.loads(l)
        p = f"{RAW}/bn_ocr_bench1000/{d['file_name']}"
        yield page(p, [dict(box=full(p), text=d["text"], hand=0)], "test", "bench1000", "MIT")


def src_rimes():
    """RIMES-2011 handwritten French letter lines (lines with chars outside the vocab, e.g. accents, drop out)."""
    for sp in ("train", "validation", "test"):
        t = pq.read_table(f"{RAW}/rimes_line/data/{sp}.parquet").to_pylist()
        for p, d in zip(dump_parquet_images(f"rimes_{sp}", [d["image"]["bytes"] for d in t]), t):
            yield page(p, [dict(box=full(p), text=d["text"], hand=1)], "val" if sp == "validation" else sp, "rimes", "MIT")


def src_textocr():
    """TextOCR scene-text number crops (train_numbers / test_numbers): noisy printed digits."""
    for sp in ("train", "test"):
        t = pq.read_table(glob.glob(f"{RAW}/textocr_numbers/data/{sp}_numbers-*.parquet")[0]).to_pylist()
        for i, (p, d) in enumerate(zip(dump_parquet_images(f"textocr_{sp}", [d["image"]["bytes"] for d in t]), t)):
            yield page(p, [dict(box=full(p), text=d["text"], hand=0)], sp if sp == "test" else hsplit(f"tx{i}", 5, 0),
                       "textocr", "CC BY 4.0")


_ARITH = re.compile(r"[0-9+\-=().]+")


def crohme_text(latex):
    """Space-tokenised CROHME LaTeX -> plain arithmetic line (digits + - × ÷ = ( ) .), or None."""
    t = latex.replace("\\times", "×").replace("\\div", "÷").replace(" ", "")
    return t if t and re.fullmatch(r"[0-9+\-=().×÷]+", t) and re.search(r"\d", t) else None


def src_crohme():
    """CROHME handwritten math (rendered ink): plain-arithmetic expressions only; train shards -> train,
    CROHME 2019 -> test."""
    for sp, pat in (("train", "train-*"), ("test", "2019-*")):
        for k, f in enumerate(sorted(glob.glob(f"{RAW}/crohme/data/{pat}"))):
            t = [d for d in pq.read_table(f).to_pylist() if crohme_text(d["label"])]
            for p, d in zip(dump_parquet_images(f"crohme_{sp}{k}", [d["image"]["bytes"] for d in t]), t):
                g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if g.mean() < 128:                                  # rendered white ink on black -> dark on white
                    g = 255 - g
                    cv2.imwrite(p, g)
                ys, xs = np.where(g < 128)
                if not len(xs):
                    continue
                g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]      # fixed 1366x768 canvas: cut the ink, and
                t = crohme_text(d["label"])                              # undo the horizontal squeeze of long
                w = max(g.shape[1], int(0.5 * g.shape[0] * len(t)))     # expressions (~0.5 h per symbol)
                g = cv2.copyMakeBorder(cv2.resize(g, (w, g.shape[0]), interpolation=cv2.INTER_CUBIC), 20, 20, 20, 20,
                                       cv2.BORDER_CONSTANT, value=255)
                p = p[:-4] + "_line.png"
                cv2.imwrite(p, g)
                box = [20, 20, g.shape[1] - 20, g.shape[0] - 20]
                yield page(p, [dict(box=box, text=crohme_text(d["label"]), hand=1)], sp, "crohme", "CROHME research")


def src_sroie():
    """SROIE receipts (FiftyOne export, 712 scans, line boxes + text): printed EN; 15% of receipts test."""
    for s in json.load(open(f"{BENCH}/sroie/samples.json"))["samples"]:
        p = f"{BENCH}/sroie/{s['filepath']}"
        W, H = s["metadata"]["width"], s["metadata"]["height"]
        lines = []
        for det in s["text_detections"]["detections"]:
            x, y, w, h = det["bounding_box"]
            if det["label"].strip():
                lines.append(dict(box=[x * W, y * H, (x + w) * W, (y + h) * H], text=det["label"], hand=0))
        yield page(p, lines, hsplit(os.path.basename(p), 5, 15), "sroie", "ICDAR SROIE research")


def group_rows(words, gap=1.5):
    """Word boxes [(x0,y0,x1,y1,text)] -> lines, and a line is cut where the horizontal gap > gap * height
    (form fields side by side)."""
    out = []
    for L in group_lines(words):
        ws = sorted(zip(L["words"], L["text"].split(" ")), key=lambda w: w[0][0]) if len(L["words"]) == len(L["text"].split(" ")) else None
        if ws is None:
            continue
        cur = [ws[0]]
        for w in ws[1:]:
            h = max(w[0][3] - w[0][1], cur[-1][0][3] - cur[-1][0][1])
            if w[0][0] - cur[-1][0][2] > gap * h:
                out.append(cur)
                cur = [w]
            else:
                cur.append(w)
        out.append(cur)
    return [dict(box=[min(b[0] for b, _ in c), min(b[1] for b, _ in c), max(b[2] for b, _ in c), max(b[3] for b, _ in c)],
                 text=" ".join(t for _, t in c), hand=0, words=[list(b) for b, _ in c]) for c in out]


def src_funsd():
    """FUNSD forms (scanned, mostly printed, some handwriting; nielsr/funsd words + boxes): words grouped into
    lines, split at wide gaps. Official train / test."""
    for sp, f in (("train", f"{RAW}/funsd_hf/data/train-00000-of-00001.parquet"),
                  ("test", f"{RAW}/funsd_hf/data/test-00000-of-00001.parquet")):
        if not os.path.exists(f):
            f = f"{BENCH}/funsd/data/{sp}-00000-of-00001.parquet"
            if not os.path.exists(f):
                continue
        t = pq.read_table(f).to_pylist()
        for p, d in zip(dump_parquet_images(f"funsd_{sp}", [d["image"]["bytes"] for d in t]), t):
            H, W = cv2.imread(p, cv2.IMREAD_GRAYSCALE).shape
            ws = [(b[0] * W / 1000, b[1] * H / 1000, b[2] * W / 1000, b[3] * H / 1000, w)      # boxes are 0..1000
                  for b, w in zip(d["bboxes"], d["words"]) if w.strip() and " " not in w.strip()]
            yield page(p, group_rows([(a, b, c, e, w.strip()) for a, b, c, e, w in ws]), sp, "funsd", "FUNSD research")


def cord_rows(gt):
    """CORD ground_truth json -> lines: words grouped by row_id; word boxes from the quads (deskewed crop)."""
    rows = {}
    for vl in json.loads(gt)["valid_line"]:
        for w in vl["words"]:
            q = w["quad"]
            rows.setdefault(w["row_id"], []).append((min(q["x1"], q["x4"]), min(q["y1"], q["y2"]),
                                                     max(q["x2"], q["x3"]), max(q["y3"], q["y4"]), w["text"]))
    out = []
    for ws in rows.values():
        ws.sort()
        out.append(dict(box=[min(w[0] for w in ws), min(w[1] for w in ws), max(w[2] for w in ws), max(w[3] for w in ws)],
                        text=" ".join(w[4] for w in ws), hand=0, words=[list(w[:4]) for w in ws]))
    return out


def src_cord():
    """CORD-v2 receipts (photos, often skewed): train shards 0-1 + validation; test from bench_raw (test only)."""
    for sp, pat in (("train", f"{RAW}/cord_train/data/train-*.parquet"), ("val", f"{RAW}/cord_train/data/validation-*.parquet"),
                    ("test", f"{BENCH}/cord/data/test-*.parquet")):
        for k, f in enumerate(sorted(glob.glob(pat))):
            t = pq.read_table(f).to_pylist()
            for p, d in zip(dump_parquet_images(f"cord_{sp}{k}", [d["image"]["bytes"] for d in t]), t):
                yield page(p, cord_rows(d["ground_truth"]), sp, "cord", "CC BY 4.0")


def src_rupom():
    t = pq.read_table(f"{BENCH}/bn_hw_rupom/data/train-00000-of-00001.parquet").to_pylist()
    for p, d in zip(dump_parquet_images("bn_hw_rupom", [d["image"]["bytes"] for d in t]), t):
        yield page(p, [dict(box=full(p), text=d["text"], hand=1)], "test", "rupom", "unstated")


def src_imgur5k():
    base = f"{RAW}/imgur5k"
    for sp in ("train", "val", "test"):
        d = json.load(open(f"{base}/repo/dataset_info/imgur5k_annotations_{sp}.json"))
        for idx, anns in d["index_to_ann_map"].items():
            p = f"{base}/images/{idx}.jpg"
            if not os.path.exists(p):
                continue
            lines = []
            for a in anns:
                t = d["ann_id"][a]["word"]
                bb = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", d["ann_id"][a]["bounding_box"])]
                if t and t != "." and "#" not in t and len(bb) == 5:
                    lines.append(dict(box=bb, text=t, hand=1))
            yield page(p, lines, sp, "imgur5k", "CC BY-NC 4.0")


def src_iiit():
    """IIIT-INDIC-HW-WORDS bn: {split}.txt rows "<FileName>, <VocabId>", label = line VocabId (0-based) of vocab.txt."""
    base = glob.glob(f"{RAW}/iiit_bn/**/vocab.txt", recursive=True)
    d = os.path.dirname(base[0])
    vocab = open(base[0], encoding="utf-8").read().split("\n")
    for sp in ("train", "val", "test"):
        for l in open(f"{d}/{sp}.txt", encoding="utf-8"):
            f, _, v = l.strip().rpartition(",")
            if f and v.strip().isdigit():
                p = f"{d}/{f.strip()}"
                yield page(p, [dict(box=full(p), text=vocab[int(v)].strip(), hand=1)], sp, "iiit", "CC BY 4.0 (AIKosh)")


SOURCES = dict(gnhk=src_gnhk, banglawriting=src_banglawriting, bnhtrd=src_bnhtrd, iam=src_iam,
               prescription=src_prescription, bnsynthwords=src_bnsynthwords, bnprinted=src_bnprinted,
               rupom=src_rupom, imgur5k=src_imgur5k, iiit=src_iiit, swapnil=src_swapnil, sroie=src_sroie,
               funsd=src_funsd, cord=src_cord, sahil=src_sahil, bench1000=src_bench1000, rimes=src_rimes,
               textocr=src_textocr, crohme=src_crohme)


def build_pages(names):
    os.makedirs(f"{OUT}/pages", exist_ok=True)
    for n in names:
        cnt = {}
        with open(f"{OUT}/pages/{n}.jsonl", "w") as f:
            for pg in SOURCES[n]():
                f.write(json.dumps(pg, ensure_ascii=False) + "\n")
                cnt[pg["split"]] = cnt.get(pg["split"], 0) + len(pg["lines"])
        print(n, "lines per split", cnt, flush=True)


# ---------------- mmap store (RAM safe: shared page cache, no per-item Python objects) ----------------
META = np.dtype([("off", "i8"), ("n", "i4"), ("w", "i4"), ("loff", "i8"), ("ln", "i4"), ("hand", "i1")])


def save_store(items, base):
    """items: iterable of (png bytes, text, hand) -> base.blob.npy / base.meta.npy / base.lab.npy (int16 ids)."""
    blob, labs, meta = bytearray(), [], []
    lo = 0
    for png, t, hand in items:
        ids = encode(t)
        if ids is None:
            continue
        meta.append((len(blob), len(png), int.from_bytes(png[16:20], "big"), lo, len(ids), hand))
        blob += png
        labs += ids
        lo += len(ids)
    # tmp + rename: a running training keeps its mmap of the old files (np.save in place would truncate them)
    for ext, arr in ((".blob.npy", np.frombuffer(bytes(blob), np.uint8)), (".lab.npy", np.array(labs, np.int16)),
                     (".meta.npy", np.array(meta, META))):
        np.save(base + ext + ".tmp.npy", arr)
        os.replace(base + ext + ".tmp.npy", base + ext)
    return len(meta)


class Store:
    """Read-only memory-mapped crop set: s[i] -> (png bytes, ids list); s.w widths; s.text(i)."""

    def __init__(self, base):
        self.blob = np.load(base + ".blob.npy", mmap_mode="r")
        self.meta = np.load(base + ".meta.npy")
        self.lab = np.load(base + ".lab.npy", mmap_mode="r")
        self.w = self.meta["w"]

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        m = self.meta[i]
        return self.blob[m["off"]:m["off"] + m["n"]].tobytes(), self.lab[m["loff"]:m["loff"] + m["ln"]].tolist()

    def text(self, i):
        from vocab import CHARS
        return "".join(CHARS[c - 1] for c in self[i][1])


def stores(split, names=None, rec=None):
    """{src: [Store, ...]} for <rec>/<src>_<split>[_<shard>].meta.npy (rec: data/ocr/rec, or rec48 for 48 px)"""
    out = {}
    for p in sorted(glob.glob(f"{rec or REC}/*_{split}*.meta.npy")):
        base = p[:-9]
        src = os.path.basename(base).split(f"_{split}")[0]
        if not names or src in names:
            out.setdefault(src, []).append(Store(base))
    return out


# ---------------- crops ----------------
def rot_crop(gray, cx, cy, w, h, deg):
    """Upright crop of a box rotated `deg` counter-clockwise, with page context (1 h left/right, 2 h above/below)."""
    mx, my = h, 2 * h
    M = cv2.getRotationMatrix2D((cx, cy), -deg, 1.0)
    M[:, 2] += (w / 2 + mx - cx, h / 2 + my - cy)
    out = cv2.warpAffine(gray, M, (int(w + 2 * mx), int(h + 2 * my)), flags=cv2.INTER_LINEAR, borderValue=255)
    return out, [mx, my, mx + w, my + h]


def line_rect(words):
    """Word boxes of one line -> rotated rect (cx, cy, length, text height, deg ccw) from a baseline fit."""
    w = np.array(words, np.float64)
    c = np.stack([(w[:, 0] + w[:, 2]) / 2, (w[:, 1] + w[:, 3]) / 2], 1)
    hs, ws = w[:, 3] - w[:, 1], w[:, 2] - w[:, 0]
    a = np.polyfit(c[:, 0], c[:, 1], 1)[0] if len(w) >= 2 and np.ptp(c[:, 0]) > 2 * np.median(hs) else 0.0
    a = float(np.clip(a, -0.5, 0.5))
    h = max(np.median(hs - abs(a) * ws), 0.6 * np.median(hs))     # axis box of a slanted word is too tall
    x0, x1 = w[:, 0].min(), w[:, 2].max()
    cx = (x0 + x1) / 2
    cy = np.mean(c[:, 1]) + a * (cx - np.mean(c[:, 0]))
    L = (x1 - x0) * np.sqrt(1 + a * a) + abs(a) * h
    return cx, cy, L, h * 1.1, -np.degrees(np.arctan(a))


def ink_png(ink):
    """Ink map [0,1] -> PNG bytes with 16 grey levels (3.5x smaller than 8 bit; GPU augmentation adds noise anyway)."""
    return cv2.imencode(".png", (np.round(ink * 15) * 17).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 3])[1].tobytes()


def ctc_ok(t, steps):
    return len(t) + sum(a == b for a, b in zip(t, t[1:])) <= steps


def crop_page(pg):
    from indra_ocr.rec_preproc import crop_line
    gray = cv2.imread(f"{ROOT}/{pg['image']}", cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return []
    out = []
    for l in pg["lines"]:
        t = canon(l["text"])
        if encode(t) is None:
            continue
        g, box = gray, l["box"]
        if l.get("words"):
            g, box = rot_crop(gray, *line_rect(l["words"]))
        elif box is None:
            box = [0, 0, gray.shape[1], gray.shape[0]]
        elif len(box) == 5:
            if box[2] < 4 or box[3] < 4:
                continue
            g, box = rot_crop(gray, *box)
        if box[2] - box[0] < 2 or box[3] - box[1] < 4:
            continue
        ink = crop_line(g, box, height=LINE_H, window=3, max_width=MAX_W)
        if ctc_ok(t, ink.shape[1] // 4):                  # CTC needs a step per char + one per repeat
            out.append((ink_png(ink), t, l.get("hand", 1)))
    return out


def build_crops(names):
    os.makedirs(REC, exist_ok=True)
    for n in names:
        pages = [json.loads(l) for l in open(f"{OUT}/pages/{n}.jsonl")]
        by = {}
        with Pool(int(os.environ.get("ORUS_PROCS", 6))) as pool:
            for pg, cr in zip(pages, pool.imap(crop_page, pages, chunksize=16)):
                by.setdefault(pg["split"], []).extend(cr)
        for sp, items in by.items():
            save_store(items, f"{REC}/{n}_{sp}")
        print(n, {k: len(v) for k, v in by.items()}, flush=True)


if __name__ == "__main__":
    cmd, names = sys.argv[1], sys.argv[2:] or list(SOURCES)
    {"pages": build_pages, "crops": build_crops}[cmd](names)
