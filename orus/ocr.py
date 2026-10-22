"""Orus-OCR-1.0-alpha: Bangla + English (+ mixed) OCR for handwritten and printed pages, on CPU.

  python ocr.py IMAGE [IMAGE ...] [--json] [--weights DIR] [--fast]
  python ocr.py export CKPT.pt OUT.onnx        # fp32 ONNX of a trained recognizer (name / vocab / cfg in metadata)
  python ocr.py selftest                       # layout + language rules, no model needed

Pipeline: deskew -> line detector (Indra OCR det.onnx) -> drop lines on pictures -> CTC line recognizer
(conv + 2-layer BiLSTM, 48 px) -> gated char 5-gram LM + word lexicon -> confidence-gated re-read -> layout.
Output: {"model", "language": bn|en|bn+en, "handwritten": "Yes"|"No", "text", "lines": [...], "ms"}
"""
import json
import math
import os
import re
import sys
import time

import cv2
import numpy as np

NAME = "Orus-OCR-1.0-alpha"
HF_REPO = "orus-research/Orus-OCR-V1.0"
WEIGHTS = ["orus_rec.onnx", "orus_lm.npz", "orus_lex.txt.gz", "orus_pic.onnx", "orus_hand.onnx", "orus_hand.json"]
BN = re.compile(r"[ঀ-৿]")
EN = re.compile(r"[A-Za-z]")


def weights_dir(path=None):
    """Folder holding the Orus weights: `path`, else $ORUS_WEIGHTS, else a (cached) download of HF_REPO."""
    path = path or os.environ.get("ORUS_WEIGHTS")
    if path:
        return path
    from huggingface_hub import snapshot_download
    return snapshot_download(HF_REPO, allow_patterns=WEIGHTS)


def _ort(path, threads):
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


# ---------------- recognizer ----------------
class Recognizer:
    """Line recognizer from an exported .onnx (ONNX Runtime, CPU) or a .pt checkpoint (PyTorch).
    lm: char n-gram LM (.npz) used on lines with confidence in [0.3, gate); lex: word lexicon (.txt.gz)."""

    def __init__(self, path, threads=4, device=None, lm=None, alpha=0.3, beta=1.0, gate=0.9, lex=None):
        from vocab import CHARS
        self.chars = CHARS
        self.lm = self.lex = None
        if lex and os.path.exists(lex):
            from ocr_words import Lexicon
            self.lex = Lexicon(lex)
        if lm and os.path.exists(lm):
            from indra_ocr.lm import NgramLM
            self.lm, self.alpha, self.beta, self.gate = NgramLM(lm), alpha, beta, gate
        self.onnx = path.endswith(".onnx")
        if self.onnx:
            self.sess = _ort(path, threads)
            meta = self.sess.get_modelmeta().custom_metadata_map
            self.height = json.loads(meta["cfg"]).get("height", 32) if "cfg" in meta else 32
            self.name = meta.get("name", NAME)
        else:
            import torch
            from ocr_model import RecNet
            self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            ck = torch.load(path, map_location=self.dev)
            self.height = ck["cfg"].get("height", 32)
            self.name = ck.get("name", NAME)
            self.net = RecNet(**ck["cfg"]).eval().to(self.dev)
            self.net.load_state_dict(ck["model"])
            torch.set_num_threads(threads)

    def logits(self, x, w):
        if self.onnx:
            return self.sess.run(["logits", "lengths"], {"x": x, "widths": w})
        import torch
        with torch.no_grad():
            lg, ln = self.net(torch.from_numpy(x).to(self.dev), torch.from_numpy(w).to(self.dev))
        return lg.float().cpu().numpy(), ln.cpu().numpy()

    def line_logprobs(self, crops, bs=64, area=64 * 1024):
        """Ink maps [3*height, W] -> [(log-probs [T, V], confidence)] per crop, width-bucketed batches."""
        from indra_ocr.rec_preproc import pad_batch, width_batches
        out = [None] * len(crops)
        for part in width_batches([c.shape[1] for c in crops], bs, max_area=area):
            x, w = pad_batch([crops[i] for i in part])
            lg, ln = self.logits(x, w)
            mx = lg.max(-1, keepdims=True)
            lp = lg - mx - np.log(np.exp(lg - mx).sum(-1, keepdims=True))
            for k, i in enumerate(part):
                l = lp[k, :int(ln[k])]
                ids = l.argmax(-1)
                keep = (ids != 0) & np.concatenate([[True], ids[1:] != ids[:-1]])
                out[i] = (l, float(np.exp(l.max(-1)[keep]).mean()) if keep.any() else 0.0)
        return out

    def read_crops(self, crops, bs=64, area=64 * 1024, tta=None):
        """Ink maps [3*height, W] -> (text, confidence) per crop. Greedy CTC; gated char-LM beam on unsure lines;
        word lexicon correction. tta: lines under this confidence are read again horizontally squeezed / stretched
        (x0.8, x1.25) and the most confident reading is kept (None = off)."""
        from vocab import decode
        res = self.line_logprobs(crops, bs, area)
        if tta:
            low = [i for i, (_, c) in enumerate(res) if c < tta]
            for f in (0.8, 1.25):
                if not low:
                    break
                var = [cv2.resize(crops[i], (max(8, int(crops[i].shape[1] * f)), crops[i].shape[0]),
                                  interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR) for i in low]
                for i, r in zip(low, self.line_logprobs(var, bs, area)):
                    if r[1] > res[i][1]:
                        res[i] = r
        out = []
        for l, conf in res:
            txt = decode(l.argmax(-1).tolist())
            steps = None
            if self.lm is not None and 0.3 <= conf < self.gate:     # gated LM rescoring of unsure lines
                from indra_ocr.lm import beam_search
                top = np.argsort(-l, -1)[:, :8]
                tok, steps = beam_search(l[:, 0], np.take_along_axis(l, top, -1), top, self.lm, self.alpha, self.beta, 8)
                txt = "".join(self.chars[c - 1] for c in tok)
            if self.lex is not None:
                if steps is None:
                    from ocr_words import greedy_steps
                    txt, steps = greedy_steps(l)
                txt = self.lex.correct(txt, steps, l)
            out.append((txt, conf))
        return out

    def read_boxes(self, items, tta=None):
        """[(gray, box)] -> [text]."""
        from indra_ocr.rec_preproc import crop_line
        return [t for t, _ in self.read_crops([crop_line(g, b, height=self.height, window=3) for g, b in items], tta=tta)]


# ---------------- pictures (text inside photos / charts is ignored) ----------------
class PictureNet:
    """Picture rectangles [P, 4] in page pixels from a ~30k-param net on a 384 px thumbnail (1/8 output grid)."""
    S, STRIDE = 384, 8

    def __init__(self, path, thr=0.5, min_area=0.01):
        self.sess, self.thr, self.min_area = _ort(path, 1), thr, min_area

    def __call__(self, gray):
        s = self.S / max(gray.shape)
        g = cv2.resize(gray, (max(1, round(gray.shape[1] * s)), max(1, round(gray.shape[0] * s))), interpolation=cv2.INTER_AREA)
        x = np.zeros((self.S, self.S), np.float32)
        x[:g.shape[0], :g.shape[1]] = 1.0 - g.astype(np.float32) / 255.0
        p = self.sess.run(None, {"x": x[None, None]})[0][0, 0]
        mask = (p > np.log(self.thr / (1 - self.thr))).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        f = self.STRIDE / s
        out = [[x0 * f, y0 * f, min((x0 + w) * f, gray.shape[1]), min((y0 + h) * f, gray.shape[0])]
               for x0, y0, w, h, a in st[1:n] if a * self.STRIDE ** 2 / (s * s) >= self.min_area * gray.size]
        return np.array(out, np.float32).reshape(-1, 4)


# ---------------- detector ----------------
class Detector:
    """Indra OCR line detector (det.onnx) + Indra's decode. Returns line boxes [N,5] in page pixels; lines centred
    inside a picture region are dropped (picture boxes kept in self.pictures)."""

    def __init__(self, path, threads=4, long_side=1024, pic=None):
        self.sess, self.long_side = _ort(path, threads), long_side
        self.pic = PictureNet(pic) if pic and os.path.exists(pic) else None

    def __call__(self, gray):
        from indra_ocr.det_post import decode, page_to_input
        x, s = page_to_input(gray, self.long_side)
        out = self.sess.run(None, {"x": x[None, None]})[0][0]
        b = decode(out[:3])
        if len(b):
            med = float(np.median(b[:, 3] - b[:, 1]))
            if med < 12 and len(b) >= 8:                  # tiny text: run again at a larger scale
                ls = int(min(2048, self.long_side * 16 / max(med, 1))) // 32 * 32
                if ls > self.long_side:
                    x, s = page_to_input(gray, ls)
                    out = self.sess.run(None, {"x": x[None, None]})[0][0]
                    b = decode(out[:3])
            b[:, :4] /= s
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, gray.shape[1])
            b[:, [1, 3]] = b[:, [1, 3]].clip(0, gray.shape[0])
        self.pictures = self.pic(gray) if self.pic else np.zeros((0, 4), np.float32)
        if len(b) and len(self.pictures):
            cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
            p = self.pictures
            inside = ((cx[:, None] > p[None, :, 0]) & (cx[:, None] < p[None, :, 2]) &
                      (cy[:, None] > p[None, :, 1]) & (cy[:, None] < p[None, :, 3])).any(1)
            b = b[~inside]
        return b


# ---------------- handwritten yes/no ----------------
class HandFlag:
    """~98k-param tile classifier (handwritten vs printed) on ink tiles of the page; page score = mean of the top 20%
    tile probabilities, compared with the threshold in orus_hand.json."""

    def __init__(self, path, threads=2):
        self.sess = _ort(path, threads)
        self.thr = json.load(open(path[:-5] + ".json"))["thr"]

    def __call__(self, img):
        import tiles as T
        g, _ = T.normalize(img)
        x, _ = T.tiles(g)
        if len(x) == 0:
            return False, 0.0
        x = ((x.astype(np.float32) / 255.0 - 0.5) / 0.25)[:, None]
        p = 1 / (1 + np.exp(-self.sess.run(None, {"tiles": x})[0].ravel()))
        s = float(np.sort(p)[-max(1, math.ceil(0.2 * len(p))):].mean())
        return s > self.thr, s


# ---------------- layout ----------------
def language(text, min_chars=3, min_share=0.05):
    b, e = len(BN.findall(text)), len(EN.findall(text))
    if b + e == 0:
        return "unknown"
    has_b = b >= min_chars and b / (b + e) >= min_share
    has_e = e >= min_chars and e / (b + e) >= min_share
    return "bn+en" if has_b and has_e else "bn" if b >= e else "en"


def chain(lines, max_gap=2.5, max_dy=0.6):
    """Join detector fragments of one slanted / curved line: left to right, a fragment continues the chain whose
    last piece ends just before it (gap < max_gap * h) at nearly the same height (|dy| < max_dy * h)."""
    L = sorted([l for l in lines if l["text"].strip()], key=lambda l: l["bbox"][0])
    chains = []
    for l in L:
        x0, y0, x1, y1 = l["bbox"]
        h, cy = y1 - y0, (y0 + y1) / 2
        best, bd = None, None
        for c in chains:
            px0, py0, px1, py1 = c[-1]["bbox"]
            ph = max(h, py1 - py0)
            gap, dy = x0 - px1, abs(cy - (py0 + py1) / 2)
            if -0.5 * ph < gap < max_gap * ph and dy < max_dy * ph and (bd is None or dy < bd):
                best, bd = c, dy
        (best.append(l) if best is not None else chains.append([l]))
    out = []
    for c in chains:
        b = [min(l["bbox"][0] for l in c), min(l["bbox"][1] for l in c), max(l["bbox"][2] for l in c), max(l["bbox"][3] for l in c)]
        if len(c) > 1:                                   # keep the first fragment's height for row grouping
            b[1], b[3] = c[0]["bbox"][1], c[0]["bbox"][3]
        out.append(dict(bbox=b, text=" ".join(l["text"].strip() for l in c), conf=min(l.get("conf", 1) for l in c),
                        lang=language(" ".join(l["text"] for l in c))))
    return out


def assemble(lines):
    """lines: [{bbox, text}] -> structured text. Reading order = rows top to bottom (boxes overlapping
    vertically share a row, left to right); indentation from the left margin; a blank line for a big gap.
    Single-column reading order: multi-column pages are read row by row across the columns."""
    L = [l for l in lines if l["text"].strip()]
    if not L:
        return ""
    h = float(np.median([l["bbox"][3] - l["bbox"][1] for l in L]))
    L.sort(key=lambda l: (l["bbox"][1] + l["bbox"][3]) / 2)
    rows = []
    for l in L:
        cy = (l["bbox"][1] + l["bbox"][3]) / 2
        if rows and abs(cy - rows[-1]["cy"]) < 0.5 * h:
            rows[-1]["items"].append(l)
        else:
            rows.append(dict(cy=cy, items=[l]))
    left = min(l["bbox"][0] for l in L)
    cw = 0.45 * h                                          # rough glyph width for indentation / gaps
    out, prev = [], None
    for r in rows:
        it = sorted(r["items"], key=lambda l: l["bbox"][0])
        top = min(l["bbox"][1] for l in it)
        if prev is not None and top - prev > 1.0 * h:
            out.append("")
        s = " " * int(round(max(0.0, it[0]["bbox"][0] - left) / cw / 2))
        for i, l in enumerate(it):
            if i:
                gap = l["bbox"][0] - it[i - 1]["bbox"][2]
                s += " " * max(1, min(8, int(round(gap / cw / 2))))
            s += l["text"].strip()
        out.append(s.rstrip())
        prev = max(l["bbox"][3] for l in it)
    return "\n".join(out)


class OCR:
    def __init__(self, weights=None, threads=4, hand=True, tta=0.8, lex=True, det=None):
        """weights: folder with the Orus weights (default: $ORUS_WEIGHTS or a download of the HF repo).
        det: Indra det.onnx (default: downloaded by indra-ocr from datalab-to/indra-ocr).
        tta=None and lex=False give the fast path (greedy + gated char LM only)."""
        from indra_ocr import model_dir
        w = weights_dir(weights)
        self.tta = tta
        self.rec = Recognizer(f"{w}/orus_rec.onnx", threads, lm=f"{w}/orus_lm.npz", lex=f"{w}/orus_lex.txt.gz" if lex else None)
        self.det = Detector(det or os.path.join(str(model_dir(None)), "det.onnx"), threads, pic=f"{w}/orus_pic.onnx")
        self.hand = HandFlag(f"{w}/orus_hand.onnx") if hand else None

    def __call__(self, img):
        from indra_ocr.pipeline import DESKEW_MIN, estimate_skew, preview, rotate_page
        from indra_ocr.rec_preproc import crop_line
        if isinstance(img, str):
            path, img = img, cv2.imread(img)
            if img is None:
                raise FileNotFoundError(path)
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t0 = time.perf_counter()
        a = estimate_skew(preview(gray))
        if abs(a) >= DESKEW_MIN:
            gray = rotate_page(gray, a)
        boxes = self.det(gray)
        t1 = time.perf_counter()
        res = self.rec.read_crops([crop_line(gray, b[:4], height=self.rec.height, window=3) for b in boxes], tta=self.tta)
        t2 = time.perf_counter()
        lines = [dict(bbox=[round(float(v), 1) for v in b[:4]], text=t, conf=round(c, 3), lang=language(t))
                 for b, (t, c) in zip(boxes, res) if c >= 0.35 and t.strip()]
        text = assemble(chain(lines))
        hw, hs = self.hand(img) if self.hand is not None else (None, None)
        t3 = time.perf_counter()
        return dict(model=self.rec.name, language=language(text), handwritten=None if hw is None else ("Yes" if hw else "No"),
                    handwritten_score=None if hs is None else round(hs, 3), text=text, lines=lines,
                    ms=dict(detect=round(1000 * (t1 - t0)), recognize=round(1000 * (t2 - t1)),
                            handwritten=round(1000 * (t3 - t2)), total=round(1000 * (t3 - t0))))


# ---------------- export ----------------
def export(ckpt, out):
    """Checkpoint from ocr_train.py -> fp32 ONNX with name / vocab / cfg / val metadata; checks argmax == PyTorch."""
    import onnx
    import onnxruntime as ort
    import torch
    from ocr_model import RecNet
    ck = torch.load(ckpt, map_location="cpu")
    net = RecNet(**ck["cfg"]).eval()
    net.load_state_dict(ck["model"])
    x, w = torch.zeros(2, 1, 3 * net.height, 256), torch.tensor([256, 200])
    torch.onnx.export(net, (x, w), out, input_names=["x", "widths"], output_names=["logits", "lengths"],
                      dynamic_axes={"x": {0: "B", 3: "W"}, "widths": {0: "B"}, "logits": {0: "B", 1: "T"},
                                    "lengths": {0: "B"}}, opset_version=17, dynamo=False)
    m = onnx.load(out)
    for k, v in dict(name=ck.get("name", NAME), vocab=json.dumps(ck["chars"], ensure_ascii=False), cfg=json.dumps(ck["cfg"]),
                     val=json.dumps(ck.get("val", {}))).items():
        m.metadata_props.add(key=k, value=v)
    onnx.save(m, out)
    xr = (torch.rand(3, 1, 3 * net.height, 320) > 0.85).float()
    wr = torch.tensor([320, 250, 100])
    with torch.no_grad():
        ref = net(xr, wr)[0].numpy()
    lg = ort.InferenceSession(out, providers=["CPUExecutionProvider"]).run(["logits"], {"x": xr.numpy(), "widths": wr.numpy()})[0]
    agree = float((lg.argmax(-1) == ref.argmax(-1)).mean())
    print(out, f"{os.path.getsize(out) / 1e6:.2f} MB", "argmax agreement vs torch", round(agree, 4))
    assert agree > 0.999


def selftest():
    assert language("আমি rice খাই and tea") == "bn+en" and language("hello world") == "en" and language("আমি ভাত") == "bn"
    L = [dict(bbox=[10, 10, 300, 30], text="Shopping list"), dict(bbox=[30, 40, 200, 60], text="1. চাল"),
         dict(bbox=[30, 70, 200, 90], text="2. eggs"), dict(bbox=[10, 140, 300, 160], text="শেষ")]
    t = assemble(L).split("\n")
    assert t[0] == "Shopping list" and t[1].startswith(" ") and "" in t, t
    print("selftest ok")


def main():
    a = sys.argv[1:]
    if a and a[0] == "export":
        return export(*a[1:3])
    if not a or a[0] == "selftest":
        return selftest()
    kw = {}
    if "--weights" in a:
        i = a.index("--weights")
        kw["weights"] = a[i + 1]
        del a[i:i + 2]
    if "--fast" in a:
        kw.update(tta=None, lex=False)
    as_json = "--json" in a
    a = [x for x in a if x not in ("--json", "--fast")]
    o = OCR(**kw)
    for f in a:
        r = o(f)
        if as_json:
            print(json.dumps(r, ensure_ascii=False, indent=1))
        else:
            print(f"== {f}\nLanguage: {r['language']}\nHandwritten: {r['handwritten']}\n{r['text']}\n({r['ms']['total']} ms)")


if __name__ == "__main__":
    main()
