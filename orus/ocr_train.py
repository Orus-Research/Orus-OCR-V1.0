"""Train the Orus-OCR line recognizer (1.0-alpha: 48 px + 2-layer BiLSTM 128 on the full data mix).

  python ocr_train.py rec --name orus_rec --init indra|scratch [--rnn 128 --height 48] [--resume CKPT] [--steps 70000]
      CTC line recognizer on the mmap line stores in data/ocr/rec<height> (real crops + synthetic), mixed by
      SOURCE_W, width-bucketed batches, GPU augmentation, bf16; best checkpoint by mean val CER of the real sets
      -> runs/<name>.pt (+ runs/<name>_last.pt and the newest 3 per-eval copies). Needs a CUDA GPU.
"""
import argparse
import glob
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from rapidfuzz.distance import Levenshtein

from ocr_model import NAME, RecNet, adapt, load_indra
from vocab import CHARS, decode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root: data/, runs/, models/
# share of each training batch per source (renormalised over the sources present); 1.0-alpha mix
SOURCE_W = dict(
    # synthetic: general font text, stitched real handwritten words, numbers / arithmetic, receipts, printed Rx,
    # book / newspaper lines, handwritten digits (MNIST) / arithmetic; 8-12% of each kind is heavily degraded
    synth_font=0.15, synth_stitch=0.12, synth_num=0.04, synth_receipt=0.03, synth_rx=0.015, synth_book=0.04,
    synth_hdigit=0.02,
    # real handwriting
    bnhtrd=0.10, iiit=0.07, banglawriting=0.04, gnhk=0.05, iam=0.05, imgur5k=0.06, rimes=0.025, prescription=0.03,
    crohme=0.005,
    # real print
    bnsynthwords=0.015, bnprinted=0.015, swapnil=0.03, sahil=0.02, sroie=0.025, cord=0.015, funsd=0.015, textocr=0.01,
    distill=0.02)                     # teacher-agreed lines of unlabeled EN pages (ocr_distill.py)
VAL = ["bnhtrd", "banglawriting", "iiit", "gnhk", "iam", "imgur5k", "prescription", "rimes", "swapnil", "sroie", "cord",
       "textocr", "synth_stitch", "synth_font", "synth_num", "synth_receipt", "synth_rx", "synth_book", "synth_hdigit"]
VAL01 = ["bnhtrd", "banglawriting", "iiit", "gnhk", "iam", "imgur5k", "prescription"]    # 0.1-alpha's val sets


def png_w(b):
    return int.from_bytes(b[16:20], "big")


class Multi:
    """Several mmap Stores of one source seen as one list: m[i] -> (png, ids, text)."""

    def __init__(self, sts):
        self.sts = sts
        self.cum = np.cumsum([0] + [len(s) for s in sts])
        self.w = np.concatenate([s.w for s in sts])

    def __len__(self):
        return int(self.cum[-1])

    def __getitem__(self, i):
        k = int(np.searchsorted(self.cum, i, side="right") - 1)
        s, j = self.sts[k], int(i - self.cum[k])
        png, ids = s[j]
        return png, ids, s.text(j)


def load_split(split, names=None, height=32):
    from ocr_data import OUT, stores
    return {k: Multi(v) for k, v in stores(split, names, f"{OUT}/rec" if height == 32 else f"{OUT}/rec{height}").items()}


class Batches(torch.utils.data.IterableDataset):
    """Endless width-bucketed batches: draw 4*B items by source weight, sort by width, cut into 4 batches."""

    def __init__(self, data, weights, area=96 * 1024, bmax=192, stretch=(0.8, 1.25)):
        self.data, self.area, self.bmax, self.stretch = data, area, bmax, stretch
        self.srcs = [s for s in weights if s in data]
        w = np.array([weights[s] for s in self.srcs])
        self.p = w / w.sum()

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        r = np.random.default_rng((wi.id if wi else 0) * 7919 + int(time.time() * 1000) % 100000)
        while True:
            src = r.choice(len(self.srcs), size=4 * self.bmax // 2, p=self.p)
            items = []
            for s in src:
                L = self.data[self.srcs[s]]
                b, ids, _ = L[int(r.integers(len(L)))]
                items.append((b, ids, r.uniform(*self.stretch)))
            items.sort(key=lambda x: png_w(x[0]) * x[2])
            for k in range(0, len(items), len(items) // 4):
                yield self.collate(items[k:k + len(items) // 4])

    def collate(self, items):
        ims, tg = [], []
        for b, ids, s in items:
            im = cv2.imdecode(np.frombuffer(b, np.uint8), 0)
            w = max(8, int(im.shape[1] * s))
            if len(ids) + sum(a == c for a, c in zip(ids, ids[1:])) > w // 4:
                w = im.shape[1]
            im = cv2.resize(im, (w, im.shape[0]), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            ims.append(im)
            tg.append(ids)
        Wm = max(i.shape[1] for i in ims)
        n = max(1, min(len(ims), self.area // max(Wm, 64)))
        ims, tg = ims[-n:], tg[-n:]                       # keep the widest (sorted) so the area cap holds
        Wm = int(math.ceil(max(i.shape[1] for i in ims) / 8) * 8)
        x = np.zeros((len(ims), 1, ims[0].shape[0], Wm), np.uint8)
        for i, im in enumerate(ims):
            x[i, 0, :, :im.shape[1]] = im
        return (torch.from_numpy(x), torch.tensor([i.shape[1] for i in ims]),
                torch.tensor([c for t in tg for c in t]), torch.tensor([len(t) for t in tg]))


def gpu_aug(x):
    """x: [B,1,3h,W] ink map in [0,1]. Small affine (keeps the line in the middle band), stroke width,
    contrast/background, blur, noise."""
    B, dev = x.shape[0], x.device
    ang = (torch.rand(B, device=dev) - 0.5) * math.radians(3)
    sc = torch.exp((torch.rand(B, device=dev) - 0.5) * 0.12)
    ar = x.shape[3] / x.shape[2]
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    ty = (torch.rand(B, device=dev) - 0.5) * 0.08
    sh = (torch.rand(B, device=dev) - 0.5) * 0.4                                # italic shear
    theta = torch.stack([torch.stack([cos, (-sin + sh) / ar, torch.zeros_like(ty)], 1),
                         torch.stack([sin * ar, cos, ty], 1)], 1)
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="zeros", align_corners=False)
    k = torch.rand(1).item()
    if k < 0.2:
        x = F.max_pool2d(x, 3, 1, 1)                                             # bolder strokes
    elif k < 0.35:
        x = -F.max_pool2d(-x, 3, 1, 1)                                           # thinner strokes
    if torch.rand(1) < 0.3:
        s = float(torch.empty(1).uniform_(0.4, 1.2))
        kk = torch.exp(-torch.arange(-2, 3, device=dev, dtype=x.dtype) ** 2 / (2 * s * s)); kk /= kk.sum()
        x = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="replicate"), (kk[:, None] * kk[None]).view(1, 1, 5, 5))
    c = 0.5 + 0.7 * torch.rand(B, 1, 1, 1, device=dev)
    b = 0.15 * torch.rand(B, 1, 1, 1, device=dev) * (torch.rand(B, 1, 1, 1, device=dev) < 0.3)
    x = x * c + b + torch.randn_like(x) * 0.06 * torch.rand(B, 1, 1, 1, device=dev)
    return x.clamp(0, 1)


def greedy(logits, lengths):
    ids = logits.argmax(-1).cpu().numpy()
    return [decode(ids[i, :lengths[i]]) for i in range(len(ids))]


@torch.no_grad()
def evaluate(model, val, dev, cap=1500, bs=64):
    model.eval()
    res = {}
    for src, L in val.items():
        items = [L[i] for i in range(min(cap, len(L)))]
        order = sorted(range(len(items)), key=lambda i: png_w(items[i][0]))
        ed = n = 0
        for k in range(0, len(order), bs):
            chunk = [items[i] for i in order[k:k + bs]]
            ims = [cv2.imdecode(np.frombuffer(b, np.uint8), 0) for b, _, _ in chunk]
            Wm = int(math.ceil(max(i.shape[1] for i in ims) / 8) * 8)
            x = np.zeros((len(ims), 1, ims[0].shape[0], Wm), np.float32)
            for i, im in enumerate(ims):
                x[i, 0, :, :im.shape[1]] = im / 255.0
            w = torch.tensor([i.shape[1] for i in ims], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                lg, ln = model(torch.from_numpy(x).to(dev), w)
            for p, (_, _, t) in zip(greedy(lg.float(), ln.tolist()), chunk):
                ed += Levenshtein.distance(p, t)
                n += len(t)
        res[src] = ed / max(1, n)
    model.train()
    return res


def train_rec(a):
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = False                  # widths vary per batch; autotune per shape is slow
    cfg = dict(c=a.c, e=a.e, depth=a.depth, rnn=a.rnn, height=a.height)
    name = NAME
    model = RecNet(**cfg)
    if a.init == "indra":
        load_indra(model, "rec.onnx")
    model.to(dev)
    if a.resume:
        miss = model.load_state_dict(adapt(torch.load(a.resume, map_location=dev)["model"], model), strict=False)
        print("resume: new (not in checkpoint)", sorted({k.split(".")[0] for k in miss.missing_keys}),
              "unused", miss.unexpected_keys, flush=True)
    print(name, "rec params", sum(p.numel() for p in model.parameters()), flush=True)
    train = load_split("train", height=a.height)
    val = {k: v for k, v in load_split("val", height=a.height).items() if k in VAL}
    print("train", {k: len(v) for k, v in train.items()}, "\nval", {k: len(v) for k, v in val.items()}, flush=True)
    W = dict(SOURCE_W)
    if a.real_only:
        W = {k: v for k, v in W.items() if not k.startswith("synth")}
    dl = torch.utils.data.DataLoader(Batches(train, W, area=int(a.area * 96 * 1024 * 32 // a.height), bmax=int(192 * a.area)), batch_size=None, num_workers=a.workers,
                                     pin_memory=True, prefetch_factor=4, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05,
                                                div_factor=20, final_div_factor=100)
    os.makedirs(f"{ROOT}/runs", exist_ok=True)
    best, t0, run = 9.9, time.time(), 0.0
    it = iter(dl)
    for step in range(1, a.steps + 1):
        x, w, tg, tl = next(it)
        x = gpu_aug(x.to(dev, non_blocking=True).float() / 255.0)
        w = w.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, ln = model(x, w)
        lp = F.log_softmax(lg.float(), -1).transpose(0, 1)
        loss = F.ctc_loss(lp, tg.to(dev), ln, tl.to(dev), blank=0, zero_infinity=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.detach() if step > 1 else loss.detach()  # no per-step GPU sync
        if step % 200 == 0:
            print(f"step {step} loss {float(run):.3f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s"
                  f" gpu {torch.cuda.max_memory_reserved() / 2**30:.1f}G", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            res = evaluate(model, val, dev)
            real = [v for k, v in res.items() if not k.startswith("synth")]
            score = float(np.mean(real))
            m01 = float(np.mean([res[k] for k in VAL01 if k in res]))
            print("val CER", {k: round(v, 4) for k, v in res.items()}, "mean real", round(score, 4),
                  "mean 0.1-sets", round(m01, 4), flush=True)
            ck = dict(model=model.state_dict(), cfg=cfg, chars=CHARS, val=res, step=step, name=name)
            torch.save(ck, f"{ROOT}/runs/{a.name}_last.pt")
            torch.save(ck, f"{ROOT}/runs/{a.name}_step{step}.pt")  # per-eval copies, keep the newest 3
            for old in sorted(glob.glob(f"{ROOT}/runs/{a.name}_step*.pt"), key=lambda p: int(p.rsplit("step", 1)[1][:-3]))[:-3]:
                os.remove(old)
            if score < best:
                best = score
                torch.save(ck, f"{ROOT}/runs/{a.name}.pt")
                print("  saved best", round(best, 4), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["rec"])
    ap.add_argument("--name", default="orus_rec")
    ap.add_argument("--init", default="indra", choices=["indra", "scratch"])
    ap.add_argument("--c", type=int, default=128)
    ap.add_argument("--e", type=int, default=256)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--height", type=int, default=32, help="line height in px; data from data/ocr/rec<h> (build it with ORUS_H=<h>)")
    ap.add_argument("--rnn", type=int, default=0, help="BiLSTM hidden size on top of the conv blocks (0 = none)")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--eval_every", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--resume", default="")
    ap.add_argument("--real_only", action="store_true")
    ap.add_argument("--area", type=float, default=1.0, help="batch size multiplier (pixels per batch)")
    train_rec(ap.parse_args())


if __name__ == "__main__":
    main()
