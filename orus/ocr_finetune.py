"""Low-LR fine-tune of a trained recognizer (e.g. the released orus_rec.pt) on your own line stores.

Thin wrapper over ocr_train.train_rec: swaps the source mix (--w mix.json, {source: weight}; sources without a
store are ignored), optionally turns on grid warp (--warp) and hard-example oversampling (--hard).
Build your stores with ocr_data.save_store (48 px ink maps, see ocr_data.crop_page) into data/ocr/rec48/
<src>_train*.blob/meta/lab.npy and <src>_val*.  Keep a replay share of the original sources to avoid forgetting.
  python ocr_finetune.py --name orus_rec_ft --resume orus_rec.pt --w mix.json --steps 8000 --lr 2e-5
Not used for the released 1.0-alpha weights (they come from ocr_train.py alone).
"""
import argparse
import json

import numpy as np

import ocr_train as T


class Resampled:
    """m[i] with chosen indices repeated (hard-example oversampling); Batches draws uniformly from len(L)."""

    def __init__(self, m, idx, rep):
        self.m, self.map = m, np.concatenate([np.arange(len(m))] + [np.asarray(idx, np.int64)] * (rep - 1))

    def __len__(self):
        return len(self.map)

    def __getitem__(self, i):
        return self.m[int(self.map[i])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="orus_rec_ft")
    ap.add_argument("--resume", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--area", type=float, default=2.0)
    ap.add_argument("--warp", action="store_true", help="grid warp (ocr_aug.grid_warp) before gpu_aug")
    ap.add_argument("--hard", default="", help="json {src: [train idx]} of hard lines to oversample")
    ap.add_argument("--hard_rep", type=int, default=3, help="each hard item appears this many times")
    ap.add_argument("--w", required=True, help="json file {source: weight} with the fine-tune mix")
    a = ap.parse_args()

    T.SOURCE_W = json.load(open(a.w))
    T.VAL = sorted(set(T.VAL) | {k for k in T.SOURCE_W if not k.startswith("synth")})
    if a.warp:
        import ocr_aug
        T.gpu_aug = ocr_aug.with_warp(T.gpu_aug)

    if a.hard:
        hard, ls = json.load(open(a.hard)), T.load_split

        def load_split(split, names=None, height=32):
            d = ls(split, names, height)
            if split == "train":
                d = {k: Resampled(v, hard[k], a.hard_rep) if k in hard and hard[k] else v for k, v in d.items()}
            return d
        T.load_split = load_split

    ns = argparse.Namespace(name=a.name, init="indra", c=128, e=256, depth=10, height=48, rnn=128, steps=a.steps, lr=a.lr,
                            eval_every=a.eval_every, workers=a.workers, resume=a.resume, real_only=False, area=a.area)
    T.train_rec(ns)


if __name__ == "__main__":
    main()
