"""Character n-gram LM for Orus-OCR (orus_lm.npz), in Indra's lm.npz backoff format
(so indra_ocr.lm.NgramLM / beam_search run on it unchanged).

  python ocr_lm.py build [order=5] [chars=6e6]   -> models/orus_lm.npz
Text: BN Wikipedia + EN wikitext + generated code-mixed / list / prescription lines (ocr_synth.gen_text).
Tokens are Orus class ids (1..V); a line starts with order-1 BOS (packed 0). Absolute discounting backoff:
P*(t|h) = (c(h,t) - D) / c(h), backoff weight b(h) = D * N1+(h) / c(h). Note: backoff weights are not
renormalised against the lower order (fine for rescoring close calls; exact KN if the LM ever decides alone).
"""
import os
import random
import sys

import numpy as np

from vocab import CHARS, canon, encode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root: data/, runs/, models/
BITS = 11
D = 0.7


def lines(total):
    import ocr_synth as S
    r = random.Random(0)
    n = 0
    budget = dict(bn=0.45 * total, en=0.30 * total, mix=0.25 * total)
    for lang in ("bn", "en"):
        k = 0
        for p in S.corpus(lang):
            words = S.words_of(lang, random.Random(p[:20])) if lang == "en" else p.split()
            for i in range(0, len(words), 10):
                t = canon(" ".join(words[i:i + 10]))
                ids = encode(t)
                if ids:
                    yield ids
                    k += len(ids)
            if k > budget[lang]:
                break
    k = 0
    while k < budget["mix"]:
        ids = encode(S.gen_text(r))
        if ids:
            yield ids
            k += len(ids)


def pack_all(seq, n, order):
    """Packed keys of every (history of length n, token) window in a BOS-padded id array."""
    m = len(seq)
    key = np.zeros(m - (order - 1), np.uint64)
    for j in range(n + 1):
        key = (key << np.uint64(BITS)) | seq[order - 1 - n + j: m - n + j].astype(np.uint64)
    return key


def build(order=5, total=6e6):
    seqs = []
    for ids in lines(total):
        seqs.append(np.concatenate([np.zeros(order - 1, np.int64), np.array(ids, np.int64) + 1]))
    # one long array; windows that cross a line start are fine because every line begins with order-1 BOS
    seq = np.concatenate(seqs)
    tok = seq[order - 1:]
    seq_ok = tok != 0                                    # windows ending in a real token only
    out = dict(order=np.int64(order), vocab_size=np.int64(len(CHARS) + 1), bits=np.int64(BITS))
    print("tokens", int(seq_ok.sum()))
    for n in range(order):
        keys = pack_all(seq, n, order)[seq_ok]
        uk, cnt = np.unique(keys, return_counts=True)
        if n >= 3:                                       # prune singletons of long orders
            keep = cnt > 1
            uk, cnt = uk[keep], cnt[keep]
        hist = uk >> np.uint64(BITS)
        hk, inv, hcnt_n = np.unique(hist, return_inverse=True, return_counts=True)   # N1+(h)
        hc = np.bincount(inv, weights=cnt)                                           # c(h)
        if n == 0:
            v = np.log(cnt / cnt.sum())
            hb = np.zeros(1)
        else:
            v = np.log((cnt - D) / hc[inv])
            hb = np.log(D * hcnt_n / hc)
        out[f"k{n}"], out[f"v{n}"] = uk, v.astype(np.float32)
        out[f"hk{n}"], out[f"hb{n}"] = hk, hb.astype(np.float32)
        print("order", n + 1, "grams", len(uk), "histories", len(hk), flush=True)
    p = f"{ROOT}/models/orus_lm.npz"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez_compressed(p, **out)
    print("saved", p, f"{os.path.getsize(p) / 1e6:.1f} MB")
    check(p)


def check(p):
    """A real BN / EN phrase must score better than its character-shuffled version."""
    from indra_ocr.lm import BOS, NgramLM
    lm = NgramLM(p)

    def score(t):
        h, s = (BOS,) * (lm.order - 1), 0.0
        for c in encode(t):
            s += lm.logp(h[-(lm.order - 1):], c)
            h = h + (c,)
        return s / len(t)
    for t in ("আমি বাংলায় গান গাই", "the patient was given", "Tab. Napa 500mg 1+0+1"):
        sh = "".join(random.Random(1).sample(t, len(t)))
        a, b = score(t), score(sh)
        print(f"{t!r}: {a:.2f} vs shuffled {b:.2f}")
        assert a > b + 0.5


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(*(int(float(x)) for x in sys.argv[2:]))
    else:
        check(f"{ROOT}/models/orus_lm.npz")
