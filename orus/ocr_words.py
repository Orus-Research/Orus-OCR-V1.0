"""Word-level lexicon correction for Orus-OCR-1.0-alpha (the WER lever on top of greedy / char-LM decoding).

  python ocr_words.py build      -> models/orus_lex.txt.gz  (word \\t count; BN + EN, lower-case EN)
  python ocr_words.py demo       -> self-check

A decoded word that is not in the lexicon (or a lexicon word with an unsure char) is replaced by the candidate that
maximises (candidates: lexicon words at edit distance 1, plus lexicon words a CTC prefix beam over the word's steps
reaches through the sorted word list, at any edit distance)
    CTC log P(word | the word's own time steps) + lam * log P_lex(word),
where the original reading competes with lam * log P_oov. The CTC score is the exact forward probability over
the steps between the neighbouring emitted characters, so a clearly written rare word or name survives.
Sources: BN Wikipedia (data/text/bn.txt), BN newspaper corpus (data/text/bn_news.txt), EN wikitext
(data/text/en.txt), /usr/share/dict words, training-split labels of the real line sets (never test splits).
"""
import bisect
import gzip
import json
import math
import os
import re
import sys
from collections import Counter

import numpy as np

from vocab import CHARS, IDX, canon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root: data/, runs/, models/
LEX = f"{ROOT}/models/orus_lex.txt.gz"
BN_L = re.compile(r"[ঀ-৿]")
BN_WORD = re.compile(r"[ঀ-ৣৰ-৾]+")            # BN letters + signs, not BN digits / dari
EN_WORD = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")
BN_ALPHA = [c for c in CHARS if re.fullmatch(r"[ঀ-ৣৰ-৾]", c)]
EN_ALPHA = list("abcdefghijklmnopqrstuvwxyz'")


def tokens(line):
    line = canon(line)
    return BN_WORD.findall(line) + [w.lower() for w in EN_WORD.findall(line)]


def build(bn_cap=300000, en_cap=150000):
    c = Counter()
    for f in ("bn.txt", "bn_news.txt", "en.txt"):
        p = f"{ROOT}/data/text/{f}"
        if os.path.exists(p):
            for l in open(p, encoding="utf-8"):
                c.update(tokens(l.replace(" @-@ ", "-")))
    for f in ("/usr/share/dict/american-english", "/usr/share/dict/british-english"):
        if os.path.exists(f):
            for w in open(f, encoding="utf-8", errors="ignore"):
                w = w.strip()
                if EN_WORD.fullmatch(w) and not w.endswith("'s"):
                    c[w.lower()] += 2
    for src in ("bnhtrd", "banglawriting", "iiit", "gnhk", "iam", "imgur5k", "prescription", "rimes", "swapnil",
                "sahil", "sroie", "cord", "funsd", "bnprinted"):
        p = f"{ROOT}/data/ocr/pages/{src}.jsonl"
        if os.path.exists(p):
            for l in open(p):
                pg = json.loads(l)
                if pg["split"] == "train":
                    for ln in pg["lines"]:
                        c.update(tokens(ln["text"]))
    bn = [(w, n) for w, n in c.most_common() if BN_L.search(w) and n >= 2 and len(w) >= 2][:bn_cap]
    en = [(w, n) for w, n in c.most_common() if not BN_L.search(w) and n >= 2 and len(w) >= 2][:en_cap]
    with gzip.open(LEX, "wt", encoding="utf-8") as f:
        for w, n in bn + en:
            f.write(f"{w}\t{n}\n")
    print("lexicon", len(bn), "BN", len(en), "EN ->", LEX)


def _lse(a, b):
    m = np.maximum(a, b)
    with np.errstate(invalid="ignore"):
        return np.where(m == -np.inf, -np.inf, m + np.log1p(np.exp(-np.abs(a - b))))


def ctc_score(lp, cands):
    """lp: [T, V] log-probs of a word's steps; cands: list of id lists. -> [C] log P(cand | lp) (CTC forward,
    batched over candidates)."""
    T = lp.shape[0]
    C = len(cands)
    U = max(len(c) for c in cands)
    L = 2 * U + 1
    lab = np.zeros((C, L), np.int64)                                  # blank = 0 at even states
    for i, c in enumerate(cands):
        lab[i, 1:2 * len(c):2] = c
    n_states = np.array([2 * len(c) + 1 for c in cands])
    valid = np.arange(L)[None] < n_states[:, None]
    skip = np.zeros((C, L), bool)                                     # s -> s+2 allowed between different labels
    skip[:, 3::2] = (lab[:, 3::2] != lab[:, 1:-2:2]) & (lab[:, 3::2] != 0)
    a = np.full((C, L), -np.inf)
    a[:, 0] = lp[0, 0]
    a[:, 1] = lp[0, lab[:, 1]]
    a[~valid] = -np.inf
    for t in range(1, T):
        prev1 = np.concatenate([np.full((C, 1), -np.inf), a[:, :-1]], 1)
        prev2 = np.concatenate([np.full((C, 2), -np.inf), a[:, :-2]], 1)
        prev2 = np.where(skip, prev2, -np.inf)
        a = _lse(_lse(a, prev1), prev2) + lp[t, lab]
        a[~valid] = -np.inf
    last = a[np.arange(C), n_states - 1]
    last2 = a[np.arange(C), np.maximum(n_states - 2, 0)]
    return _lse(last, last2)


class Lexicon:
    def __init__(self, path=LEX, lam=0.6, oov=-24.0, beam=True, known=True, known_conf=0.8):  # tuned on val
        """known: also re-check lexicon words with a char under known_conf (real-word errors); a reading that is a
        lexicon word keeps its own prior. Tuned with scripts/tune_lex.py."""
        self.logp, self.lam, self.oov, self.beam, self.known, self.known_conf = {}, lam, oov, beam, known, known_conf
        self.sorted = {True: [], False: []}
        if not os.path.exists(path):
            return
        cnt = {}
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for l in f:
                w, n = l.rstrip("\n").split("\t")
                cnt[w] = int(n)
        tot = {True: 0, False: 0}
        for w, n in cnt.items():
            tot[bool(BN_L.search(w))] += n
        self.logp = {w: math.log(n / tot[bool(BN_L.search(w))]) for w, n in cnt.items()}

        self.sorted = {True: sorted(w for w in self.logp if BN_L.search(w)), False: sorted(w for w in self.logp if not BN_L.search(w))}

    def _is_prefix(self, q, bn):
        ws = self.sorted[bn]
        i = bisect.bisect_left(ws, q)
        return i < len(ws) and ws[i].startswith(q)

    def beam_words(self, lp, bn, beam=16, k=6, floor=-7.0):
        """Lexicon words that a CTC prefix beam over the word's steps reaches, any edit distance away from the
        greedy reading. lp: [T, V] log-probs of the span. The prefix tree is the sorted word list (bisect)."""
        T = lp.shape[0]
        alpha = BN_ALPHA if bn else EN_ALPHA
        ids = {}
        for ch in alpha:                                     # EN: lower + upper case classes read as one letter
            c = [IDX[x] for x in {ch, ch.upper()} if x in IDX]
            ids[ch] = c
        col = {ch: np.logaddexp.reduce(lp[:, c], 1) if len(c) > 1 else lp[:, c[0]] for ch, c in ids.items()}
        chars = list(col)
        M = np.stack([col[ch] for ch in chars], 1)              # [T, |alpha|]
        beams = {"": (0.0, -np.inf)}
        for t in range(T):
            b0 = lp[t, 0]
            top = np.argsort(-M[t])[:k]
            cand = [(chars[j], M[t, j]) for j in top if M[t, j] > floor]
            nxt = {}

            def add(p, b, nb):
                o = nxt.get(p)
                nxt[p] = (b, nb) if o is None else (np.logaddexp(o[0], b), np.logaddexp(o[1], nb))
            for p, (pb, pnb) in beams.items():
                tot = np.logaddexp(pb, pnb)
                add(p, tot + b0, -np.inf)
                if p:
                    add(p, -np.inf, pnb + col[p[-1]][t])
                for ch, v in cand:
                    q = p + ch
                    if not self._is_prefix(q, bn):
                        continue
                    add(q, -np.inf, (pb if p and ch == p[-1] else tot) + v)
            beams = dict(sorted(nxt.items(), key=lambda kv: -np.logaddexp(*kv[1]))[:beam])
        return [p for p in beams if p in self.logp]

    def __bool__(self):
        return bool(self.logp)

    def edits1(self, w):
        alpha = BN_ALPHA if BN_L.search(w) else EN_ALPHA
        sp = [(w[:i], w[i:]) for i in range(len(w) + 1)]
        out = {a + b[1:] for a, b in sp if b}
        out |= {a + c + b[1:] for a, b in sp if b for c in alpha}
        out |= {a + c + b for a, b in sp for c in alpha}
        out |= {a + b[1] + b[0] + b[2:] for a, b in sp if len(b) > 1}
        out.discard(w)
        return out

    def candidates(self, w):
        return {x for x in self.edits1(w) if x in self.logp}      # edit distance 1 only (the CTC beam covers larger edits)

    def correct(self, text, steps, lp):
        """text: decoded line; steps: emission step per char of text (-1 for chars without one); lp [T, V]
        log-probs of the line. Returns the corrected line."""
        if not self.logp or not text:
            return text
        T = lp.shape[0]
        out = list(text)
        repl = []
        for m in list(BN_WORD.finditer(text)) + list(re.finditer(r"[A-Za-z]+", text)):
            w = m.group()
            key = w.lower() if not BN_L.search(w) else w
            if len(w) < 2 or (key in self.logp and not self.known):
                continue
            if not BN_L.search(w) and w[1:].lower() != w[1:] and not w.isupper():
                continue                                             # mixed case: likely a code / name
            s0, s1 = m.start(), m.end()
            before = [steps[i] for i in range(s0) if steps[i] >= 0]
            after = [steps[i] for i in range(s1, len(text)) if steps[i] >= 0]
            t0 = before[-1] + 1 if before else 0
            t1 = after[0] if after else T
            if t1 - t0 < 1:
                continue
            if key in self.logp:                                     # known word: re-check only if a char is unsure
                pr = [lp[steps[i], IDX[text[i]]] for i in range(s0, s1) if steps[i] >= 0]
                if not pr or min(pr) > math.log(self.known_conf):
                    continue
            cands = self.candidates(key)
            if self.beam:
                cands |= set(self.beam_words(lp[t0:t1], bool(BN_L.search(w))))
            cands.discard(key)
            cands = list(cands)[:200]
            if not cands:
                continue
            forms = [self._case(c, w) for c in cands]
            ids = [[IDX[ch] for ch in f] for f in [w] + forms]
            sc = ctc_score(lp[t0:t1], ids)
            prior = np.array([self.logp.get(key, self.oov)] + [self.logp[c] for c in cands])
            tot = sc + self.lam * prior
            k = int(np.argmax(tot))
            if k > 0:
                repl.append((s0, s1, forms[k - 1]))
        for s0, s1, f in sorted(repl, reverse=True):
            out[s0:s1] = list(f)
        return "".join(out)

    @staticmethod
    def _case(c, w):
        if BN_L.search(c):
            return c
        if w.isupper() and len(w) > 1:
            return c.upper()
        return c[0].upper() + c[1:] if w[0].isupper() else c


def greedy_steps(lp):
    """Greedy CTC path -> (text, emission step per char)."""
    ids = lp.argmax(-1)
    txt, st, prev = [], [], 0
    for t, i in enumerate(ids):
        if i != prev and i != 0:
            txt.append(CHARS[i - 1])
            st.append(t)
        prev = i
    return "".join(txt), st


def demo():
    lex = Lexicon.__new__(Lexicon)
    lex.logp, lex.lam, lex.oov, lex.beam, lex.known = {"hello": math.log(0.01), "world": math.log(0.01), "help": math.log(0.01)}, 0.6, -16.0, True, False
    lex.known_conf = 0.9
    lex.sorted = {True: [], False: sorted(lex.logp)}
    V = len(CHARS) + 1

    def frames(s, noise_at=None, alt=None):
        lp = np.full((2 * len(s) + 1, V), -9.0)
        lp[:, 0] = -0.05
        for i, ch in enumerate(s):
            lp[2 * i + 1, :] = -9.0
            lp[2 * i + 1, IDX[ch]] = -0.05
            if i == noise_at:
                lp[2 * i + 1, IDX[ch]] = -0.9
                lp[2 * i + 1, IDX[alt]] = -0.6
        return lp
    lp = frames("hello world", 1, "a")                      # 'hallo' read greedily, 'e' close behind
    t, st = greedy_steps(lp)
    assert t == "hallo world", t
    assert lex.correct(t, st, lp) == "hello world"
    lp = frames("zorbax world")                             # confident unknown word survives
    t, st = greedy_steps(lp)
    assert lex.correct(t, st, lp) == "zorbax world"
    sc = ctc_score(frames("ab"), [[IDX["a"], IDX["b"]], [IDX["a"]], [IDX["a"], IDX["a"], IDX["b"]]])
    assert sc[0] > sc[1] and sc[0] > sc[2], sc
    print("ocr_words demo ok")


if __name__ == "__main__":
    {"build": build, "demo": demo}[sys.argv[1]]()
