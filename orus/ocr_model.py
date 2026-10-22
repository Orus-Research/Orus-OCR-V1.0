"""Orus-OCR networks: CTC line recognizer (conv blocks + optional BiLSTM) + the BN/EN vocabulary.

The conv part of RecNet reproduces Indra OCR 1.1 `rec.onnx` (tall-window context branch) / `rec_world.onnx`
(no context) layer for layer, so Indra weights load as a warm start (`load_indra`). 1.0-alpha adds a 2-layer
BiLSTM (SeqLSTM) on top and reads 48 px lines.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vocab import CHARS, decode, encode

NAME = "Orus-OCR-1.0-alpha"


def indra_dir():
    """Indra OCR model folder (downloaded and cached by the indra-ocr package from datalab-to/indra-ocr)."""
    from indra_ocr import model_dir
    return str(model_dir(None))


# ---------------- recognizer ----------------
class Block(nn.Module):
    def __init__(self, c, e, k=9):
        super().__init__()
        self.dw = nn.Conv1d(c, c, k, padding=k // 2, groups=c)
        self.expand = nn.Conv1d(c, e, 1)
        self.fc1, self.fc2 = nn.Conv1d(e, e // 4, 1), nn.Conv1d(e // 4, e, 1)
        self.project = nn.Conv1d(e, c, 1)

    def forward(self, x, m, msum):
        h = F.hardswish(self.expand(self.dw(x)))
        s = (h * m).sum(2, keepdim=True) / msum                  # masked squeeze-excite
        h = h * torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return (x + self.project(h)) * m


class Context(nn.Module):
    """Tall-window branch: 4 attention heads over the 24 rows of the 96-px window."""
    def __init__(self, c, rows=24):
        super().__init__()
        self.patch = nn.Conv2d(1, 32, 4, 4)
        self.dw = nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False)
        self.pw = nn.Conv2d(32, 64, 1)
        self.score = nn.Conv2d(64, 4, 1)
        self.row_prior = nn.Parameter(torch.zeros(1, 4, rows, 1))
        self.proj = nn.Conv1d(64, c, 1)

    def forward(self, x):
        h = F.hardswish(self.pw(self.dw(F.hardswish(self.patch(x)))))   # B,64,24,T
        a = torch.softmax(self.score(h) + self.row_prior, 2)            # B,4,24,T
        B, C, R, T = h.shape
        h = (h.view(B, 4, C // 4, R, T) * a.unsqueeze(2)).sum(3)
        return self.proj(h.reshape(B, C, T))


class SeqLSTM(nn.Module):
    """Bidirectional LSTM over valid timesteps only, without packing (ONNX friendly): the backward direction
    runs on each sequence reversed inside its own length. Added as a zero-initialised residual, so a model
    warm-started from a conv-only checkpoint starts out identical to it."""
    def __init__(self, c, hidden, layers=2):
        super().__init__()
        self.f = nn.ModuleList(nn.LSTM(c if i == 0 else 2 * hidden, hidden, batch_first=True) for i in range(layers))
        self.b = nn.ModuleList(nn.LSTM(c if i == 0 else 2 * hidden, hidden, batch_first=True) for i in range(layers))
        self.proj = nn.Conv1d(2 * hidden, c, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h, lengths, m):
        T = h.shape[2]
        t = torch.arange(T, device=h.device)[None]
        rev = torch.where(t < lengths[:, None], lengths[:, None] - 1 - t, t)        # [B,T], an involution
        y = h.transpose(1, 2)                                                        # [B,T,C]
        for f, b in zip(self.f, self.b):
            yr = torch.gather(y, 1, rev[:, :, None].expand(-1, -1, y.shape[2]))
            ob = b(yr)[0]
            ob = torch.gather(ob, 1, rev[:, :, None].expand(-1, -1, ob.shape[2]))
            y = torch.cat([f(y)[0], ob], 2)
        return h + self.proj(y.transpose(1, 2)) * m


class RecNet(nn.Module):
    """x: [B,1,height*window,W] ink map (1 = ink); widths: [B] valid px. Returns logits [B,T,V], lengths [B].
    rnn > 0 adds a SeqLSTM (hidden size rnn, 2 layers) between the conv blocks and the head (1.0-alpha: rnn=128,
    height=48)."""
    def __init__(self, vocab=len(CHARS) + 1, c=128, e=256, depth=10, window=3, k=9, rnn=0, height=32):
        super().__init__()
        self.seq = SeqLSTM(c, rnn) if rnn else None
        self.window, self.height = window, height
        self.stem = nn.Conv2d(1, c, (height, 4), (height, 4))
        self.stem_bn = nn.BatchNorm1d(c)
        self.context = Context(c, rows=3 * height // 4) if window == 3 else None
        self.blocks = nn.ModuleList(Block(c, e, k) for _ in range(depth))
        self.head = nn.Conv1d(c, vocab, 1)

    def forward(self, x, widths):
        mid = x[:, :, self.height:2 * self.height] if self.window == 3 else x
        h = self.stem(mid).squeeze(2)
        if self.context is not None:
            h = h + self.context(x)
        T = h.shape[2]
        lengths = torch.clamp(torch.div(widths + 3, 4, rounding_mode="floor"), max=T)
        m = (torch.arange(T, device=x.device)[None] < lengths[:, None]).unsqueeze(1).to(h.dtype)
        msum = m.sum(2, keepdim=True).clamp(min=1)
        h = F.hardswish(self.stem_bn(h)) * m
        for b in self.blocks:
            h = b(h, m, msum)
        if self.seq is not None:
            h = self.seq(h, lengths, m)
        return self.head(h).transpose(1, 2), lengths


def adapt(sd, model):
    """State dict of another line height (e.g. a 32 px checkpoint) -> this model: the stem kernel and the
    context row prior are resized along the height axis (stem scaled so its response to the same ink is kept)."""
    own = model.state_dict()
    out = {}
    for k, v in sd.items():
        if k in own and own[k].shape != v.shape and k == "stem.weight":
            v = F.interpolate(v, size=own[k].shape[2:], mode="bilinear", align_corners=False) * (v.shape[2] / own[k].shape[2])
        elif k in own and own[k].shape != v.shape and k.endswith("row_prior"):
            v = F.interpolate(v, size=own[k].shape[2:], mode="bilinear", align_corners=False)
        out[k] = v
    return out


def indra_vocab(name="rec.onnx"):
    import json
    meta = json.load(open(f"{indra_dir()}/meta.json"))
    v = meta["vocab_world" if "world" in name else "vocab"]
    return (json.loads(v) if isinstance(v, str) else v)["chars"]


def load_indra(model, name="rec.onnx", map_head=True):
    """Copy Indra ONNX weights into RecNet. Anonymous (BN-folded) convs are matched by graph order.
    map_head: copy head rows for chars shared with Indra's vocab (others keep random init)."""
    import onnx
    from onnx import numpy_helper as nh
    g = onnx.load(f"{indra_dir()}/{name}").graph
    W = {i.name: torch.from_numpy(nh.to_array(i).copy()) for i in g.initializer}
    sd = model.state_dict()
    put = lambda k, v: sd[k].copy_(v.view_as(sd[k]))
    for k, v in W.items():
        if not k.startswith("rec.") or k.startswith("rec.head"):
            continue
        k2 = k[4:].replace(".se.", ".")
        if k2 in sd and sd[k2].shape == v.shape:
            put(k2, v)
    anon = []  # (conv node, weight, bias) in graph order
    for n in g.node:
        if n.op_type == "Conv" and n.input[1].startswith("onnx::"):
            anon.append((n.name, W[n.input[1]], W[n.input[2]] if len(n.input) > 2 else None))
    for name_, w, b in anon:
        if "context/patch" in name_:
            put("context.patch.weight", w), put("context.patch.bias", b)
        elif "context/pw" in name_:
            put("context.pw.weight", w), put("context.pw.bias", b)
        elif "/dw/" in name_:
            i = int(name_.split("blocks.")[1].split("/")[0])
            if i < len(model.blocks):
                put(f"blocks.{i}.dw.weight", w), put(f"blocks.{i}.dw.bias", b)
    if map_head:
        hw, hb = W["rec.head.weight"], W["rec.head.bias"]
        if hw.shape == sd["head.weight"].shape:
            put("head.weight", hw), put("head.bias", hb)
        else:
            iv = {c: i + 1 for i, c in enumerate(indra_vocab(name))}
            rows = [0] + [iv.get(c, -1) for c in CHARS]
            for r, j in enumerate(rows):
                if j >= 0:
                    sd["head.weight"][r] = hw[j].view(-1, 1)
                    sd["head.bias"][r] = hb[j]
    model.load_state_dict(sd)
    return model


def demo():
    """Check: RecNet with Indra weights and Indra's own vocab reproduces ORT logprobs."""
    import onnxruntime as ort
    for name, win in (("rec.onnx", 3), ("rec_world.onnx", 1)):
        V = len(indra_vocab(name)) + 1
        net = load_indra(RecNet(vocab=V, window=win), name).eval()
        x = (torch.rand(3, 1, 32 * win, 200) > 0.8).float()
        w = torch.tensor([200, 150, 77])
        with torch.no_grad():
            lg, ln = net(x, w)
        s = ort.InferenceSession(f"{indra_dir()}/{name}", providers=["CPUExecutionProvider"])
        o = dict(zip([o.name for o in s.get_outputs()], s.run(None, {"x": x.numpy(), "widths": w.numpy()})))
        lp = torch.log_softmax(lg, -1).numpy()
        top = np.take_along_axis(lp, o["topi"], -1)
        err = np.abs(top - o["topv"]).max()
        assert (ln.numpy() == o["lengths"]).all(), (ln, o["lengths"])
        assert err < 1e-3, err
        print(name, "matches ORT, max |dlogp| =", float(err), "params", sum(p.numel() for p in net.parameters()))
    m = load_indra(RecNet(), "rec.onnx")
    assert encode("আমি  Rice খাই।") is not None and decode([0, 5, 5, 0, 6]) == CHARS[4] + CHARS[5]
    print("Orus vocab", len(CHARS), "params", sum(p.numel() for p in m.parameters()))


if __name__ == "__main__":
    demo()
