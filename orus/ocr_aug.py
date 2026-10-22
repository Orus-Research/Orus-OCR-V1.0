"""Opt-in extra GPU augmentations for the Orus fine-tune (nothing here is used by default; see ocr_finetune.py).

Existing ocr_train.gpu_aug already does: small rotation / scale / shear (affine), stroke thicken / thin, blur,
contrast / background, gaussian noise.  Existing ocr_synth does elastic warp only on synthetic font text.
Missing for REAL handwriting lines: local non-rigid warp, baseline wave, and per-word horizontal stretch.
"""
import math

import torch
import torch.nn.functional as F


def grid_warp(x, p=0.5, ax=0.10, ay=0.05, wave=0.04):
    """x [B,1,3h,W] ink map. With probability p per sample: smooth random displacement field (elastic / grid warp) with
    peak horizontal amplitude `ax` and vertical `ay` (fractions of the line height h = H/3), plus a slow sinusoidal
    baseline wave of amplitude `wave`*h. Control grid ~ one node per 0.7 line heights horizontally."""
    B, _, H, W = x.shape
    h = H / 3.0
    gw, gh = max(3, int(W / (0.7 * h)) + 1), 5
    d = (torch.rand(B, 2, gh, gw, device=x.device) * 2 - 1)
    d = F.interpolate(d, (H, W), mode="bicubic", align_corners=True)
    amp = torch.tensor([ax * h * 2 / W, ay * h * 2 / H], device=x.device).view(1, 2, 1, 1)   # px -> normalised coords
    d = d * amp
    u = torch.linspace(0, 1, W, device=x.device).view(1, 1, W)
    ph, fr = torch.rand(B, 1, 1, device=x.device) * 2 * math.pi, 0.5 + 2.5 * torch.rand(B, 1, 1, device=x.device)
    wv = (torch.sin(2 * math.pi * fr * u * W / (4 * h) / 3 + ph) * wave * h * 2 / H) * torch.rand(B, 1, 1, device=x.device)
    d[:, 1] += wv                                                                   # wave: vertical shift by column
    d = d * (torch.rand(B, 1, 1, 1, device=x.device) < p)
    eye = torch.eye(2, 3, device=x.device).unsqueeze(0).expand(B, 2, 3)
    grid = F.affine_grid(eye, x.shape, align_corners=False) + d.permute(0, 2, 3, 1)
    return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


def with_warp(base_aug, **kw):
    """Wrap a gpu_aug(x) so that grid_warp runs first (geometry before photometry)."""
    return lambda x: base_aug(grid_warp(x, **kw))


if __name__ == "__main__":               # python ocr_aug.py out.png : warp preview on an existing 48 px store
    import sys
    import cv2
    import numpy as np
    from ocr_data import Store, OUT
    s = Store(f"{OUT}/rec48/iam_val")
    ims = [cv2.imdecode(np.frombuffer(s[i][0], np.uint8), 0) for i in range(6)]
    Wm = max(i.shape[1] for i in ims)
    x = torch.zeros(6, 1, ims[0].shape[0], Wm)
    for k, im in enumerate(ims):
        x[k, 0, :, :im.shape[1]] = torch.from_numpy(im) / 255.0
    y = grid_warp(x, p=1.0)
    out = np.vstack([np.vstack([(x[k, 0].numpy() * 255), (y[k, 0].numpy() * 255)]) for k in range(6)]).astype(np.uint8)
    cv2.imwrite(sys.argv[1], out)
