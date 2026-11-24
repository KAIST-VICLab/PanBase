# SPDX-License-Identifier: Apache-2.0
"""Quality indices with the conventions of the PanBase tables (PanCollection protocol).

Reduced resolution (with ground truth), on images normalised to [0, 1], after removing a 30-pixel border
(``rr_crop``: ``x[..., 29:H-30, 29:W-30]``, i.e. 197 x 197 of a 256 x 256 image, as in PanCollection's MATLAB demos):
    psnr   peak signal-to-noise ratio, peak 1, MSE over all bands and pixels
    sam    spectral angle mapper in degrees (mean over pixels, float64)
    ergas  ERGAS with ratio 4 (per-band RMSE normalised by the reference band mean)
    ssim   mean over bands of a Gaussian-window SSIM (11 x 11, sigma 1.5)
    q2n    Q2^n (Q4 / Q8) on 32 x 32 blocks, in digital numbers  [third_party.vivone_assessment.q2n]
Full resolution (no reference), uncropped, in digital numbers:
    hqnr, d_lambda, d_s   HQNR = (1 - D_lambda)(1 - D_s), block size 32  [third_party.vivone_assessment.HQNR]

psnr / sam / ergas / ssim are implemented here (Apache-2.0).  q2n and HQNR call the vendored Python assessment toolbox
of G. Vivone (non-commercial licence, see third_party/vivone_assessment).  All functions take batches ``(B, C, H, W)``
as torch tensors or NumPy arrays and return one value per image (NumPy float64 arrays).
"""
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from .sensors import RATIO, get_dataset

RR_DIM_CUT = 30


def _t(x):
    return x if torch.is_tensor(x) else torch.from_numpy(np.ascontiguousarray(x))


def rr_crop(x, dim_cut: int = RR_DIM_CUT):
    """Remove the border used by the PanCollection RR protocol: ``x[..., dim_cut-1 : H-dim_cut, dim_cut-1 : W-dim_cut]``."""
    h, w = x.shape[-2], x.shape[-1]
    if h <= 2 * dim_cut or w <= 2 * dim_cut:
        raise ValueError(f"dim_cut {dim_cut} is too large for a {h}x{w} image")
    return x[..., dim_cut - 1:h - dim_cut, dim_cut - 1:w - dim_cut]


def psnr(x, y, data_range: float = 1.0):
    x, y = _t(x), _t(y)
    mse = torch.mean((x - y) ** 2, dim=(1, 2, 3)).clamp_min(1e-20)
    return (20 * np.log10(data_range) - 10 * torch.log10(mse)).double().numpy()


def sam(x, y, eps: float = 1e-12):
    """Mean spectral angle in degrees, computed in float64."""
    x, y = _t(x).double(), _t(y).double()
    num = (x * y).sum(1)
    den = x.norm(dim=1) * y.norm(dim=1) + eps
    a = torch.arccos((num / den).clamp(-1.0, 1.0))
    return torch.rad2deg(a).flatten(1).mean(1).numpy()


def ergas(x, y, ratio: int = RATIO, eps: float = 1e-8):
    x, y = _t(x), _t(y)
    mse = ((x - y) ** 2).mean(dim=(2, 3))
    mu = y.mean(dim=(2, 3)).abs().clamp_min(eps)
    return (100.0 / ratio * torch.sqrt((mse / mu ** 2).mean(1))).double().numpy()


def ssim(x, y, data_range: float = 1.0, w: int = 11, sigma: float = 1.5):
    x, y = _t(x), _t(y)
    c = x.shape[1]
    g = torch.arange(w, dtype=x.dtype, device=x.device) - (w - 1) / 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k = (g[:, None] @ g[None, :]).expand(c, 1, w, w).contiguous()
    mu_x = F.conv2d(x, k, groups=c)
    mu_y = F.conv2d(y, k, groups=c)
    xx = F.conv2d(x * x, k, groups=c) - mu_x ** 2
    yy = F.conv2d(y * y, k, groups=c) - mu_y ** 2
    xy = F.conv2d(x * y, k, groups=c) - mu_x * mu_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (xx + yy + c2))
    return s.flatten(1).mean(1).double().numpy()


def q2n(pred, gt, divisor: float, block: int = 32, shift: int = 32):
    """Q2^n of each image (inputs normalised; evaluated in digital numbers, as the index requires)."""
    from .third_party.vivone_assessment.q2n import q2n as _q2n

    g = _t(gt).detach().double().cpu().numpy() * divisor
    p = _t(pred).detach().double().cpu().numpy() * divisor
    return np.asarray([_q2n(g[i].transpose(1, 2, 0), p[i].transpose(1, 2, 0), block, shift)[0]
                       for i in range(g.shape[0])], dtype=np.float64)


def rr_metrics(pred, gt, dataset: str, crop: bool = True, with_q2n: bool = True) -> Dict[str, np.ndarray]:
    """Reduced-resolution indices of each image: psnr, ssim, sam, ergas (and q2n)."""
    spec = get_dataset(dataset)
    pred, gt = _t(pred), _t(gt)
    if crop:
        pred, gt = rr_crop(pred), rr_crop(gt)
    out = dict(psnr=psnr(pred, gt), ssim=ssim(pred, gt), sam=sam(pred, gt), ergas=ergas(pred, gt, RATIO))
    if with_q2n:
        out["q2n"] = q2n(pred, gt, spec.divisor)
    return out


def fr_metrics(fused, ms, lms, pan, dataset: str, block: int = 32) -> Dict[str, np.ndarray]:
    """Full-resolution indices of each image: hqnr, d_lambda, d_s.

    fused, lms (B, C, H, W); ms (B, C, H/4, W/4); pan (B, 1, H, W); all normalised to [0, 1].  The arrays are converted
    to digital numbers (``x.double() * divisor``) and scored with the toolbox's HQNR(ps_ms, ms, msexp, pan, S, sensor,
    ratio), sensor = DatasetSpec.hqnr_sensor.
    """
    from .third_party.vivone_assessment.HQNR import HQNR

    spec = get_dataset(dataset)
    to = lambda x: _t(x).detach().double().cpu().numpy() * spec.divisor   # noqa: E731
    f, m, l, p = to(fused), to(ms), to(lms), to(pan)
    if f.shape != l.shape or f.shape[-2:] != p.shape[-2:] or m.shape[-1] * RATIO != f.shape[-1]:
        raise ValueError(f"shape mismatch: fused {f.shape}, ms {m.shape}, lms {l.shape}, pan {p.shape}")
    # q2n (inside D_lambda) may normalise its first argument in place, and HQNR passes the same ms to D_s: hand it a copy.
    res = [HQNR(f[i].transpose(1, 2, 0), np.array(m[i].transpose(1, 2, 0)), l[i].transpose(1, 2, 0), p[i, 0],
                block, spec.hqnr_sensor, RATIO) for i in range(f.shape[0])]
    pick = lambda j: np.asarray([r[j] for r in res], dtype=np.float64)    # noqa: E731
    return dict(hqnr=pick(0), d_lambda=pick(1), d_s=pick(2))
