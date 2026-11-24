# SPDX-License-Identifier: Apache-2.0
"""MATLAB-compatible bicubic resampling, ``imresize(x, scale, 'bicubic')``, for torch tensors.

Used by the toolbox versions of BDSD-PC, PRACS and AWLP in place of the HS-Pan-Toolbox's ``Utils/imresize_bicubic.py``
(a copy of a third-party file that carries no licence).  Written for this package from the algorithm:

* separable: first along rows (dim -2), then along columns (dim -1); an axis of length n becomes ceil(n * s);
* output sample i (0-based) sits at input coordinate u_i = (i + 0.5) / s - 0.5;
* Keys' cubic convolution kernel with a = -0.5; when shrinking (s < 1) the kernel is stretched by 1/s (antialiasing);
* taps j = floor(u_i) - K // 2 + 1 + t, t = 0 .. K-1, with K = ceil(4 / s) + 2 when shrinking and K = 6 otherwise;
  weights k((u_i - j) * min(s, 1)) normalised to sum 1;
* taps outside the image are mirrored with the edge sample repeated (MATLAB's symmetric extension: ..., x1, x0 | x0,
  x1, ...).

As in the toolbox, the computation is done in float32 whatever the input dtype and cast back to the input dtype.  For
the power-of-two scales used here every unnormalised weight and every weight sum is exact in float32, so the only
rounding before the weighted sum is one correctly rounded division per weight; the weighted sum runs over the tap axis
of a contiguous (N, K, rows, cols) array.  This reproduces the toolbox's output bit for bit.
"""
import math
from typing import Optional, Tuple

import torch


def _keys(t: torch.Tensor, a: float = -0.5) -> torch.Tensor:
    """Keys' cubic convolution kernel."""
    s = t.abs()
    s2 = s * s
    s3 = s2 * s
    near = ((a + 2) * s3 - (a + 3) * s2 + 1) * (s <= 1).to(t.dtype)
    far = (a * s3 - 5 * a * s2 + 8 * a * s - 4 * a) * ((s > 1) & (s <= 2)).to(t.dtype)
    return near + far


def _taps(n_out: int, scale: float, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """First tap index (n_out,) and normalised weights (K, n_out) of one axis."""
    stretch = scale if scale < 1 else 1.0
    K = (math.ceil(4 / scale) if scale < 1 else 4) + 2
    u = (torch.arange(n_out, dtype=dtype) + 0.5) / scale - 0.5
    first = torch.floor(u) - (K // 2) + 1
    offset = u - first                                                  # u_i - first tap
    d = torch.stack([offset - t for t in range(K)])                     # (K, n_out): u_i - (first + t)
    w = _keys(d * stretch)
    w = w / w.sum(dim=0, keepdim=True)
    return first.long(), w


def _mirror(j: torch.Tensor, n: int) -> torch.Tensor:
    """Symmetric extension with the edge repeated (valid for |overhang| <= n)."""
    j = torch.where(j < 0, -j - 1, j)
    return torch.where(j >= n, 2 * n - 1 - j, j)


def _axis(x: torch.Tensor, axis: int, n_out: int, scale: float) -> torch.Tensor:
    """Resample x (N, 1, H, W) along axis -2 or -1."""
    first, w = _taps(n_out, scale, x.dtype)
    K = w.shape[0]
    n_in = x.shape[axis]
    idx = _mirror(first[None, :] + torch.arange(K)[:, None], n_in)       # (K, n_out)
    if axis == -2:
        sample = x[:, 0][:, idx, :]                                      # (N, K, n_out, W)
        weight = w.view(1, K, n_out, 1)
    else:
        sample = x[:, 0][:, :, idx].permute(0, 2, 1, 3).contiguous()     # (N, K, H, n_out)
        weight = w.view(1, K, 1, n_out)
    return (sample * weight).sum(dim=1, keepdim=True)


def imresize(x: torch.Tensor, scale: Optional[float] = None, sizes: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """Bicubic resize of a (H, W), (C, H, W) or (B, C, H, W) tensor by ``scale`` (or to ``sizes``), MATLAB semantics."""
    if (scale is None) == (sizes is None):
        raise ValueError("give exactly one of scale or sizes")
    shape = x.shape
    h, w = shape[-2], shape[-1]
    if sizes is None:
        sizes = (math.ceil(h * scale), math.ceil(w * scale))
        scales = (scale, scale)
    else:
        scales = (sizes[0] / h, sizes[1] / w)
    dtype = x.dtype
    y = x.reshape(-1, 1, h, w).float()
    if scales[0] != 1:
        y = _axis(y, -2, sizes[0], scales[0])
    if scales[1] != 1:
        y = _axis(y, -1, sizes[1], scales[1])
    return y.reshape(tuple(shape[:-2]) + (y.shape[-2], y.shape[-1])).to(dtype)
