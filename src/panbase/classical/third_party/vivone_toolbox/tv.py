# SPDX-License-Identifier: LicenseRef-Vivone-Toolbox-NonProfit
#
# Python port of TV_pansharpen.m from G. Vivone et al.'s pansharpening benchmark toolbox ("Pansharpening Toolbox for
# Distribution", companion of IEEE GRSM 9(1):53-81, 2021), which the PanCollection MATLAB test toolbox redistributes
# byte-identically as MetricCode/TV/TV_pansharpen.m; per-sensor parameters from PanCollection's
# MetricCode/00-build/build_TV.m.
# Terms: the upstream MATLAB toolbox states only "Copyright (c) 2020. All rights reserved. This work should only be used
# for nonprofit purposes." (README.txt).  This port is a derivative of that code and keeps those terms; it is NOT
# covered by the Apache-2.0 licence of the rest of the PanBase repository.  See ../__init__.py.
"""TV pansharpening (VO).

Method: F. Palsson, J. R. Sveinsson, M. O. Ulfarsson, "A new pansharpening algorithm based on total variation",
IEEE GRSL 11(1):318-322, 2014.

A line-by-line port of the official MATLAB code; the algorithm is NOT re-derived from the paper.
Official code: TV_pansharpen.m (Vivone et al. toolbox; identical copy in the PanCollection MATLAB test toolbox as
shipped e.g. with https://github.com/WangXueyang-uestc/ARConv @ f568bef7e45f8a7c3758b2fa8e9d2b62f4b62617,
MetricCode/TV/TV_pansharpen.m) and MetricCode/00-build/build_TV.m (per-sensor parameters and the call
    I_TV = TV_pansharpen(data.I_MS_LR, data.I_PAN, alpha, lambda, c, maxiter, w)
i.e. the method takes the LRMS (`ms_lr`) and the PAN; the interpolated `ms` is not used).

Parameters (build_TV.m, keyed by opts.sensor; the PanCollection demos set 'WV3' / 'QB' / 'GF2'; WV2 has no demo, so it
takes build_TV's 'WV3','WV2' branch):
  {'IKONOS','QB','GF2'} -> w = [0.1091 0.2127 0.2928 0.3854], c = 8, alpha = 1.064, maxiter = 10, lambda = 0.47106
  {'WV3','WV2'}         -> w = [0.0657 0.1012 0.1537 0.1473 0.1245 0.1545 0.1338 0.1192], c = 8, alpha = 0.75,
                           maxiter = 50, lambda = 1e-3
Inputs are digital numbers (the h5 scale the MATLAB demos feed; lambda assumes it).  TV calls no sensor MTF (its
operator is imresize 'bilinear'), so the sensor Nyquist gains do not enter.

Official-code behaviours reproduced ON PURPOSE (they are what PanCollection ran):
  * cIDDTz(z, c) computes `cIddtz = c*z - ddtz` but RETURNS `ddtz` (= D D^T z), so the z update is
    z <- (D b + D D^T z) ./ W, not the (c I - D D^T) z of the MM derivation.
  * DTz / cIDDTz pair z(:,:,i) with z(:,:,i+4) -- a hard-coded 4.  Correct for 4 bands; for 8 bands (WV3/WV2) bands 1-4
    take the Dx planes of bands 5-8 as their "Dy" part and bands 5-8 take the Dy planes of bands 1-4.
  * decimate = imresize(x, 0.25, 'bilinear') (MATLAB: antialiased triangle of half-width 4, indices mirrored at the
    border); interpolate = imresize(x, 4, 'bilinear').  Emulated exactly below (MATLAB's imresize `contributions`
    algorithm; rows first, then columns).
  * float64 throughout; x0 = 0, z0 = 0; no clipping (the caller clamps).
Deviations: none known.
Verification (PanBase project, 2026-10-14): the unmodified build_TV.m -> TV_pansharpen.m run in GNU Octave 10.3 (with a
transcription of MATLAB's imresize) on the same inputs agrees with fuse() to <= 1.6e-12 DN (relative <= 8e-16) on
WV3/WV2/QB/GF2 test images.  On the 20 WV3 reduced-resolution PanCollection test images this gives SAM 5.692 /
ERGAS 4.855 / Q8 0.7955, as reported for "TV" in CANNet (CVPR 2024), Tab. 1.
"""
import numpy as np
import torch

FAMILY = "VO"
REF = "Palsson et al., IEEE GRSL 2014"

# build_TV.m, verbatim
_P4 = dict(w=(0.1091, 0.2127, 0.2928, 0.3854), c=8.0, alpha=1.064, maxiter=10, lam=0.47106)            # 'IKONOS','QB','GF2'
_P8 = dict(w=(0.0657, 0.1012, 0.1537, 0.1473, 0.1245, 0.1545, 0.1338, 0.1192), c=8.0, alpha=0.75, maxiter=50,
           lam=1.0000e-03)                                                                               # 'WV3','WV2'
SENSOR = {"wv3": "WV3", "wv2": "WV2", "qb": "QB", "gf2": "GF2"}                 # opts.sensor of the PanCollection demos
PARAMS = {"IKONOS": _P4, "QB": _P4, "GF2": _P4, "WV3": _P8, "WV2": _P8}


# ---------------------------------------------------------------- MATLAB imresize(x, scale, 'bilinear'), exact
def _triangle(x):
    # imresize.m: f = (x+1) .* ((-1 <= x) & (x < 0)) + (1-x) .* ((0 <= x) & (x <= 1));
    return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))


def _contributions(in_length, out_length, scale, antialiasing=True):
    """imresize.m `contributions` with kernel = @triangle, kernel_width = 2 ('bilinear').  Returns (weights, 0-based idx)."""
    kernel_width = 2.0
    if scale < 1 and antialiasing:
        h = lambda t: scale * _triangle(scale * t)                              # noqa: E731
        kernel_width = kernel_width / scale
    else:
        h = _triangle
    x = np.arange(1, out_length + 1, dtype=np.float64)[:, None]
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(np.ceil(kernel_width)) + 2
    indices = left + np.arange(P, dtype=np.float64)[None, :]
    weights = h(u - indices)
    s = np.zeros((out_length, 1))
    for p in range(P):                                                          # sum(weights, 2), sequential
        s[:, 0] += weights[:, p]
    weights = weights / s
    aux = np.concatenate([np.arange(1, in_length + 1), np.arange(in_length, 0, -1)])
    indices = aux[np.mod(indices - 1, aux.size).astype(np.int64)]              # mirror out-of-range indices
    keep = np.any(weights != 0, axis=0)                                         # drop all-zero columns
    return weights[:, keep], indices[:, keep] - 1


_CACHE = {}


def _resize_along(x, dim, out_length, scale):
    key = (x.shape[dim], out_length, scale)
    if key not in _CACHE:
        w, idx = _contributions(x.shape[dim], out_length, scale)
        _CACHE[key] = (torch.from_numpy(w), torch.from_numpy(idx))
    w, idx = _CACHE[key]
    w = w.to(x.dtype)
    shape = [1] * x.dim()
    shape[dim] = out_length
    out = None
    for p in range(w.shape[1]):                                                 # imresizemex: sum_p w(k,p) * in(idx(k,p))
        t = x.index_select(dim, idx[:, p]) * w[:, p].reshape(shape)
        out = t if out is None else out + t
    return out


def imresize_bilinear(x, scale):
    """MATLAB imresize(x, scale, 'bilinear') on the last two dims (rows = dim -2 first, then columns = dim -1)."""
    H, W = x.shape[-2], x.shape[-1]
    oh, ow = int(np.ceil(scale * H)), int(np.ceil(scale * W))
    return _resize_along(_resize_along(x, -2, oh, scale), -1, ow, scale)


# ---------------------------------------------------------------- TV_pansharpen.m (x: (C,H,W), MATLAB (:,:,i) = x[i])
def _decimate(x):
    return imresize_bilinear(x, 0.25)                                          # y = imresize(x,0.25,'bilinear');


def _interpolate(x):
    return imresize_bilinear(x, 4.0)                                           # y = imresize(x,4,'bilinear');


def _Dx(v):                                                                     # [diff(v,1,2) zeros(size(v,1),1)]
    return torch.cat([v[..., :, 1:] - v[..., :, :-1], torch.zeros_like(v[..., :, :1])], dim=-1)


def _Dy(v):                                                                     # [diff(v); zeros(1,size(v,2))]
    return torch.cat([v[..., 1:, :] - v[..., :-1, :], torch.zeros_like(v[..., :1, :])], dim=-2)


def _DyT(v):
    u0 = -v[..., :1, :]                                                         # u0 = -v(1,:);
    u1 = -(v[..., 1:, :] - v[..., :-1, :])                                      # u1 = -diff(v);
    u2 = v[..., -2:-1, :]                                                       # u2 = v(end-1,:);
    return torch.cat([u0, u1[..., :-1, :], u2], dim=-2)                         # [u0; u1(1:(end-1),:); u2]


def _DxT(v):                                                                    # DyT(v')'
    return _DyT(v.transpose(-2, -1)).transpose(-2, -1)


def _computeDb(b):                                                              # [Dx(b_i) ..., Dy(b_i) ...]
    return torch.cat([_Dx(b), _Dy(b)], dim=0)


def _DTz(z):
    n = z.shape[0] // 2                                                         # for i=1:size(z,3)/2
    return torch.stack([_DxT(z[i]) + _DyT(z[i + 4]) for i in range(n)])        # DxT(z(:,:,i))+DyT(z(:,:,i+4))  (sic: 4)


def _cIDDTz(z, c):
    ddtz = _computeDb(_DTz(z))
    cIddtz = c * z - ddtz                                                       # noqa: F841  computed, then discarded:
    return ddtz                                                                 # the official function returns ddtz


def _computeH(x, w):
    yms = _decimate(x)
    ypan = torch.zeros_like(x[0])
    for i in range(x.shape[0]):
        ypan = ypan + w[i] * x[i]                                               # ypan=ypan+w(i)*x(:,:,i);
    return yms, ypan


def _adjointH(yms, ypan, w):
    up = _interpolate(yms)
    return torch.stack([up[i] + w[i] * ypan for i in range(yms.shape[0])])     # interpolate(yms(:,:,i))+w(i)*ypan


def _computeb(yms, ypan, xk, alpha, w):
    Hxms, Hxpan = _computeH(xk, w)
    return alpha * xk + _adjointH(yms - Hxms, ypan - Hxpan, w)


def _znext(z0, x0, b, alpha, lam, c):
    Wt = 2 * alpha / lam * torch.sqrt(_Dx(x0) ** 2 + _Dy(x0) ** 2) + c         # 2* alpha/lambda * sqrt(Dx.^2+Dy.^2)+c
    W = torch.cat([Wt, Wt], dim=0)                                              # W(:,:,i) and W(:,:,i+size(x0,3))
    return (_computeDb(b) + _cIDDTz(z0, c)) / W


def _xnext(z1, b, alpha):
    return (b - _DTz(z1)) / alpha


def tv_pansharpen(yms, ypan, alpha, lam, c, maxiter, w):
    """x = TV_pansharpen(yms, ypan, alpha, lambda, c, maxiter, w).  yms (C,h,w), ypan (H,W), float64 DN -> x (C,H,W)."""
    C = yms.shape[0]
    z = torch.zeros((2 * C,) + tuple(ypan.shape), dtype=yms.dtype)
    x = torch.zeros((C,) + tuple(ypan.shape), dtype=yms.dtype)
    for _ in range(maxiter):
        b = _computeb(yms, ypan, x, alpha, w)
        z = _znext(z, x, b, alpha, lam, c)
        x = _xnext(z, b, alpha)
    return x


def fuse(ms, ms_lr, pan, ds):
    p = PARAMS[SENSOR[ds]]
    yms = ms_lr[0].double().cpu()
    ypan = pan[0, 0].double().cpu()
    assert yms.shape[0] == len(p["w"]), (ds, yms.shape, len(p["w"]))
    assert tuple(ypan.shape) == (4 * yms.shape[1], 4 * yms.shape[2]), (tuple(yms.shape), tuple(ypan.shape))
    x = tv_pansharpen(yms, ypan, p["alpha"], p["lam"], p["c"], p["maxiter"], [float(v) for v in p["w"]])
    return x.unsqueeze(0)
