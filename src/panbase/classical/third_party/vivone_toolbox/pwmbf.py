# SPDX-License-Identifier: LicenseRef-Vivone-Toolbox-NonProfit AND LicenseRef-Rice-RWT
#
# Python port of PWMBF/{PWMBF.m, compute_PhiTX.m, compute_PhiX.m} of G. Vivone et al.'s pansharpening benchmark
# toolbox ("Pansharpening Toolbox for Distribution", companion of IEEE GRSM 9(1):53-81, 2021; the PanCollection MATLAB
# test toolbox and DLPan-Toolbox ship identical copies), of the Rice Wavelet Toolbox routines it calls
# (rwt/bin/daubcqf.m and the MEX sources lib/src/rdwt.c, irdwt.c of mrdwt / mirdwt), and of the 'bicubic' /
# 'bilinear' resampling algorithm of MATLAB's imresize.
# Terms: the upstream MATLAB toolbox states only "Copyright (c) 2020. All rights reserved. This work should only be used
# for nonprofit purposes." (README.txt).  This port is a derivative of that code and keeps those terms; it is NOT
# covered by the Apache-2.0 licence of the rest of the PanBase repository.  See ../__init__.py.
#
# The Rice Wavelet Toolbox portions (daubcqf, mrdwt, mirdwt and their helpers) are derived from code distributed under:
#   Copyright (c) 2000 RICE UNIVERSITY. All rights reserved.
#
#   This software is distributed and licensed to you on a non-exclusive basis, free-of-charge. Redistribution and use
#   in source and binary forms, with or without modification, are permitted provided that the following conditions
#   are met:
#
#   1. Redistribution of source code must retain the above copyright notice, this list of conditions and the
#      following disclaimer.
#   2. Redistribution in binary form must reproduce the above copyright notice, this list of conditions and the
#      following disclaimer in the documentation and/or other materials provided with the distribution.
#   3. Neither the name of the University nor the names of its contributors may be used to endorse or promote products
#      derived from this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY WILLIAM MARSH RICE UNIVERSITY, HOUSTON, TEXAS, AND CONTRIBUTORS AS IS AND ANY EXPRESS
#   OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL RICE UNIVERSITY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
#   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTIONS) HOWEVER CAUSED AND ON ANY
#   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE), PRODUCT
#   LIABILITY, OR OTHERWISE ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
#   SUCH DAMAGE.
"""PWMBF: model-based fusion using PCA and wavelets (VO).

Method: F. Palsson, J. R. Sveinsson, M. O. Ulfarsson, J. A. Benediktsson, "Model-based fusion of multi- and
hyperspectral images using PCA and wavelets", IEEE TGRS 53(5):2652-2663, 2015.

Ported line by line from the official code (not re-derived from the paper):
  * PWMBF.m, compute_PhiTX.m, compute_PhiX.m ('rwt' branch) of Vivone et al.'s benchmark toolbox (the PanCollection
    copies are byte-identical apart from line endings);
  * rwt (Rice Wavelet Toolbox): mrdwt / mirdwt MEX = lib/src/rdwt.c, irdwt.c; filter = rwt/bin/daubcqf.m;
  * MATLAB imresize 'bicubic' / 'bilinear' (antialiasing on when shrinking, the default).

Parameters: r = 3, wavelet = 1, degrade = 0, called as PWMBF(I_PAN, I_MS_LR, ratio, r, wavelet, degrade) with ratio 4
(Vivone toolbox, Fusion_Algorithms_Reduced_Resolution.m and Fusion_Algorithms_Full_Resolution.m); L = 4 levels,
type 'rwt' (undecimated DWT) and daubcqf(4, 'min') are fixed inside PWMBF.m.  PanCollection ships the PWMBF folder
without a build script, so the Vivone call is the only official parameter set.  PWMBF calls no sensor MTF, so the
sensor gains do not enter.  Inputs: I_MS_LR (= ms_lr) and I_PAN in digital numbers, as in the official call; PWMBF
upsamples the LRMS itself (bicubic), so the 23-tap lms (= ms) is not used.

Port details (all the same operations as the official code; differences are floating-point round-off):
  * MATLAB's column-major reshape / vec are followed with order='F'.
  * The accumulation order of rdwt.c / irdwt.c (x0 = x0 + x*c,  x0 = (x0 + a) + b) is kept.
  * imresize: MATLAB contributions() (u = x/scale + 0.5(1-1/scale), symmetric border aux = [1:n n:-1:1], all-zero
    weight columns removed) + imresizemex's sequential sum, dimension order [1 2].
  * mad(x) = MATLAB default flag 0 = mean(|x - mean(x)|);  inv(scalar) = 1/scalar.
  * SVD sign / order ambiguity: the output is invariant to the signs of singular-vector pairs.
Deviations: none in the algorithm.
Verification (PanBase project, 2026-10-14): the unmodified PWMBF.m with the official rwt C code compiled as MEX in
GNU Octave 10.3, on the same inputs: max |d| 9.0e-12 / 2.6e-11 / 3.0e-11 / 3.9e-11 DN on WV3 / QB / GF2 / WV2
reduced-resolution test image 0, 2.1e-10 / 8.6e-11 DN on WV3 / GF2 full-resolution image 0; mrdwt / mirdwt are
bit-identical to the official rdwt.c / irdwt.c built with gcc.
"""
import numpy as np
import torch

FAMILY = "VO"
REF = "Palsson et al., IEEE TGRS 2015"

RATIO = 4
R_PC, WAVELET, DEGRADE = 3, 1, 0          # Vivone 2021 toolbox, Fusion_Algorithms_*_Resolution.m
L_LEVELS = 4                              # PWMBF.m: L=4; type='rwt'


# --------------------------------------------------------------------------------------------
# MATLAB imresize (double input; 'bicubic' / 'bilinear'; antialiasing on when shrinking)
# --------------------------------------------------------------------------------------------
def _cubic(x):
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return ((1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1)
            + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((1 < absx) & (absx <= 2)))


def _triangle(x):
    return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))


_KERNELS = {"bicubic": (_cubic, 4.0), "bilinear": (_triangle, 2.0)}


def _contributions(in_length, out_length, scale, kernel, kernel_width, antialiasing=True):
    if scale < 1 and antialiasing:
        h = lambda x: scale * kernel(scale * x)                                       # noqa: E731
        kernel_width = kernel_width / scale
    else:
        h = kernel
    x = np.arange(1, out_length + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(np.ceil(kernel_width)) + 2
    indices = left[:, None] + np.arange(P, dtype=np.float64)[None, :]                 # 1-based
    weights = h(u[:, None] - indices)
    wsum = weights[:, 0].copy()
    for p in range(1, P):                                                             # sum(weights, 2)
        wsum = wsum + weights[:, p]
    weights = weights / wsum[:, None]
    aux = np.concatenate([np.arange(1, in_length + 1), np.arange(in_length, 0, -1)])
    indices = aux[np.mod(indices.astype(np.int64) - 1, aux.size)]
    keep = np.any(weights != 0, axis=0)
    return weights[:, keep], indices[:, keep] - 1                                     # 0-based


def _resize_along_dim(A, dim, weights, indices):
    A = np.moveaxis(A, dim, 0)
    bshape = (weights.shape[0],) + (1,) * (A.ndim - 1)
    out = np.zeros((weights.shape[0],) + A.shape[1:])
    for p in range(weights.shape[1]):                                                 # imresizemex: sequential sum
        out = out + weights[:, p].reshape(bshape) * A[indices[:, p]]
    return np.moveaxis(out, 0, dim)


def imresize(A, scale, method="bicubic"):
    """MATLAB imresize(A, scale, method) for a double array (H, W[, C]); scalar scale."""
    kernel, kw = _KERNELS[method]
    B = np.asarray(A, dtype=np.float64)
    for dim in (0, 1):                                                                # order = sort([s s]) = [1 2]
        in_len = B.shape[dim]
        out_len = int(np.ceil(scale * in_len))
        w, ind = _contributions(in_len, out_len, scale, kernel, kw)
        B = _resize_along_dim(B, dim, w, ind)
    return B


# --------------------------------------------------------------------------------------------
# Rice Wavelet Toolbox: daubcqf.m, rdwt.c (mrdwt), irdwt.c (mirdwt) -- 2-D case
# --------------------------------------------------------------------------------------------
def daubcqf(N, TYPE="min"):
    """rwt/bin/daubcqf.m, first output h_0 (min/max phase; 'mid' not needed here)."""
    if N % 2:
        raise ValueError("No Daubechies filter exists for ODD length")
    K = N // 2
    a = 1.0
    p = np.array([1.0])
    q = np.array([1.0])
    h_0 = np.array([1.0, 1.0])
    for j in range(1, K):
        a = -a * 0.25 * (j + K - 1) / j
        h_0 = np.r_[0.0, h_0] + np.r_[h_0, 0.0]
        p = np.r_[0.0, -p] + np.r_[p, 0.0]
        p = np.r_[0.0, -p] + np.r_[p, 0.0]
        q = np.r_[0.0, q, 0.0] + a * p
    q = np.sort(np.roots(q))
    qt = q[:K - 1]
    if TYPE == "mid":
        raise NotImplementedError("daubcqf 'mid' is not used by PWMBF")
    h_0 = np.convolve(h_0, np.real(np.poly(qt)))
    h_0 = np.sqrt(2) * h_0 / np.sum(h_0)
    if TYPE == "max":
        h_0 = h_0[::-1].copy()
    if abs(np.sum(h_0 ** 2)) - 1 > 1e-4:
        raise ValueError('Numerically unstable for this value of "N".')
    return h_0


def _rdwt_convolution(x, s, coeff_low, coeff_high, axis):
    """rdwt_convolution on every stride-s, circularly extended subsequence along `axis`."""
    n = coeff_low.size
    lo = np.zeros_like(x)
    hi = np.zeros_like(x)
    for j in range(n):
        xs = np.roll(x, -j * s, axis=axis)                                            # x_in[j+i]
        lo = lo + xs * coeff_low[n - 1 - j]
        hi = hi + xs * coeff_high[n - 1 - j]
    return lo, hi


def mrdwt(x, h, L):
    """[yl, yh] = mrdwt(x, h, L) for a 2-D x; yh = [LH HL HH] per level, levels left to right."""
    x = np.asarray(x, dtype=np.float64)
    nrows, ncols = x.shape
    h = np.asarray(h, dtype=np.float64).ravel()
    coeff_low = h[::-1].copy()                                                        # rdwt_coefficients
    coeff_high = h.copy()
    coeff_high[0::2] = -coeff_high[0::2]
    yl = x.copy()
    yh = np.zeros((nrows, 3 * ncols * L))
    s = 1
    for level in range(1, L + 1):
        cc = 3 * ncols * (level - 1)
        lo, hi = _rdwt_convolution(yl, s, coeff_low, coeff_high, axis=1)              # go by rows
        yl = lo
        yh[:, cc:cc + ncols] = hi
        ll, lh = _rdwt_convolution(yl, s, coeff_low, coeff_high, axis=0)              # go by columns
        hl, hh = _rdwt_convolution(yh[:, cc:cc + ncols], s, coeff_low, coeff_high, axis=0)
        yl = ll
        yh[:, cc:cc + ncols] = lh
        yh[:, cc + ncols:cc + 2 * ncols] = hl
        yh[:, cc + 2 * ncols:cc + 3 * ncols] = hh
        s *= 2
    return yl, yh


def _irdwt_convolution(lo, hi, s, coeff_low, coeff_high, axis):
    """irdwt_convolution: x0 = x0 + (low[j+i]*cl[n-1-j]) + (high[j+i]*ch[n-1-j]), circular prefix."""
    n = coeff_low.size
    out = np.zeros_like(lo)
    for j in range(n):
        sh = (n - 1 - j) * s                                                          # buf[j+i] = d[i+j-(n-1)]
        out = (out + np.roll(lo, sh, axis=axis) * coeff_low[n - 1 - j]) + np.roll(hi, sh, axis=axis) * coeff_high[n - 1 - j]
    return out


def mirdwt(yl, yh, h, L):
    """x = mirdwt(yl, yh, h, L) for a 2-D signal."""
    yl = np.asarray(yl, dtype=np.float64)
    yh = np.asarray(yh, dtype=np.float64)
    nrows, ncols = yl.shape
    h = np.asarray(h, dtype=np.float64).ravel()
    n = h.size
    coeff_low = h / 2                                                                 # irdwt_coefficients
    coeff_high = h[::-1] / 2
    coeff_high[1:n + 1:2] = -coeff_high[1:n + 1:2]
    x = yl.copy()
    s = 2 ** (L - 1)
    for level in range(L, 0, -1):
        cc = 3 * ncols * (level - 1)
        x_low = _irdwt_convolution(x, yh[:, cc:cc + ncols], s, coeff_low, coeff_high, axis=0)         # by columns
        x_high = _irdwt_convolution(yh[:, cc + ncols:cc + 2 * ncols], yh[:, cc + 2 * ncols:cc + 3 * ncols],
                                    s, coeff_low, coeff_high, axis=0)
        x = _irdwt_convolution(x_low, x_high, s, coeff_low, coeff_high, axis=1)                        # by rows
        s //= 2
    return x


def compute_PhiTX(X, L, h):
    """compute_PhiTX.m, case 'rwt'."""
    M, T = X.shape
    n = int(round(np.sqrt(M)))
    PhiTX = np.zeros(((3 * L + 1) * M, T))
    for k in range(T):
        xl, xh = mrdwt(X[:, k].reshape(n, n, order="F"), h, L)
        PhiTX[:, k] = np.concatenate([xl, xh], axis=1).reshape(-1, order="F") / 2
    return PhiTX


def compute_PhiX(X, L, h):
    """compute_PhiX.m, case 'rwt'."""
    N, r = X.shape
    M = N // (3 * L + 1)
    n = int(round(np.sqrt(M)))
    PhiX = np.zeros((M, r))
    for k in range(r):
        PhiX[:, k] = mirdwt(X[:M, k].reshape(n, n, order="F"), X[M:, k].reshape(n, 3 * L * n, order="F"),
                            h, L).reshape(-1, order="F") * 2
    return PhiX


def _mad(x):
    """MATLAB mad(x) (flag 0): mean absolute deviation."""
    return np.mean(np.abs(x - np.mean(x)))


# --------------------------------------------------------------------------------------------
# PWMBF.m
# --------------------------------------------------------------------------------------------
def PWMBF(Pan, Low, ratio, r, wavelet, degrade):
    L = L_LEVELS
    Low = np.asarray(Low, dtype=np.float64)
    Pan = np.asarray(Pan, dtype=np.float64)
    if Pan.ndim == 2:
        Pan = Pan[:, :, None]
    N = Pan.shape[0]
    Q = Pan.shape[2]
    nb = Low.shape[2]
    if r > nb:
        raise ValueError("Number of PCs greater than number of bands")
    X = Pan
    Ylow = Low
    if degrade:
        X = imresize(Pan, 1 / ratio)
        Ylow = imresize(Low, 1 / ratio)
        N = N // 4
    Y = imresize(Ylow, ratio, "bicubic")                                              # Upsample Y
    Xtilde = imresize(imresize(X, 1 / ratio, "bilinear"), ratio, "bicubic")           # Degrade X
    N2 = N * N
    X = X.reshape(N2, Q, order="F")
    Xtilde = Xtilde.reshape(N2, Q, order="F")
    Y = Y.reshape(N2, nb, order="F")

    F, D, Rt = np.linalg.svd(Y, full_matrices=False)                                  # [F, D, R] = svd(Y,'econ')
    G = F * D                                                                         # G = F*D
    U = Rt.T                                                                          # U = R
    wfilter = daubcqf(4, "min")

    if wavelet:
        x = compute_PhiTX(Xtilde, L, wfilter)
        x0 = compute_PhiTX(X, L, wfilter)
        y = compute_PhiTX(G[:, :r], L, wfilter)
        yl = y[:N2, :]
        zh = np.zeros((3 * L * N2, r))
        for p in range(r):
            for j in range(1, 3 * L + 1):
                xh = x[j * N2:(j + 1) * N2, :]
                xh0 = x0[j * N2:(j + 1) * N2, :]
                yh = y[j * N2:(j + 1) * N2, p]
                Cyy = (yh @ yh) / N2
                Cyx = (yh @ xh) / N2                                                  # 1 x Q
                Cxx = (xh.T @ xh) / N2                                                # Q x Q
                Cn = (_mad(np.abs(yh)) / 0.6745) ** 2
                inv_Cxx = np.linalg.inv(Cxx)
                Cy_x = Cyy - (Cyx @ inv_Cxx) @ Cyx
                if Q > 1:
                    CyxiCxx = Cyx @ inv_Cxx
                    mu_zx = xh @ CyxiCxx
                    mu_zx0 = xh0 @ CyxiCxx
                else:
                    c = (Cyx @ inv_Cxx)[0]
                    mu_zx = c * xh[:, 0]
                    mu_zx0 = c * xh0[:, 0]
                ymu = yh - mu_zx
                CC = Cy_x * (1.0 / (Cy_x + Cn))                                       # Cy_x*inv(Cy_x+Cn)
                zh[(j - 1) * N2:j * N2, p] = mu_zx0 + ymu * CC
        z = np.concatenate([yl, zh], axis=0)
        B = compute_PhiX(z, L, wfilter)
        G[:, :r] = B                                                                  # deg = 0 branch
        Zhat = G @ U.T
    else:
        Cn = 0.0
        yh = G[:, :r]
        xh = Xtilde
        xh0 = X
        Cyy = (yh.T @ yh) / N2
        Cyx = (yh.T @ xh) / N2
        Cxx = (xh.T @ xh) / N2
        inv_Cxx = np.linalg.inv(Cxx)
        Cy_x = Cyy - (Cyx @ inv_Cxx) @ Cyx.T
        CyxiCxx = Cyx @ inv_Cxx
        mu_zx = xh @ CyxiCxx.T
        mu_zx0 = xh0 @ CyxiCxx.T
        ymu = yh - mu_zx
        CC = np.linalg.solve((Cy_x + Cn).T, Cy_x.T).T                                 # Cy_x/(Cy_x+Cn)
        B = mu_zx0 + ymu @ CC
        G[:, :r] = B
        Zhat = G @ U.T
    return Zhat.reshape(N, N, nb, order="F")


def fuse(ms, ms_lr, pan, ds):
    low = ms_lr[0].permute(1, 2, 0).cpu().double().numpy()                           # I_MS_LR (h, w, C), DN
    P = pan[0, 0].cpu().double().numpy()                                              # I_PAN (H, W), DN
    Z = PWMBF(P, low, RATIO, R_PC, WAVELET, DEGRADE)
    return torch.from_numpy(np.ascontiguousarray(Z.transpose(2, 0, 1)))[None].to(ms.dtype)
