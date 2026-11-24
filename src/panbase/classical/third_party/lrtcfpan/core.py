# SPDX-License-Identifier: GPL-3.0-only
#
# NumPy port of the official MATLAB code of LRTCFPan (lib/inc_LRTCFPan/*.m of https://github.com/zhongchengwu/code_LRTCFPan,
# commit ddaaa607da43a104e60f8198a022193be5c89436; copyright Zhong-Cheng Wu, Ting-Zhu Huang, Liang-Jian Deng, Jie Huang,
# Jocelyn Chanussot, Gemine Vivone).  The upstream repository is licensed under the GNU General Public License v3
# (full text in LICENSE next to this file); this port is a derivative work under the same licence and is NOT covered by
# the Apache-2.0 licence of the rest of the PanBase repository.
"""LRTCFPan (Wu, Huang, Deng, Huang, Chanussot, Vivone, IEEE TIP 2023) -- line-by-line NumPy port of the official MATLAB code.

Each function below mirrors one file of lib/inc_LRTCFPan/ (same name, same statement order).  Arrays keep MATLAB's
H x W x bands layout; MATLAB's 1-based indices are written as 0-based slices.  Data must be in [0, 1] (demo_LRTCFPan.m).
The algorithm files are the 2023-02-23 upload of the official repository (later commits touch comments / README only).

MATLAB built-ins are reproduced with MATLAB (R2018a, the paper's platform) semantics:
  fspecial('gaussian'), kaiser, fwind1 (Huang's rotation, linear interp1) + fsamp2, psf2otf (incl. its real() round-off test),
  imfilter(.,'replicate' | 'circular') = correlation, 'same' size,  imresize(., 'nearest' | 'box') via MATLAB's contribution
  weights (no antialiasing for 'nearest' or for upscaling),  std2 (N-1) / mean2,  uint32() (round half away from zero,
  saturate, NaN -> 0),  hist(y, nbins) (edges miny + k*binwidth, bins ( ], first bin closed),  svd(., 'econ'),
  fft(., [], 3) with MATLAB's real narrowing of slices whose imaginary part is exactly zero, and ifft / ifft2 of conjugate-
  symmetric arrays returning real (done here with rfft/irfft, i.e. the same r2c/c2r transforms MATLAB uses for them).
Differences to MATLAB are therefore at floating-point round-off level (FFT/LAPACK/BLAS summation order), except where noted.
The FFT worker count follows OMP_NUM_THREADS (default 1).
"""
import os

import numpy as np
import scipy.fft as sfft

EPS = float(np.finfo(np.float64).eps)          # MATLAB eps
_WORKERS = max(1, int(os.environ.get("OMP_NUM_THREADS", "1") or 1))


# =============================================================================== MATLAB built-ins
def matlab_round(x):
    """MATLAB round(): half away from zero (scalar)."""
    return float(np.sign(x) * np.floor(abs(x) + 0.5))


def fspecial_gaussian(n, sigma):
    """fspecial('gaussian', n, sigma)."""
    siz = (n - 1) / 2
    x, y = np.meshgrid(np.arange(-siz, siz + 1), np.arange(-siz, siz + 1))
    arg = -(x * x + y * y) / (2 * sigma * sigma)
    h = np.exp(arg)
    h[h < EPS * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h / sumh
    return h


def kaiser(n, beta=0.5):
    """kaiser(n) (MATLAB default beta = 0.5); same closed form as MATLAB's kaiser.m."""
    return np.kaiser(n, beta)


def fsamp2(hd):
    """fsamp2(Hd) (frequency sampling)."""
    hd = np.rot90(np.fft.fftshift(np.rot90(hd, 2)), 2)       # inverse fftshift
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)                                        # rotate for consistency with freqz2
    return h


def fwind1(hd, w1):
    """fwind1(Hd, win) with one 1-D window: Huang's rotation method (interp1 'linear'), then fsamp2(Hd) .* w."""
    n = len(w1)
    t = np.arange(-(n - 1) / 2, (n - 1) / 2 + 1) * (2 / (n - 1))
    t1, t2 = np.meshgrid(t, t)
    t12 = np.sqrt(t1 * t1 + t2 * t2)
    d = (t12 < t[0]) | (t12 > t[-1])
    t12[d] = 0
    w = np.interp(t12.ravel(), t, w1).reshape(t12.shape)
    w[d] = 0
    return fsamp2(hd) * w


def mtf_fir(gnyq, ratio, n=41):
    """The 5 lines shared by MTF_lrms.m / MTF_pan.m / FFT_kernel.m:  real(fwind1(fspecial gaussian / max, kaiser(N)))."""
    fcut = 1 / ratio
    alpha = np.sqrt(((n - 1) * (fcut / 2)) ** 2 / (-2 * np.log(gnyq)))
    H = fspecial_gaussian(n, alpha)
    Hd = H / H.max()
    h = fwind1(Hd, kaiser(n))
    return np.real(h)


class ImfilterReplicate:
    """imfilter(I(:,:,b), h_b, 'replicate') for every band b of an H x W x B array (correlation, 'same' size).

    Computed as the 'valid' part of an FFT convolution of the edge-padded image with the 180-degree-rotated kernel
    (identical to MATLAB's direct correlation up to round-off).  Kernels: k x k x B (or k x k x 1, broadcast)."""

    def __init__(self, kernels, H, W):
        if kernels.ndim == 2:
            kernels = kernels[:, :, None]
        k = kernels.shape[0]
        assert kernels.shape[1] == k and k % 2 == 1
        self.k, self.p, self.H, self.W = k, k // 2, H, W
        self.fs = (sfft.next_fast_len(H + 2 * self.p + k - 1, real=True), sfft.next_fast_len(W + 2 * self.p + k - 1, real=True))
        self.K = sfft.rfft2(kernels[::-1, ::-1, :], s=self.fs, axes=(0, 1), workers=_WORKERS)

    def __call__(self, x):
        squeeze = x.ndim == 2
        if squeeze:
            x = x[:, :, None]
        assert x.shape[:2] == (self.H, self.W), (x.shape, self.H, self.W)
        p, k = self.p, self.k
        xp = np.pad(x, ((p, p), (p, p), (0, 0)), mode="edge")
        y = sfft.irfft2(sfft.rfft2(xp, s=self.fs, axes=(0, 1), workers=_WORKERS) * self.K, s=self.fs, axes=(0, 1),
                        workers=_WORKERS)
        y = y[k - 1:k - 1 + self.H, k - 1:k - 1 + self.W, :]
        return y[:, :, 0] if squeeze else y


def imfilter_circular_rows(t, kernel):
    """imfilter(t, kernel_row, 'circular') for a 1 x L row kernel: correlation along each row, periodic boundary."""
    L = len(kernel)
    c = (L + 1) // 2 - 1                                       # MATLAB centre floor((L+1)/2), 0-based
    n = t.shape[1]
    tp = np.concatenate([t[:, n - c:], t, t[:, :L - 1 - c]], axis=1) if c else t
    out = np.zeros_like(t)
    for m in range(L):
        out += kernel[m] * tp[:, m:m + n]
    return out


def _imresize_contrib(in_len, out_len, scale, antialias):
    """MATLAB imresize contributions() for the box kernel (used by both 'nearest' and 'box'), kernel width 1."""
    kernel = lambda x: ((-0.5 <= x) & (x < 0.5)).astype(np.float64)
    kw = 1.0
    if scale < 1 and antialias:
        h = lambda x: scale * kernel(scale * x)
        kw = kw / scale
    else:
        h = kernel
    x = np.arange(1, out_len + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kw / 2)
    P = int(np.ceil(kw)) + 2
    ind = left[:, None] + np.arange(P)[None, :]
    weights = h(u[:, None] - ind)
    weights = weights / weights.sum(axis=1, keepdims=True)
    aux = np.concatenate([np.arange(1, in_len + 1), np.arange(in_len, 0, -1)])
    ind = aux[np.mod(ind - 1, len(aux)).astype(np.int64)]
    keep = np.any(weights != 0, axis=0)
    return weights[:, keep], ind[:, keep].astype(np.int64) - 1


def imresize(A, out_hw, method, scale=None):
    """imresize(A, scale | [rows cols], 'nearest' | 'box') for 2-D or H x W x B arrays (MATLAB algorithm)."""
    assert method in ("nearest", "box")
    antialias = method == "box"                                 # MATLAB: Antialiasing defaults to false for 'nearest'
    out = A
    for dim in (0, 1):
        in_len = A.shape[dim]
        s = scale if scale is not None else out_hw[dim] / in_len
        o_len = int(np.ceil(in_len * scale)) if scale is not None else int(out_hw[dim])
        w, ind = _imresize_contrib(in_len, o_len, s, antialias)
        moved = np.moveaxis(out, dim, 0)
        res = np.zeros((o_len,) + moved.shape[1:])
        for p in range(w.shape[1]):
            res += w[:, p].reshape((-1,) + (1,) * (moved.ndim - 1)) * moved[ind[:, p]]
        out = np.moveaxis(res, 0, dim)
    return out


def psf2otf(psf, out_size):
    """psf2otf(psf, outSize), including MATLAB's 'discard imaginary part within round-off' test."""
    if np.all(psf == 0):
        return np.zeros(out_size)
    pad = np.zeros(out_size)
    pad[:psf.shape[0], :psf.shape[1]] = psf
    pad = np.roll(pad, (-(psf.shape[0] // 2), -(psf.shape[1] // 2)), axis=(0, 1))
    otf = np.fft.fft2(pad)
    n_elem = psf.size
    n_ops = sum(k * np.log2(k) * (n_elem / k) for k in psf.shape)
    if np.max(np.abs(otf.imag)) / np.max(np.abs(otf)) <= n_ops * EPS:
        otf = otf.real
    return otf


def matlab_uint32(x):
    """double(uint32(x)): round half away from zero, saturate to [0, 2^32-1], NaN -> 0."""
    x = np.asarray(x, dtype=np.float64)
    y = np.where(np.isnan(x), 0.0, x)
    f = np.floor(y)
    r = np.where(y - f >= 0.5, f + 1, f)
    return np.clip(r, 0.0, 4294967295.0)


def matlab_hist(y, nbins):
    """[nn, centres] = hist(y, nbins) for a real vector y (MATLAB R2018a hist.m: histc on [-Inf, xx + eps(xx)], merge ends)."""
    y = np.asarray(y, dtype=np.float64).ravel()
    nbins = int(nbins)
    if nbins < 1:
        raise ValueError("hist: number of bins must be >= 1 (MATLAB would fail here)")
    miny, maxy = float(np.min(y)), float(np.max(y))
    if miny == maxy:
        miny = miny - np.floor(nbins / 2) - 0.5
        maxy = maxy + np.ceil(nbins / 2) - 0.5
    binwidth = (maxy - miny) / nbins
    xx = miny + binwidth * np.arange(nbins + 1)
    xx[-1] = maxy
    centres = xx[:-1] + binwidth / 2
    edges = np.concatenate([[-np.inf], xx + np.spacing(np.abs(xx))])
    # histc: n(k) = #{edges(k) <= y < edges(k+1)},  n(end) = #{y == edges(end)}
    idx = np.searchsorted(edges, y, side="right") - 1
    nn = np.zeros(len(edges))
    inside = idx < len(edges) - 1
    np.add.at(nn, idx[inside], 1)
    nn[-1] += np.count_nonzero(y == edges[-1])
    nn[1] += nn[0]
    nn[-2] += nn[-1]
    return nn[1:-1], centres


# =============================================================================== official sensor tables
def gnyq_ms(sensor, nbands, tag=""):
    """GNyq of MTF_lrms.m (FFT_kernel.m has the same table; its 'none' = 0.3)."""
    if sensor == "QB":
        return np.array([0.34, 0.32, 0.30, 0.22])
    if sensor == "IKONOS":
        return 0.3 * np.ones(nbands)
    if sensor == "GeoEye1":
        return np.array([0.23, 0.23, 0.23, 0.23])
    if sensor == "WV2":
        return np.array([0.35] * 7 + [0.27])
    if sensor == "WV3":
        return np.array([0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315])
    if sensor == "none":
        return np.array([0.35] * 7 + [0.27]) if tag == "WV2" else 0.3 * np.ones(nbands)
    raise ValueError(f"MTF_lrms: unknown sensor {sensor!r} (GNyq undefined in the official code)")


def gnyq_pan(sensor):
    """GNyq of MTF_pan.m."""
    tab = {"QB": 0.15, "IKONOS": 0.17, "GeoEye1": 0.16, "WV4": 0.16, "WV2": 0.11, "WV3": 0.14, "none": 0.15}
    if sensor not in tab:
        raise ValueError(f"MTF_pan: unknown sensor {sensor!r}")
    return tab[sensor]


# =============================================================================== lib/inc_LRTCFPan/*.m
def parasetting(sensor, type_, I_lrms, I_pan):
    """parasetting.m (verbatim values)."""
    l = I_lrms.shape[2]
    sz = tuple(I_pan.shape)
    paras = dict(sz=sz, Nways=(sz[0], sz[1], l), tol=2 * 1e-5, maxit=200)
    table = {  # sensor: {type: (lambda1, lambda2, lambda3, eta1, eta2, eta3, lower, block_size)}
        "none": {"R": (0.05, 18, 1e-4, 1e-4, 8.1, 1.8, 0, (8, 8)),        # Guangzhou dataset (GF-2)
                 "F": (1, 50, 1e-4, 1e-4, 2.1, 4.7, 0.4, (10, 10))},
        "QB":   {"R": (0.11, 65, 1e-4, 1e-4, 1.1, 8.7, 0, (8, 8)),        # Indianapolis dataset (QB)
                 "F": (0.4, 75, 0.1, 1e-3, 2.1, 6.7, 0.8, (10, 10))},
        "WV3":  {"R": (0.14, 56, 1e-4, 1e-4, 4.2, 8.3, 0, (8, 8)),        # Rio dataset (WV-3)
                 "F": (1.1, 36, 1e-4, 1e-4, 6.2, 3.3, 0.6, (10, 10))},
    }
    if sensor not in table or type_ not in table[sensor]:
        raise ValueError(f"parasetting: no parameters for sensor={sensor!r} type={type_!r} in the official code")
    l1, l2, l3, e1, e2, e3, lower, bs = table[sensor][type_]
    paras.update(lambda_=(l1, l2, l3), eta=(e1, e2, e3), lower=lower, block_size=bs)
    return paras


def interp23tap(I, ratio):
    """interp23tap.m."""
    if 2 ** round(np.log2(float(ratio))) != ratio:
        raise ValueError("Error: Only resize factors power of 2")
    r, c, b = I.shape
    CDF23 = 2. * np.array([0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0,
                           0.000807762146, 0, -0.000060081482])
    CDF23 = np.concatenate([CDF23[1:][::-1], CDF23])
    BaseCoeff = CDF23
    first = True
    for z in range(1, int(ratio // 2) + 1):
        I1LRU = np.zeros(((2 ** z) * r, (2 ** z) * c, b))
        if first:
            I1LRU[1::2, 1::2, :] = I
            first = False
        else:
            I1LRU[0::2, 0::2, :] = I
        for ii in range(b):
            t = I1LRU[:, :, ii]
            t = imfilter_circular_rows(t.T, BaseCoeff)
            I1LRU[:, :, ii] = imfilter_circular_rows(t.T, BaseCoeff)
        I = I1LRU
    return I


def mapping_block(block_lrms, block_pan):
    """mapping_block.m."""
    S = block_lrms.shape[2]
    out = np.zeros(block_pan.shape + (S,))
    sp, mp = np.std(block_pan, ddof=1), np.mean(block_pan)
    for i in range(S):
        temp = block_lrms[:, :, i]
        out[:, :, i] = (np.std(temp, ddof=1) / sp) * (block_pan - mp) + np.mean(temp)
    return out


def hist_mapping(I_lrms, I_pan, ratio, blocksize):
    """hist_mapping.m."""
    br, bc = int(blocksize[0]), int(blocksize[1])
    nr, nc, S = I_lrms.shape
    Nr, Nc = I_pan.shape
    high_pan = np.zeros((Nr, Nc, S))
    while nr % br != 0:
        br -= 1
    while nc % bc != 0:
        bc -= 1
    gridx, gridX = range(0, nr - br + 1, br), range(0, Nr - ratio * br + 1, ratio * br)
    gridy, gridY = range(0, nc - bc + 1, bc), range(0, Nc - ratio * bc + 1, ratio * bc)
    for i in range(len(gridx)):
        for j in range(len(gridy)):
            high_pan[gridX[i]:gridX[i] + ratio * br, gridY[j]:gridY[j] + ratio * bc, :] = mapping_block(
                I_lrms[gridx[i]:gridx[i] + br, gridy[j]:gridy[j] + bc, :],
                I_pan[gridX[i]:gridX[i] + ratio * br, gridY[j]:gridY[j] + ratio * bc])
    return high_pan


class MTFlrms:
    """MTF_lrms.m for a fixed image size (kernels built once; imfilter 'replicate')."""

    def __init__(self, sensor, tag, ratio, H, W, nbands):
        g = gnyq_ms(sensor, nbands, tag)
        self.f = ImfilterReplicate(np.stack([mtf_fir(g[ii], ratio) for ii in range(nbands)], axis=2), H, W)

    def __call__(self, I_MS):
        return self.f(I_MS)


def MTF_lrms(I_MS, sensor, tag, ratio):
    return MTFlrms(sensor, tag, ratio, I_MS.shape[0], I_MS.shape[1], I_MS.shape[2])(I_MS)


def MTF_pan(I_PAN, sensor, ratio):
    """MTF_pan.m."""
    return ImfilterReplicate(mtf_fir(gnyq_pan(sensor), ratio), I_PAN.shape[0], I_PAN.shape[1])(I_PAN)


def linstretch(CoeffsToSup, tol):
    """linstretch (sub-function of Suppl_coeffs.m)."""
    times = 1500
    CoeffsToSup = times * CoeffsToSup
    N, M, L = CoeffsToSup.shape
    CoeffsSuped = np.zeros((N, M, L))
    NM = N * M
    for i in range(L):
        b = matlab_uint32(CoeffsToSup[:, :, i])
        nb = np.max(b) - np.min(b)
        if nb > 5e7:
            raise ValueError(f"linstretch: hist with {nb:.0f} bins (MATLAB would run out of memory)")
        hb, levelb = matlab_hist(b, nb)
        chb = np.cumsum(hb)
        k1 = np.nonzero(chb > NM * tol[i][0])[0]
        k2 = np.nonzero(chb < NM * tol[i][1])[0]
        if len(k1) == 0 or len(k2) == 0:
            raise ValueError("linstretch: empty find() -> MATLAB would fail at t(1)/t(2) assignment")
        t1 = np.ceil(levelb[k1[0]])
        t2 = np.ceil(levelb[k2[-1]])
        b = np.where(b < t1, t1, b)
        b = np.where(b > t2, t2, b)
        CoeffsSuped[:, :, i] = b
    return CoeffsSuped / times


def Suppl_coeffs(CoeffsToSup, tol1):
    """Suppl_coeffs.m, nargin == 2 branch (the only one LRTCFPan uses)."""
    CoeffsToSup = np.asarray(CoeffsToSup, dtype=np.float64)
    L = CoeffsToSup.shape[2]
    tol = [list(tol1)] * L
    return linstretch(CoeffsToSup, tol)


def Compute_inj_comps(I_lrms, I_pan, paras):
    """Compute_inj_comps.m."""
    sz = paras["sz"]
    ratio = paras["ratio"]
    Nways = paras["Nways"]
    lower = paras["lower"]
    sensor = paras["sensor"]
    block_size = paras["block_size"]
    h, w = sz[0] // ratio, sz[1] // ratio
    assert (h * ratio, w * ratio) == tuple(sz)
    P_highdim = hist_mapping(I_lrms, I_pan, ratio, (h, w))
    Phigh_lp = MTF_lrms(P_highdim, sensor, "", ratio)
    Phigh_lp_done = imresize(Phigh_lp, None, "nearest", scale=1 / ratio)
    pan_f = ImfilterReplicate(mtf_fir(gnyq_pan(sensor), ratio), sz[0], sz[1])
    Phigh_lp_ori = np.zeros_like(P_highdim)
    Phigh_de = np.zeros((h, w, Nways[2]))
    for band in range(Nways[2]):
        op_band = pan_f(P_highdim[:, :, band])
        Phigh_lp_ori[:, :, band] = op_band
        Phigh_de[:, :, band] = imresize(op_band, None, "nearest", scale=1 / ratio)
    Phigh_de_lp = MTF_lrms(Phigh_de, sensor, "", ratio)
    I_LRMS_lp = MTF_lrms(I_lrms, sensor, "", ratio)
    I_LRMS_hp = I_lrms - I_LRMS_lp
    Phigh_de_hp = Phigh_lp_done - Phigh_de_lp
    if h % block_size[0] or w % block_size[1]:
        # Block_index = [sz/ratio./block_size, Nways(3)] is non-integer -> zeros(Block_index) errors in MATLAB
        raise ValueError(f"block_size {block_size} does not tile the {h}x{w} LRMS (official code cannot run)")
    nb1, nb2 = h // block_size[0], w // block_size[1]
    Coeffs_de = np.zeros((nb1, nb2, Nways[2]))
    for ii in range(nb1):
        for jj in range(nb2):
            r0, c0 = block_size[0] * ii, block_size[1] * jj
            for kk in range(Nways[2]):
                block_lrms = I_LRMS_hp[r0:r0 + block_size[0], c0:c0 + block_size[1], kk].ravel(order="F")
                block_pan = Phigh_de_hp[r0:r0 + block_size[0], c0:c0 + block_size[1], kk].ravel(order="F")
                with np.errstate(divide="ignore", invalid="ignore"):
                    s = 1.0 / np.dot(block_pan, block_pan)
                    Coeffs_de[ii, jj, kk] = np.dot(s * block_pan, block_lrms)
    Coeffs_de[Coeffs_de < 0] = 0
    Coeffs_de_sup = Suppl_coeffs(Coeffs_de, (lower, 1))
    Coeffs = imresize(Coeffs_de_sup, sz, "box")
    Inj_comps = Coeffs * (P_highdim - Phigh_lp_ori)
    return Inj_comps


def FFT_kernel(sf, sensor, Nways):
    """FFT_kernel.m -> dict(fft_B, fft_BT) (full spectra, as MATLAB stores them)."""
    sz = (Nways[0], Nways[1])
    g = gnyq_ms(sensor, Nways[2], "")                         # FFT_kernel's table == MTF_lrms's table (tag '')
    OTF = np.zeros(Nways, dtype=np.complex128)
    for i in range(Nways[2]):
        psf = mtf_fir(g[i], sf)
        OTF[:, :, i] = psf2otf(psf, sz)
    real = np.all(OTF.imag == 0)
    fft_B = OTF.real.copy() if real else OTF
    fft_BT = np.conj(fft_B) if not real else fft_B.copy()
    return dict(fft_B=fft_B, fft_BT=fft_BT)


class _Half:
    """rfft2-domain helpers for X_solver / B_filter.  MATLAB's ifft2 of these exactly conjugate-symmetric spectra returns
    real (c2r); rfft2/irfft2 are the same r2c/c2r transforms."""

    def __init__(self, par, Nways):
        H, W = Nways[0], Nways[1]
        self.s = (H, W)
        self.B = par["fft_B"][:, :W // 2 + 1, :]
        self.BT = par["fft_BT"][:, :W // 2 + 1, :]

    def F(self, a):
        return sfft.rfft2(a, axes=(0, 1), workers=_WORKERS)

    def Fi(self, a):
        return sfft.irfft2(a, s=self.s, axes=(0, 1), workers=_WORKERS)


def B_filter(z, half):
    """B_filter.m: real(ifft2(fft2(z) .* fft_B)) per band."""
    return half.Fi(half.F(z) * half.B)


def X_solver(U, V, Z, Theta_1, Theta_2, Theta_3, half, opts):
    """X_solver.m (all bands at once; per-element arithmetic in MATLAB's order)."""
    eta = opts["eta"]
    F = half.F
    FFT_molecular = (eta[0] * F(U) - F(Theta_1) + eta[1] * F(V) - F(Theta_2) + eta[2] * F(Z) * half.BT
                     - F(Theta_3) * half.BT)
    FFT_denominator = eta[2] * half.B * half.BT + (eta[0] + eta[1])
    X_out = half.Fi(FFT_molecular / FFT_denominator)
    X_out[X_out > 1] = 1
    X_out[X_out < 0] = 0
    return X_out


def ClosedWNNM(SigmaY, C, oureps):
    """ClosedWNNM.m."""
    temp = (SigmaY - oureps) ** 2 - 4 * (C - oureps * SigmaY)
    ind = np.nonzero(temp > 0)[0]
    svp = len(ind)
    SigmaX = np.maximum(SigmaY[ind] - oureps + np.sqrt(temp[ind]), 0) / 2
    return SigmaX, svp


def WNNM(Y, C, NSig):
    """WNNM.m."""
    U, SigmaY, Vh = np.linalg.svd(Y, full_matrices=False)
    X = np.zeros_like(Y)
    TempC = C * NSig ** 2
    SigmaX, svp = ClosedWNNM(SigmaY, TempC, EPS)
    if svp >= 1:
        X = (U[:, :svp] * SigmaX) @ Vh[:svp, :]
    return X


def _narrow(a):
    """MATLAB drops an all-zero imaginary part when indexing a complex array."""
    return a.real.copy() if np.all(a.imag == 0) else a


def Log_prox_tnn(Y, rho):
    """Log_prox_tnn.m.  Only the slices 1..floor(n3/2)+1 are ever computed by the official code (the rest are their
    conjugates), so fft(Y,[],3) is taken as rfft and the inverse as irfft (MATLAB returns real here as well)."""
    n1, n2, n3 = Y.shape
    Yf = sfft.rfft(Y, axis=2, workers=_WORKERS)
    X = np.zeros_like(Yf)
    X[:, :, 0] = WNNM(_narrow(Yf[:, :, 0]), rho, 1)
    halfn3 = int(matlab_round(n3 / 2))
    for i in range(2, halfn3 + 1):                            # MATLAB i = 2..halfn3 (1-based)
        X[:, :, i - 1] = WNNM(_narrow(Yf[:, :, i - 1]), rho, 1)
    if n3 % 2 == 0:
        i = halfn3 + 1
        X[:, :, i - 1] = WNNM(_narrow(Yf[:, :, i - 1]), rho, 1)
    return sfft.irfft(X, n=n3, axis=2, workers=_WORKERS)


def T_solver(Z, I_lrms, opts):
    """T_solver.m."""
    ratio = opts["ratio"]
    s0 = 3
    lam = opts["lambda_"]
    T = Log_prox_tnn(Z, lam[2] / (2 * lam[1]))
    T[s0 - 1::ratio, s0 - 1::ratio, :] = I_lrms
    return T


def inc_LRTCFPan(I_lrms, I_pan, paras, info=None):
    """inc_LRTCFPan.m.  `info` (optional dict) receives the iteration count and the last relative change."""
    tol = paras["tol"]
    eta = paras["eta"]
    maxit = paras["maxit"]
    Nways = paras["Nways"]
    ratio = paras["ratio"]
    sensor = paras["sensor"]
    lam = paras["lambda_"]
    I_incos = Compute_inj_comps(I_lrms, I_pan, paras)
    par = FFT_kernel(ratio, sensor, Nways)
    half = _Half(par, Nways)
    mtf_hr = MTFlrms(sensor, "", ratio, Nways[0], Nways[1], Nways[2])
    U = np.zeros(Nways)
    V = U.copy()
    Z = U.copy()
    T = U.copy()
    Theta_1 = U.copy()
    Theta_2 = V.copy()
    Theta_3 = Z.copy()
    X = interp23tap(I_lrms, ratio)
    X_k = X
    it, Rel_Err = 0, np.nan
    for it in range(1, maxit + 1):
        X = X_solver(U, V, Z, Theta_1, Theta_2, Theta_3, half, paras)
        U = Log_prox_tnn(X + Theta_1 / eta[0], 1 / eta[0])
        V = (2 * lam[0] * (Z + I_incos) + eta[1] * X + Theta_2) / (2 * lam[0] + eta[1])
        Z = (2 * lam[0] * (V - I_incos) + 2 * lam[1] * T + eta[2] * B_filter(X, half) + Theta_3) / (
            2 * (lam[0] + lam[1]) + eta[2])
        T = T_solver(Z, I_lrms, paras)
        Theta_1 = Theta_1 + eta[0] * (X - U)
        Theta_2 = Theta_2 + eta[1] * (X - V)
        Theta_3 = Theta_3 + eta[2] * (mtf_hr(X) - Z)
        with np.errstate(divide="ignore", invalid="ignore"):
            Rel_Err = np.linalg.norm((X - X_k).ravel()) / np.linalg.norm(X_k.ravel())
        X_k = X
        if Rel_Err < tol:
            break
    if info is not None:
        info.update(iterations=it, rel_err=float(Rel_Err))
    return X
