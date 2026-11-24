# SPDX-License-Identifier: LicenseRef-Vivone-Toolbox-NonProfit
#
# Python port of FE-HPM/{FE_HPM.m, FE.m} and Tools/{LPfilter.m, ndwt2_working.m, indwt2_working.m,
# estimation_alpha.m, interp23tap.m} of G. Vivone et al.'s pansharpening benchmark toolbox ("Pansharpening Toolbox for
# Distribution", companion of IEEE GRSM 9(1):53-81, 2021; the PanCollection MATLAB test toolbox and DLPan-Toolbox ship
# identical copies).  ndwt2_working.m / indwt2_working.m are themselves modified copies of the (obsolete) MATLAB Wavelet
# Toolbox functions ndwt2 / indwt2.
# Terms: the upstream MATLAB toolbox states only "Copyright (c) 2020. All rights reserved. This work should only be used
# for nonprofit purposes." (README.txt).  This port is a derivative of that code and keeps those terms; it is NOT
# covered by the Apache-2.0 licence of the rest of the PanBase repository.  See ../__init__.py.
"""FE-HPM: pansharpening by semiblind deconvolution (filter estimation + HPM injection), VO.

Method: G. Vivone, M. Simoes, M. Dalla Mura, R. Restaino, J. M. Bioucas-Dias, G. A. Licciardi, J. Chanussot,
"Pansharpening based on semiblind deconvolution", IEEE TGRS 53(4):1997-2010, 2015.
FE.m estimates the detail-extraction filter h from the data: start from the a-trous LP of the PAN, regress it on
[MS, 1] (global LS) to get the intensity I, edge-taper PAN and I, solve the Tikhonov + gradient-regularised
deconvolution h = F^-1{ conj(P) I / (|P|^2 + lambda + mu |G|^2) }, threshold, cut a 25x25 support around the peak,
normalise to unit sum, and repeat (5 iterations, re-filtering the PAN with the new h).  FE_HPM.m then equalises the
PAN to every band (mean/std), low-passes it with h, decimates (imresize nearest, 1/ratio), re-interpolates with the
23-tap polynomial and injects with HPM:  F_b = MS_b * P_b / (P_b,LP + eps).

Ported line by line (no change of algorithm or of operation order).  MATLAB built-ins are re-implemented with MATLAB's
semantics: imfilter(A,h,'replicate') (correlation, same size, edge replication), imfilter(A,h,'circular'),
imresize(A,1/r,'nearest') (MATLAB contributions() with the box kernel, no antialiasing -> rows/cols 3:4:end),
edgetaper(I,PSF), psf2otf, fft2(x,R,C), fftshift, mean2, std2 (N-1), max + ind2sub (column-major, first max),
backslash (least squares).

Parameters: PanCollection's toolbox has no FE-HPM wrapper, so the values are those of Vivone's benchmark toolbox
(Fusion_Algorithms_Reduced_Resolution.m and Fusion_Algorithms_Full_Resolution.m, identical):
  tap = 25, num_iter_max = 5, lambda = 1e5, mu = 1e5, threshold = 1e-3, filtername 'Basic',
  called as FE_HPM(I_MS, I_PAN, ratio, max(tap,3*ratio), lambda, mu, threshold, max(num_iter_max,3), 'Basic').
ratio = 4.  No sensor-dependent parameter (no MTF gains), so `ds` is unused; `ms_lr` is unused (FE_HPM takes the
interpolated MS, here PanCollection's lms).  Inputs are digital numbers (lambda, mu and threshold assume that scale).

Deviations: none in the algorithm.  Floating-point summation order of FFTs / LS / filters differs from MATLAB
(round-off level).  Where MATLAB would raise (peak too close to the border for the 25x25 cut), this raises too.
Verification (PanBase project, 2026-10-14): the official FE_HPM.m run in GNU Octave 10.3 (edgetaper replaced by a
transcription of MATLAB's algorithm) agrees with fuse() on all 160 PanCollection test inputs (20 RR + 20 FR of
WV3/QB/GF2/WV2): after the [0, divisor] clamp max |d| <= 7.2e-6 DN.
"""
import numpy as np
import torch
from scipy import ndimage

FAMILY = "VO"
REF = "Vivone et al., IEEE TGRS 2015"

RATIO = 4
TAP, NUM_ITER_MAX, LAMBDA, MU, THRESHOLD, FILTERNAME = 25, 5, 1e5, 1e5, 1e-3, "Basic"   # Vivone benchmark toolbox
EPS = np.finfo(np.float64).eps                                                         # MATLAB eps


def fuse(ms, ms_lr, pan, ds):
    I_MS = ms[0].detach().double().cpu().permute(1, 2, 0).numpy()                      # H, W, C (MATLAB layout)
    I_PAN = pan[0, 0].detach().double().cpu().numpy()                                 # H, W
    F, _ = FE_HPM(I_MS, I_PAN, RATIO, max(TAP, 3 * RATIO), LAMBDA, MU, THRESHOLD, max(NUM_ITER_MAX, 3), FILTERNAME)
    return torch.from_numpy(np.ascontiguousarray(F.transpose(2, 0, 1)))[None].to(torch.float64)


# ----------------------------------------------------------------------------------------------- FE_HPM.m
def FE_HPM(I_MS, I_PAN, ratio, tap, lam, mu, th, num_iter, filtername):
    imageHR = np.array(I_PAN, dtype=np.float64)
    I_MS = np.array(I_MS, dtype=np.float64)
    nBands = I_MS.shape[2]

    # Equalization
    imageHR = np.repeat(imageHR[:, :, None], nBands, axis=2)
    for ii in range(nBands):
        x = imageHR[:, :, ii]
        imageHR[:, :, ii] = (x - _mean2(x)) * (_std2(I_MS[:, :, ii]) / _std2(x)) + _mean2(I_MS[:, :, ii])

    PSF_l = FE(I_MS, I_PAN, ratio, tap, lam, mu, th, num_iter, filtername)

    PAN_LP = np.zeros(imageHR.shape)
    for ii in range(nBands):
        PAN_LP[:, :, ii] = _imfilter_replicate(imageHR[:, :, ii], PSF_l)
        t = _imresize_nearest(PAN_LP[:, :, ii], 1 / ratio)
        PAN_LP[:, :, ii] = _interp23tap(t, ratio)

    I_Fus = I_MS * (imageHR / (PAN_LP + EPS))
    return I_Fus, PSF_l


# ----------------------------------------------------------------------------------------------- FE.m
def _derivative_filters(filtername):
    if filtername == "Naive2":
        gv = np.array([[-1.0], [1.0]]); gh = np.array([[-1.0, 1.0]])
    elif filtername == "Naive3":
        gv = np.array([[-1.0], [0.0], [1.0]]); gh = np.array([[-1.0, 0.0, 1.0]])
    elif filtername == "Prewitt":
        gv = np.zeros((3, 3)); gv[0, :] = -1; gv[2, :] = 1
        gh = np.zeros((3, 3)); gh[:, 0] = -1; gh[:, 2] = 1
    elif filtername == "Sobel":
        gv = np.array([[-1.0, -2.0, -1.0], [0, 0, 0], [1.0, 2.0, 1.0]])
        gh = np.array([[-1.0, 0, 1.0], [-2.0, 0, 2.0], [-1.0, 0, 1.0]])
    else:                                                                              # 'Basic' and otherwise
        gv = np.zeros((2, 2)); gv[0, :] = -1; gv[1, :] = 1
        gh = np.zeros((2, 2)); gh[:, 0] = -1; gh[:, 1] = 1
    return gv, gh


def FE(I_MS, I_PAN, ratio, tap, lam, mu, th, num_iter, filtername):
    sum_tap = 0 if tap % 2 == 0 else 1
    tap = int(np.floor(tap / 2))
    R_SIZE, C_SIZE = I_PAN.shape

    gv, gh = _derivative_filters(filtername)
    gvf = np.fft.fft2(gv, s=(R_SIZE, C_SIZE))
    ghf = np.fft.fft2(gh, s=(R_SIZE, C_SIZE))
    gvf2 = np.conj(gvf) * gvf
    ghf2 = np.conj(ghf) * ghf
    gf2sum = gvf2 + ghf2

    H_E = np.array(I_PAN, dtype=np.float64)
    MS1 = np.concatenate([I_MS, np.ones(I_MS.shape[:2] + (1,))], axis=2)             # cat(3, I_MS, ones)
    box = np.ones((tap, tap)) / (tap ** 2)
    PSF_l = None
    for jj in range(1, num_iter + 1):
        # Filter PAN to estimate alpha set
        if jj == 1:
            PAN_LP = _LPfilter(H_E, ratio)
        else:
            PAN_LP = _imfilter_replicate(H_E, PSF_l)
        # Estimate alpha
        alpha = _estimation_alpha_global(MS1, PAN_LP)
        It_E = np.sum(MS1 * alpha[None, None, :], axis=2)
        # Edge taper (H_E is overwritten: the taper accumulates over the iterations, as in FE.m)
        H_E = _edgetaper(H_E, box)
        It_E = _edgetaper(It_E, box)
        # Filter estimation
        FH = np.fft.fft2(H_E)
        PSF = np.real(np.fft.fftshift(np.fft.ifft2(np.conj(FH) * np.fft.fft2(It_E) /
                                                   (np.abs(FH) ** 2 + lam + mu * gf2sum))))
        # Thresholding
        PSF[PSF < th] = 0
        # Cut using the support dimension and center (MATLAB max: first maximum in column-major order)
        rm, cm = np.unravel_index(int(np.argmax(PSF.ravel(order="F"))), PSF.shape, order="F")
        r0, r1, c0, c1 = rm - tap, rm + tap + sum_tap, cm - tap, cm + tap + sum_tap
        if r0 < 0 or c0 < 0 or r1 > R_SIZE or c1 > C_SIZE:
            raise IndexError(f"FE: PSF peak at ({rm},{cm}) too close to the border (MATLAB would error)")
        PSF_l = PSF[r0:r1, c0:c1]
        PSF_l = PSF_l / np.sum(PSF_l.ravel(order="F"))
    return PSF_l


# ----------------------------------------------------------------------------------------------- Tools/*.m
def _mean2(x):
    return float(np.mean(x))


def _std2(x):
    return float(np.std(x, ddof=1))


def _estimation_alpha_global(I_MS, I_PAN):
    """estimation_alpha(I_MS, I_PAN, 'global'):  alpha = ILRc \\ IHc  (least squares)."""
    IHc = I_PAN.reshape(-1, order="F")
    ILRc = I_MS.reshape(-1, I_MS.shape[2], order="F")
    return np.linalg.lstsq(ILRc, IHc, rcond=None)[0]


def _imfilter_replicate(A, h):
    """imfilter(A, h, 'replicate'): correlation, 'same', centre floor((size(h)+1)/2) (odd h here)."""
    assert h.shape[0] % 2 == 1 and h.shape[1] % 2 == 1, h.shape
    return ndimage.correlate(A, h, mode="nearest")


def _imresize_nearest(A, scale):
    """imresize(A, scale, 'nearest') for a 2-D double: MATLAB contributions() with the box kernel, no antialiasing."""
    def idx(in_len):
        out_len = int(np.ceil(in_len * scale))
        x = np.arange(1, out_len + 1, dtype=np.float64)
        u = x / scale + 0.5 * (1 - 1 / scale)
        left = np.floor(u - 0.5)                                                      # kernel_width = 1
        ind = left[:, None] + np.arange(3)[None, :]                                    # P = ceil(1) + 2
        d = u[:, None] - ind
        w = (-0.5 <= d) & (d < 0.5)                                                    # box(x)
        assert (w.sum(1) == 1).all()
        return np.clip((ind * w).sum(1), 1, in_len).astype(np.int64) - 1
    return A[np.ix_(idx(A.shape[0]), idx(A.shape[1]))]


_CDF23 = 2.0 * np.array([0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0,
                         0.000807762146, 0, -0.000060081482])
_BASECOEFF = np.concatenate([_CDF23[1:][::-1], _CDF23])                                # 23 taps


def _interp23tap(I, ratio):
    """interp23tap.m for a 2-D image (zero insertion + circular 23-tap filtering, first along rows then columns)."""
    assert 2 ** round(np.log2(ratio)) == ratio
    r, c = I.shape
    first = True
    for z in range(1, ratio // 2 + 1):
        I1LRU = np.zeros(((2 ** z) * r, (2 ** z) * c))
        if first:
            I1LRU[1::2, 1::2] = I
            first = False
        else:
            I1LRU[0::2, 0::2] = I
        t = ndimage.correlate1d(I1LRU, _BASECOEFF, axis=0, mode="wrap")              # imfilter(t', BaseCoeff, 'circular')
        I = ndimage.correlate1d(t, _BASECOEFF, axis=1, mode="wrap")                   # imfilter(t', BaseCoeff, 'circular')
    return I


def _psf2otf(psf, out_size):
    psf_size = np.array(psf.shape)
    if np.all(psf == 0):
        return np.zeros(out_size)
    pad = np.zeros(out_size)
    pad[:psf_size[0], :psf_size[1]] = psf
    pad = np.roll(pad, shift=tuple(int(s) for s in -(psf_size // 2)), axis=(0, 1))   # circshift(psf, -floor(size/2))
    otf = np.fft.fft2(pad)
    n_elem = float(np.prod(psf_size))
    n_ops = sum(k * np.log2(k) * (n_elem / k) for k in psf_size)
    if np.max(np.abs(otf.imag)) / np.max(np.abs(otf)) <= n_ops * EPS:
        otf = otf.real
    return otf


def _edgetaper(I, PSF):
    """MATLAB edgetaper(I, PSF) (Image Processing Toolbox algorithm; Octave's edgetaper is a different function)."""
    size_i = I.shape
    alpha = np.ones(size_i)
    for n in range(2):
        if PSF.shape[n] == 1:
            continue
        proj = PSF.sum(axis=1 - n)                                                     # projection on dimension n
        z = np.real(np.fft.ifft(np.abs(np.fft.fft(proj, size_i[n] - 1)) ** 2))
        z = np.concatenate([z, z[:1]]) / np.max(z)                                     # z([1:end 1]) / max(z(:))
        shape = [1, 1]
        shape[n] = size_i[n]
        alpha = alpha * (1 - z).reshape(shape)
    otf = _psf2otf(PSF, size_i)
    return alpha * I + (1 - alpha) * np.real(np.fft.ifft2(np.fft.fft2(I) * otf))


# LPfilter.m: a-trous ('sym' extension) non-decimated wavelet, details zeroed, inverse  (ndwt2_working / indwt2_working)
def _conv_full_axis1(X, f):
    out = np.zeros((X.shape[0], X.shape[1] + len(f) - 1))
    for k, fk in enumerate(f):
        out[:, k:k + X.shape[1]] += fk * X
    return out


def _wdec1D(X, Lo, Hi, perm):
    if perm:
        X = X.T
    lf = len(Lo)
    lx = X.shape[1]
    lc = lx + lf - 1
    X = np.concatenate([X[:, np.arange(lf - 2, -1, -1)], X, X[:, np.arange(lx - 1, lx - lf - 1, -1)]], axis=1)  # 'sym'
    L = _conv_full_axis1(X, Lo)
    H = _conv_full_axis1(X, Hi)
    lenL = L.shape[1]
    first, last = lf, lenL - lf + 1
    L, H = L[:, first - 1:last], H[:, first - 1:last]
    lenL = L.shape[1]
    first = 1 + int(np.floor((lenL - lc) / 2))
    last = first + lc - 1
    L, H = L[:, first - 1:last], H[:, first - 1:last]
    if perm:
        L, H = L.T, H.T
    return L, H


def _decFUNC(X, LoD, HiD):
    a_Lo, d_Hi = _wdec1D(X, LoD[0], HiD[0], False)
    d11, d12 = _wdec1D(a_Lo, LoD[1], HiD[1], True)
    d21, d22 = _wdec1D(d_Hi, LoD[1], HiD[1], True)
    return [[d11, d12], [d21, d22]]


def _ndwt2(X, level, LoD, HiD):
    sizes = [None] * (level + 1)
    sizes[level] = X.shape
    cfs = None
    for k in range(1, level + 1):
        dec = _decFUNC(X, LoD, HiD)
        X = dec[0][0]
        sizes[level - k] = X.shape
        dec4 = [dec[0][0], dec[1][0], dec[0][1], dec[1][1]]                           # reshape(dec, 4, 1, 1)
        cfs = dec4 if k == 1 else dec4 + cfs[1:]
    return cfs, sizes


def _wrec1D(X, F, perm):
    if perm:
        X = X.T
    X = _conv_full_axis1(X, F)
    if perm:
        X = X.T
    return X


def _recFUNC(dec, sINI, Lo, Hi):
    W = [_wrec1D(dec[i][0], Lo[1], True) + _wrec1D(dec[i][1], Hi[1], True) for i in range(2)]
    X = (_wrec1D(W[0], Lo[0], False) + _wrec1D(W[1], Hi[0], False)) / 4
    sREC = np.array(X.shape)
    sINI = np.array(sINI)
    F = np.floor((sREC - sINI) / 2).astype(int)
    C = np.ceil((sREC - sINI) / 2).astype(int)
    return X[F[0]:sREC[0] - C[0], F[1]:sREC[1] - C[1]]


def _indwt2(cfs, sizes, level, LoR, HiR):
    cfs = list(cfs)
    idxBeg = 0
    X = None
    for k in range(1, level + 1):
        idxEnd = idxBeg + 3
        c4 = cfs[idxBeg:idxEnd + 1]
        dec = [[c4[0], c4[2]], [c4[1], c4[3]]]                                         # reshape(cfs(..), 2, 2)
        X = _recFUNC(dec, sizes[k], LoR, HiR)
        cfs[idxEnd] = X
        idxBeg = idxEnd
    return X


def _LPfilter(HRPan, ratio):
    h = np.array([1, 4, 6, 4, 1]) / 16
    g = np.array([0, 0, 1, 0, 0]) - h
    htilde = np.array([1, 4, 6, 4, 1]) / 16
    gtilde = np.array([0, 0, 1, 0, 0]) + htilde
    h, g, htilde, gtilde = np.sqrt(2) * h, np.sqrt(2) * g, np.sqrt(2) * htilde, np.sqrt(2) * gtilde
    Levels = int(np.ceil(np.log2(ratio)))
    cfs, sizes = _ndwt2(np.asarray(HRPan, dtype=np.float64), Levels, [h, h], [g, g])
    cfs = [cfs[0]] + [np.zeros(c.shape) for c in cfs[1:]]                              # WT.dec{2:end} = 0
    return _indwt2(cfs, sizes, Levels, [htilde, htilde], [gtilde, gtilde])
