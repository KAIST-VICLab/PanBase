# SPDX-License-Identifier: Apache-2.0
"""MTF-matched low-pass kernels for the PanCollection observation model.

Every multispectral (MS) band b is modelled with a Gaussian modulation transfer function (MTF) whose amplitude at the
Nyquist frequency of the MS grid equals ``gnyq[b]`` (Aiazzi et al., PE&RS 2006; Vivone et al., IEEE TGRS 2015 and IEEE
GRSM 2021).  The discrete kernel is the classical frequency-sampling / window design of an ``N x N`` FIR filter
(N = 41 by default):

1. Desired response.  On the centred integer frequency grid ``k = -(N-1)/2 .. (N-1)/2`` (both axes) the response is the
   Gaussian ``G(k) = exp(-|k|^2 / (2 alpha^2))`` with ``alpha = sqrt(((N-1) / (2 r))^2 / (-2 ln gnyq))``, so that
   ``G = gnyq`` at ``|k| = (N-1) / (2 r)``, the MS Nyquist frequency for the resolution ratio ``r``.  Values below
   machine epsilon times the peak are set to zero, the array is normalised to unit sum and then to unit peak.
2. Frequency sampling.  The spatial filter is the real part of the centred inverse 2-D DFT of that response (zero
   frequency moved to the origin before the inverse transform, back to the centre after it), rotated by 180 degrees.
3. Window.  The filter is multiplied by a circularly symmetric window obtained by evaluating the 1-D Kaiser window
   (length N, beta = 0.5) at the radius ``rho = sqrt(t1^2 + t2^2)`` of the normalised coordinates ``t in [-1, 1]``
   (linear interpolation, zero outside ``rho > 1``).

This is a re-implementation from the published description of the design, written for this package.  It reproduces the
kernels that generated the PanCollection reduced-resolution data (``D(GT) = MS`` to round-off on the WV3 / GF2 splits),
which is what makes the projection of PanBase exact.

The Nyquist gains below are the published sensor values used by PanCollection (WorldView-3, QuickBird, WorldView-2) and
the customary default 0.3 for GaoFen-2, which has no published table.
"""
from __future__ import annotations

import numpy as np

#: Nyquist gains of the MS bands (PanCollection band order).  ``wv2`` is the WorldView-2 operator; PanBase's WV2
#: coefficients are the WV3 ones (no WV2 training split exists).
GNYQ = {
    "wv3": [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315],
    "qb": [0.34, 0.32, 0.30, 0.22],
    "gf2": [0.3, 0.3, 0.3, 0.3],
    "wv2": [0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.27],
}


def _gaussian_response(n: int, alpha: float) -> np.ndarray:
    """Centred n x n Gaussian of standard deviation `alpha` (grid units), unit sum, tiny values zeroed."""
    half = (n - 1.0) / 2.0
    k = np.arange(-half, half + 1.0)
    g = np.exp(-(k[None, :] * k[None, :] + k[:, None] * k[:, None]) / (2.0 * alpha * alpha))
    g[g < np.finfo(g.dtype).eps * g.max()] = 0.0
    s = g.sum()
    if s != 0:
        g = g / s
    return g


def _frequency_sampling(hd: np.ndarray) -> np.ndarray:
    """Spatial FIR filter whose DFT samples equal the centred desired response `hd` (odd size)."""
    h = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(hd)))
    return np.real(h[::-1, ::-1])


def _radial_window(w1: np.ndarray) -> np.ndarray:
    """Circularly symmetric 2-D window from a symmetric 1-D window (linear interpolation in the radius)."""
    n = w1.shape[0]
    t = np.arange(-(n - 1) / 2, (n - 1) / 2 + 1) * 2 / (n - 1)
    t1, t2 = np.meshgrid(t, t)
    rho = np.sqrt(t1 ** 2 + t2 ** 2)
    outside = (rho < t[0]) | (rho > t[-1])
    rho[outside] = 0.0
    w = np.interp(rho.ravel(), t, w1).reshape(rho.shape)
    w[outside] = 0.0
    return w


def mtf_kernel(gnyq: float, ratio: int = 4, size: int = 41) -> np.ndarray:
    """(size, size) float64 MTF-matched low-pass kernel of one band (see the module docstring)."""
    if not 0.0 < float(gnyq) < 1.0:
        raise ValueError(f"gnyq must lie in (0, 1), got {gnyq}")
    if size % 2 != 1:
        raise ValueError("the kernel size must be odd")
    fcut = 1.0 / float(ratio)
    alpha = np.sqrt(((size - 1) * (fcut / 2)) ** 2 / (-2 * np.log(float(gnyq))))
    g = _gaussian_response(size, alpha)
    hd = g / np.max(g)
    return _frequency_sampling(hd) * _radial_window(np.kaiser(size, 0.5))


def mtf_kernels(gnyq, ratio: int = 4, size: int = 41) -> np.ndarray:
    """(C, size, size) float64 kernels, one per entry of `gnyq`."""
    return np.stack([mtf_kernel(g, ratio, size) for g in gnyq])
