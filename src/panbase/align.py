# SPDX-License-Identifier: Apache-2.0
"""Input-only PAN -> MS translation estimate (used by the MS-geometry base s_ms).

Per image, a global sub-pixel translation tau (in PAN pixels, (y, x)) with  P_LP(x + tau) ~ I(x),  where

* ``P_LP`` is the PAN blurred with the mean MS MTF (no decimation) and
* ``I = sum_b alpha_b L_b + beta`` is the least-squares synthetic intensity of the interpolated MS ``L`` that best explains
  ``P_LP``.

The estimate is a band-limited (nu = 2 r |f| < 1.5), Hann-windowed phase correlation for the integer peak with a
parabolic sub-pixel fit (|tau| <= max_shift), refined by Gauss-Newton (Lucas-Kanade) steps on the windowed residual
``P_LP(x + tau) - I_tau(x)`` with the intensity refitted after every step, until the largest step of the batch is below
1e-3 px (at most 10 steps, each clamped to +-2 px).  Shifts are applied as Fourier phase ramps on a reflect-extended,
odd-sized grid.

Note: the stopping rule and the reflect halo of :func:`fourier_shift` callers are shared by the images of one batch
(this is how the paper's bases were computed); a different batch composition changes tau by less than about 1e-3 px.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def synth_intensity(L: torch.Tensor, PL: torch.Tensor):
    """(N, 1, H, W) least-squares fit of PL on [L_1 .. L_C, 1] per sample (centred, solved in float64)."""
    N, C, H, W = L.shape
    X = L.flatten(2).transpose(1, 2).double()                                          # (N, HW, C)
    y = PL.flatten(2).transpose(1, 2).double()                                         # (N, HW, 1)
    w = torch.ones_like(y)
    sw = w.sum(1, keepdim=True).clamp_min(1e-12)
    mx, my = (w * X).sum(1, keepdim=True) / sw, (w * y).sum(1, keepdim=True) / sw
    Xc, yc = (X - mx) * w, (y - my) * w
    XtX = Xc.transpose(1, 2) @ Xc
    dg = XtX.diagonal(dim1=1, dim2=2).mean(-1)
    reg = (1e-8 * dg + 1e-12 * float(dg.max().clamp_min(1e-30)))[:, None, None] * torch.eye(C, device=L.device,
                                                                                            dtype=X.dtype)
    a = torch.linalg.solve(XtX + reg, Xc.transpose(1, 2) @ yc)                         # (N, C, 1)
    inten = (X @ a + (my - mx @ a)).transpose(1, 2).reshape(N, 1, H, W)
    return inten.to(L.dtype)


def fourier_shift(d: torch.Tensor, tau: torch.Tensor, halo: int = 16) -> torch.Tensor:
    """d'(x) = d(x + tau) per sample (tau (N, 2) in pixels, (y, x)); reflect halo, odd-sized FFT grid, real output."""
    N, C, H, W = d.shape
    p = int(halo)
    padb = (p, p + (1 - (W + 2 * p) % 2), p, p + (1 - (H + 2 * p) % 2))                 # make both sizes odd
    de = F.pad(d, padb, mode="reflect")
    Hp, Wp = de.shape[-2:]
    fy = torch.fft.fftfreq(Hp, device=d.device, dtype=d.dtype)[:, None]
    fx = torch.fft.fftfreq(Wp, device=d.device, dtype=d.dtype)[None, :]
    ph = 2 * torch.pi * (fy[None] * tau[:, 0, None, None] + fx[None] * tau[:, 1, None, None])     # (N, Hp, Wp)
    out = torch.fft.ifft2(torch.fft.fft2(de) * torch.polar(torch.ones_like(ph), ph)[:, None])
    return out.real[..., p:p + H, p:p + W]


def _phase_correlation(a: torch.Tensor, b: torch.Tensor, max_shift: float, nu_max: float, r: float) -> torch.Tensor:
    """tau (N, 2) with a(x + tau) ~ b(x): whitened, band-limited cross-power peak + separable parabolic refinement."""
    N, _, H, W = a.shape
    wy = torch.hann_window(H, periodic=False, device=a.device, dtype=a.dtype)
    wx = torch.hann_window(W, periodic=False, device=a.device, dtype=a.dtype)
    win = wy[:, None] * wx[None, :]
    za = (a - a.mean((-2, -1), keepdim=True)) * win
    zb = (b - b.mean((-2, -1), keepdim=True)) * win
    Rx = torch.fft.fft2(za) * torch.fft.fft2(zb).conj()
    Rx = Rx / Rx.abs().clamp_min(1e-12 * Rx.abs().amax((-2, -1), keepdim=True))
    fy = torch.fft.fftfreq(H, device=a.device, dtype=a.dtype)[:, None]
    fx = torch.fft.fftfreq(W, device=a.device, dtype=a.dtype)[None, :]
    Rx = Rx * (2 * r * torch.sqrt(fy ** 2 + fx ** 2) < nu_max).to(a.dtype)
    c = torch.fft.ifft2(Rx).real[:, 0]                                                 # c(tau) = sum_x a(x + tau) b(x)
    ys = torch.fft.fftfreq(H, 1.0 / H, device=a.device, dtype=a.dtype)
    xs = torch.fft.fftfreq(W, 1.0 / W, device=a.device, dtype=a.dtype)
    far = (ys.abs()[:, None] > max_shift) | (xs.abs()[None, :] > max_shift)
    idx = c.masked_fill(far[None], -1e30).flatten(1).argmax(1)
    iy, ix = idx // W, idx % W

    def parabola(cm1, c0, cp1):
        den = cm1 - 2 * c0 + cp1
        return torch.where(den.abs() > 1e-12, 0.5 * (cm1 - cp1) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)

    n = torch.arange(N, device=a.device)
    c0 = c[n, iy, ix]
    dy = parabola(c[n, (iy - 1) % H, ix], c0, c[n, (iy + 1) % H, ix])
    dx = parabola(c[n, iy, (ix - 1) % W], c0, c[n, iy, (ix + 1) % W])
    return torch.stack([ys[iy] + dy, xs[ix] + dx], -1)


def _gauss_newton_step(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """-(A + eps I)^{-1} b per sample in float64; a tile without gradient energy gets a zero step."""
    A64, b64 = A.double(), b.double()
    dg = A64.diagonal(dim1=1, dim2=2).mean(-1)
    flat = (dg <= 0) | (dg <= 1e-12 * float(dg.max().clamp_min(1e-300)))
    eps = torch.where(flat, torch.ones_like(dg), 1e-9 * dg)
    step = -torch.linalg.solve(A64 + eps[:, None, None] * torch.eye(2, device=A.device, dtype=torch.float64), b64)
    return torch.where(flat[:, None], torch.zeros_like(step), step).to(A.dtype)


def _refine(PL: torch.Tensor, L: torch.Tensor, tau: torch.Tensor, iters: int = 10, halo: int = 16) -> torch.Tensor:
    """Gauss-Newton on  min_tau sum_x w(x) [PL(x + tau) - I_tau(x)]^2  (separable Hann weight, spectral derivatives)."""
    N, _, H, W = PL.shape
    wy = torch.hann_window(H, periodic=False, device=PL.device, dtype=PL.dtype)
    wx = torch.hann_window(W, periodic=False, device=PL.device, dtype=PL.dtype)
    w = (wy[:, None] * wx[None, :])[None, None]
    ey = torch.tensor([0.5, 0.0], device=PL.device, dtype=PL.dtype)
    ex = torch.tensor([0.0, 0.5], device=PL.device, dtype=PL.dtype)
    for _ in range(iters):
        halo = max(halo, 16 + int(float(tau.abs().max())) + 1)
        Ps = fourier_shift(PL, tau, halo)
        inten = synth_intensity(L, Ps)
        gy = fourier_shift(PL, tau + ey, halo) - fourier_shift(PL, tau - ey, halo)
        gx = fourier_shift(PL, tau + ex, halo) - fourier_shift(PL, tau - ex, halo)
        e = Ps - inten
        J = torch.stack([gy, gx], -1)                                                  # (N, 1, H, W, 2)
        A = torch.einsum("nchwi,nchwj->nij", J * w[..., None], J)
        b = torch.einsum("nchwi,nchw->ni", J * w[..., None], e)
        step = _gauss_newton_step(A, b)
        tau = tau + step.clamp(-2.0, 2.0)
        halo = max(halo, 16 + int(float(tau.abs().max())) + 1)
        if float(step.abs().max()) < 1e-3:                                             # every image of the batch converged
            break
    return tau


def estimate_shift(L: torch.Tensor, PL: torch.Tensor, ratio: float = 4.0, max_shift: float | None = None,
                   nu_max: float = 1.5, iters: int = 10) -> torch.Tensor:
    """tau (N, 2), in PAN pixels, aligning the low-pass PAN to the MS geometry:  PL(x + tau) ~ MS structure(x).

    L: (N, C, H, W) interpolated MS;  PL: (N, 1, H, W) MTF-low-passed PAN.  `max_shift` defaults to min(32, min(H, W) // 4).
    """
    H, W = PL.shape[-2:]
    if max_shift is None:
        max_shift = float(min(32, min(H, W) // 4))
    inten = synth_intensity(L, PL)
    tau = _phase_correlation(PL, inten, float(max_shift), nu_max, float(ratio))
    if iters:
        tau = _refine(PL, L, tau, iters)
    return tau


__all__ = ["estimate_shift", "fourier_shift", "synth_intensity"]
