# SPDX-License-Identifier: Apache-2.0
"""Detail forms of PanBase and their FFT band split.

With ``P`` the PAN, ``L`` the interpolated MS (PAN grid), ``m`` the low-resolution MS and ``k`` the MTF kernel of the
mean Nyquist gain of the MS bands:

    P_LP  = k * pad_replicate(P)                    PAN low-pass on the PAN grid (no decimation)
    d     = P - P_LP                                PAN detail

    fs    r^1_b = d                                 additive (component substitution)
    hpm   r^2_b = (L_b / P_LP) d                    multiplicative (high-pass modulation)
    cbd   r^3_b = beta_b d                          local regression (context-based decision):
              beta_b = cov_7x7(m_b, P_lr) / var_7x7(P_lr) on the LR grid, clipped to [-2, 4], upsampled bilinearly,
              P_lr = D_mean(P) (MTF of the mean gain, replicate padding, decimation at the operator's offset)

MS geometry (s_ms): the same three forms with the PAN quantities shifted by the input-estimated translation tau
(:mod:`panbase.align`):  d -> d(x + tau),  P_LP -> P_LP(x + tau),  P -> P(x + tau) inside beta.

Every form is split into three frequency bands by a sum-preserving FFT partition (``F_1 + F_2 + F_3 = identity``):
masks ``1 - a_1, a_1 - a_2, a_2`` with ``a_j = sigmoid((nu - kappa_j) / t)``, ``kappa = (1, 2)``, ``t = 0.15`` and
``nu = 2 r |f|`` (f in cycles per PAN pixel of the reflect-extended grid; nu = 1 is the MS Nyquist radius).  The image is
reflect-extended by ``max(4, min(H, W) // 4)`` pixels before the transform and cropped back afterwards.

Small floors that guard divisions (``1e-3 * mean |P_LP|`` and ``1e-6 * var(P_lr)``) are computed over the whole batch, as
for the paper's bases; they only act on (near-)constant regions.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .align import estimate_shift, fourier_shift
from .mtf import mtf_kernel
from .projection import MTFObservation

FORMS = ("fs", "hpm", "cbd")


class FFTBands:
    """Split (N, C, H, W) maps into (N, C, J, H, W) frequency bands that sum back to the input."""

    def __init__(self, ratio: float, kappas=(1.0, 2.0), t: float = 0.15):
        self.r, self.kappas, self.t = float(ratio), [float(k) for k in sorted(kappas)], float(t)
        self.J = len(self.kappas) + 1
        self._cache = {}

    def spec(self) -> dict:
        return dict(kappas=self.kappas, t=self.t, coordinate="nu = 2 r |f|, f = fftfreq of the reflect-extended grid",
                    extension="reflect", halo="max(4, min(H, W) // 4)")

    def masks(self, Hp: int, Wp: int, device, dtype) -> torch.Tensor:
        key = (Hp, Wp, str(device), dtype)
        if key not in self._cache:
            fy = torch.fft.fftfreq(Hp, dtype=torch.float64)[:, None]
            fx = torch.fft.rfftfreq(Wp, dtype=torch.float64)[None, :]
            nu = 2.0 * self.r * torch.sqrt(fx ** 2 + fy ** 2)
            a = [torch.sigmoid((nu - k) / self.t) for k in self.kappas]
            W = [1.0 - a[0]] + [a[j - 1] - a[j] for j in range(1, len(a))] + [a[-1]]
            self._cache[key] = torch.stack(W).to(device=device, dtype=dtype)
        return self._cache[key]

    def split(self, d: torch.Tensor) -> torch.Tensor:
        N, C, H, W = d.shape
        p = max(4, min(H, W) // 4)
        de = F.pad(d, (p, p, p, p), mode="reflect")
        Hp, Wp = de.shape[-2:]
        Wm = self.masks(Hp, Wp, d.device, d.dtype)                                    # (J, Hp, Wp // 2 + 1)
        out = torch.fft.irfft2(torch.fft.rfft2(de)[:, :, None] * Wm[None, None], s=(Hp, Wp))
        return out[..., p:p + H, p:p + W]


class DetailForms:
    """The three regressors of one geometry for a sensor (operator D, kernel of the mean gain)."""

    def __init__(self, obs: MTFObservation, geometry: str = "pan"):
        if geometry not in ("pan", "ms"):
            raise ValueError("geometry must be 'pan' (s_pan) or 'ms' (s_ms)")
        self.geometry = geometry
        self.obs = obs
        g = obs.gnyq
        # The PAN low-pass and the LR PAN use the mean Nyquist gain of the MS bands.  The paper's implementation wrote
        # that mean in two ways (sum / len for P_LP, numpy.mean for P_lr); for WV3 and WV2 they differ by one ulp, which
        # changes a few float32 kernel entries by one ulp, so both are kept to reproduce the paper's bases exactly.
        k = mtf_kernel(float(sum(g) / len(g)), obs.ratio, obs.kernel_size)
        self.lp_weight = torch.as_tensor(np.ascontiguousarray(k), dtype=obs.dtype, device=obs.device)[None, None]
        self.obs_pan = MTFObservation([float(np.mean(g))], obs.ratio, obs.offset, obs.kernel_size, obs.dtype, obs.device)
        self.bands = FFTBands(obs.ratio)

    def pan_lowpass(self, P: torch.Tensor) -> torch.Tensor:
        w = self.lp_weight.to(device=P.device, dtype=P.dtype)
        return F.conv2d(F.pad(P, (w.shape[-1] // 2,) * 4, mode="replicate"), w)

    def cbd_gain(self, m: torch.Tensor, P: torch.Tensor, win: int = 7) -> torch.Tensor:
        """beta (N, C, H, W): local LR regression coefficient of m_b on the LR PAN, upsampled to the PAN grid."""
        P_lr = self.obs_pan.D(P)
        h, w = min(P_lr.shape[-2], m.shape[-2]), min(P_lr.shape[-1], m.shape[-1])
        P_lr, mm = P_lr[..., :h, :w], m[..., :h, :w]
        box = lambda x: F.avg_pool2d(F.pad(x, (win // 2,) * 4, mode="replicate"), win, 1)
        mp, pp = box(mm), box(P_lr)
        cov = box(mm * P_lr) - mp * pp
        var = (box(P_lr * P_lr) - pp * pp).clamp_min(1e-6 * float(P_lr.var()))
        beta = (cov / var).clamp(-2.0, 4.0)
        return F.interpolate(beta, size=P.shape[-2:], mode="bilinear", align_corners=False)

    def regressors(self, L: torch.Tensor, m: torch.Tensor, P: torch.Tensor, tau: torch.Tensor | None = None):
        """[r_fs (N, 1, H, W), r_hpm (N, C, H, W), r_cbd (N, C, H, W)] and the translation tau (None for s_pan).

        `tau` (s_ms only): (N, 2) PAN -> MS translation in PAN pixels, (y, x); estimated from the inputs if None."""
        PL = self.pan_lowpass(P)
        dP = P - PL
        if self.geometry == "pan":
            hpm = L / PL.clamp_min(1e-3 * float(PL.abs().mean())) * dP
            return [dP, hpm, self.cbd_gain(m, P) * dP], None
        H, W = P.shape[-2:]
        if tau is None:
            tau = estimate_shift(L, PL, self.obs.ratio, max_shift=float(min(32, min(H, W) // 4)))
        else:
            tau = torch.as_tensor(tau, device=P.device, dtype=P.dtype).reshape(P.shape[0], 2)
        halo = 16 + int(np.ceil(float(tau.abs().max())))                              # larger than the shift: no wrap
        dA = fourier_shift(dP, tau, halo)
        PLa = fourier_shift(PL, tau, halo)
        hpm = (L / PLa.clamp_min(1e-3 * float(PLa.abs().mean()))) * dA
        Pa = fourier_shift(P, tau, halo)
        return [dA, hpm, self.cbd_gain(m, Pa) * dA], tau


__all__ = ["FFTBands", "DetailForms", "FORMS"]
