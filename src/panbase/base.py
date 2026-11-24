# SPDX-License-Identifier: Apache-2.0
"""PanBase: a calibrated, observation-consistent physical estimate s of the high-resolution MS image.

    u_b = L_b + sum_{i in forms} sum_{j in bands} g_{b,i,j} F_j(r^i_b)
    s   = clip( Pi(u; m), 0, max_value )

* ``L``  interpolated MS on the PAN grid (PanCollection's ``lms``), ``m`` the low-resolution MS (``ms``), ``P`` the PAN.
* ``r^i_b``  the three detail forms of :mod:`panbase.details` (additive, multiplicative, local regression) in PAN
  geometry (``s_pan``) or shifted to the MS geometry by an input-estimated translation (``s_ms``).
* ``F_j``  the three-band FFT partition;  ``g``  9 calibrated scalars per MS band (fitted on training data, see
  :mod:`panbase.fit`);  ``Pi``  the projection onto ``{x : D x = m}`` (:mod:`panbase.projection`).

No network, no GPU needed.  The base is scale-equivariant (every floor is relative), so the inputs can be given in any
consistent unit as long as ``max_value`` (the clipping bound) is in the same unit:

* PanCollection h5 arrays divided by 2047 (WV3, QB, WV2) or 1023 (GF2), the usual training convention: ``max_value=1``;
* raw h5 values (DN): ``max_value=2047`` or ``1023`` (the paper's bases were computed this way).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .details import FORMS, DetailForms
from .mtf import GNYQ
from .projection import MTFObservation

FORMAT = "panbase-coefficients/1"
GEOMETRIES = {"pan": "s_pan", "ms": "s_ms"}
DATA_RANGE = {"wv3": 2047.0, "qb": 2047.0, "gf2": 1023.0, "wv2": 2047.0}


def _as_tensor(x, name: str, ndim_img: int):
    if x is None:
        raise ValueError(f"`{name}` is required")
    t = torch.as_tensor(np.asarray(x)) if not torch.is_tensor(x) else x
    if t.dim() == ndim_img - 1:
        t = t[None]
    if t.dim() != ndim_img:
        raise ValueError(f"`{name}` must be (N, C, H, W) or (C, H, W); got shape {tuple(t.shape)}")
    return t


class PanBase:
    """Network-free base estimator for one sensor and geometry.

    Parameters
    ----------
    gains : array (C, 3, 3)
        ``g[b, i, j]``: MS band b, detail form i in ``("fs", "hpm", "cbd")``, frequency band j (low, mid, high).
    gnyq : list of float
        Nyquist gains of the C MS bands (the sensor operator D).
    geometry : "pan" or "ms"
        ``"pan"`` = s_pan (detail aligned to the PAN), ``"ms"`` = s_ms (detail shifted to the MS geometry).
    ratio, offset : int
        Resolution ratio and decimation offset of D (PanCollection: 4 and 2).
    dtype, device :
        Compute precision and device (float32 on CPU by default, as for the paper's bases).
    """

    def __init__(self, gains, gnyq, geometry: str = "pan", ratio: int = 4, offset: int | None = None,
                 dtype: torch.dtype = torch.float32, device="cpu", meta: dict | None = None):
        g = np.asarray(gains, dtype=np.float64)
        if g.ndim != 3 or g.shape[1:] != (3, 3) or g.shape[0] != len(gnyq):
            raise ValueError(f"gains must have shape (C, 3, 3) with C = len(gnyq) = {len(gnyq)}; got {g.shape}")
        if geometry not in GEOMETRIES:
            raise ValueError("geometry must be 'pan' or 'ms'")
        self.gains = torch.as_tensor(g)
        self.geometry = geometry
        self.obs = MTFObservation(gnyq, ratio, offset, 41, dtype, device)
        self.forms = DetailForms(self.obs, geometry)
        self.dtype, self.device = dtype, torch.device(device)
        self.meta = dict(meta or {})

    # ------------------------------------------------------------------ construction / I/O
    @classmethod
    def from_pretrained(cls, path, dataset: str | None = None, geometry: str | None = None, *, dtype=torch.float32,
                        device="cpu") -> "PanBase":
        """Load released coefficients.

        `path` is the release directory holding ``config.json`` (then `dataset` -- 'wv3', 'qb', 'gf2' or 'wv2' -- and
        `geometry` -- 'pan' for s_pan or 'ms' for s_ms -- pick the entry), or a single coefficient file
        (``panbase_<dataset>_<geometry>.json``, which carries its own geometry and operator).  WV2 has no training split:
        its directory entry uses the WV3 coefficients with the WV2 operator (zero-shot), as in the paper.
        """
        path = os.fspath(path)
        gnyq = None
        if os.path.isdir(path):
            if dataset is None or geometry is None:
                raise ValueError("pass `dataset` and `geometry` when loading from the release directory")
            cfg = json.load(open(os.path.join(path, "config.json")))
            if dataset not in cfg["datasets"]:
                raise ValueError(f"unknown dataset '{dataset}'; available: {sorted(cfg['datasets'])}")
            ent = cfg["datasets"][dataset]
            if geometry not in ent["coefficients"]:
                raise ValueError(f"geometry must be one of {sorted(ent['coefficients'])}, got '{geometry}'")
            gnyq = ent["operator"]["gnyq"]
            path = os.path.join(path, ent["coefficients"][geometry])
        rec = json.load(open(path))
        if rec.get("format") != FORMAT:
            raise ValueError(f"{path}: not a {FORMAT} file")
        if geometry is not None and rec["geometry"] != geometry:
            raise ValueError(f"{path} holds geometry '{rec['geometry']}', asked for '{geometry}'")
        op = rec["operator"]
        name = dataset or rec.get("dataset")
        meta = dict(source_file=os.path.basename(path), dataset=name, coefficients_from=rec.get("dataset"),
                    data_range=DATA_RANGE.get(name))
        return cls(rec["gains"], gnyq or op["gnyq"], rec["geometry"], op["ratio"], op["offset"], dtype, device, meta)

    def coefficients(self, **extra) -> dict:
        """Serializable record of this estimator (the released file format)."""
        geo = "PAN" if self.geometry == "pan" else "MS (PAN quantities shifted by the input-estimated translation tau)"
        return dict(format=FORMAT, geometry=self.geometry, base=GEOMETRIES[self.geometry], forms=list(FORMS),
                    detail_geometry=geo, bands=["low (nu < 1)", "mid (1 < nu < 2)", "high (nu > 2)"],
                    gains_index="gains[b][form][band]",
                    gains=self.gains.tolist(), operator=self.obs.spec(), partition=self.forms.bands.spec(),
                    projection=dict(tol=1e-12, maxiter=200, stop="max over the batch of ||res||^2/||res_0||^2 < tol"),
                    **extra)

    def save(self, path, **extra) -> None:
        with open(path, "w") as fh:
            json.dump(self.coefficients(**extra), fh, indent=1)

    # ------------------------------------------------------------------ the base
    @torch.no_grad()
    def __call__(self, ms, pan, lms, *, max_value: float = 1.0, clip: bool = True, batch_size: int | None = None,
                 check_range: bool = True, tau=None, return_info: bool = False):
        """Compute s.

        Parameters
        ----------
        ms : (N, C, h, w) or (C, h, w)
            Low-resolution MS (PanCollection ``ms``), h = H / ratio.
        pan : (N, 1, H, W), (1, H, W) or (H, W)
            PAN (PanCollection ``pan``).
        lms : (N, C, H, W) or (C, H, W)
            MS interpolated to the PAN grid (PanCollection ``lms``, 23-tap interpolation); the base adds calibrated
            PAN detail to it and projects the result.
        max_value : float
            Upper clipping bound in the units of the inputs: 1.0 for PanCollection arrays divided by 2047 (WV3, QB,
            WV2) or 1023 (GF2); 2047 / 1023 for raw h5 values.
        clip : bool
            Clip to [0, max_value] (the paper's bases are clipped).  Without clipping, ``D s = ms`` holds to the solver
            tolerance.
        batch_size : int or None
            Process this many images at a time (None: all at once).  The paper's bases used 8 images for the 256 x 256
            reduced-resolution tests and 2 for the 512 x 512 full-resolution tests; s_ms (and, at round-off level,
            s_pan) depends slightly on the batch composition, see :mod:`panbase.align`.
        check_range : bool
            Refuse inputs whose MS maximum exceeds 1.5 * max_value (DN passed with max_value = 1).
        tau : (N, 2) array or None
            s_ms only: PAN -> MS translation per image in PAN pixels, (y, x), to use instead of the input-only estimate.
        return_info : bool
            Also return ``dict(tau=..., consistency=...)``: the PAN->MS translation per image (s_ms only, PAN pixels,
            (y, x)) and ||D s - ms|| / ||ms|| before clipping.

        Returns
        -------
        s with the shape and container of `lms` (numpy array, or torch tensor on the device of `lms`), in the compute
        dtype of the estimator (float32 by default).
        """
        was_numpy = not torch.is_tensor(lms)
        out_device = torch.device("cpu") if was_numpy else lms.device
        single = (lms.ndim if was_numpy else lms.dim()) == 3
        L = _as_tensor(lms, "lms", 4)
        m = _as_tensor(ms, "ms", 4)
        P = pan
        P = torch.as_tensor(np.asarray(P)) if not torch.is_tensor(P) else P
        if P.dim() == 2:
            P = P[None, None]
        elif P.dim() == 3:
            P = P[None] if single else P[:, None]
        C = self.gains.shape[0]
        if L.shape[1] != C or m.shape[1] != C:
            raise ValueError(f"these coefficients are for {C} MS bands; got lms {tuple(L.shape)}, ms {tuple(m.shape)}")
        if P.shape[1] != 1 or P.shape[-2:] != L.shape[-2:] or P.shape[0] != L.shape[0] or m.shape[0] != L.shape[0]:
            raise ValueError(f"inconsistent shapes: ms {tuple(m.shape)}, pan {tuple(P.shape)}, lms {tuple(L.shape)}")
        r = self.obs.ratio
        if L.shape[-2] % r or L.shape[-1] % r or m.shape[-2] * r != L.shape[-2] or m.shape[-1] * r != L.shape[-1]:
            raise ValueError(f"PAN-grid size {tuple(L.shape[-2:])} must be {r} x the MS size {tuple(m.shape[-2:])}")
        if check_range and float(m.max()) > 1.5 * float(max_value):
            raise ValueError(f"ms reaches {float(m.max()):.4g} > 1.5 * max_value = {1.5 * max_value:.4g}: pass max_value "
                             "in the units of the inputs (e.g. 2047 for raw WV3/QB/WV2 h5 values, 1023 for GF2)")
        dev = self.device
        bs = L.shape[0] if not batch_size else int(batch_size)
        outs, taus, cons = [], [], []
        g = self.gains.to(device=dev, dtype=self.dtype)
        for i in range(0, L.shape[0], bs):
            Lb, mb, Pb = (t[i:i + bs].to(device=dev, dtype=self.dtype) for t in (L, m, P))
            tb = None if tau is None else torch.as_tensor(np.asarray(tau) if not torch.is_tensor(tau) else tau)[i:i + bs]
            regs, tb = self.forms.regressors(Lb, mb, Pb, tb)
            u = Lb.clone()
            for f, reg in enumerate(regs):
                parts = self.forms.bands.split(reg)                                 # (n, C or 1, 3, H, W)
                u = u + (g[None, :, f, :, None, None] * parts).sum(2)
            s = self.obs.project(u, mb)
            if return_info:
                cons.append(self.obs.consistency(s, mb).cpu())
                taus.append(None if tb is None else tb.cpu())
            outs.append((s.clamp(0.0, float(max_value)) if clip else s).cpu())
        s = torch.cat(outs)
        if not was_numpy:
            s = s.to(out_device)
        if single:
            s = s[0]
        if was_numpy:
            s = s.numpy()
        if not return_info:
            return s
        info = dict(tau=None if self.geometry == "pan" else torch.cat(taus), consistency=torch.cat(cons))
        return s, info

    def __repr__(self) -> str:
        return (f"PanBase({GEOMETRIES[self.geometry]}, C={self.gains.shape[0]}, gnyq={self.obs.gnyq}, ratio={self.obs.ratio}, "
                f"dtype={self.dtype}, device={self.device})")


def pancollection_base(gains, dataset: str, geometry: str = "pan", **kw) -> PanBase:
    """PanBase with the PanCollection operator of `dataset` ('wv3', 'qb', 'gf2' or 'wv2')."""
    return PanBase(gains, GNYQ[dataset], geometry, 4, 2, **kw)


__all__ = ["PanBase", "pancollection_base", "DATA_RANGE", "FORMAT"]
