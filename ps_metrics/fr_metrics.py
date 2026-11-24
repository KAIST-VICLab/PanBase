"""Full-resolution (no-reference) pansharpening metrics: HQNR, D_lambda, D_s.

**Why this file exists.**  The project reports two surfaces.  RR compares against a ground
truth manufactured by Wald degradation; FR has no ground truth and scores the fused image
against the observations it came from.  Until now FR existed only in the PanCollection
harness (`baselines/lib/pancoll.hqnr`), so the pretraining screening -- which is where
architecture arms are chosen -- could only be read in RR-PSNR.  An arm selected on RR is
selected on how well it fits a synthetic degradation, which is not the claim the paper
makes.  This module is the FR half, and both callers share it so the two surfaces cannot
drift apart.

**One implementation, not two.**  The numbers come from Vivone et al.'s Pansharpening
Toolbox (the field's reference implementation), vendored under
`baselines/repos/AWFLN/Pansharpening_Toolbox_Assessment_Python`.  `pancoll.hqnr` is a thin
unit-converting wrapper around `hqnr_dn` below; nothing else computes D_lambda or D_s.

**Everything is DN.**  D_s convolves with an MTF-shaped kernel and compares block Q indices;
D_lambda compares inter-band Q indices.  Both are ratio-like and would survive a global
scale, but the toolbox's SSIM call inside D_s is *not* scale-free (see `_skimage_shim`), so
feeding normalised tensors would silently change the published quantity.  Callers invert
their own normalisation and hand DN in.
"""
from __future__ import annotations
import os as _os  # 공개본 (make_release): 뿌리 = $PANBASE_ROOT (기본: 이 저장소)
_GB = _os.environ.get("PANBASE_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))

import sys
from pathlib import Path

import numpy as np

#: The vendored toolbox.  `q2n.py` uses package imports and `HQNR.py` flat ones, so both the
#: repository root and the package directory have to be importable.
_TOOLBOX = Path((_GB + "/repos/AWFLN"))
_ROOTS = (_TOOLBOX, _TOOLBOX / "Pansharpening_Toolbox_Assessment_Python")

#: Corpus `sensor_id` -> the toolbox's sensor string.  `genMTF` knows QB, IKONOS,
#: GeoEye1/WV4, WV2 and WV3 by name and falls back to a flat ``GNyq = 0.3`` for anything
#: else.  Most of this corpus is "anything else": there is no published MTF table for
#: Landsat ETM+/OLI or SPOT 6/7 in this toolbox.
#:
#: The fallback is **named, not hidden**.  `toolbox_sensor` returns `"none"` for those
#: sensors, which is the same string the PanCollection table already uses for GF2 -- so the
#: pretraining FR numbers and the GF2 column of the baseline table rest on exactly the same
#: assumption, and a reader who distrusts GNyq=0.3 distrusts both together.  Reporting them
#: as if the MTF were known would be the failure mode; falling back is not.
TOOLBOX_SENSOR = {
    "worldview2": "WV2",
    "worldview3": "WV3",
    "quickbird": "QB",
    "ikonos": "IKONOS",
    "geoeye1": "GeoEye1",
}


def toolbox_sensor(sensor_id: str) -> str:
    """Toolbox sensor string for a corpus `sensor_id`; `"none"` means the GNyq=0.3 fallback."""
    return TOOLBOX_SENSOR.get(str(sensor_id).lower(), "none")


def _np_shim() -> None:
    """The toolbox uses `np.int` and friends, removed in NumPy 1.24.

    Restoring the aliases is preferred to editing the vendored source: the behaviour is
    identical (both resolve to the builtin) and the upstream file stays byte-for-byte what
    the commit record says it is.
    """
    for a, b in (("int", int), ("float", float), ("bool", bool), ("object", object)):
        if not hasattr(np, a):
            setattr(np, a, b)


def _skimage_shim() -> None:
    """`D_s` calls skimage's SSIM without `data_range`.

    The skimage of the era applied `dtype_range[float64] = (-1, 1)`, i.e. **data_range = 2**,
    to float input; modern skimage raises instead.  Every published D_s / HQNR number was
    produced under the old behaviour, so the old value is injected rather than picking a new
    one -- a different choice here would make our table incomparable with the literature.
    """
    import skimage.metrics as _sm
    if getattr(_sm, "_panbase_patched", False):
        return
    _orig = _sm.structural_similarity

    def _ssim(a, b, **kw):
        kw.setdefault("data_range", 2.0)
        return _orig(a, b, **kw)

    _sm.structural_similarity = _ssim
    _sm._panbase_patched = True


# NOTICE (release copy): `D_lambda_K` below is a modified re-implementation of D_lambda_K.py from the
# Python port of the Pansharpening Toolbox, whose notice reads:
#     Copyright (c) 2020 Vivone Gemine.
#     All rights reserved. This work should only be used for nonprofit purposes.
# Modification: the decimation slice is [ratio//2::ratio] (identical to the original at ratio 4).
# The toolbox itself is not redistributed; scripts/fetch_repos.sh fetches it with the AWFLN repository.
def _d_lambda_k_ratio_safe(_MTF, _q2n):
    """`D_lambda_K` with a decimation that is correct for ratio 2 as well as ratio 4.

    The vendored function decimates the MTF-blurred fusion with

        fused_degraded[int(ratio/2) : -1 : int(ratio)]

    The `-1` stop drops the last candidate sample.  At ratio 4 on an `N = 4n` image the last
    candidate is `N-2`, which the stop at `N-1` still admits, so the slice yields `n` and the
    bug is invisible -- and **every published pansharpening number, this project's included,
    is ratio 4**.  At ratio 2 the last candidate is `N-1`, which the stop excludes, so the
    slice yields `n-1` and `q2n` is handed a 63-row image to compare against a 64-row MS.

    `[ratio // 2 :: ratio]` is the intended decimation and is **identical at ratio 4**, which
    `tests/test_fr_metrics.py` pins against the vendored function directly.  This corpus is
    97% ratio 2 on the `dev` surface (the Landsat family), so dropping those rows instead
    would leave 168 of 9,367.

    The upstream file is not edited: it stays byte-for-byte what the commit record says.
    """
    def D_lambda_K(fused, ms, ratio, sensor, S):
        if fused.shape[0] != ratio * ms.shape[0] or fused.shape[1] != ratio * ms.shape[1]:
            raise ValueError(f"fused {fused.shape[:2]} != ratio {ratio} x ms {ms.shape[:2]}")
        g = _MTF(fused, sensor, ratio)
        g = g[int(ratio) // 2::int(ratio), int(ratio) // 2::int(ratio), :]
        return 1.0 - _q2n(ms, g, S, S)[0]
    return D_lambda_K


def _load():
    for r in _ROOTS:
        if str(r) not in sys.path:
            sys.path.insert(0, str(r))
    _np_shim()
    _skimage_shim()
    import HQNR as _HQ
    from MTF import MTF as _MTF
    from q2n import q2n as _q2n
    # `HQNR.py` did `from D_lambda_K import D_lambda_K`, so the name to rebind lives in the
    # HQNR module's namespace, not in D_lambda_K's.
    _HQ.D_lambda_K = _d_lambda_k_ratio_safe(_MTF, _q2n)
    _H = _HQ.HQNR
    import D_s as _Ds
    # `D_s.py` binds `ssim` at import time, before the shim above could patch it.  Rebinding
    # the module attribute is what actually makes the shim take effect inside D_s.
    _Ds.ssim = __import__("skimage.metrics", fromlist=["x"]).structural_similarity
    return _H


def toolbox_hr_shift(sensor_id: str, ratio: float, contract: str | None = None) -> int:
    """HR shift (px) that puts a product on the toolbox's registration, LR pixel i <-> HR r·i + r//2.

    The vendored `D_lambda_K` samples the MTF-filtered fusion at `[r//2::r]`.  A pixel-is-point product (Landsat;
    `panfm_corpus.degradation.GRID_POINT_SENSORS`, RR contract v2) has LR pixel i at HR r·i, so its HR arrays are moved
    by r//2 px (exactly 1 at ratio 2) -- otherwise D_lambda compares each MS pixel with the fusion half an MS pixel
    away.  Pixel-is-area products keep the toolbox's own convention (0): the 0.5-HR-px gap between area centres and
    `[r//2::r]` is common to every published number, so it is not "corrected" here."""
    from panfm_corpus import degradation as dg
    return int(ratio) // 2 if dg.grid_convention(sensor_id, contract) == "point" else 0


def _shift_hr(a: np.ndarray, s: int) -> np.ndarray:
    """a'[..., y, x] = a[..., y − s, x − s] with edge replication (integer s >= 0)."""
    if s == 0:
        return a
    a = np.concatenate([np.repeat(a[..., :1, :], s, axis=-2), a[..., :-s, :]], axis=-2)
    return np.concatenate([np.repeat(a[..., :, :1], s, axis=-1), a[..., :, :-s]], axis=-1)


def hqnr_dn(fused_dn, ms_dn, lms_dn, pan_dn, sensor: str, ratio: int = 4, S: int = 32, hr_shift: int = 0):
    """HQNR, D_lambda, D_s for one batch, all arrays in DN.

    Shapes, channel-first, batch-first::

        fused_dn (B, C, H, W)    the fused estimate at PAN resolution
        ms_dn    (B, C, h, w)    the observed MS, h = H // ratio
        lms_dn   (B, C, H, W)    `ms_dn` upsampled to PAN resolution
        pan_dn   (B, 1, H, W)    the observed PAN

    `sensor` is a toolbox sensor string (see `toolbox_sensor`), `S` the block size.  `hr_shift`
    (`toolbox_hr_shift`; 0 = historical behaviour) moves fused, lms and pan onto the toolbox registration;
    `lms_dn` must then be the MS upsampled on the product's own grid (`pan_align.upsample_grid`).
    Returns three float64 arrays of length B.  `HQNR = (1 - D_lambda) * (1 - D_s)`.
    """
    _H = _load()
    f = np.asarray(fused_dn, dtype=np.float64)
    m = np.asarray(ms_dn, dtype=np.float64)
    l = np.asarray(lms_dn, dtype=np.float64)
    p = np.asarray(pan_dn, dtype=np.float64)
    if hr_shift:
        f, l, p = _shift_hr(f, int(hr_shift)), _shift_hr(l, int(hr_shift)), _shift_hr(p, int(hr_shift))
    if not (f.ndim == m.ndim == l.ndim == 4 and p.ndim == 4):
        raise ValueError(f"expected 4-D (B,C,H,W); got {f.shape} {m.shape} {l.shape} {p.shape}")
    if f.shape != l.shape:
        raise ValueError(f"fused {f.shape} and lms {l.shape} must match at PAN resolution")
    if f.shape[-2:] != p.shape[-2:]:
        raise ValueError(f"fused {f.shape[-2:]} and pan {p.shape[-2:]} must match")
    if m.shape[-2] * ratio != f.shape[-2] or m.shape[-1] * ratio != f.shape[-1]:
        raise ValueError(f"ms {m.shape[-2:]} * {ratio} != fused {f.shape[-2:]}")
    # The toolbox wants HWC, and `interp23` inside it assumes a single-band PAN is 2-D.
    # `m[i]` is copied, not passed as a view.  The vendored `q2n` normalises its first
    # argument **in place** (`onions_quality.py:31` writes back into a slice of it), and
    # `HQNR` hands the same `ms` to `D_s` afterwards.  On this project's data the effect
    # measures exactly 0.000e+00 -- q2n rebuilds the array whenever its block grid needs
    # padding, which is the case here -- so this copy changes no recorded number.  It is
    # here because "zero on the shapes we happen to use" is not a property anyone should
    # have to re-derive before adding a sensor with a different tile size.
    out = [_H(f[i].transpose(1, 2, 0), np.array(m[i].transpose(1, 2, 0)),
              l[i].transpose(1, 2, 0), p[i, 0], S, sensor, ratio)
           for i in range(f.shape[0])]
    pick = lambda j: np.asarray([o[j] for o in out], dtype=np.float64)
    return pick(0), pick(1), pick(2)
