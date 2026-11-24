# SPDX-License-Identifier: Apache-2.0
"""Uniform entry point for the twelve classical pansharpening methods of the PanBase paper (Table 2).

    hrms = run(method, lms, pan, ms, dataset, resolution=None)

Inputs (PanCollection conventions, one image or a batch; torch tensors or NumPy arrays):
    lms      (C, H, W) or (B, C, H, W)   MS interpolated to the PAN grid (PanCollection ``lms``, 23-tap kernel)
    pan      (H, W), (1, H, W), (B, H, W) or (B, 1, H, W)
    ms       (C, h, w) or (B, C, h, w)   low-resolution MS (PanCollection ``ms``), h = H / 4
    dataset  'wv3' | 'qb' | 'gf2' | 'wv2'  -- selects the normalisation divisor, the sensor MTF gains and the
             per-sensor parameters (see panbase.classical.sensors)
    resolution  'reduced' or 'full'; only LRTCFPan uses it (its official parameter sets differ between the two).
All arrays are normalised as in PanCollection: DN / 2047 for wv3, qb and wv2, DN / 1023 for gf2.

Processing (identical to the pipeline that produced the PanBase tables): per image, the inputs are converted back to
digital numbers with ``(x * divisor).double()`` (in the input's dtype, then float64), the method runs on CPU in float64,
and the result is cleaned with ``nan_to_num(nan=0, posinf=divisor, neginf=0)``, clamped to ``[0, divisor]`` and
divided by the divisor.  The output has the type of ``lms`` (torch / NumPy), float32, the same layout as ``lms``.

Methods and the code they run (see third_party/ for origin and licence of each):
    CS   BT-H, BDSD-PC, GSA, PRACS               third_party.hs_pan_toolbox  (HS-Pan-Toolbox, CC BY-NC-SA 4.0)
    MRA  AWLP, MTF-GLP-FS, MTF-GLP-HPM, MF       third_party.hs_pan_toolbox
    VO   TV, FE-HPM, PWMBF                       third_party.vivone_toolbox  (ports of Vivone et al.'s MATLAB toolbox)
         LRTCFPan                                third_party.lrtcfpan        (port of the authors' code, GPL-3.0)
Method names are matched case-insensitively, ignoring '-', '_' and spaces ('mtf_glp_hpm', 'fehpm', ...).
"""
import types
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch

from .sensors import RATIO, get_dataset

#: method -> (family, reference)
METHODS = OrderedDict([
    ("BT-H", ("CS", "Lolli et al., IEEE GRSL 2017")),
    ("BDSD-PC", ("CS", "Vivone, IEEE TIP 2019")),
    ("GSA", ("CS", "Aiazzi et al., IEEE TGRS 2007")),
    ("PRACS", ("CS", "Choi et al., IEEE TGRS 2011")),
    ("AWLP", ("MRA", "Otazu et al., IEEE TGRS 2005")),
    ("MTF-GLP-FS", ("MRA", "Vivone et al., IEEE TIP 2018")),
    ("MTF-GLP-HPM", ("MRA", "Aiazzi et al., PE&RS 2006")),
    ("MF", ("MRA", "Restaino et al., IEEE TIP 2016")),
    ("TV", ("VO", "Palsson et al., IEEE GRSL 2014")),
    ("FE-HPM", ("VO", "Vivone et al., IEEE TGRS 2015")),
    ("PWMBF", ("VO", "Palsson et al., IEEE TGRS 2015")),
    ("LRTCFPan", ("VO", "Wu et al., IEEE TIP 2023")),
])

_TOOLBOX_FN = {"BT-H": "BT_H", "BDSD-PC": "BDSD_PC", "GSA": "GSA", "PRACS": "PRACS", "AWLP": "AWLP",
               "MTF-GLP-FS": "MTF_GLP_FS", "MTF-GLP-HPM": "MTF_GLP_HPM", "MF": "MF"}
_VO_MODULE = {"TV": "tv", "FE-HPM": "fe_hpm", "PWMBF": "pwmbf"}
_TOOLBOX_SENSOR = {"wv3": "WV3", "qb": "QB", "gf2": "none", "wv2": "WV2"}     # informative only (gains are explicit)
_RES = {"reduced": "reduced", "rr": "reduced", "full": "full", "fr": "full"}


def _key(s: str) -> str:
    return "".join(ch for ch in str(s).upper() if ch not in "-_ ")


_ALIASES = {_key(m): m for m in METHODS}


def canonical_name(method: str) -> str:
    """The table name of ``method`` ('mtf_glp_hpm' -> 'MTF-GLP-HPM')."""
    try:
        return _ALIASES[_key(method)]
    except KeyError:
        raise ValueError(f"unknown method {method!r}; available: {', '.join(METHODS)}") from None


def list_methods(family: Optional[str] = None):
    """Method names in table order, optionally restricted to one family ('CS', 'MRA' or 'VO')."""
    return [m for m, (fam, _) in METHODS.items() if family is None or fam == family.upper()]


def fuse_dn(method: str, lms: torch.Tensor, pan: torch.Tensor, ms: torch.Tensor, dataset: str,
            resolution: Optional[str] = None) -> torch.Tensor:
    """Run one method on ONE image in digital numbers, without any output post-processing.

    Same argument order as run(): lms (1, C, H, W) interpolated MS, pan (1, 1, H, W), ms (1, C, H/4, W/4) LRMS, all
    float64 CPU tensors in digital numbers.  Returns the raw fused image (1, C, H, W), float64 DN (it may contain values
    outside the valid range, or non-finite values; run() applies the table's output recipe).
    """
    name = canonical_name(method)
    spec = get_dataset(dataset)
    ds = spec.name
    if name in _TOOLBOX_FN:
        from .third_party import hs_pan_toolbox as tb
        od = types.SimpleNamespace(ms=lms.clone(), ms_lr=ms.clone(), pan=pan.clone(), ratio=RATIO,
                                   sensor=_TOOLBOX_SENSOR[ds], dataset=_TOOLBOX_SENSOR[ds],
                                   ms_gnyq=[float(g) for g in spec.ms_gnyq], pan_gnyq=float(spec.pan_gnyq))
        return getattr(tb, _TOOLBOX_FN[name])(od)
    if name in _VO_MODULE:
        from .third_party import vivone_toolbox as vt
        return getattr(vt, _VO_MODULE[name]).fuse(lms.clone(), ms.clone(), pan.clone(), ds)
    if name == "LRTCFPan":
        from .third_party import lrtcfpan
        if resolution is None or str(resolution).lower() not in _RES:
            raise ValueError("LRTCFPan needs resolution='reduced' or 'full' (its official parameter sets differ)")
        return lrtcfpan.fuse(lms.clone(), ms.clone(), pan.clone(), ds, kind=_RES[str(resolution).lower()])
    raise AssertionError(name)


def _to_tensor(x) -> torch.Tensor:
    return (x if torch.is_tensor(x) else torch.from_numpy(np.ascontiguousarray(x))).detach().cpu()


def run(method: str, lms, pan, ms, dataset: str, resolution: Optional[str] = None, clip: bool = True):
    """Fuse ``lms`` / ``pan`` / ``ms`` (normalised to [0, 1]) with a classical method; see the module docstring.

    Returns the pansharpened image(s), normalised, float32, with the type and layout of ``lms``.  ``clip=False``
    skips the NaN cleaning and the [0, 1] clamp.
    """
    name = canonical_name(method)
    spec = get_dataset(dataset)
    if resolution is not None and str(resolution).lower() not in _RES:
        raise ValueError(f"resolution must be 'reduced' or 'full', got {resolution!r}")
    numpy_out = not torch.is_tensor(lms)
    L, P, M = _to_tensor(lms), _to_tensor(pan), _to_tensor(ms)
    single = L.dim() == 3
    if single:
        L, M = L[None], M[None]
        if P.dim() == 2:
            P = P[None, None]
        elif P.dim() == 3:
            P = P[None]
    elif P.dim() == 3:
        P = P[:, None]
    if not (L.dim() == M.dim() == P.dim() == 4):
        raise ValueError(f"expected lms (B,C,H,W), ms (B,C,h,w), pan (B,1,H,W); got {tuple(L.shape)}, "
                         f"{tuple(M.shape)}, {tuple(P.shape)}")
    B, C, H, W = L.shape
    if C != spec.bands or M.shape[:2] != (B, C) or tuple(P.shape) != (B, 1, H, W) \
            or (M.shape[2] * RATIO, M.shape[3] * RATIO) != (H, W):
        raise ValueError(f"{spec.name}: expected {spec.bands} bands and ratio {RATIO}; got lms {tuple(L.shape)}, "
                         f"ms {tuple(M.shape)}, pan {tuple(P.shape)}")
    div = spec.divisor
    out = []
    with torch.no_grad():
        for i in range(B):
            lms_dn = (L[i:i + 1] * div).double()
            ms_dn = (M[i:i + 1] * div).double()
            pan_dn = (P[i:i + 1] * div).double()
            f = fuse_dn(name, lms_dn, pan_dn, ms_dn, spec.name, resolution).double()
            if tuple(f.shape) != (1, C, H, W):
                raise RuntimeError(f"{name} returned shape {tuple(f.shape)}, expected {(1, C, H, W)}")
            if clip:
                f = torch.nan_to_num(f, nan=0.0, posinf=float(div), neginf=0.0).clamp(0, div)
            out.append((f / div).float()[0])
    y = torch.stack(out)
    if single:
        y = y[0]
    return y.numpy() if numpy_out else y
