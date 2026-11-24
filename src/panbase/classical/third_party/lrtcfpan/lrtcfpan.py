# SPDX-License-Identifier: GPL-3.0-only
#
# Port of demo_LRTCFPan.m of the official LRTCFPan code (https://github.com/zhongchengwu/code_LRTCFPan, commit
# ddaaa607da43a104e60f8198a022193be5c89436), GNU GPL v3 (see LICENSE next to this file).  NOT covered by the Apache-2.0
# licence of the rest of the PanBase repository.
"""LRTCFPan -- low-rank tensor completion based variational pansharpening (VO).

Method : Z.-C. Wu, T.-Z. Huang, L.-J. Deng, J. Huang, J. Chanussot, G. Vivone, "LRTCFPan: Low-rank tensor completion based
         framework for pansharpening", IEEE TIP 32:1640-1655, 2023.
Code   : official MATLAB code of the first author, https://github.com/zhongchengwu/code_LRTCFPan (commit ddaaa60; the
         algorithm files are the 2023-02-23 upload).  core.py ports lib/inc_LRTCFPan/ one function per .m file.
fuse() = demo_LRTCFPan.m:
             paras = parasetting(sensor, type, I_lrms, I_pan); paras.ratio = 4; paras.sensor = sensor;
             I_fused = inc_LRTCFPan(I_lrms, I_pan, paras)
         with I_lrms = LRMS, I_pan = PAN, both divided by the PanCollection divisor (2047; GF2 1023) -- the demo requires
         data in [0, 1] (X_solver clips X to [0, 1]); the demo's own GF-2 data are exactly GT/1023.  Output x divisor.
         The interpolated `ms` is not used: the official code builds its own interp23tap(I_lrms) start.

Parameters (all from the official parasetting.m; the paper's hyper-parameter table, R / F rows, is identical):
  dataset  sensor key (parasetting)  MTF sensor (MTF_lrms/MTF_pan/FFT_kernel)   MS GNyq                     PAN GNyq
  wv3      'WV3'                     'WV3'                                      .325 .355 .36 .35 .365 .36 .335 .315  .14
  qb       'QB'                      'QB'                                       .34 .32 .30 .22             .15
  gf2      'none' (Guangzhou, GF-2)  'none'                                     .3 x 4                      .15
  wv2      'WV3'  (DEVIATION 1)      'WV2'                                      .35 x 7, .27                .11
  The MTF gains coincide with panbase.classical.sensors (= the CS/MRA rows) for every dataset (asserted below).
  Reduced-resolution images use the 'R' set, full-resolution images the 'F' set (the official code and the paper tune
  them separately):
    R: WV3 l=(.14, 56, 1e-4) eta=(1e-4, 4.2, 8.3)  QB l=(.11, 65, 1e-4) eta=(1e-4, 1.1, 8.7)  GF2 l=(.05, 18, 1e-4) eta=(1e-4, 8.1, 1.8);
       lower 0, block 8x8
    F: WV3 l=(1.1, 36, 1e-4) eta=(1e-4, 6.2, 3.3) lower .6   QB l=(.4, 75, .1) eta=(1e-3, 2.1, 6.7) lower .8
       GF2 l=(1, 50, 1e-4) eta=(1e-4, 2.1, 4.7) lower .4;  block 10x10 -> 8x8 (DEVIATION 2)
  tol 2e-5, maxit 200, ratio 4 (parasetting.m / demo).
  The caller passes kind='reduced' (-> 'R') or kind='full' (-> 'F'); nothing is inferred from the image size.

Deviations from the official code (only these two):
  1. WV2: parasetting.m has no WV2 entry (only 'none'/GF-2, 'QB', 'WV3' -> MATLAB would stop with undefined paras.eta).
     WV2 uses the 8-band WV3 parameter set; its MTF uses the official 'WV2' gains (identical to the CS/MRA rows).
  2. FR block size: the 'F' sets use 10x10 blocks, tuned on the paper's 400x400 FR images (LRMS 100x100).  PanCollection FR
     LRMS are 128x128; 128/10 is not an integer and the official Compute_inj_comps.m fails at zeros(Block_index).  We use
     8x8, the other block size the paper names ("the blocksize can be selected from {8x8, 10x10}", Sec. V-C) and the only
     one of the two that tiles 128.  RR (64x64 LRMS, 8x8) runs unchanged.
Verification (PanBase project, 2026-10-14): agrees with the official MATLAB code run in GNU Octave to ~1e-10 DN.
Runtime: about 0.5-1 min per 256x256 reduced-resolution image and 2-4 min per 512x512 full-resolution image on one CPU
core (200 iterations at most).
"""
import numpy as np
import torch

from . import core

FAMILY = "VO"
REF = "Wu et al., IEEE TIP 2023"

DIV = {"wv3": 2047.0, "qb": 2047.0, "gf2": 1023.0, "wv2": 2047.0}
PARAM_KEY = {"wv3": "WV3", "qb": "QB", "gf2": "none", "wv2": "WV3"}     # parasetting.m switch (wv2: deviation 1)
MTF_SENSOR = {"wv3": "WV3", "qb": "QB", "gf2": "none", "wv2": "WV2"}    # MTF_lrms / MTF_pan / FFT_kernel switch
FR_BLOCK = (8, 8)                                                        # deviation 2 (only if the official size does not tile)
VERBOSE = False

# The table's sensor MTF gains (panbase.classical.sensors) -- must equal the official tables used here.
_MS_GNYQ = {"wv3": [0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315], "qb": [0.34, 0.32, 0.30, 0.22],
            "gf2": [0.3] * 4, "wv2": [0.35] * 7 + [0.27]}
_PAN_GNYQ = {"wv3": 0.14, "qb": 0.15, "wv2": 0.11, "gf2": 0.15}
for _ds in DIV:
    _nb = len(_MS_GNYQ[_ds])
    assert np.array_equal(core.gnyq_ms(MTF_SENSOR[_ds], _nb), np.array(_MS_GNYQ[_ds])), _ds
    assert core.gnyq_pan(MTF_SENSOR[_ds]) == _PAN_GNYQ[_ds], _ds


def paras_for(ds, kind, I_lrms, I_pan):
    """demo_LRTCFPan.m's parameter block for our dataset / resolution (plus the two documented deviations)."""
    typ = {"reduced": "R", "full": "F"}[kind]
    paras = core.parasetting(PARAM_KEY[ds], typ, I_lrms, I_pan)
    paras["ratio"] = 4
    paras["sensor"] = MTF_SENSOR[ds]
    h, w = I_lrms.shape[:2]
    bs = paras["block_size"]
    if h % bs[0] or w % bs[1]:
        assert typ == "F" and h % FR_BLOCK[0] == 0 and w % FR_BLOCK[1] == 0, (ds, kind, h, w, bs)
        paras["block_size"] = FR_BLOCK
        paras["block_size_official"] = bs
    return paras


def fuse(ms, ms_lr, pan, ds, kind, info=None):
    """(1, C, H, W) float64 DN.  kind = 'reduced' (official 'R' parameters) or 'full' ('F' parameters)."""
    if kind not in ("reduced", "full"):
        raise ValueError("LRTCFPan needs kind='reduced' or kind='full' (RR uses the official 'R' parameters, FR the 'F' ones)")
    div = DIV[ds]
    I_lrms = ms_lr[0].permute(1, 2, 0).double().cpu().numpy() / div      # h x w x C, [0, 1]
    I_pan = pan[0, 0].double().cpu().numpy() / div                        # H x W, [0, 1]
    paras = paras_for(ds, kind, I_lrms, I_pan)
    inf = {} if info is None else info
    X = core.inc_LRTCFPan(I_lrms, I_pan, paras, info=inf)
    inf.update(paras={k: v for k, v in paras.items() if k not in ("sz", "Nways")})
    if VERBOSE:
        print(f"  [LRTCFPan {ds} {kind[0].upper()}] it={inf['iterations']} relchg={inf['rel_err']:.2e} "
              f"block={paras['block_size'][0]}", flush=True)
    return torch.from_numpy(np.ascontiguousarray(X.transpose(2, 0, 1) * div))[None]
