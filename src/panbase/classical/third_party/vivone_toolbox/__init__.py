# SPDX-License-Identifier: LicenseRef-Vivone-Toolbox-NonProfit
"""VO methods ported to Python from G. Vivone et al.'s MATLAB pansharpening benchmark toolbox.

Source: G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, J. Chanussot,
"A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and
emerging pansharpening methods", IEEE Geoscience and Remote Sensing Magazine 9(1):53-81, 2021,
doi:10.1109/MGRS.2020.3019315; toolbox "Pansharpening Toolbox for Distribution" (openremotesensing.net, November 2020;
zip sha256 b635bf36560ea52e46f3e8401aca7370cf5e70407bcd0ebea4660759c29b7d50).  The PanCollection MATLAB test toolbox
redistributes the same files.

Terms stated by the toolbox (README.txt), which this folder keeps:
    "Copyright (c) 2020. All rights reserved. This work should only be used for nonprofit purposes."
No further licence is granted upstream.  These Python files are line-by-line ports (derivative works) and are NOT
covered by the Apache-2.0 licence of the rest of the PanBase repository.

    tv.py      TV pansharpening (Palsson et al., IEEE GRSL 2014): TV/TV_pansharpen.m; parameters of PanCollection's
               MetricCode/00-build/build_TV.m
    fe_hpm.py  FE-HPM (Vivone et al., IEEE TGRS 2015): FE-HPM/FE_HPM.m, FE.m; Tools/LPfilter.m, ndwt2_working.m,
               indwt2_working.m, estimation_alpha.m, interp23tap.m
    pwmbf.py   PWMBF (Palsson et al., IEEE TGRS 2015): PWMBF/PWMBF.m, compute_PhiTX.m, compute_PhiX.m and the Rice
               Wavelet Toolbox (PWMBF/rwt: bin/daubcqf.m, lib/src/rdwt.c, irdwt.c; BSD-style Rice University licence,
               reproduced in pwmbf.py)

Each module exposes fuse(ms, ms_lr, pan, ds) -> (1, C, H, W) float64 in digital numbers, where ms = interpolated MS
(1, C, H, W), ms_lr = LRMS (1, C, H/4, W/4), pan = (1, 1, H, W), all float64 DN, ds in {'wv3', 'qb', 'gf2', 'wv2'}.
"""
from . import fe_hpm, pwmbf, tv

__all__ = ["tv", "fe_hpm", "pwmbf"]
