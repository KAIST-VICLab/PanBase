# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# Adapted from the HS-Pan-Toolbox of M. Ciotola, G. Guarino, G. Vivone, G. Poggi, J. Chanussot, A. Plaza and
# G. Scarpa, "Hyperspectral Pansharpening: Critical Review, Tools and Future Perspectives", IEEE GRSM, 2024,
# https://github.com/matciotola/hyperspectral_pansharpening_toolbox (commit 679c33ae254a72850589ac028bbbe723c3b6f499),
# file CS/Brovey.py.
# Copyright (c) 2024 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# Licensed under CC BY-NC-SA 4.0 (see LICENSE.md in the folder above). Upstream README: "This software should be used,
# reproduced and modified only for informational and nonprofit purposes." This file is NOT covered by the Apache-2.0
# licence of the rest of the PanBase repository.
#
# Modifications for the PanBase release (2026): package-relative imports only.
#
# BT-H: Brovey transform with haze correction (Lolli et al., IEEE GRSL 2017).
import torch

from ..Utils.spectral_tools import LPfilterGauss
from ..Utils.pansharpening_aux_tools import estimation_alpha


def BT_H(ordered_dict):
    ms = torch.clone(ordered_dict.ms)
    pan = torch.clone(ordered_dict.pan)
    ratio = ordered_dict.ratio

    min_ms = torch.amin(ms, dim=(2, 3), keepdim=True)

    pan_lp = LPfilterGauss(pan, ratio)
    alphas = estimation_alpha(ms, pan_lp)

    img = torch.sum((ms - min_ms) * alphas, dim=1, keepdim=True)

    img_hr = (pan - torch.mean(pan_lp, dim=(2, 3), keepdim=True)) * (
                torch.std(img, dim=(2, 3), keepdim=True) / torch.std(pan_lp, dim=(2, 3),
                                                                     keepdim=True)) + torch.mean(img, dim=(2, 3),
                                                                                                 keepdim=True)

    ms_minus_min = ms - min_ms
    ms_minus_min = torch.clip(ms_minus_min, 0, ms_minus_min.max())

    fused = ms_minus_min * (img_hr / (img + torch.finfo(torch.float64).eps)).repeat(1, ms.shape[1], 1,
                                                                                           1) + min_ms

    return fused
