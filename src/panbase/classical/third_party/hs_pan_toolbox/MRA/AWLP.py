# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# Adapted from the HS-Pan-Toolbox of M. Ciotola, G. Guarino, G. Vivone, G. Poggi, J. Chanussot, A. Plaza and
# G. Scarpa, "Hyperspectral Pansharpening: Critical Review, Tools and Future Perspectives", IEEE GRSM, 2024,
# https://github.com/matciotola/hyperspectral_pansharpening_toolbox (commit 679c33ae254a72850589ac028bbbe723c3b6f499),
# file MRA/AWLP.py.
# Copyright (c) 2024 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# Licensed under CC BY-NC-SA 4.0 (see LICENSE.md in the folder above). Upstream README: "This software should be used,
# reproduced and modified only for informational and nonprofit purposes." This file is NOT covered by the Apache-2.0
# licence of the rest of the PanBase repository.
#
# Modifications for the PanBase release (2026): package-relative imports;
#   * `imresize` comes from panbase.classical._resample (Apache-2.0, written for this package, bit-identical to the
#     toolbox's Utils/imresize_bicubic.py, which is a copy of a third-party file without a licence and is not
#     redistributed here).
#
# AWLP: additive wavelet luminance proportional (Otazu et al., IEEE TGRS 2005).
import torch

from ..Utils.spectral_tools import LPFilter
from ...._resample import imresize


def AWLP(ordered_dict):
    ms = torch.clone(ordered_dict.ms)
    pan = torch.clone(ordered_dict.pan)
    ratio = ordered_dict.ratio

    mean_low = torch.mean(ms, dim=1, keepdim=True)

    img_intensity = ms / (mean_low + torch.finfo(ms.dtype).eps)

    pan = pan.repeat(1, ms.shape[1], 1, 1)

    pan_lp = imresize(imresize(pan, 1 / ratio), ratio)

    pan = (pan - torch.mean(pan, dim=(2, 3), keepdim=True)) * (
                torch.std(ms, dim=(2, 3), keepdim=True) / torch.std(pan_lp, dim=(2, 3),
                                                                           keepdim=True)) + torch.mean(ms,
                                                                                                       dim=(2, 3),
                                                                                                       keepdim=True)

    pan_lpp = []
    for i in range(pan.shape[1]):
        pan_lpp.append(LPFilter(pan[:, i, None, :, :].type(ms.dtype), ratio))

    pan_lpp = torch.cat(pan_lpp, dim=1)

    details = pan - pan_lpp

    fused = details * img_intensity + ms

    return fused
