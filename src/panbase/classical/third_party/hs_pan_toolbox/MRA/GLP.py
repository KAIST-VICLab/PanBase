# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# Adapted from the HS-Pan-Toolbox of M. Ciotola, G. Guarino, G. Vivone, G. Poggi, J. Chanussot, A. Plaza and
# G. Scarpa, "Hyperspectral Pansharpening: Critical Review, Tools and Future Perspectives", IEEE GRSM, 2024,
# https://github.com/matciotola/hyperspectral_pansharpening_toolbox (commit 679c33ae254a72850589ac028bbbe723c3b6f499),
# file MRA/GLP.py.
# Copyright (c) 2024 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# Licensed under CC BY-NC-SA 4.0 (see LICENSE.md in the folder above). Upstream README: "This software should be used,
# reproduced and modified only for informational and nonprofit purposes." This file is NOT covered by the Apache-2.0
# licence of the rest of the PanBase repository.
#
# Modifications for the PanBase release (2026): the MTF filter takes the sensor's MS Nyquist gains
# `ordered_dict.ms_gnyq` (upstream: `mtf(bands_hr, sensor, ratio)`, gains fixed at 0.3); MTF_GLP_HPM_R (not part of
# the PanBase tables) was removed.
#
# MTF-GLP-FS: GLP with MTF-matched filter and full-scale regression injection (Vivone et al., IEEE TIP 2018).
# MTF-GLP-HPM: GLP with MTF-matched filter and high-pass modulation injection (Aiazzi et al., PE&RS 2006).
import torch
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode as Inter

from ..Utils.spectral_tools import mtf, LPfilterGauss
from ..Utils.pansharpening_aux_tools import batch_cov
from ..Utils.interpolator_tools import ideal_interpolator


def MTF_GLP_FS(ordered_dict):
    ms = torch.clone(ordered_dict.ms)
    pan = torch.clone(ordered_dict.pan)
    ratio = ordered_dict.ratio

    _, c, h, w = ms.shape

    bands_hr = pan.repeat(1, c, 1, 1)

    bands_hr_lp = mtf(bands_hr, ordered_dict.ms_gnyq, ratio)
    bands_hr_lr_lr = resize(bands_hr_lp, [h // ratio, w // ratio], interpolation=Inter.NEAREST_EXACT)

    bands_hr_lr = ideal_interpolator(bands_hr_lr_lr, ratio)

    low_covs = []
    high_covs = []

    for i in range(ms.shape[1]):
        points_lr = torch.cat([ms[:, i, None, :, :], bands_hr[:, i, None, :, :]], dim=1)
        points_lr = torch.flatten(points_lr, start_dim=2)

        points_hr = torch.cat([bands_hr_lr[:, i, None, :, :], bands_hr[:, i, None, :, :]], dim=1)
        points_hr = torch.flatten(points_hr, start_dim=2)

        low_covs.append(batch_cov(points_lr.transpose(1, 2))[:, None, :, :])
        high_covs.append(batch_cov(points_hr.transpose(1, 2))[:, None, :, :])

    low_covs = torch.cat(low_covs, dim=1)
    high_covs = torch.cat(high_covs, dim=1)

    gamma = low_covs[:, :, 0, 1] / high_covs[:, :, 0, 1]
    gamma = gamma[:, :, None, None]
    fused = ms + gamma * (bands_hr - bands_hr_lr)

    return fused


def MTF_GLP_HPM(ordered_dict):
    ms = torch.clone(ordered_dict.ms)
    pan = torch.clone(ordered_dict.pan)
    ratio = ordered_dict.ratio

    _, c, h, w = ms.shape

    bands_hr = pan.repeat(1, c, 1, 1)

    bands_hr = (bands_hr - torch.mean(bands_hr, dim=(2, 3), keepdim=True)) * (
            torch.std(ms, dim=(2, 3), keepdim=True) / torch.std(LPfilterGauss(bands_hr, ratio), dim=(2, 3),
                                                                       keepdim=True)) + torch.mean(ms,
                                                                                                   dim=(2, 3),
                                                                                                   keepdim=True)

    bands_hr_lp = mtf(bands_hr, ordered_dict.ms_gnyq, ratio)
    bands_hr_lr_lr = resize(bands_hr_lp, [h // ratio, w // ratio], interpolation=Inter.NEAREST_EXACT)
    bands_hr_lr = ideal_interpolator(bands_hr_lr_lr, ratio)

    fused = ms * torch.clip((bands_hr / (bands_hr_lr + torch.finfo(ms.dtype).eps)), 0, 10.0)
    return fused
