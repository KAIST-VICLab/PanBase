# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# Adapted from the HS-Pan-Toolbox of M. Ciotola, G. Guarino, G. Vivone, G. Poggi, J. Chanussot, A. Plaza and
# G. Scarpa, "Hyperspectral Pansharpening: Critical Review, Tools and Future Perspectives", IEEE GRSM, 2024,
# https://github.com/matciotola/hyperspectral_pansharpening_toolbox (commit 679c33ae254a72850589ac028bbbe723c3b6f499),
# files CS/BDSD.py and (normalize / denormalize) Utils/dl_tools.py.
# Copyright (c) 2024 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# Licensed under CC BY-NC-SA 4.0 (see LICENSE.md in the folder above). Upstream README: "This software should be used,
# reproduced and modified only for informational and nonprofit purposes." This file is NOT covered by the Apache-2.0
# licence of the rest of the PanBase repository.
#
# Modifications for the PanBase release (2026):
#   * the MS and PAN MTF filters take the sensor Nyquist gains `ordered_dict.ms_gnyq` / `ordered_dict.pan_gnyq`
#     (upstream: `mtf(gt, sensor, ratio)` / `mtf_pan(pan, sensor, ratio)` with gains fixed at 0.3 / 0.15);
#   * `imresize` comes from panbase.classical._resample (Apache-2.0, written for this package, bit-identical to the
#     toolbox's Utils/imresize_bicubic.py, which is a copy of a third-party file without a licence and is not
#     redistributed here).
#   * cvxpy is imported lazily inside `lsqlin`; the tqdm progress bar was dropped; normalize / denormalize are copied
#     here from Utils/dl_tools.py (whose other imports are not needed).
#   The constrained least squares is solved by cvxpy with the PIQP solver, exactly as upstream; results therefore depend
#   (at solver-tolerance level) on the installed cvxpy / piqp versions.
#
# BDSD-PC: band-dependent spatial detail with physical constraints (Vivone, IEEE TIP 2019).
import torch
from torch.nn.functional import interpolate

from ...._resample import imresize
from ..Utils.spectral_tools import mtf, mtf_pan


def normalize(tensor):
    return tensor / (2 ** 16)


def denormalize(tensor):
    return tensor * (2 ** 16)


def lsqlin(C, d, A, b):
    import cvxpy as cp

    A = A.squeeze(0).detach().cpu().numpy()
    b = b.squeeze().detach().cpu().numpy()
    C = C.squeeze(0).detach().cpu().numpy()
    d = d.squeeze().detach().cpu().numpy()

    x = cp.Variable(C.shape[1])

    constraints = [A @ x <= b]
    objective = cp.Minimize(cp.sum_squares(C @ x - d))
    prob = cp.Problem(objective, constraints)

    _ = prob.solve(solver='PIQP', verbose=False, warm_start=False)

    x = torch.tensor(x.value)

    return x


def BDSD_PC(ordered_dict):

    ms, pan, ratio = ordered_dict.ms, ordered_dict.pan, ordered_dict.ratio

    ms = normalize(ms)
    pan = normalize(pan)

    gt = imresize(ms, scale=1/ratio)
    ms_lr = mtf(gt, ordered_dict.ms_gnyq, ratio)
    pan_lp = mtf_pan(pan, ordered_dict.pan_gnyq, ratio)
    pan_lr = interpolate(pan_lp, scale_factor=1/ratio, mode='nearest-exact')

    fused = []
    for i in range(ms.shape[1]):
        h1 = gt[:, i:i+1, :, :].transpose(2, 3).flatten(2).transpose(1, 2)
        h2 = ms_lr[:, i:i+1, :, :].transpose(2, 3).flatten(2).transpose(1, 2)
        h = torch.cat([pan_lr, ms_lr], dim=1).transpose(2, 3).flatten(2).transpose(1, 2)
        A = torch.eye(h.shape[2], dtype=h.dtype, device=h.device)[None, :, :].repeat(h.shape[0], 1, 1)
        A[:, 0, 0] = -1
        b = torch.zeros((1, h.shape[2], 1), dtype=h.dtype, device=h.device).repeat(h.shape[0], 1, 1)

        gamma = lsqlin(h, h1 - h2, A, b)
        fused.append(ms[:, i:i+1, :, :] +
                     torch.reshape(torch.cat([pan, ms], dim=1).transpose(2, 3).flatten(2).transpose(1, 2)
                     @
                     gamma, (ms.shape[0], 1, ms.shape[3], ms.shape[2])).transpose(2, 3))

    fused = torch.cat(fused, dim=1)

    fused = denormalize(fused)

    return fused
