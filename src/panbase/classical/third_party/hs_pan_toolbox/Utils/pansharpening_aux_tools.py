# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# Adapted from the HS-Pan-Toolbox of M. Ciotola, G. Guarino, G. Vivone, G. Poggi, J. Chanussot, A. Plaza and
# G. Scarpa, "Hyperspectral Pansharpening: Critical Review, Tools and Future Perspectives", IEEE GRSM, 2024,
# https://github.com/matciotola/hyperspectral_pansharpening_toolbox (commit 679c33ae254a72850589ac028bbbe723c3b6f499),
# file Utils/pansharpening_aux_tools.py.
# Copyright (c) 2024 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# Licensed under CC BY-NC-SA 4.0 (see LICENSE.md in the folder above). Upstream README: "This software should be used,
# reproduced and modified only for informational and nonprofit purposes." This file is NOT covered by the Apache-2.0
# licence of the rest of the PanBase repository.
#
# Modifications for the PanBase release (2026): `eps` and `mldivide` (unused here) were removed. The remaining
# functions are unchanged; note that estimation_alpha returns float32 coefficients, as upstream does.
import torch


def estimation_alpha(ms, pan):

    ms_f = torch.flatten(ms, start_dim=2).double()
    pan_f = torch.flatten(pan, start_dim=2).double()

    alpha = torch.linalg.lstsq(ms_f.transpose(1, 2), pan_f.transpose(1, 2)).solution
    return alpha[:, :, :, None].float()


def batch_cov(points):
    B, N, D = points.size()
    mean = points.mean(dim=1).unsqueeze(1)
    diffs = (points - mean).reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    return bcov  # (B, D, D)


def regress(y, X):
    y = y.double()
    X = X.double()

    Q, R = torch.linalg.qr(X)
    b = torch.linalg.lstsq(R, torch.bmm(Q.transpose(1, 2), y)).solution

    return b
