# SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""CS and MRA pansharpening methods adapted from the HS-Pan-Toolbox.

Source: M. Ciotola, G. Guarino, G. Vivone, G. Poggi, J. Chanussot, A. Plaza, G. Scarpa, "Hyperspectral
Pansharpening: Critical Review, Tools and Future Perspectives", IEEE Geoscience and Remote Sensing Magazine, 2024,
doi:10.1109/MGRS.2024.3509139; code https://github.com/matciotola/hyperspectral_pansharpening_toolbox, commit
679c33ae254a72850589ac028bbbe723c3b6f499.

Copyright (c) 2024 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International licence; the full text is in
LICENSE.md next to this file.  The upstream README adds: "This software should be used, reproduced and modified only
for informational and nonprofit purposes."  Everything in this folder keeps those terms; it is NOT covered by the
Apache-2.0 licence of the rest of the PanBase repository.  The toolbox's Utils/imresize_bicubic.py (a copy of S. Son's
"bicubic_pytorch", which has no licence) is NOT included: the three methods that resample (BDSD-PC, PRACS, AWLP)
use panbase.classical._resample (Apache-2.0), verified bit-identical to it.

What was changed (per file headers): package-relative imports; the MTF-matched filters take the sensor Nyquist gains as
an argument (the toolbox hard-codes 0.3 for MS and 0.15 for PAN); unused functions removed.  The numerical code of the
methods is otherwise unchanged.

Entry points (all take a namespace with ``ms`` = interpolated MS, ``ms_lr`` = LRMS, ``pan`` (1, 1, H, W), ``ratio``,
``ms_gnyq``, ``pan_gnyq``; float64 tensors in digital numbers; batch of one image):
    CS:  BT_H, BDSD_PC, GSA, PRACS
    MRA: AWLP, MTF_GLP_FS, MTF_GLP_HPM, MF
"""
import importlib

# Entry points are imported on first use, so that an optional dependency (kornia for MF, cvxpy + piqp for BDSD-PC)
# is only needed by the method that uses it.
_ENTRY = {"BT_H": ".CS.Brovey", "BDSD_PC": ".CS.BDSD", "GSA": ".CS.GS", "PRACS": ".CS.PRACS", "AWLP": ".MRA.AWLP",
          "MTF_GLP_FS": ".MRA.GLP", "MTF_GLP_HPM": ".MRA.GLP", "MF": ".MRA.MF"}

__all__ = list(_ENTRY)


def __getattr__(name):
    if name in _ENTRY:
        return getattr(importlib.import_module(_ENTRY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
