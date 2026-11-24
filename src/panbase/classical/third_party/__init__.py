# SPDX-License-Identifier: Apache-2.0
"""Third-party code used by panbase.classical, kept apart because it is NOT under the Apache-2.0 licence.

Each sub-folder states its origin and terms in its ``__init__.py`` and file headers:

    hs_pan_toolbox/     CS + MRA methods (BT-H, BDSD-PC, GSA, PRACS, AWLP, MTF-GLP-FS, MTF-GLP-HPM, MF), adapted from the
                        HS-Pan-Toolbox (GRIP-UNINA); CC BY-NC-SA 4.0 (non-commercial, share-alike).
    vivone_toolbox/     TV, FE-HPM, PWMBF: Python ports of G. Vivone et al.'s MATLAB benchmark toolbox ("nonprofit
                        purposes only"; PWMBF also ports the Rice Wavelet Toolbox, BSD-style licence).
    lrtcfpan/           LRTCFPan: NumPy port of the authors' MATLAB code; GNU GPL v3.
    vivone_assessment/  Q2n and HQNR / D_lambda / D_s of G. Vivone's Python assessment toolbox (nonprofit licence;
                        imresize.py is MIT).

In short: the classical methods and the reference metrics may be used for research and other non-commercial purposes
only.  This file itself is Apache-2.0.
"""
