# SPDX-License-Identifier: LicenseRef-Vivone-PyToolbox
"""Q2n and HQNR (D_lambda, D_s) of G. Vivone's Python pansharpening assessment toolbox, as used for the PanBase tables.

Source: "Pansharpening_Toolbox_Assessment_Python" by Gemine Vivone (Python counterpart of the MATLAB quality indices of
G. Vivone et al., IEEE JSTARS 14:6102-6118, 2021 and IEEE GRSM 9(1):53-81, 2021), as redistributed in
https://github.com/yotick/AWFLN (commit 210232c0836f7808adadbbab6b59a1123fc18c55).  Upstream file names are kept.

Terms (LICENSE.txt in this folder, kept verbatim): reproduction, modification and usage are allowed free of charge
provided that (i) the software is used, reproduced and modified only for informational and nonprofit purposes,
(ii) any reproduction or modification retains all original notices, and (iii) reference to the original author is
given whenever results obtained with it are made public.  This folder is NOT covered by the Apache-2.0 licence of the
rest of the PanBase repository.  imresize.py is fatheral/matlab_imresize (MIT, notice in the file); tools.py adapts
GRIP-UNINA code as stated in its header.

Changes (listed in each file header): package-relative imports, `np.int` -> `int`, `scipy.ndimage.filters.correlate`
-> `scipy.ndimage.correlate`, and an explicit `data_range=2.0` in the two SSIM calls of D_s (the implicit value of the
scikit-image versions with which the published D_s / HQNR numbers were produced).
"""
# No re-exports: the submodules HQNR.py and q2n.py share their names with the functions they define (import them as
# ``from panbase.classical.third_party.vivone_assessment.HQNR import HQNR``); HQNR needs scikit-image, q2n does not.
