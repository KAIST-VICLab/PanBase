# SPDX-License-Identifier: Apache-2.0
"""PanBase: network-free, calibrated and observation-consistent base estimates for pansharpening.

    >>> from panbase import PanBase
    >>> pb = PanBase.from_pretrained("path/to/panbase", dataset="wv3", geometry="pan")    # s_pan
    >>> s = pb(ms, pan, lms, max_value=1.0)       # PanCollection arrays / 2047, shapes (N,8,64,64), (N,1,256,256), (N,8,256,256)

Modules: :mod:`panbase.base` (the estimator), :mod:`panbase.projection` (observation operator D and projection Pi),
:mod:`panbase.details` (detail forms and FFT bands), :mod:`panbase.align` (PAN -> MS translation for s_ms),
:mod:`panbase.mtf` (MTF kernels), :mod:`panbase.fit` (calibrating the coefficients on training data).
"""
from .base import DATA_RANGE, PanBase, pancollection_base
from .mtf import GNYQ, mtf_kernel, mtf_kernels
from .projection import MTFObservation, pancollection_operator

__version__ = "1.0.0"
__all__ = ["PanBase", "pancollection_base", "MTFObservation", "pancollection_operator", "mtf_kernel", "mtf_kernels",
           "GNYQ", "DATA_RANGE", "__version__"]
