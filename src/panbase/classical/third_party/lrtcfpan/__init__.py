# SPDX-License-Identifier: GPL-3.0-only
"""LRTCFPan (VO), NumPy port of the authors' official MATLAB code.

Source: Z.-C. Wu, T.-Z. Huang, L.-J. Deng, J. Huang, J. Chanussot, G. Vivone, "LRTCFPan: Low-rank tensor completion
based framework for pansharpening", IEEE TIP 32:1640-1655, 2023; code https://github.com/zhongchengwu/code_LRTCFPan
(commit ddaaa607da43a104e60f8198a022193be5c89436), released under the GNU General Public License v3 (LICENSE in this
folder).  The port (core.py: lib/inc_LRTCFPan/*.m; lrtcfpan.py: demo_LRTCFPan.m) is a derivative work under the same
licence.  It is NOT covered by the Apache-2.0 licence of the rest of the PanBase repository; distributing it together
with Apache-2.0 code is allowed (Apache-2.0 is GPLv3-compatible), but the combination is then subject to the GPL.
"""
from .lrtcfpan import fuse

__all__ = ["fuse"]
