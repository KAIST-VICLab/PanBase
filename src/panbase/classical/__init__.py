# SPDX-License-Identifier: Apache-2.0
"""Classical (network-free) pansharpening methods of the PanBase paper, Table 2.

Twelve methods on the PanCollection test sets (WorldView-3, QuickBird, GaoFen-2; WorldView-2 zero-shot):

    CS   BT-H, BDSD-PC, GSA, PRACS
    MRA  AWLP, MTF-GLP-FS, MTF-GLP-HPM, MF
    VO   TV, FE-HPM, PWMBF, LRTCFPan

Usage::

    from panbase.classical import run, load_test_set, rr_metrics
    data = load_test_set("/path/to/PanCollection", "wv3", "reduced")     # float32, DN / 2047
    hrms = run("MTF-GLP-HPM", data["lms"], data["pan"], data["ms"], "wv3", resolution="reduced")
    scores = rr_metrics(hrms, data["gt"], "wv3")                         # psnr, ssim, sam, ergas, q2n per image

``run`` accepts torch tensors or NumPy arrays normalised as in PanCollection (DN / 2047 for wv3, qb, wv2;
DN / 1023 for gf2) and returns the fused image(s) in the same normalisation, float32, clamped to [0, 1].
``resolution`` ('reduced' | 'full') is required by LRTCFPan only.  Everything runs on CPU in float64.

Licensing: the files in this folder are Apache-2.0, but the method implementations and the Q2n / HQNR code live in
``third_party/`` and keep their upstream terms (CC BY-NC-SA 4.0, Vivone's non-commercial terms, GPL-3.0); in practice
the methods may be used for research and other non-commercial purposes only.  See ``third_party/__init__.py``.
"""
from .api import METHODS, canonical_name, fuse_dn, list_methods, run
from .data import find_test_file, load_test_set
from .metrics import fr_metrics, rr_crop, rr_metrics
from .sensors import DATASETS, RATIO, DatasetSpec, get_dataset

__all__ = ["run", "fuse_dn", "METHODS", "list_methods", "canonical_name", "load_test_set", "find_test_file",
           "rr_metrics", "fr_metrics", "rr_crop", "DATASETS", "DatasetSpec", "get_dataset", "RATIO"]
