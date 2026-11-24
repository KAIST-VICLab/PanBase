# SPDX-License-Identifier: Apache-2.0
"""Reader for the PanCollection test sets (HDF5).

PanCollection (https://github.com/liangjiandeng/PanCollection) ships one HDF5 file per sensor and resolution with the
datasets (N images, channel-first, digital numbers stored as float64):

    reduced resolution (RR):  test_<ds>_multiExm1.h5            gt (N,C,256,256), lms (N,C,256,256), ms (N,C,64,64),
                                                                pan (N,1,256,256)
    full resolution (FR):     test_<ds>_OrigScale_multiExm1.h5  lms (N,C,512,512), ms (N,C,128,128), pan (N,1,512,512)

``lms`` is ``ms`` interpolated to the PAN grid (23-tap polynomial kernel), ``gt`` the RR reference.  The loader returns
float32 arrays normalised as ``np.asarray(h5[key], dtype=np.float32) / divisor`` -- exactly the recipe used for the
PanBase tables (the methods then receive ``(x * divisor).double()``).
"""
import glob
import os
from typing import Dict, Iterable, Optional

import numpy as np

from .sensors import get_dataset

FILES = {"reduced": ("reduced_examples_h5", "test_{p}_multiExm1.h5"),
         "full": ("full_examples_h5", "test_{p}_OrigScale_multiExm1.h5")}
SPLITS = tuple(FILES)


def find_test_file(data_root: str, dataset: str, split: str = "reduced") -> str:
    """Path of the PanCollection test file of ``dataset`` / ``split`` under ``data_root``.

    ``data_root`` may be the .h5 file itself, or a directory.  Tried in order: ``<root>/<WV3|QB|GF2|WV2>/<sub>/<file>``,
    ``<root>/<FOLDER>/<file>``, ``<root>/<file>``, then a recursive search for ``<file>`` (``test_wv3_multiExm1.h5``,
    ``test_wv3_OrigScale_multiExm1.h5``, ...).
    """
    spec = get_dataset(dataset)
    if split not in FILES:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if data_root is None:
        raise ValueError("data_root is not set (pass --data-root or set PANCOLLECTION_ROOT)")
    if os.path.isfile(data_root):
        return data_root
    sub, pat = FILES[split]
    fname = pat.format(p=spec.name)
    for cand in (os.path.join(data_root, spec.folder, sub, fname), os.path.join(data_root, spec.folder, fname),
                 os.path.join(data_root, fname)):
        if os.path.isfile(cand):
            return cand
    hits = sorted(glob.glob(os.path.join(data_root, "**", fname), recursive=True))
    if len(hits) > 1:
        hits = [h for h in hits if f"{os.sep}{spec.folder}{os.sep}" in h] or hits
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise FileNotFoundError(f"{fname} not found under {data_root}")
    raise FileNotFoundError(f"{fname} is ambiguous under {data_root}: {hits}")


def load_test_set(data_root: str, dataset: str, split: str = "reduced",
                  indices: Optional[Iterable[int]] = None) -> Dict[str, np.ndarray]:
    """Load a PanCollection test set, normalised to [0, 1].

    Returns a dict with float32 arrays ``lms``, ``ms``, ``pan`` (and ``gt`` for the reduced-resolution split), plus
    ``indices`` (int array of the image numbers) and ``path``.  ``indices`` selects a subset of the images.
    """
    import h5py

    spec = get_dataset(dataset)
    path = find_test_file(data_root, dataset, split)
    out: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        n = f["pan"].shape[0]
        idx = np.arange(n) if indices is None else np.asarray(sorted({int(i) for i in indices}), dtype=np.int64)
        if idx.size and (idx.min() < 0 or idx.max() >= n):
            raise IndexError(f"image indices {idx.tolist()} out of range 0..{n - 1}")
        for k in ("gt", "lms", "ms", "pan"):
            if k in f:
                a = f[k][()] if indices is None else f[k][idx.tolist()]
                out[k] = np.asarray(a, dtype=np.float32) / spec.divisor
    if out["ms"].shape[1] != spec.bands:
        raise ValueError(f"{path}: {out['ms'].shape[1]} bands, expected {spec.bands} for {dataset}")
    out["indices"] = idx
    out["path"] = path
    return out
