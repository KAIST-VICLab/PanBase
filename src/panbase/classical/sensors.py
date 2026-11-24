# SPDX-License-Identifier: Apache-2.0
"""Per-dataset constants of the PanCollection benchmark, as used for the classical rows of the PanBase tables.

All four PanCollection test sets have a PAN / MS scale ratio of 4.  Arrays are normalised as
``x = DN / divisor`` (11-bit WorldView-3, QuickBird and WorldView-2: 2047; 10-bit GaoFen-2: 1023).

MTF-matched filters.  Several methods (BDSD-PC, MTF-GLP-FS, MTF-GLP-HPM, LRTCFPan) model the sensor's modulation
transfer function by a Gaussian whose amplitude at the Nyquist frequency is ``GNyq``:

    dataset  MS GNyq per band                                         PAN GNyq   source
    wv3      0.325 0.355 0.360 0.350 0.365 0.360 0.335 0.315          0.14       Vivone et al. toolbox (WV3)
    qb       0.34 0.32 0.30 0.22                                      0.15       Vivone et al. toolbox (QB)
    gf2      0.3 x 4                                                  0.15       no published table: generic 0.3 / 0.15
    wv2      0.35 x 7, 0.27                                           0.11       Vivone et al. toolbox (WV2)

HQNR sensor.  The full-resolution index D_lambda (Khan's protocol) filters the fused image with the MS MTF of the
toolbox sensor ``hqnr_sensor``.  WV2 is scored with the WV3 gains: the 'WV2' branch of the Python assessment toolbox
is broken for any number of bands, and WV2 / WV3 share the same 8-band layout and ratio.  GF2 uses the toolbox's
generic gains ('none').
"""
from dataclasses import dataclass
from typing import Dict, Tuple

RATIO = 4


@dataclass(frozen=True)
class DatasetSpec:
    """Constants of one PanCollection test set."""

    name: str                      # 'wv3' | 'qb' | 'gf2' | 'wv2'
    sensor: str                    # sensor name
    folder: str                    # PanCollection folder name
    bands: int                     # number of MS bands
    divisor: float                 # DN / divisor -> [0, 1]
    ms_gnyq: Tuple[float, ...]     # MS MTF amplitude at Nyquist, per band
    pan_gnyq: float                # PAN MTF amplitude at Nyquist
    hqnr_sensor: str               # sensor string of the HQNR toolbox (D_lambda MTF)


DATASETS: Dict[str, DatasetSpec] = {
    "wv3": DatasetSpec("wv3", "WorldView-3", "WV3", 8, 2047.0,
                       (0.325, 0.355, 0.360, 0.350, 0.365, 0.360, 0.335, 0.315), 0.14, "WV3"),
    "qb": DatasetSpec("qb", "QuickBird", "QB", 4, 2047.0, (0.34, 0.32, 0.30, 0.22), 0.15, "QB"),
    "gf2": DatasetSpec("gf2", "GaoFen-2", "GF2", 4, 1023.0, (0.3, 0.3, 0.3, 0.3), 0.15, "none"),
    "wv2": DatasetSpec("wv2", "WorldView-2", "WV2", 8, 2047.0, (0.35,) * 7 + (0.27,), 0.11, "WV3"),
}


def get_dataset(name: str) -> DatasetSpec:
    """The DatasetSpec of ``name`` (case-insensitive: 'wv3', 'qb', 'gf2', 'wv2')."""
    key = str(name).lower()
    if key not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {sorted(DATASETS)}")
    return DATASETS[key]
