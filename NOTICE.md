# Third-party notices

This repository contains **no upstream source files**. `scripts/fetch_repos.sh` retrieves each repository below from its
origin at the pinned commit and applies our patch; the code you obtain is governed by that repository's own terms. The
patches in `patches/` are diffs: they quote a few lines of the original files as context and apply only to a copy you
fetched yourself.

| Fetched as | Repository | Commit | Upstream licence | Our patch |
|---|---|---|---|---|
| PanNet | https://github.com/Zeyu-Zhu/PGCU | `987de8e6ff4fd8248142d1a06f6b6300779e45fe` | MIT | – |
| LAGConv | https://github.com/liangjiandeng/LAGConv | `5d142a26d1df43a7c0324cd93615143651007b6a` | no licence file (all rights reserved by the authors) | – |
| MutInf | https://github.com/manman1995/Mutual-Information-driven-Pan-sharpening | `5915c65f3f24ddfcf5d528d80d7b35b7f2ae9de3` | no licence file (all rights reserved by the authors) | – |
| MDCUN | https://github.com/yggame/MDCUN | `82e81dde20a99b921ede7625207947ccadca64cb` | no licence file (all rights reserved by the authors) | `patches/MDCUN.patch` |
| PanFlowNet | https://github.com/yggame/PanFlowNet | `84afb1a59a82113ded09ea2b863ba1faae8af0eb` | no licence file (all rights reserved by the authors) | `patches/PanFlowNet.patch` |
| MSDDN | https://github.com/alexhe101/MSDDN | `834bfb9bc2e1a4c3b30efb8891c1e5b5326e6c19` | no licence file (all rights reserved by the authors) | `patches/MSDDN.patch` |
| HFIN | https://github.com/JosephTiTan/HFIN | `73a4cbb9309ba02bf2b76b8d0b8a6bef518199da` | no licence file (all rights reserved by the authors) | `patches/HFIN.patch` |
| UTeRM | https://github.com/mtntruong/UTeRM | `b0149012a39ef95f5699dbf7d0d6dad4d84dcd57` | GPL | – |
| BNNPan | https://github.com/coder-JMHou/BNNPan | `0257e9a021222b2d532269e1860ef560fc1fa042` | no licence file (all rights reserved by the authors) | – |
| U-Know-DiffPAN | https://github.com/KAIST-VICLab/U-Know-DiffPAN | `e3f7b9d079d0753ec16af2a44855e44fd3c2307b` | MIT | – |
| PAN-Crafter | https://github.com/KAIST-VICLab/PAN-Crafter | `e12fb5ef09051d19bc21f3fef6080a05cf7b223a` | licence file with non-commercial / research-only terms | – |
| PINO | https://github.com/ez4lionky/PINO | `6793a1380d5d31fc2a50329a71056c422a18a2f7` | no licence file (all rights reserved by the authors) | – |
| PanDiT | https://github.com/para133/PanDIT | `985a4ea1d9b5fc39e897e94030848f1b94f2b8f8` | no licence file (all rights reserved by the authors) | `patches/PanDIT.patch` |
| MMMamba | https://github.com/Gracewangyy/MMMamba | `74c09612903075e7da803719a16fb4a89a7fed98` | no licence file (all rights reserved by the authors) | `patches/MMMamba.patch` |
| TS-Pan | https://github.com/lzm-01/TS-Pan | `e47cf0d09c79ee551adc64b9af18f0c746969676` | Apache-2.0 | `patches/TS-Pan.patch` |
| AWFLN | https://github.com/yotick/AWFLN | `210232c0836f7808adadbbab6b59a1123fc18c55` | no licence file (all rights reserved by the authors) | – |

## Derived code in our files

- `ps_metrics/fr_metrics.py`: `D_lambda_K` is a modified re-implementation of `D_lambda_K.py` from the Python port of the
  Pansharpening Toolbox (Copyright (c) 2020 Vivone Gemine. All rights reserved. This work should only be used for
  nonprofit purposes). It keeps those terms. The FR metrics call the toolbox itself, which `scripts/fetch_repos.sh`
  fetches with the AWFLN repository; it is not redistributed here.
- `lib/specs/pino.py`: `make_coord` is written from its definition (cell centres of a regular grid); no code of the PINO
  repository is copied.

## Third-party code inside this repository

- `src/panbase/classical/third_party/hs_pan_toolbox/` (14 files): third-party code redistributed under CC BY-NC-SA 4.0 (`LICENSE.md`; HS-Pan-Toolbox); the header of each file names its origin and our changes. Our Apache-2.0 licence does not cover these files.
- `src/panbase/classical/third_party/lrtcfpan/` (3 files): third-party code redistributed under GPL-3.0 (`LICENSE`; code_LRTCFPan); the header of each file names its origin and our changes. Our Apache-2.0 licence does not cover these files.
- `src/panbase/classical/third_party/vivone_assessment/` (14 files): third-party code redistributed under the licence of the Pansharpening Toolbox's Python assessment code (`LICENSE.txt`: nonprofit reproduction and modification, original notices kept, author credited); the header of each file names its origin and our changes. Our Apache-2.0 licence does not cover these files.
- `src/panbase/classical/third_party/vivone_toolbox/` (4 files): third-party code redistributed under the terms of G. Vivone's MATLAB Pansharpening Toolbox, quoted in `__init__.py` ("All rights reserved. This work should only be used for nonprofit purposes."); the header of each file names its origin and our changes. Our Apache-2.0 licence does not cover these files.

In short, the classical methods other than LRTCFPan, and the FR metrics, may be used for non-commercial (nonprofit) purposes only; LRTCFPan is GPL-3.0.
