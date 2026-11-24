# Model zoo

All 129 released checkpoints (Hugging Face `JeonghyeokDo/PanBase`). RR / FR values are the paper's table values (PanCollection test sets, 20 images each); every checkpoint's test output was re-run and reproduces the RR PSNR (`verification_2026_11_23` in its `config.json`).

| Backbone | Arm | Train | RR PSNR (test) | FR HQNR (test) | WV2 RR PSNR / HQNR (zero-shot) | Weights (Hugging Face) |
|---|---|---|---|---|---|---|
| PanNet | original | WV3 | 34.29 | 0.934 | 28.51 / 0.968 | `backbones/PanNet/original/wv3/` |
| PanNet | original | QB | 33.83 | 0.843 |  | `backbones/PanNet/original/qb/` |
| PanNet | original | GF2 | 36.87 | 0.915 |  | `backbones/PanNet/original/gf2/` |
| PanNet | s_pan | WV3 | 34.96 | 0.967 | 29.10 / 0.958 | `backbones/PanNet/s_pan/wv3/` |
| PanNet | s_pan | QB | 34.20 | 0.857 |  | `backbones/PanNet/s_pan/qb/` |
| PanNet | s_pan | GF2 | 37.98 | 0.927 |  | `backbones/PanNet/s_pan/gf2/` |
| PanNet | s_ms | WV3 | 35.22 | 0.967 | 29.40 / 0.963 | `backbones/PanNet/s_ms/wv3/` |
| PanNet | s_ms | QB | 36.41 | 0.881 |  | `backbones/PanNet/s_ms/qb/` |
| PanNet | s_ms | GF2 | 40.06 | 0.963 |  | `backbones/PanNet/s_ms/gf2/` |
| LAGConv | original | WV3 | 37.22 | 0.944 | 28.71 / 0.949 | `backbones/LAGConv/original/wv3/` |
| LAGConv | original | QB | 37.96 | 0.923 |  | `backbones/LAGConv/original/qb/` |
| LAGConv | original | GF2 | 42.47 | 0.915 |  | `backbones/LAGConv/original/gf2/` |
| LAGConv | s_pan | WV3 | 37.34 | 0.958 | 29.17 / 0.903 | `backbones/LAGConv/s_pan/wv3/` |
| LAGConv | s_pan | QB | 38.11 | 0.911 |  | `backbones/LAGConv/s_pan/qb/` |
| LAGConv | s_pan | GF2 | 42.82 | 0.949 |  | `backbones/LAGConv/s_pan/gf2/` |
| LAGConv | s_ms | WV3 | 37.29 | 0.962 | 29.07 / 0.936 | `backbones/LAGConv/s_ms/wv3/` |
| LAGConv | s_ms | QB | 38.12 | 0.900 |  | `backbones/LAGConv/s_ms/qb/` |
| LAGConv | s_ms | GF2 | 43.02 | 0.969 |  | `backbones/LAGConv/s_ms/gf2/` |
| MutInf | original | WV3 | 35.77 | 0.913 | 28.87 / 0.902 | `backbones/MutInf/original/wv3/` |
| MutInf | original | QB | 36.64 | 0.876 |  | `backbones/MutInf/original/qb/` |
| MutInf | original | GF2 | 36.65 | 0.943 |  | `backbones/MutInf/original/gf2/` |
| MutInf | s_pan | WV3 | 37.05 | 0.967 | 29.01 / 0.940 | `backbones/MutInf/s_pan/wv3/` |
| MutInf | s_pan | QB | 38.33 | 0.905 |  | `backbones/MutInf/s_pan/qb/` |
| MutInf | s_pan | GF2 | 43.59 | 0.973 |  | `backbones/MutInf/s_pan/gf2/` |
| MutInf | s_ms | WV3 | 36.97 | 0.964 | 29.23 / 0.935 | `backbones/MutInf/s_ms/wv3/` |
| MutInf | s_ms | QB | 38.29 | 0.927 |  | `backbones/MutInf/s_ms/qb/` |
| MutInf | s_ms | GF2 | 43.03 | 0.984 |  | `backbones/MutInf/s_ms/gf2/` |
| MDCUN | original | WV3 | 36.95 | 0.961 | 28.96 / 0.938 | `backbones/MDCUN/original/wv3/` |
| MDCUN | original | QB | 35.99 | 0.914 |  | `backbones/MDCUN/original/qb/` |
| MDCUN | original | GF2 | 42.04 | 0.953 |  | `backbones/MDCUN/original/gf2/` |
| MDCUN | s_pan | WV3 | 37.12 | 0.967 | 30.05 / 0.961 | `backbones/MDCUN/s_pan/wv3/` |
| MDCUN | s_pan | QB | 37.66 | 0.943 |  | `backbones/MDCUN/s_pan/qb/` |
| MDCUN | s_pan | GF2 | 42.57 | 0.964 |  | `backbones/MDCUN/s_pan/gf2/` |
| MDCUN | s_ms | WV3 | 36.92 | 0.961 | 28.79 / 0.941 | `backbones/MDCUN/s_ms/wv3/` |
| MDCUN | s_ms | QB | 38.08 | 0.937 |  | `backbones/MDCUN/s_ms/qb/` |
| MDCUN | s_ms | GF2 | 43.15 | 0.985 |  | `backbones/MDCUN/s_ms/gf2/` |
| PanFlowNet | original | WV3 | 35.76 | 0.918 | 27.78 / 0.869 | `backbones/PanFlowNet/original/wv3/` |
| PanFlowNet | original | QB | 36.23 | 0.853 |  | `backbones/PanFlowNet/original/qb/` |
| PanFlowNet | original | GF2 | 38.56 | 0.904 |  | `backbones/PanFlowNet/original/gf2/` |
| PanFlowNet | s_pan | WV3 | 36.53 | 0.954 | 28.19 / 0.936 | `backbones/PanFlowNet/s_pan/wv3/` |
| PanFlowNet | s_pan | QB | 37.87 | 0.831 |  | `backbones/PanFlowNet/s_pan/qb/` |
| PanFlowNet | s_pan | GF2 | 40.12 | 0.928 |  | `backbones/PanFlowNet/s_pan/gf2/` |
| PanFlowNet | s_ms | WV3 | 36.56 | 0.865 | 28.22 / 0.912 | `backbones/PanFlowNet/s_ms/wv3/` |
| PanFlowNet | s_ms | QB | 37.93 | 0.909 |  | `backbones/PanFlowNet/s_ms/qb/` |
| PanFlowNet | s_ms | GF2 | 40.19 | 0.934 |  | `backbones/PanFlowNet/s_ms/gf2/` |
| MSDDN | original | WV3 | 35.62 | 0.956 | 28.65 / 0.931 | `backbones/MSDDN/original/wv3/` |
| MSDDN | original | QB | 36.47 | 0.910 |  | `backbones/MSDDN/original/qb/` |
| MSDDN | original | GF2 | 38.72 | 0.945 |  | `backbones/MSDDN/original/gf2/` |
| MSDDN | s_pan | WV3 | 36.66 | 0.928 | 28.82 / 0.920 | `backbones/MSDDN/s_pan/wv3/` |
| MSDDN | s_pan | QB | 38.11 | 0.941 |  | `backbones/MSDDN/s_pan/qb/` |
| MSDDN | s_pan | GF2 | 41.80 | 0.956 |  | `backbones/MSDDN/s_pan/gf2/` |
| MSDDN | s_ms | WV3 | 36.85 | 0.942 | 28.68 / 0.899 | `backbones/MSDDN/s_ms/wv3/` |
| MSDDN | s_ms | QB | 38.08 | 0.946 |  | `backbones/MSDDN/s_ms/qb/` |
| MSDDN | s_ms | GF2 | 41.86 | 0.971 |  | `backbones/MSDDN/s_ms/gf2/` |
| HFIN | original | WV3 | 36.53 | 0.917 | 28.39 / 0.903 | `backbones/HFIN/original/wv3/` |
| HFIN | original | QB | 35.17 | 0.796 |  | `backbones/HFIN/original/qb/` |
| HFIN | original | GF2 | 38.08 | 0.909 |  | `backbones/HFIN/original/gf2/` |
| HFIN | s_pan | WV3 | 37.26 | 0.965 | 29.47 / 0.952 | `backbones/HFIN/s_pan/wv3/` |
| HFIN | s_pan | QB | 37.42 | 0.924 |  | `backbones/HFIN/s_pan/qb/` |
| HFIN | s_pan | GF2 | 40.95 | 0.920 |  | `backbones/HFIN/s_pan/gf2/` |
| HFIN | s_ms | WV3 | 37.40 | 0.966 | 28.68 / 0.922 | `backbones/HFIN/s_ms/wv3/` |
| HFIN | s_ms | QB | 38.03 | 0.896 |  | `backbones/HFIN/s_ms/qb/` |
| HFIN | s_ms | GF2 | 42.56 | 0.966 |  | `backbones/HFIN/s_ms/gf2/` |
| UTeRM | original | WV3 | 35.96 | 0.942 | 28.85 / 0.954 | `backbones/UTeRM/original/wv3/` |
| UTeRM | original | QB | 37.26 | 0.949 |  | `backbones/UTeRM/original/qb/` |
| UTeRM | original | GF2 | 41.89 | 0.940 |  | `backbones/UTeRM/original/gf2/` |
| UTeRM | s_pan | WV3 | 36.45 | 0.956 | 29.13 / 0.922 | `backbones/UTeRM/s_pan/wv3/` |
| UTeRM | s_pan | QB | 37.40 | 0.941 |  | `backbones/UTeRM/s_pan/qb/` |
| UTeRM | s_pan | GF2 | 42.56 | 0.966 |  | `backbones/UTeRM/s_pan/gf2/` |
| UTeRM | s_ms | WV3 | 36.46 | 0.963 | 29.45 / 0.946 | `backbones/UTeRM/s_ms/wv3/` |
| UTeRM | s_ms | QB | 37.54 | 0.894 |  | `backbones/UTeRM/s_ms/qb/` |
| UTeRM | s_ms | GF2 | 42.69 | 0.974 |  | `backbones/UTeRM/s_ms/gf2/` |
| BNNPan | original | WV3 | 36.73 | 0.936 | 27.35 / 0.755 | `backbones/BNNPan/original/wv3/` |
| BNNPan | original | QB | 37.38 | 0.856 |  | `backbones/BNNPan/original/qb/` |
| BNNPan | original | GF2 | 42.12 | 0.927 |  | `backbones/BNNPan/original/gf2/` |
| BNNPan | s_pan | WV3 | 37.05 | 0.969 | 29.69 / 0.949 | `backbones/BNNPan/s_pan/wv3/` |
| BNNPan | s_pan | QB | 37.71 | 0.880 |  | `backbones/BNNPan/s_pan/qb/` |
| BNNPan | s_pan | GF2 | 42.69 | 0.942 |  | `backbones/BNNPan/s_pan/gf2/` |
| U-Know-DiffPAN | original | WV3 | 37.44 | 0.964 | 29.20 / 0.966 | `backbones/U-Know-DiffPAN/original/wv3/` |
| U-Know-DiffPAN | original | QB | 37.00 | 0.939 |  | `backbones/U-Know-DiffPAN/original/qb/` |
| U-Know-DiffPAN | original | GF2 | 43.24 | 0.966 |  | `backbones/U-Know-DiffPAN/original/gf2/` |
| U-Know-DiffPAN | s_pan | WV3 | 37.83 | 0.959 | 29.80 / 0.947 | `backbones/U-Know-DiffPAN/s_pan/wv3/` |
| U-Know-DiffPAN | s_pan | QB | 37.45 | 0.818 |  | `backbones/U-Know-DiffPAN/s_pan/qb/` |
| U-Know-DiffPAN | s_pan | GF2 | 44.05 | 0.976 |  | `backbones/U-Know-DiffPAN/s_pan/gf2/` |
| U-Know-DiffPAN | s_ms | WV3 | 37.89 | 0.963 | 30.04 / 0.960 | `backbones/U-Know-DiffPAN/s_ms/wv3/` |
| U-Know-DiffPAN | s_ms | QB | 37.90 | 0.871 |  | `backbones/U-Know-DiffPAN/s_ms/qb/` |
| PAN-Crafter | original | WV3 | 37.48 | 0.964 | 29.04 / 0.949 | `backbones/PAN-Crafter/original/wv3/` |
| PAN-Crafter | original | QB | 35.75 | 0.826 |  | `backbones/PAN-Crafter/original/qb/` |
| PAN-Crafter | original | GF2 | 44.33 | 0.971 |  | `backbones/PAN-Crafter/original/gf2/` |
| PAN-Crafter | s_pan | WV3 | 37.54 | 0.962 | 29.90 / 0.937 | `backbones/PAN-Crafter/s_pan/wv3/` |
| PAN-Crafter | s_pan | QB | 37.74 | 0.846 |  | `backbones/PAN-Crafter/s_pan/qb/` |
| PAN-Crafter | s_pan | GF2 | 44.80 | 0.979 |  | `backbones/PAN-Crafter/s_pan/gf2/` |
| PAN-Crafter | s_ms | WV3 | 37.60 | 0.960 | 28.88 / 0.933 | `backbones/PAN-Crafter/s_ms/wv3/` |
| PAN-Crafter | s_ms | QB | 37.93 | 0.935 |  | `backbones/PAN-Crafter/s_ms/qb/` |
| PAN-Crafter | s_ms | GF2 | 45.00 | 0.981 |  | `backbones/PAN-Crafter/s_ms/gf2/` |
| PINO | original | WV3 | 36.42 | 0.959 | 29.76 / 0.946 | `backbones/PINO/original/wv3/` |
| PINO | original | QB | 37.37 | 0.925 |  | `backbones/PINO/original/qb/` |
| PINO | original | GF2 | 42.48 | 0.979 |  | `backbones/PINO/original/gf2/` |
| PINO | s_pan | WV3 | 37.26 | 0.961 | 30.14 / 0.953 | `backbones/PINO/s_pan/wv3/` |
| PINO | s_pan | QB | 38.12 | 0.935 |  | `backbones/PINO/s_pan/qb/` |
| PINO | s_pan | GF2 | 44.20 | 0.986 |  | `backbones/PINO/s_pan/gf2/` |
| PINO | s_ms | WV3 | 37.55 | 0.954 | 30.43 / 0.941 | `backbones/PINO/s_ms/wv3/` |
| PINO | s_ms | QB | 38.03 | 0.945 |  | `backbones/PINO/s_ms/qb/` |
| PINO | s_ms | GF2 | 44.66 | 0.984 |  | `backbones/PINO/s_ms/gf2/` |
| PanDiT | original | WV3 | 36.26 | 0.964 | 29.15 / 0.978 | `backbones/PanDiT/original/wv3/` |
| PanDiT | original | QB | 37.10 | 0.944 |  | `backbones/PanDiT/original/qb/` |
| PanDiT | original | GF2 | 42.00 | 0.954 |  | `backbones/PanDiT/original/gf2/` |
| PanDiT | s_pan | WV3 | 36.13 | 0.965 | 30.17 / 0.962 | `backbones/PanDiT/s_pan/wv3/` |
| PanDiT | s_pan | QB | 36.67 | 0.953 |  | `backbones/PanDiT/s_pan/qb/` |
| PanDiT | s_pan | GF2 | 42.16 | 0.982 |  | `backbones/PanDiT/s_pan/gf2/` |
| PanDiT | s_ms | WV3 | 36.47 | 0.962 | 30.37 / 0.962 | `backbones/PanDiT/s_ms/wv3/` |
| PanDiT | s_ms | QB | 37.40 | 0.837 |  | `backbones/PanDiT/s_ms/qb/` |
| PanDiT | s_ms | GF2 | 42.34 | 0.978 |  | `backbones/PanDiT/s_ms/gf2/` |
| MMMamba | original | WV3 | 36.23 | 0.952 | 28.89 / 0.932 | `backbones/MMMamba/original/wv3/` |
| MMMamba | original | QB | 36.97 | 0.788 |  | `backbones/MMMamba/original/qb/` |
| MMMamba | original | GF2 | 39.68 | 0.928 |  | `backbones/MMMamba/original/gf2/` |
| MMMamba | s_pan | WV3 | 37.05 | 0.952 | 28.88 / 0.899 | `backbones/MMMamba/s_pan/wv3/` |
| MMMamba | s_pan | QB | 37.75 | 0.916 |  | `backbones/MMMamba/s_pan/qb/` |
| MMMamba | s_pan | GF2 | 42.72 | 0.961 |  | `backbones/MMMamba/s_pan/gf2/` |
| MMMamba | s_ms | WV3 | 36.70 | 0.954 | 27.96 / 0.877 | `backbones/MMMamba/s_ms/wv3/` |
| TS-Pan | original | WV3 | 37.90 | 0.946 | 29.59 / 0.920 | `backbones/TS-Pan/original/wv3/` |
| TS-Pan | original | QB | 38.53 | 0.743 |  | `backbones/TS-Pan/original/qb/` |
| TS-Pan | original | GF2 | 41.94 | 0.944 |  | `backbones/TS-Pan/original/gf2/` |
| TS-Pan | s_pan | WV3 | 37.51 | 0.955 | 30.10 / 0.930 | `backbones/TS-Pan/s_pan/wv3/` |
| TS-Pan | s_pan | QB | 38.54 | 0.942 |  | `backbones/TS-Pan/s_pan/qb/` |
| TS-Pan | s_pan | GF2 | 44.34 | 0.931 |  | `backbones/TS-Pan/s_pan/gf2/` |
| TS-Pan | s_ms | WV3 | 37.83 | 0.934 | 30.15 / 0.935 | `backbones/TS-Pan/s_ms/wv3/` |
| TS-Pan | s_ms | QB | 38.49 | 0.868 |  | `backbones/TS-Pan/s_ms/qb/` |
| TS-Pan | s_ms | GF2 | 44.56 | 0.979 |  | `backbones/TS-Pan/s_ms/gf2/` |

## Upstream code

| Backbone | Upstream repository | Pinned commit | Upstream code licence | Our patch |
|---|---|---|---|---|
| PanNet | https://github.com/Zeyu-Zhu/PGCU | `987de8e6ff4f` | MIT | – |
| LAGConv | https://github.com/liangjiandeng/LAGConv | `5d142a26d1df` | no licence file (all rights reserved by the authors) | – |
| MutInf | https://github.com/manman1995/Mutual-Information-driven-Pan-sharpening | `5915c65f3f24` | no licence file (all rights reserved by the authors) | – |
| MDCUN | https://github.com/yggame/MDCUN | `82e81dde20a9` | no licence file (all rights reserved by the authors) | `patches/MDCUN.patch` |
| PanFlowNet | https://github.com/yggame/PanFlowNet | `84afb1a59a82` | no licence file (all rights reserved by the authors) | `patches/PanFlowNet.patch` |
| MSDDN | https://github.com/alexhe101/MSDDN | `834bfb9bc2e1` | no licence file (all rights reserved by the authors) | `patches/MSDDN.patch` |
| HFIN | https://github.com/JosephTiTan/HFIN | `73a4cbb9309b` | no licence file (all rights reserved by the authors) | `patches/HFIN.patch` |
| UTeRM | https://github.com/mtntruong/UTeRM | `b0149012a39e` | GPL | – |
| BNNPan | https://github.com/coder-JMHou/BNNPan | `0257e9a02122` | no licence file (all rights reserved by the authors) | – |
| U-Know-DiffPAN | https://github.com/KAIST-VICLab/U-Know-DiffPAN | `e3f7b9d079d0` | MIT | – |
| PAN-Crafter | https://github.com/KAIST-VICLab/PAN-Crafter | `e12fb5ef0905` | licence file with non-commercial / research-only terms | – |
| PINO | https://github.com/ez4lionky/PINO | `6793a1380d5d` | no licence file (all rights reserved by the authors) | – |
| PanDiT | https://github.com/para133/PanDIT | `985a4ea1d9b5` | no licence file (all rights reserved by the authors) | `patches/PanDIT.patch` |
| MMMamba | https://github.com/Gracewangyy/MMMamba | `74c096129030` | no licence file (all rights reserved by the authors) | `patches/MMMamba.patch` |
| TS-Pan | https://github.com/lzm-01/TS-Pan | `e47cf0d09c79` | Apache-2.0 | `patches/TS-Pan.patch` |
