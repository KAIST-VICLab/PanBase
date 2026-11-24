# Data: PanCollection

Download PanCollection (the WorldView-3, QuickBird and GaoFen-2 training / validation / test sets and the WorldView-2
test sets) from its official page and arrange it as

```
$PANCOLLECTION_ROOT/
  WV3/train_wv3.h5  WV3/valid_wv3.h5
  WV3/reduced_examples_h5/test_wv3_multiExm1.h5  WV3/full_examples_h5/test_wv3_OrigScale_multiExm1.h5
  QB/...   GF2/...   (same pattern: train_qb.h5, ..., test_gf2_OrigScale_multiExm1.h5)
  WV2/reduced_examples_h5/test_wv2_multiExm1.h5  WV2/full_examples_h5/test_wv2_OrigScale_multiExm1.h5
```

Values are digital numbers; the code divides by 2047 (WorldView-3, QuickBird, WorldView-2) or 1023 (GaoFen-2).
PanBase bases for training and testing are written next to the repository by `scripts/make_bases.py`
(`data_skip/S_PAN/...`, `data_skip/S_MS/...`, same relative paths as PanCollection).
PAN-Crafter additionally needs the low-pass PAN distributed with its repository; `scripts/fetch_repos.sh` unpacks it.
