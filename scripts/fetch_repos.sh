#!/usr/bin/env bash
# Fetch every upstream repository at its pinned commit and apply our patch.
# Upstream source is NOT part of this release -- each repository keeps its own licence (NOTICE.md).
#
#   bash scripts/fetch_repos.sh     # the 15 backbones of the paper tables + the metric toolbox
#
# Env: PANBASE_ROOT   repository root (default: the parent of this script's directory)
#      PANBASE_FETCH_MIRROR clone from <mirror>/<dir> instead of GitHub (offline test)
#      PYTHON              interpreter used to unpack PAN-Crafter's lpan archive (default: python)
# Re-running is safe: existing clones are checked out again and an already applied patch is skipped.
set -euo pipefail
ROOT="${PANBASE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
ALL=0
for a in "$@"; do case "$a" in --all) ALL=1 ;; *) echo "usage: $0 [--all]" >&2; exit 2 ;; esac; done
MIRROR="${PANBASE_FETCH_MIRROR:-}"
mkdir -p "$ROOT/repos"
cd "$ROOT/repos"

fetch() {  # <name> <dir> <url> <commit> [patch]
  local name="$1" dir="$2" url="$3" commit="$4" patch="${5:-}"
  echo "== $name  ($dir @ ${commit:0:7})"
  if [ ! -e "$dir/.git" ]; then
    if [ -n "$MIRROR" ]; then git clone -q --shared "$MIRROR/$dir" "$dir"; else git clone -q "$url" "$dir"; fi
  fi
  git -C "$dir" -c advice.detachedHead=false checkout -q "$commit"
  [ "$(git -C "$dir" rev-parse HEAD)" = "$commit" ] || { echo "!! $dir: HEAD is not $commit" >&2; return 1; }
  if [ -n "$patch" ]; then
    if git -C "$dir" apply --reverse --check "$ROOT/$patch" 2>/dev/null; then
      echo "   patch already applied ($patch)"
    else
      git -C "$dir" apply "$ROOT/$patch" && echo "   patch applied ($patch)"
    fi
  fi
}

fetch "PanNet" "PGCU" "https://github.com/Zeyu-Zhu/PGCU" "987de8e6ff4fd8248142d1a06f6b6300779e45fe"
fetch "LAGConv" "LAGConv" "https://github.com/liangjiandeng/LAGConv" "5d142a26d1df43a7c0324cd93615143651007b6a"
fetch "MutInf" "Mutual-Information-driven-Pan-sharpening" "https://github.com/manman1995/Mutual-Information-driven-Pan-sharpening" "5915c65f3f24ddfcf5d528d80d7b35b7f2ae9de3"
fetch "MDCUN" "MDCUN" "https://github.com/yggame/MDCUN" "82e81dde20a99b921ede7625207947ccadca64cb" "patches/MDCUN.patch"
fetch "PanFlowNet" "PanFlowNet" "https://github.com/yggame/PanFlowNet" "84afb1a59a82113ded09ea2b863ba1faae8af0eb" "patches/PanFlowNet.patch"
fetch "MSDDN" "MSDDN" "https://github.com/alexhe101/MSDDN" "834bfb9bc2e1a4c3b30efb8891c1e5b5326e6c19" "patches/MSDDN.patch"
fetch "HFIN" "HFIN" "https://github.com/JosephTiTan/HFIN" "73a4cbb9309ba02bf2b76b8d0b8a6bef518199da" "patches/HFIN.patch"
fetch "UTeRM" "UTeRM" "https://github.com/mtntruong/UTeRM" "b0149012a39ef95f5699dbf7d0d6dad4d84dcd57"
fetch "BNNPan" "BNNPan" "https://github.com/coder-JMHou/BNNPan" "0257e9a021222b2d532269e1860ef560fc1fa042"
fetch "U-Know-DiffPAN" "U-Know-DiffPAN" "https://github.com/KAIST-VICLab/U-Know-DiffPAN" "e3f7b9d079d0753ec16af2a44855e44fd3c2307b"
fetch "PAN-Crafter" "PAN-Crafter" "https://github.com/KAIST-VICLab/PAN-Crafter" "e12fb5ef09051d19bc21f3fef6080a05cf7b223a"
fetch "PINO" "PINO" "https://github.com/ez4lionky/PINO" "6793a1380d5d31fc2a50329a71056c422a18a2f7"
fetch "PanDiT" "PanDIT" "https://github.com/para133/PanDIT" "985a4ea1d9b5fc39e897e94030848f1b94f2b8f8" "patches/PanDIT.patch"
fetch "MMMamba" "MMMamba" "https://github.com/Gracewangyy/MMMamba" "74c09612903075e7da803719a16fb4a89a7fed98" "patches/MMMamba.patch"
fetch "TS-Pan" "TS-Pan" "https://github.com/lzm-01/TS-Pan" "e47cf0d09c79ee551adc64b9af18f0c746969676" "patches/TS-Pan.patch"
fetch "AWFLN (provides the Pansharpening Toolbox used by the FR metrics)" "AWFLN" "https://github.com/yotick/AWFLN" "210232c0836f7808adadbbab6b59a1123fc18c55"

# PAN-Crafter's low-pass PAN (lpan) ships inside its repository; the harness reads it from $ROOT/data_lpan.
if [ -f PAN-Crafter/pan_h5/pan_h5.zip ] && [ ! -f "$ROOT/data_lpan/test_wv3_multiExm1_pan.h5" ]; then
  "${PYTHON:-python}" -m zipfile -e PAN-Crafter/pan_h5/pan_h5.zip "$ROOT/data_lpan" && echo "== data_lpan/ unpacked"
fi

if [ "$ALL" = 1 ]; then
  echo "-- extra methods (wired, never trained by us; failures are reported and skipped)"
fi
echo "done.  Next: build the CUDA extensions a method needs (CUDA_OPS.md), then see README.md."
