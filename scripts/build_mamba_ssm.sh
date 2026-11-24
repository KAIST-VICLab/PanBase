#!/usr/bin/env bash
# MMMamba 가 쓰는 상위 mamba_ssm 1.2.0.post1 을 sm_86/89 로 빌드한다 (CUDA_OPS.md).
# 소스: git clone https://github.com/state-spaces/mamba $B/.toolchain/src/mamba && git -C ... checkout v1.2.0.post1
#       (setup.py 에 sm_86/89 를 넣는다 -- CUDA_OPS.md 의 '걸렸던 지점')
set -euo pipefail
B="${PANBASE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"   # 저장소 뿌리 (CUDA_OPS.md)
T="${CUDA_HOME:-$B/.toolchain/cuda_home}"                     # CUDA 12.6 툴킷 (bin/ include/ lib64/ nvvm/)
PY="${PYTHON:-python}"
export CUDA_HOME=$T CUDA_PATH=$T PATH=$T/bin:$PATH
export LD_LIBRARY_PATH=$T/lib64:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=$T/lib64:${LIBRARY_PATH:-}
export MAX_JOBS=4 MAMBA_FORCE_BUILD=TRUE
$PY -m pip install --no-deps --no-build-isolation "${MAMBA_SRC:-$B/.toolchain/src/mamba}"
echo "### 검증"
$PY - <<'PYX'
import torch
from mamba_ssm.modules.mamba_simple import Mamba
m = Mamba(d_model=64).cuda()
x = torch.randn(2, 16, 64, device="cuda")
y = m(x); print("mamba forward OK", tuple(y.shape))
y.sum().backward(); print("backward OK")
PYX
