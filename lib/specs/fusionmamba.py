"""FusionMamba -- Efficient Remote Sensing Image Fusion with State Space Model (TGRS 2024).

저장소: repos/FusionMamba (github.com/PSRben/FusionMamba).

의존성이 까다롭다.  `Mamba-block/readme.md` 가 요구하는 것은 상위 `mamba-ssm` 이 아니라
**Vision-Mamba(hustvl/Vim @ b9cf48f) 의 `mamba-1p1p1`** 이고, 거기에 저장소가 제공하는
`mamba_simple.py` 를 덮어쓴 것이다.  상위 mamba_ssm 에는 `bimamba_inner_fn` 과
`mamba_inner_fn_no_out_proj` 가 없어서 import 자체가 안 된다 (게다가 그 `except`
절은 값 5개를 이름 4개에 푸는 버그라 ImportError 가 ValueError 로 바뀐다).
causal_conv1d 도 1.1.0 이어야 한다 -- 1.2.x 는 `causal_conv1d_fwd` 인자 목록이 다르다.

그래서 그 조합을 **별도 prefix** `.toolchain/vim_mamba` 에 설치하고, 이 spec 이
빌드할 때만 sys.path 앞에 끼운다.  상위 site-packages 의 mamba_ssm 1.2.0.post1 은
그대로 두므로 MMMamba 가 영향을 받지 않는다.  빌드 절차는 `CUDA_OPS.md`.
"""
import os as _os  # 공개본 (make_release): 뿌리 = $PANBASE_ROOT (기본: 이 저장소)
_GB = _os.environ.get("PANBASE_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
import sys
from pathlib import Path

VIM = (_GB + "/.toolchain/vim_mamba")


def register(SPECS, REPOS, _load, _pkg):
    ROOT = Path(REPOS) / "FusionMamba"

    def build(ms_ch=4, ratio=4):
        if ratio != 4:
            raise ValueError(f"FusionMamba 저장소 설정은 x4 뿐이다 (ratio={ratio})")
        import torch.nn as nn
        # Vim 포크가 상위 mamba_ssm 을 이겨야 한다.  이미 상위가 적재돼 있으면
        # 지우고 다시 잡는다 -- 안 그러면 조용히 상위가 쓰인다.
        # 파이썬 패키지뿐 아니라 **컴파일된 확장 모듈**까지 비워야 한다.
        # `causal_conv1d_cuda` / `selective_scan_cuda` 는 최상위 이름이라
        # `causal_conv1d` 만 지우면 상위 1.2.0 확장이 그대로 남고, 포크의
        # causal_conv1d 가 그것을 다시 집어 인자 목록이 어긋난다
        # (`causal_conv1d_fwd(): incompatible function arguments`).
        _drop = ("mamba_ssm", "causal_conv1d", "causal_conv1d_cuda",
                 "selective_scan_cuda", "selective_scan_cuda_oplus")
        for k in [k for k in sys.modules
                  if k in _drop or any(k.startswith(d + ".") for d in _drop)]:
            sys.modules.pop(k)
        sys.path.insert(0, VIM)
        sys.path.insert(0, str(ROOT))
        _pkg(ROOT / "model")
        from model.u2net import U2Net

        BASE = 64                       # train.py 기본값 --H/--W 64
        net = U2Net(32, 1, ms_ch, H=BASE, W=BASE, scale=4)
        # 구성 시점의 H/W 를 기억해 둔다.  아래 래퍼가 이 값을 기준으로 비례시킨다.
        for m in net.modules():
            if hasattr(m, "input_h"):
                m._base_h, m._base_w = m.input_h, m.input_w

        class Wrap(nn.Module):
            """평가 크기(256/512)에서도 돌게 만든다.

            `input_h/input_w` 는 `rearrange` 의 인자로만 쓰이고 **파라미터는 그것에
            의존하지 않는다**(직접 확인).  즉 크기 고정은 순전파의 재배열에만 있고
            가중치에는 없다.  그래서 들어온 PAN 크기에 맞춰 비례로 다시 써 주면
            수식은 한 글자도 안 바뀐 채 임의 크기에서 돈다.  H=64 에서는 원본과
            비트 단위로 동일하다."""
            def __init__(self, net): super().__init__(); self.net = net
            def forward(self, lrms, pan):
                # **조건 없이** 매번 다시 쓴다.  전에는 `if f != 1.0` 로 감싸 두어,
                # 64x64 로 돌아올 때 직전 호출이 남긴 값이 그대로 남았다.  평가 순서가
                # RR(256) -> FR(512) -> valid(64) 라, 마지막 valid 에서
                # `rearrange(..., h=512, w=512)` 가 4096 토큰을 받고 죽었다.  학습을
                # 다 마치고 RR/FR 수치까지 찍은 뒤였다(FusionMamba/qb, 4.1 시간).
                # 크기는 상태이므로 항상 입력에 맞춰야 한다.
                f = pan.shape[-2] / BASE
                for m in self.net.modules():
                    if hasattr(m, "_base_h"):
                        m.input_h = max(1, int(round(m._base_h * f)))
                        m.input_w = max(1, int(round(m._base_w * f)))
                return self.net(lrms, pan)

        return Wrap(net)

    SPECS["FusionMamba"] = dict(
        tier="confirmed",
        build=build, venue="TGRS 2024", loss="l1", batch=32, epochs=500,
        # train.py 기본값: lr 5e-4, wd 0, StepLR(step=200, gamma=0.5).
        # 손실은 순수 L1 -- `--use_ergas` 기본값이 False 다 (train.py:135).
        opt=dict(kind="adam", lr=5e-4, weight_decay=0.0),
        sched=dict(kind="step", step_size=200, gamma=0.5),
        # forward(ms, pan) 의 ms 는 **저해상** MS 다.  train.py:45-46 의 summary 가
        # input_size 를 (B,C,16,16) / (B,1,64,64) 로 잡아 두어 확인된다.
        forward=lambda m, ms_up, pan, lrms, ctx: m(lrms, pan),
        diff="네트워크 코드 수정 없음. 두 가지를 기록한다. (1) 의존성: 상위 "
             "mamba-ssm 이 아니라 Vim @ b9cf48f 의 mamba-1p1p1 + 저장소가 준 "
             "mamba_simple.py + causal_conv1d 1.1.0 조합이 필요하다. 상위를 "
             "덮어쓰지 않도록 별도 prefix 에 넣고 빌드 시에만 sys.path 에 끼운다. "
             "(2) `U2Net(..., H, W, ...)` 이 구성 시점에 공간 크기를 받지만 "
             "**파라미터는 거기에 의존하지 않는다** -- `input_h/input_w` 는 "
             "`rearrange` 인자로만 쓰인다. 256/512 평가를 위해 들어온 PAN 크기에 "
             "비례해 그 값만 다시 쓰는 래퍼를 씌웠다. H=64 에서는 원본과 동일하다.")
