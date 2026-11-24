"""UAPN (TGRS 2023): 불확실성(aleatoric/epistemic) 유도 적응 합성곱 팬샤프닝.

가장 틀리기 쉬운 곳 — 저장소의 인자 이름 `lms` 는 **저해상 MS**(우리 `lrms`)다.  보통의
관례("lms = 업샘플된 MS")와 반대다.  모델이 내부에서 4배 bicubic 업샘플을 직접 하므로
(`UAPN_arch.py:35`), 우리 `ms_up` 을 넣으면 4H 로 부풀어 조용히 틀린 크기를 낸다.
두 번째로 틀리기 쉬운 곳 — 손실이 7항이다.  `L1(out, gt)` 하나로 줄이면 논문의 불확실성
메커니즘이 통째로 사라진다.
"""
import numpy as np
import torch

_CACHE = {}          # 모듈 재실행 금지: ARCH_REGISTRY 가 같은 이름 두 번 등록을 assert 로 막는다
_EVAL_SEED = 20260919


def register(SPECS, REPOS, _load, _pkg):
    import sys

    ROOT = REPOS / "UAPN"

    def _arch():
        if "m" not in _CACHE:
            # UAPN_arch.py 상단이 `sys.path.append("..")` 후 `from Registry import ARCH_REGISTRY`
            # 를 한다.  그 상대 경로는 cwd 가 저장소 안일 때만 맞으므로 우리가 직접 넣어 준다.
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            m = _load(ROOT / "model" / "UAPN_arch.py", "uapn_arch")

            def _random_mask(self, x, q):
                """원본(UAPN_arch.py:77-81)은 `torch.tensor(mask).cuda()` 로 장치를 박아 둔다.
                난수 호출(np.random.binomial(n=1, p=1-q, size=C))과 그 결과값은 그대로 두고
                텐서를 놓을 장치만 x 를 따라가게 했다 -- GPU 에서는 완전한 no-op 이고,
                CPU/멀티GPU 에서 죽지 않게만 한다.  마스크 값도 난수 소비량도 불변."""
                mask = np.random.binomial(n=1, p=1 - q, size=x.shape[1])
                mask = torch.as_tensor(mask, device=x.device)
                return x * mask.reshape(1, -1, 1, 1)        # rearrange("C -> 1 C 1 1") 과 동일

            m.UncertaintyEstimation.random_mask = _random_mask
            _CACHE["m"] = m
        return _CACHE["m"]

    def build(ms_ch=4, ratio=4):
        # channels=12 는 config_1.py:26 의 본 모델(UAPN-B).  channels=8 은 소형 변종(UAPN-S)
        # 이라 쓰지 않는다.  bands 는 patches 로 뚫은 인자 -- 기본 4 에서 파라미터 수가
        # 78,992 로 원본과 한 개도 다르지 않다(감사 로그의 NO-OP CHECK).
        return _arch().UAPN(channels=12, bands=ms_ch)

    # ---------------------------------------------------------------- 손실 (7항)
    def _l1w(y1, y2, W):
        """process.py:67-74  class L1Loss -- 가중 L1.  W=1 이면 평범한 mean-L1."""
        return torch.mean(torch.abs(y1 - y2) * W)

    def _uloss(y1, y2, AU):
        """process.py:76-85  class UncertaintyLoss -- 이분산 가우시안 음의 로그가능도.
        0.5*exp(-AU)*(y1-y2)^2 + 0.5*AU.  AU 는 로그분산 역할을 한다."""
        return torch.mean(0.5 * torch.exp(-1.0 * AU) * torch.pow(y1 - y2, 2) + 0.5 * AU)

    def train_step(model, b, ctx, lossfn):
        """process.py:102-114 를 한 항도 빼지 않고 그대로 옮긴 것.  lossfn 은 일부러 쓰지
        않는다 -- 저장소의 L1 은 가중 L1 이고, 하네스의 lossfn 이 무엇이든 목적함수는
        저장소 값이어야 한다.

        계수: 1.0 / 0.1 / 0.01 / 0.1 / 0.01 / 0.01 / 0.01  (전부 7항)
        `ms` 가 저해상 MS 이고 `gt` 가 정답이다 -- 저장소의 이름과 반대이니 주의."""
        gt = b["gt"]
        AU, EU, hrms1, hrms2 = model(pan=b["pan"], lms=b["ms"])     # lms 자리에 **저해상** MS

        # 1. result loss
        loss = _l1w(hrms2, gt.detach(), 1)
        # 2-5. uncertainty-related loss -- 두 UAFM 단계 각각에 심층 감독 + NLL
        loss = loss + _l1w(hrms1[0], gt.detach(), 1) * 0.1 + _uloss(hrms1[0], gt.detach(), AU[0]) * 0.01
        loss = loss + _l1w(hrms1[1], gt.detach(), 1) * 0.1 + _uloss(hrms1[1], gt.detach(), AU[1]) * 0.01
        # 6-7. focus loss -- 가중치를 detach 하지 않아 AU/EU 로도 기울기가 흐른다.
        #      짝이 비대칭이다: (hrms1[1], U1) 과 (hrms2, U2).  process.py:114 그대로다.
        U1 = 0.5 * AU[0] + 0.5 * EU[0]
        U2 = 0.5 * AU[1] + 0.5 * EU[1]
        loss = loss + (_l1w(hrms1[1], gt.detach(), U1) + _l1w(hrms2, gt.detach(), U2)) * 0.01
        return hrms2, loss

    # ---------------------------------------------------------------- 평가 forward
    def forward(m, ms_up, pan, lrms, ctx):
        """평가용.  네 번째 반환값(out)이 예측이다.  ms_up 은 쓰지 않는다 -- 모델이
        lrms 를 직접 bicubic 업샘플해 잔차 기준으로 삼는다(UAPN_arch.py:35,47).

        두 가지를 감싼다.
        (1) MC 드롭아웃이 eval() 에서도 살아 있어 출력이 확률적이다(설계, 저장소의
            validate() 도 그대로 돈다).  마스크가 NumPy 전역 RNG 에서 나오므로
            torch.manual_seed 로는 재현되지 않는다.  고정 시드를 걸고 끝나면 전역
            상태를 되돌려 학습 쪽 난수열을 건드리지 않는다.
        (2) UAConv 의 F.unfold/repeat 가 (B, 2*channels*9, H*W) 를 실체화한다.
            하네스가 bs=64 로 부르는데 H=256 이면 UAConv 하나당 수 GB 라 OOM 이 난다.
            배치를 쪼개되 **청크마다 같은 시드**를 걸어, 한 번에 돌린 것과 비트 단위로
            같은 마스크를 쓰게 했다(마스크가 (C,) 모양이라 배치 크기와 무관하므로
            난수열이 정확히 일치한다 -- draw 20회의 RNG 상태가 같음을 확인했다).
            남는 차이는 cuDNN 이 배치 크기마다 다른 합성곱 알고리즘을 고르는 데서 온다:
            전역 기본값인 cudnn TF32 가 켜져 있으면 1e-2, 끄면 1.3e-5 다.  우리 청크가
            만든 것이 아니라 eval 배치 크기를 바꾸면 어느 베이스라인에나 똑같이 생기는
            잡음이므로, 전역 TF32 설정은 여기서 건드리지 않는다(건드리면 다른
            베이스라인까지 조용히 바뀐다).
        """
        B, _, H, W = pan.shape
        per = max(1, min(B, int(2.0e9 // (H * W * 4320))))   # 4320 B/픽셀 = 2*12*9*4*5(동시 텐서)
        state = np.random.get_state()
        try:
            outs = []
            for i in range(0, B, per):
                np.random.seed(_EVAL_SEED)
                outs.append(m(pan=pan[i:i + per], lms=lrms[i:i + per])[3])
        finally:
            np.random.set_state(state)
        return torch.cat(outs, 0)

    # ---------------------------------------------------------------- 옵티마이저
    def make_opt(model, o):
        """train.py:71  `torch.optim.AdamW(model.parameters(), lr=BASE_LR)` -- 다른 인자를
        하나도 넘기지 않으므로 betas=(0.9,0.999), eps=1e-8, **weight_decay=0.01** 이 전부
        torch 기본값이다.  하네스 기본 빌더는 Adam + weight_decay=0 이라, 이걸 안 주면
        감쇠가 통째로 사라진다.  config_1.py:42-45 의 OPTIMIZER 블록(MOMENTUM=0.9 등)은
        train.py 가 한 번도 읽지 않는 죽은 설정이니 절대 배선하지 않는다."""
        return torch.optim.AdamW(model.parameters(), lr=o["lr"])

    SPECS["UAPN"] = dict(
        tier="extra",
        build=build, venue="TGRS 2023", loss="l1", batch=4, epochs=1000,
        opt=dict(kind="adamw", lr=5e-3, betas=(0.9, 0.999), weight_decay=0.01),
        make_opt=make_opt,
        sched=dict(kind="step", step_size=300, gamma=0.5),   # train.py:72-73 / config_1.py:39-40
        aug=None,                                            # process.py:44-61 -- 증강 전혀 없음
        val_cap=256,                                         # UAConv 가 무거워 중간 valid 는 부분집합
        forward=forward,
        train_step=train_step,
        diff="repos/UAPN/model/UAPN_arch.py 에 `bands` 인자 하나를 10줄로 꿰었다(밴드 4 "
             "하드코딩 6곳: conv_ms/conv_out, UAFM.condition_channels, UncertaintyEstimation "
             "의 out/aue 헤드).  전부 '밴드 수' 자리일 뿐이고, 불확실성 추정기와 UAConv "
             "(unfold-재가중-matmul, T=10, q=0.2)는 한 줄도 건드리지 않았다.  4밴드 "
             "파라미터 수가 78,992 로 패치 전후 동일하다(구조적 no-op), 8밴드는 82,708. "
             "그 밖에 random_mask 의 `.cuda()` 만 x.device 를 따르게 했다(난수 호출·값 불변). "
             "손실은 저장소 7항 전부(1.0/0.1/0.01/0.1/0.01/0.01/0.01), focus 항의 비대칭 "
             "짝 (hrms1[1],U1)+(hrms2,U2) 까지 그대로다.  옵티마이저는 AdamW(lr=5e-3, "
             "weight_decay=0.01 기본값), StepLR(300, 0.5).  평가는 MC 드롭아웃이 eval() "
             "에서도 살아 있어 확률적이므로 NumPy 시드를 고정(전역 상태는 복원)하고, "
             "메모리 때문에 배치를 쪼개되 청크마다 같은 시드를 걸어 한 번에 돌린 것과 "
             "동일한 마스크(난수열 일치 확인)를 쓴다 -- 남는 차이 1e-2 는 cuDNN 의 배치별 "
             "알고리즘/TF32 잡음이지 청크 탓이 아니다(TF32 끄면 1.3e-5).  기울기 클리핑 10.0 은 하네스 공통 기본값이다(저장소는 "
             "클리핑 없음).")
