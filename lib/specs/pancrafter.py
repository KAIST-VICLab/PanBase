"""PAN-Crafter: Learning Modality-Consistent Alignment for PAN-Sharpening (ICCV 2025).

저장소: repos/PAN-Crafter (github.com/KAIST-VICLab/PAN-Crafter), licence: conflict (see NOTICE.md).

왜 배선이 늦었나.  `TODO_baselines.md:33` 에 "`forward(pan, lpan, ms, s)` 의 `lpan`
과 `s` 의미를 학습 코드에서 확인해야 한다"고 적혀 있었다.  둘 다 확인됐다.

`lpan` -- **MS 해상도의 저역 PAN**.  PanCollection h5 에는 없고, 저자들이 형제 파일
`<split>_pan.h5` 로 저장소에 동봉했다 (`pan_h5/pan_h5.zip`, 12개 = 3 데이터셋 x
train/valid/reduced/full).  행 수가 PanCollection 과 정확히 일치한다.
`baselines/data_lpan/` 에 풀어 두었고 `pancoll.load()` 가 읽어 `ctx["lpan"]` 으로
넘긴다.  코퍼스(`$PAN_DATASETS`)는 읽기 전용이라 거기 두지 않았다.

  **직접 만들지 않는 이유.**  bicubic 다운샘플이 아니다.  gf2 reduced 20장에서
  저자 파일과 비교하면 상대오차가
      gauss(sigma=1.7) + 4x4 평균   0.637%     <- Wald 규약의 MTF 열화
      4x4 평균 / INTER_AREA         1.418%
      bicubic resize                3.092%
  다.  저자 파일이 있는데 근사치로 학습시키면 이 방법만 다른 입력을 받게 된다.

`s` -- **스위치 인덱스**.  모델 안에서 `c = self.s_token[s.long()]` 가 되어 학습되는
토큰이 조건으로 들어간다 (pancrafter.py:320).  학습은 배치를 두 배로 불려 한쪽은
PAN 을, 한쪽은 HRMS 를 **같은 헤드로** 내게 한다 (train.py:128-152):

    switch = [0]*B ++ [1]*B
      switch=0 절반 -> 목표 PAN(밴드 수만큼 복제), 잔차 기준선 upsample(lpan)
      switch=1 절반 -> 목표 GT,                   잔차 기준선 upsample(ms)
    loss = w_off * L1(pan, out[:B]) + L1(gt, out[B:]),  w_off = 1.0

    평가는 언제나 switch=1 (train.py:207).

그래서 실효 배치가 설정값의 **두 배**다 (48 -> 96).  설정 파일이 `batch_size: 48
# duplicate` 라고 직접 적어 두었다.

**레시피 이탈 둘을 고쳤다 (2026-09-25).**  배선하며 빠뜨린 것이다.

  증강 -- 저자 feeder 가 학습 split 에만 건다 (feeders/feeder.py:64-111, 173-174).  세 데이터셋 yaml 이
  같다: crop True · crop_ratio 0.75 · ms_size 16 · hflip · vflip · rot 전부 True.
    - MS 격자에서 무작위 크롭: ratio ~ U[0.75, 1), ms_p = np.round(16*ratio) (짝수 반올림 -> 12..16),
      ms_x · ms_y 균등.  PAN 격자는 4배.  cv2.INTER_CUBIC 으로 원래 크기로 되돌린다.
    - hflip · vflip 은 **무작위가 아니다** -- 플래그가 켜져 있으면 언제나 건다 (= 고정 rot180).
    - 그 뒤 rot90 k ~ {0,1,2,3}.  결과는 네 회전뿐이다 (뒤집힌 거울상은 없다 -- 하네스 aug='d4' 와 다르다).
    - gt · lms · ms · lpan · pan 에 **같은 추첨**을 표본마다 건다.
  저자 코드를 옮겨 적지 않고 **그 함수를 그대로 부른다** (`feeder_aug`).  모듈의 `random` 만 step 마다
  `random.Random(seed)` 로 바꿔 끼운다 -- seed 는 하네스 기본 CPU 난수에서 뽑으므로 학습기가 state.pt 에
  저장 · 복원하고(exact 이어 돌리기) warm 에서는 다시 심는다.  batch 복원추출은 하네스 공통 그대로다.

  입력 범위 -- 저자는 gt · lms · ms · lpan · pan 을 전부 [-1,1] (= 2x/max_pixel - 1) 로 넣고
  (feeder.py:156-163), 잔차 bicubic(ms) 를 더한 출력도 그 공간이며 L1 도 그 공간이다 (train.py:137-152).
  하네스 데이터는 x/나눗수 라 [0,1] 이다 (나눗수 = 저자 max_pixel: wv3 · qb 2047, gf2 1023).  그래서
  스펙 안에서 2x-1 로 옮겨 넣고, 예측만 (y+1)/2 로 되돌려 하네스 지표(rr_dim_cut30)에 넘긴다 -- PINO 와
  같은 방식이다 (lib/specs/pino.py `_norm`).  평가 규약은 바뀌지 않는다.
"""
import random

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# 저장소는 num_iter=50,000 **반복**으로 학습한다.  우리 하네스는 epochs x spe 이므로
# 데이터셋마다 가장 가까운 정수 epoch 을 쓴다 (전부 0.25% 이내).
#   wv3 N=9,714  spe=203 -> 246 ep = 49,938 step  (-0.12%)
#   qb  N=17,139 spe=358 -> 140 ep = 50,120 step  (+0.24%)
#   gf2 N=19,809 spe=413 -> 121 ep = 49,973 step  (-0.05%)
_EPOCHS = {"wv3": 246, "qb": 140, "gf2": 121}

#: 학습 증강 -- config/pancrafter_{wv3,qb,gf2}.yaml:9-16 `train_feeder_args` 그대로 (세 데이터셋이 같다).
#: 스펙의 `aug` 로 레시피 스냅샷에 실린다.  **하네스 학습기는 이 값을 해석하지 않는다** (학습기는 "d4" 만
#: 해석한다) -- 증강은 train_step 이 `feeder_aug` 로 건다.  그래서 `by` 를 적어 둔다.  저자 feeder 에 넣는
#: 속성도 여기서 읽는다 -- 기록된 값과 실제로 쓰는 값이 한 곳에서 나온다.
_AUG = dict(by="train_step", src="feeders/feeder.py:64-111", crop=True, crop_ratio=0.75, ms_size=16,
            hflip=True, vflip=True, rot=True, interp="cv2.INTER_CUBIC")
#: register() 가 채운다: 저장소 경로 · 적재기.  저자 feeder 모듈은 처음 쓸 때 한 번 적재한다
#: (cv2 · h5py 를 끌고 오므로 등록 시점에 적재하지 않는다 -- 실패하면 스펙 등록 자체가 빠진다).
_REPO = {}


def _sw(n, v, ref):
    return torch.full((n,), float(v), device=ref.device, dtype=ref.dtype)


def _n11(x):
    """[0,1] -> [-1,1].  저자 np2tensor (feeder.py:162) `2x/max_pixel - 1` 과 같은 값이다."""
    return x * 2.0 - 1.0


def _n01(y):
    """[-1,1] -> [0,1].  저자 test_*_save (train.py:276) `(clip(y,-1,1)+1)/2` 의 clip 은 하네스 평가의
    clamp(0,1) 이 맡는다 (train_pancoll val · FR) -- 같은 연산이다."""
    return (y + 1.0) / 2.0


def _feeder():
    """저자 `PanFeeder` 를 **__init__ 없이** 만든다 (h5 를 열지 않는다).  augment 가 읽는 속성만 `_AUG` 로."""
    if "fe" not in _REPO:
        mod = _REPO["load"](_REPO["root"] / "feeders" / "feeder.py", "pancrafter_feeder")
        fe = object.__new__(mod.PanFeeder)
        for k in ("crop", "crop_ratio", "ms_size", "hflip", "vflip", "rot"):
            setattr(fe, k, _AUG[k])
        _REPO.update(mod=mod, fe=fe)
    return _REPO["mod"], _REPO["fe"]


def feeder_aug(seed, gt, lms, ms, lpan, pan):
    """저자 `PanFeeder.augment` (feeders/feeder.py:64-111) 를 표본마다 **그대로** 부른다.

    (B,C,H,W) 텐서 다섯 개 -> 같은 모양 · 같은 장치.  표본 i 의 다섯 배열은 한 번의 추첨을 공유한다.
    저자는 파이썬 `random` 전역으로 표본마다 (crop 비율, ms_x, ms_y, rot) 를 뽑는다.  그 모듈의 `random`
    을 `random.Random(seed)` 로 바꿔 끼우고 표본을 순서대로 넘긴다 -- 저자 DataLoader 의 한 워커가 배치를
    채우는 소비 순서와 같다.  cv2 는 float32 [0,1] 에서 돈다 (저자는 원시 float64); 끝의 [-1,1] 값 차이는
    float32 반올림(2.4e-7) 수준이다 (tests/test_pancrafter_recipe.py).
    """
    mod, fe = _feeder()
    assert pan.shape[-1] == 4 * fe.ms_size and ms.shape[-1] == fe.ms_size, (
        f"저자 크롭은 ms {fe.ms_size} / pan {4 * fe.ms_size} 학습 패치를 가정한다", tuple(ms.shape), tuple(pan.shape))
    mod.random = random.Random(int(seed))
    src = [x.detach().cpu().numpy() for x in (gt, lms, ms, lpan, pan)]
    out = [[] for _ in src]
    for i in range(gt.shape[0]):
        # 저자는 h5 를 (N,H,W,C) 로 돌려 두고 표본을 np.array 로 복사해 넘긴다 (feeder.py:35-44, 166-172).
        r = fe.augment(*(np.ascontiguousarray(a[i].transpose(1, 2, 0)) for a in src))
        for o, a in zip(out, r):
            o.append(a.transpose(2, 0, 1))
    return tuple(torch.from_numpy(np.ascontiguousarray(np.stack(o))).to(x.device, x.dtype)
                 for o, x in zip(out, (gt, lms, ms, lpan, pan)))


def _lpan(ctx):
    lp = ctx.get("lpan")
    if lp is None:
        # **SystemExit 가 아니라 RuntimeError 다.**  전에는 SystemExit 였는데, 그것은
        # `except Exception` 에 안 잡힌다.  wv3 런은 학습 뒤 WV2 zero-shot 을 도는데 WV2
        # 에는 저자 lpan 파일이 없다 -- 그러면 zero-shot 루프를 뚫고 프로세스가 죽어
        # **결과 JSON 을 쓰기 전에** 수십 시간 학습이 날아가고 재큐 고리에 든다
        # (MMMamba/qb 와 같은 부류).  RuntimeError 면 zero-shot 은
        # "건너뜀"으로 남고 런은 끝난다.  학습 데이터에 lpan 이 없을 때는 첫 step 에서
        # 똑같이 크게 죽으므로 원래 의도(엉뚱한 입력으로 학습하지 않기)는 그대로다.
        raise RuntimeError(
            "PAN-Crafter 는 `lpan` 이 필요하다.  `baselines/data_lpan/` 에 저자 파일이 "
            "풀려 있는지 확인하라 (repos/PAN-Crafter/pan_h5/pan_h5.zip).  WV2 처럼 저자가 "
            "lpan 을 배포하지 않은 데이터셋에서는 평가할 수 없다.")
    return lp


def register(SPECS, REPOS, _load, _pkg):
    ROOT = Path(REPOS) / "PAN-Crafter"
    _REPO.update(root=ROOT, load=_load)

    def build(ms_ch=4, ratio=4):
        import sys
        if ratio != 4:
            raise ValueError("PAN-Crafter 는 scale_factor=4 가 forward 에 박혀 있다.")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        m = _load(ROOT / "model" / "pancrafter.py", "pancrafter_model")
        # 설정 파일의 model_args 그대로.  out_channels 만 밴드 수로 바꾼다
        # (저장소는 데이터셋별 yaml 로 같은 일을 한다).
        return m.PANCrafter(in_channels=1, out_channels=ms_ch,
                            hidden_size=[128, 128, 128, 128], s_embed_size=128,
                            dropout=0.2, use_scale_shift_norm=True,
                            depth=[2, 2, 2, 2], pan_ks=3, ms_ks=3, ka=3)

    def forward(m, ms_up, pan, lrms, ctx):
        """평가 경로.  저장소와 같이 switch=1 이고 res=True 라 upsample(ms) 를 더한다 (train.py:207-209).

        입력은 [-1,1] 로 옮겨 넣고 (저자 feeder), 예측은 [0,1] 로 되돌려 하네스 지표에 넘긴다."""
        pan, lpan, ms = _n11(pan), _n11(_lpan(ctx)), _n11(lrms)
        out = m(pan, lpan, ms, _sw(pan.shape[0], 1.0, pan))
        return _n01(out + F.interpolate(ms, scale_factor=4, mode="bicubic"))

    def train_step(m, b, ctx, lossfn):
        """배치를 두 배로 불려 PAN 절반과 HRMS 절반을 한 스텝에 학습한다 (train.py:128-152).

        `lossfn` 을 쓰지 않는다 -- 두 항이 서로 다른 목표를 보기 때문이다.
        """
        gt, lms, ms, pan = b["gt"], b["lms"], b["ms"], b["pan"]
        lpan = _lpan(ctx)
        # 증강 (저자 feeder 가 학습 split 에만 건다).  seed 는 하네스 기본 CPU 난수에서 step 마다 하나 --
        # 학습기가 state.pt 에 저장 · 복원하고 warm 이어 돌리기에서 다시 심는다 (표본 추출 g 는 건드리지 않는다).
        seed = int(torch.randint(0, 2 ** 62, (1,)))
        gt, lms, ms, lpan, pan = feeder_aug(seed, gt, lms, ms, lpan, pan)
        # 저자 feeder 는 다섯 입력을 전부 [-1,1] 로 넣는다 -- 잔차 기준선 · 목표 · L1 도 그 공간이다.
        gt, ms, lpan, pan = _n11(gt), _n11(ms), _n11(lpan), _n11(pan)
        B, nb = gt.shape[0], gt.shape[1]

        s = torch.cat([_sw(B, 0.0, gt), _sw(B, 1.0, gt)], 0)
        out = m(pan.repeat(2, 1, 1, 1), lpan.repeat(2, 1, 1, 1), ms.repeat(2, 1, 1, 1), s)

        # res=True: 절반마다 잔차 기준선이 다르다 (train.py:147).
        sv = s.view(-1, 1, 1, 1)
        res_ms = F.interpolate(ms.repeat(2, 1, 1, 1), scale_factor=4, mode="bicubic")
        res_pan = F.interpolate(lpan.repeat(2, 1, 1, 1), scale_factor=4, mode="bicubic")
        out = out + res_ms * sv + res_pan.repeat(1, nb, 1, 1) * (1.0 - sv)

        W_OFF = 1.0
        loss_pan = (pan.repeat(1, nb, 1, 1) - out[:B]).abs().mean() * W_OFF
        loss_ms = (gt - out[B:]).abs().mean()
        # 표에 실리는 예측은 HRMS 쪽(switch=1)이다.  하네스 계약대로 [0,1] 로 돌려준다 (학습기는 손실만 쓴다).
        return _n01(out[B:]), loss_pan + loss_ms

    def make_opt(m, o):
        """train.py:61  `torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)`.

        **AdamW 다 -- 분리된(decoupled) weight decay.**  예전 하네스는 `opt["kind"]` 를 읽지 않고
        `torch.optim.Adam(..., weight_decay=0.01)` 을 지었다.  Adam 의 wd 는 기울기에 더해지는 결합
        L2 이고, 적응 분모로 나뉘지 않은 채 매 스텝 들어가 가중치의 85~91% 를 0 쪽으로 무너뜨렸다
        (진단).  하네스도 이제 kind 를 지키지만(`lib/harness.make_optimizer`) 여기 명시해 둔다 --
        (1) 옛 하네스가 이 스펙을 읽어도 AdamW 가 되고, (2) 스펙 함수가 바뀌어 code_digest 가 달라지므로
        옛 옵티마이저로 낸 결과 **세 칸 모두**(wv3 은 sched 가 안 바뀐다)가 게이트에서 낡음이 된다."""
        return torch.optim.AdamW(m.parameters(), lr=o["lr"], weight_decay=o["weight_decay"])

    SPECS["PAN-Crafter"] = dict(
        tier="confirmed",
        build=build, venue="ICCV 2025", loss="l1_switch",
        batch=48, epochs=_EPOCHS["wv3"],
        # 코사인 지평(T_max)도 **그 데이터셋의 epoch 수**다.  저장소는 세 데이터셋 모두
        # num_iter=50,000 에 맞춰 스케줄러를 만든다(train.py:64-68 num_training_steps=num_iter,
        # config/pancrafter_{wv3,qb,gf2}.yaml:49-50) -- 학습 끝에서 학습률이 0 에 닿는다.  예전에는
        # epochs 만 덮어 qb(140) · gf2(121) 가 T_max=246 인 채 끝나 학습률이 최대의 39% · 51% 에서 멈췄다.
        per_dataset={d: dict(epochs=e, sched=dict(T_max=e)) for d, e in _EPOCHS.items()},
        opt=dict(kind="adamw", lr=1e-4, weight_decay=0.01),
        make_opt=make_opt,
        sched=dict(kind="cosine", T_max=_EPOCHS["wv3"], eta_min=0.0),
        # 기록용 (학습기는 해석하지 않는다 -- train_step 이 건다).  `None` 이던 값이 바뀌어 옛 레시피 결과는
        # code_digest 와 함께 이 키로도 낡음이 된다.
        aug=_AUG,
        forward=forward, train_step=train_step, aux_loss=True,
        diff="네트워크 코드 수정 없음.  다음을 기록한다.  "
             "(1) 추가 입력 `lpan`(MS 해상도 저역 PAN)은 PanCollection h5 에 없어 "
             "저자가 동봉한 `pan_h5/pan_h5.zip` 12개 파일을 `baselines/data_lpan/` 에 "
             "풀어 쓴다.  직접 만들지 않았다 -- bicubic 이 아니라 MTF 열화라 "
             "gf2 reduced 에서 bicubic 은 3.09%, gauss(1.7)+4x4평균은 0.64% 어긋난다.  "
             "(2) 학습은 배치를 2배로 복제하고 switch=[0..0,1..1] 을 붙여 같은 헤드가 "
             "PAN 과 HRMS 를 모두 내게 한다(w_off=1.0).  실효 배치가 48 이 아니라 96 이다 -- "
             "저장소 yaml 이 `batch_size: 48  # duplicate` 라고 적어 둔 그대로다.  "
             "(3) 저장소는 num_iter=50,000 반복이고 우리는 epoch 단위라 데이터셋별로 "
             "가장 가까운 정수 epoch 을 쓴다(wv3 246 / qb 140 / gf2 121, 오차 0.25% 이내).  "
             "(4) cosine 스케줄러를 저장소는 매 반복, 우리는 매 epoch step() 한다.  지평은 데이터셋별 "
             "epoch 수다(T_max = 246/140/121).  저장소의 warmup 100 반복(=0.5 epoch 미만)은 재현하지 "
             "않았다.  (5) 옵티마이저는 저장소와 같은 AdamW(lr 1e-4, wd 0.01) -- make_opt 로 명시한다.  "
             "(6) 학습 증강은 저자 feeder(feeders/feeder.py:64-111)를 표본마다 그대로 부른다: MS 격자 "
             "0.75~1.0 무작위 크롭을 cv2.INTER_CUBIC 으로 되돌림 · hflip+vflip(무작위가 아니라 언제나 -- 고정 "
             "rot180) · 무작위 rot90, gt/lms/ms/lpan/pan 에 같은 추첨.  추첨은 저자와 같은 파이썬 random 함수이고 "
             "seed 만 하네스 난수에서 step 마다 뽑는다.  (7) 입력 범위는 저자와 같은 [-1,1](2x/max_pixel-1)이고 "
             "L1 도 그 공간이다.  하네스 데이터가 [0,1] 이라 스펙 안에서 옮기고 예측만 [0,1] 로 되돌려 하네스 "
             "평가(rr_dim_cut30)에 넘긴다.  (6)(7) 은 2026-09-25 에 고쳤다 -- 그 전 레시피(증강 없음 · "
             "[0,1])의 결과는 버리지 않고 공개용으로 따로 둔다(results.bak_20260925_s41).  (8) 저자는 체크포인트를 "
             "**test** ERGAS 로 고른다(main.py:165-177, 5 에폭마다 reduced test).  우리는 시험셋으로 고르지 않고 "
             "valid PSNR 로 고른다.")

    # --- global skip 변형 연구 (2026-10-09) -------------------------------------------------------------
    # 같은 백본 · 같은 레시피 · 같은 손실 · 같은 증강.  바뀌는 것은 HRMS 쪽 잔차 기준선(global skip) 하나다:
    # 저자의 bicubic(lrms) 대신 하네스가 넘기는 `ms_up` -- `train_pancoll --skip S_PAN|S_MS` 일 때 Stage-1 base S.
    # PAN 쪽(switch=0) 기준선 bicubic(lpan)은 skip 이 아니라 그대로 둔다.  원래 스펙의 함수는 건드리지 않는다
    # (code_digest 가 바뀌면 본 칸이 게이트에서 낡음이 된다) -- 그래서 이름이 다른 스펙이다.
    def forward_sk(m, ms_up, pan, lrms, ctx):
        pan, lpan, ms = _n11(pan), _n11(_lpan(ctx)), _n11(lrms)
        out = m(pan, lpan, ms, _sw(pan.shape[0], 1.0, pan))
        return _n01(out + _n11(ms_up))

    def train_step_sk(m, b, ctx, lossfn):
        gt, lms, ms, pan = b["gt"], b["lms"], b["ms"], b["pan"]
        lpan = _lpan(ctx)
        seed = int(torch.randint(0, 2 ** 62, (1,)))
        gt, lms, ms, lpan, pan = feeder_aug(seed, gt, lms, ms, lpan, pan)      # lms(= S) 도 같은 추첨으로 증강된다
        gt, lms, ms, lpan, pan = _n11(gt), _n11(lms), _n11(ms), _n11(lpan), _n11(pan)
        B, nb = gt.shape[0], gt.shape[1]
        s = torch.cat([_sw(B, 0.0, gt), _sw(B, 1.0, gt)], 0)
        out = m(pan.repeat(2, 1, 1, 1), lpan.repeat(2, 1, 1, 1), ms.repeat(2, 1, 1, 1), s)
        sv = s.view(-1, 1, 1, 1)
        res_pan = F.interpolate(lpan.repeat(2, 1, 1, 1), scale_factor=4, mode="bicubic")
        out = out + lms.repeat(2, 1, 1, 1) * sv + res_pan.repeat(1, nb, 1, 1) * (1.0 - sv)
        loss_pan = (pan.repeat(1, nb, 1, 1) - out[:B]).abs().mean() * 1.0
        loss_ms = (gt - out[B:]).abs().mean()
        return _n01(out[B:]), loss_pan + loss_ms

    SPECS["PAN-Crafter-sk"] = dict(
        SPECS["PAN-Crafter"], tier="variant", forward=forward_sk, train_step=train_step_sk,
        diff=SPECS["PAN-Crafter"]["diff"] + "  [PAN-Crafter-sk] global skip 변형 연구(2026-10-09): HRMS 잔차 기준선이 "
             "bicubic(lrms) 대신 하네스의 ms_up(--skip 의 Stage-1 base)이다.  나머지는 PAN-Crafter 와 같다.")
