"""PanDiT (TGRS 2025) -- 잔차 gt-lms 위의 조건부 DDPM(DiT 디노이저), 평가는 DDIM-25 샘플링.

틀리기 쉬운 한 가지: **조건 스택의 웨이블릿 채널은 0-255 스케일이다.**  저장소 데이터로더는
`pywt.wavedec2` 를 uint8 배열에 돌린 뒤 `ToTensor` 로 감싸는데, float64 는 255 로 나뉘지 않는다
(`data/PanDataset.py:86-92`).  그래서 cond 의 앞 C+1 채널(lms, pan)은 [0,1] 인데 뒤 C+3 채널
(웨이블릿)은 ~255배 크다.  의도된 설계는 아닐 것이나 공개 모델은 그 입력으로 학습됐고 첫
컨볼루션이 거기 맞춰져 있다 -- [0,1] 웨이블릿을 먹이면 조건 신호의 2/3 이 조용히 255배 바뀐다.
두 번째로 틀리기 쉬운 것: 확산 타깃도 출력도 **잔차**다.  `+ lms` 를 잊으면 신호가 통째로 사라진다.
"""
from __future__ import annotations
import sys
import torch, torch.nn as nn, torch.nn.functional as F

# ---------------------------------------------------------------- 저장소 상수
# 전부 train.py 의 argparse 기본값이다 (train.py:264-291).  train() 시그니처의 파이썬
# 기본값(image_size=128, n_steps=3000, batch=128, lr=1e-5)은 __main__ 이 항상 덮어쓰는
# 죽은 값이라 읽으면 안 된다 (train.py:298-316).
_N_TIMESTEP  = 500          # train.py:273
_SCHEDULE    = "cosine"     # train.py:282 (schedule_type)
_COSINE_S    = 8e-3         # train.py:79
_DDIM        = "ddim25"     # train.py:212  (predict.py 의 "ddim2" 아님)
_PRED_MODE   = "x_start"    # train.py:71   (predict.py:62 의 "noise" 아님 -- 저장소 내부 모순)
_PATCH       = 8            # train.py:271
_INNER_CH    = 16           # train.py:283
_NOISE_CH    = 128          # train.py:284
_LAYERS      = 12           # train.py:285
_HEADS       = 16           # train.py:286
_MLP_RATIO   = 4            # train.py:287
_T_HIDDEN    = 4            # train.py:288
_EMA_DECAY   = 0.995        # train.py:94, utils/optim_utils.py:43-58
_EMA_START   = 20_000       # train.py:95  (이 전까지는 EMA 가 아니라 하드 카피)
_TRAIN_SIZE  = 64           # PanCollection 학습 패치 크기 (= 우리 harness 의 H)
# 일정: max_iterations 200,000 (train.py:274), MultiStepLR([100k, 200k, 300k], 0.2) 를 iteration 마다
# step (train.py:100-102, 163).  harness 는 에폭 경계에서 sched.step() 하므로 아래 SPECS 의
# epochs · milestones 로 옮긴다 (batch 6, spe = ceil(N/6); 200k · 100k 에 가장 가까운 에폭).

# 우리 쪽 운영 상수 (방법이 아니라 실행 예산에 관한 것)
_TOK_BUDGET  = 1024         # 샘플링 시 한 청크의 최대 토큰 수 = B*(H/8)^2  (~4.7GiB)
_EVAL_SEED   = 20260919     # valid 분할(64x64)에서만 DDIM 초기 잡음을 고정 -- valid PSNR 을 평가 간 비교 가능하게


# ------------------------------------------------------------------ Haar(db1)
def _haar(x):
    """pywt.wavedec2(x, "db1", level=1, axes=[0,1]) 와 **수치적으로 동일**한 1단계 Haar.

    반환 (cA, cH, cV, cD).  직교 정규화(/2)까지 포함해 pywt 와 1e-13 이내로 일치하는 것을
    확인했다 -- PIL/pywt 왕복(호스트 동기화)을 없애려는 것뿐, 값은 바뀌지 않는다."""
    a = x[..., 0::2, 0::2]; b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]; d = x[..., 1::2, 1::2]
    return ((a + b + c + d) / 2, (a + b - c - d) / 2,
            (a - b + c - d) / 2, (a - b - c + d) / 2)


def _cond(lms, pan):
    """train.py:143-151 의 조건 스택 (2C+4 채널).

    웨이블릿은 저장소와 같이 **0-255 스케일**에서 계산한다 (모듈 docstring 참조)."""
    cA, _, _, _ = _haar(lms * 255.0)          # lms_main : C 채널
    _, ph, pv, pd = _haar(pan * 255.0)        # pan 디테일
    wav = torch.cat([cA, ph, pd, pv], 1)      # 저장소 순서: h, d, v (data/PanDataset.py:91-92)
    wav = F.interpolate(wav, size=lms.shape[-1], mode="bicubic")
    return torch.cat([lms, pan, wav], 1).float()   # einops.pack(..., "b * h w") 와 동일


def _share(dst, src):
    """dst 의 파라미터/버퍼 텐서를 src 의 **같은 객체**로 갈아 끼운다 (복사 아님).

    PanDiT 은 생성 시점에 공간 크기가 박히지만(LayerNorm[C,H,H], Rearrange n1=H/8,
    위치 인코딩 토큰 수) 그 자리들엔 학습 텐서가 없다 -- 64/256/512 로 만든 모델의
    state_dict 키·shape 가 완전히 같다는 것이 감사에서 확인됐다.  그래서 평가용 크기별
    사본은 파라미터를 **공유**하면 되고, 그러면 1.57GB 짜리 가중치 복사도, EMA 갱신을
    사본에 다시 밀어 넣는 동기화도 필요 없다."""
    for get, name in ((nn.Module.named_parameters, "parameter"),
                      (nn.Module.named_buffers, "buffer")):
        s = dict(get(src)); d = dict(get(dst))
        assert set(s) == set(d), f"{name} key set differs between sizes"
        for k, t in s.items():
            assert d[k].shape == t.shape, f"{name} {k} shape differs: {d[k].shape} vs {t.shape}"
            m = dst
            for q in k.split(".")[:-1]:
                m = getattr(m, q)
            setattr(m, k.split(".")[-1], t)


# ------------------------------------------------------------------- 래퍼 모듈
class _PanDiT(nn.Module):
    """harness 계약(하나의 nn.Module, forward 는 임의 H 에서 한 텐서)을 PanDiT 의
    확산 학습 + DDIM 평가에 맞춘 래퍼.

    등록되는 것: `net`(학습되는 디노이저), `ema_net`(평가·보고에 쓰이는 EMA 가중치).
    등록되지 않는 것(plain dict/list 에 보관):
      * GaussianDiffusion 인스턴스들 -- 학습용과 샘플링용을 **따로** 둔다.
        `ddim_sample_loop` 가 `space_new_betas` 로 자기 betas/num_timesteps 버퍼를
        파괴적으로 덮어쓰기 때문에(diffusion_ddpm_pan.py:569-612), 한 인스턴스를
        공유하면 검증 한 번 뒤의 학습이 25스텝 스케줄로 돌아간다.  또 이 객체들은
        `net` 을 자기 서브모듈로 들고 있어서, 트리에 올리면 같은 파라미터가 두 번
        등록된다.
      * 평가 크기별 사본 -- 파라미터를 `ema_net` 과 공유하므로 옵티마이저·state_dict 에
        다시 나타나면 안 된다.
    `ema_net` 은 requires_grad=False 라 옵티마이저(make_opt: net 만)와 grad clip 에서
    자동으로 빠지지만, **체크포인트에는 들어가야 한다** -- 평가가 그 가중치로 돌기 때문이다.
    그래서 리스트가 아니라 정식 서브모듈로 등록한다."""

    def __init__(self, PanDiT, GaussianDiffusion, make_beta_schedule, ms_ch):
        super().__init__()
        self.ms_ch = int(ms_ch)
        self.train_size = _TRAIN_SIZE
        self._cls = [PanDiT, GaussianDiffusion, make_beta_schedule]   # off-tree
        self.net = self._make(self.train_size)
        self.ema_net = self._make(self.train_size)
        self.ema_net.load_state_dict(self.net.state_dict())
        self.ema_net.requires_grad_(False)
        self._train_diff = []     # off-tree: [GaussianDiffusion(self.net)]
        self._views = {}          # off-tree: (H, device) -> (net_view, GaussianDiffusion)

    # -- 구성 -------------------------------------------------------------
    def _make(self, H):
        return self._cls[0](
            in_channel=self.ms_ch, out_channel=self.ms_ch,
            image_size=H, patch_size=_PATCH,
            inner_channel=_INNER_CH, noise_level_channel=_NOISE_CH,
            lms_channel=self.ms_ch, pan_channel=1,
            time_hidden_ratio=_T_HIDDEN, num_dit_layers=_LAYERS,
            num_heads=_HEADS, mlp_ratio=_MLP_RATIO,
            with_noise_level_emb=True, self_condition=True)

    def _betas(self):
        return self._cls[2](schedule=_SCHEDULE, n_timestep=_N_TIMESTEP, cosine_s=_COSINE_S)

    def _diffusion(self, net, H, dev):
        d = self._cls[1](net, image_size=H, channels=self.ms_ch,
                         pred_mode=_PRED_MODE, loss_type="l1", device=dev,
                         clamp_range=(0, 1), self_condition=True)
        d.set_new_noise_schedule(None, dev, betas=self._betas())
        return d

    def _apply(self, *a, **k):
        # .to(device)/.float() 등으로 실제 텐서가 바뀌면 공유 사본은 더 이상 유효하지 않다.
        out = super()._apply(*a, **k)
        self._views.clear(); self._train_diff.clear()
        return out

    def _view(self, H, dev):
        key = (int(H), str(dev))
        if key not in self._views:
            if int(H) == self.train_size:
                view = self.ema_net                       # 이미 올바른 크기 -- 그대로 쓴다
            else:
                # CPU 에 만든 뒤 텐서를 통째로 공유한다.  그 무작위 초기화는 곧 버려지므로 전역 CPU 난수를
                # 소비하지 않게 fork_rng 안에서 짓는다 -- 아니면 크기별 첫 호출만 잡음 흐름이 밀려, 같은
                # seed 로 다시 뽑아도 그 크기를 처음 쓰는 프로세스인지에 따라 출력이 달라진다 (CPU 추론).
                with torch.random.fork_rng(devices=[]):
                    view = self._make(int(H))
                _share(view, self.ema_net)
                view.requires_grad_(False)
            view.eval()
            self._views[key] = (view, self._diffusion(view, int(H), dev))
        return self._views[key]

    # -- 학습 -------------------------------------------------------------
    @torch.no_grad()
    def ema_update(self, step):
        """utils/optim_utils.py:43-58 그대로.  저장소는 opt.step() **뒤에**
        `update(iterations)`(0-기반, 증가 전)를 부른다.  우리 harness 는 train_step 안에서
        opt.step() 을 할 수 없으므로, 다음 스텝 맨 앞에서 부른다 -- 그 시점의 파라미터가
        정확히 직전 opt.step() 직후의 값이라 갱신 수열 자체는 같다 (1-기반 step s 에서
        저장소의 iteration = s-2).  유일한 차이는 **평가 시점**이다: 우리 EMA 는 직전
        스텝까지만 반영하므로 저장소 대비 opt.step() 정확히 한 번만큼 뒤처진다
        (decay 0.995 에서 무시할 수 있고, 하드 카피 구간에서도 한 스텝 차이다)."""
        if step < 2:
            return                                        # 아직 opt.step() 이 없었다
        it = step - 2
        for p, q in zip(self.net.parameters(), self.ema_net.parameters()):
            if it > _EMA_START:
                q.data = q.data * _EMA_DECAY + p.data * (1 - _EMA_DECAY)
            else:
                q.data = p.data.clone().detach()

    def train_step(self, batch, ctx):
        """train.py:142-156.  손실은 GaussianDiffusion.p_losses 안에 있는 항 **하나**뿐이다:
        L1(gt-lms, 디노이저가 예측한 잔차).  p2 가중치는 gamma=0 이라 전부 1.0 이고
        decoder_nll 은 리터럴 0.0 이라 둘 다 무연산 -- 감사에서 수치로 확인됐다.
        보조항·정규화항은 없다."""
        self.ema_update(int(ctx.get("step", 1)))
        lms, pan, gt = batch["lms"], batch["pan"], batch["gt"]
        if not self._train_diff:
            self._train_diff.append(self._diffusion(self.net, lms.shape[-1], lms.device))
        loss, recon = self._train_diff[0](gt - lms, cond=_cond(lms, pan))
        return (recon + lms).detach(), loss             # recon+lms 는 로깅용 (train.py:156)

    # -- 평가 -------------------------------------------------------------
    @torch.no_grad()
    def sample(self, lms, pan, fixed_noise):
        """train.py:212-214.  DDIM-25 역확산으로 잔차를 뽑고 lms 를 더해 [0,1] 로 자른다.
        보고 수치는 저장소와 같이 **EMA 가중치**로 낸다.

        `fixed_noise` (forward 가 입력 크기로 정한다 -- SPECS 의 forward 참조):
          True  -- valid 분할(64x64, 체크포인트 선택).  단 **타일 추론 중이면 시험이다** (eval_tile=64):
                   시험 영상의 64 타일도 64x64 라 크기로는 valid 와 구별되지 않는다 -- `harness.in_tiled_eval()`
                   이 켜져 있으면 False 로 본다 (전역 난수, seed-0 규칙).  초기 잡음을 fork_rng 안에서
                   `_EVAL_SEED + 덩어리 위치` 로 뽑는다: valid PSNR 이 평가마다 같은 잡음으로
                   재어지고, 학습 난수열(CPU · CUDA)은 건드리지 않는다.
          False -- 시험 집합(RR 256 · FR 512 · zero-shot · 공개 덤프 · 그림).  저장소처럼(train.py:212,
                   seed 없음) **전역 난수**에서 뽑는다.  그래야 dump_outputs 의 seed-0 규칙(bs 1,
                   집합마다 한 번 manual_seed(0), 차례로 뽑기)이 실제로 먹는다 -- 내부 seed 를 걸면
                   bs 1 에서 모든 장면이 같은 seed(_EVAL_SEED+0)로 뽑히는데 기록에는 seed 0 이 적힌다.
        DDIM eta=0 이라 step 마다 뽑는 noise_like 는 0 이 곱해져 출력에 닿지 않지만 난수는 소비한다
        (diffusion_ddpm_pan.py:600-606) -- 저장소 그대로다."""
        H = int(lms.shape[-1]); dev = lms.device
        assert H % _PATCH == 0, f"H={H} must be divisible by patch_size {_PATCH}"
        view, diff = self._view(H, dev)
        view.eval()
        cs = max(1, _TOK_BUDGET // max(1, (H // _PATCH) ** 2))
        devs = list(range(torch.cuda.device_count())) if dev.type == "cuda" else []
        out = []
        for i in range(0, lms.shape[0], cs):
            l, p = lms[i:i + cs], pan[i:i + cs]
            # ddim_sample_loop 는 자기 betas 를 파괴한다 -- 매 호출 전에 500스텝 코사인으로 되돌린다
            diff.set_new_noise_schedule(None, dev, betas=self._betas())
            if fixed_noise and not _tiled():
                with torch.random.fork_rng(devices=devs):
                    torch.manual_seed(_EVAL_SEED + i)   # 초기 잡음 고정(평가 간 비교 가능)
                    res = diff(_cond(l, p), mode="ddim_sample", section_counts=_DDIM)
            else:
                res = diff(_cond(l, p), mode="ddim_sample", section_counts=_DDIM)
            out.append((res + l).clip(0, 1))
        return torch.cat(out, 0)


def _tiled() -> bool:
    """harness 의 타일 추론 안인가 (시험 영상의 64 타일 -- valid 분할이 아니다)."""
    try:
        import harness as _H
    except ImportError:
        return False
    return _H.in_tiled_eval()


# --------------------------------------------------------------------- 등록
def register(SPECS, REPOS, _load, _pkg):
    ROOT = REPOS / "PanDIT"

    def _imports():
        """`models` / `utils` 는 다른 저장소와 이름이 겹치므로 필요한 동안만 설치하고
        원래 sys.modules 상태로 되돌린다.  클래스 객체는 이미 바인딩돼 있어 안전하다."""
        keys = lambda: [k for k in sys.modules
                        if k in ("models", "utils") or k.startswith(("models.", "utils."))]
        saved = {k: sys.modules[k] for k in keys()}
        try:
            _pkg(ROOT / "models", "models")     # models.PanVIT_modules
            _pkg(ROOT / "utils", "utils")       # utils.loss_utils (diffusion 이 import)
            PV = _load(ROOT / "models" / "PanVIT.py", "pandit_panvit")
            DD = _load(ROOT / "diffusion" / "diffusion_ddpm_pan.py", "pandit_ddpm")
        finally:
            for k in keys():
                del sys.modules[k]
            sys.modules.update(saved)
        # DDIM 루프가 청크마다 tqdm 막대를 찍는다 -- 로그만 더럽히므로 끈다 (수치 무관).
        _t = DD.tqdm
        DD.tqdm = lambda *a, **k: _t(*a, **{**k, "disable": True})
        return PV.PanDiT, DD.GaussianDiffusion, DD.make_beta_schedule

    def build(ms_ch=4, ratio=4):
        PanDiT, GD, mbs = _imports()
        return _PanDiT(PanDiT, GD, mbs, ms_ch)

    def make_opt(model, o):
        # train.py:98 -- AdamW, 디노이저만.  EMA 사본은 requires_grad=False 지만
        # 옵티마이저에 아예 넘기지 않는 편이 분명하다.
        return torch.optim.AdamW(model.net.parameters(), lr=o["lr"],
                                 betas=o.get("betas", (0.9, 0.999)),
                                 weight_decay=o.get("weight_decay", 0.0))

    SPECS["PanDiT"] = dict(
        tier="confirmed",   # (2026-09-28): 스모크 통과, 표에 올린다
        build=build, venue="TGRS 2025", loss="l1",
        batch=6,                            # train.py:276
        # 저장소는 epoch 이 아니라 200,000 **iteration** 이다 (train.py:274).  spe = ceil(N/6):
        #   wv3 124 x 1619 = 200,756 · qb 70 x 2857 = 199,990 · gf2 61 x 3302 = 201,422
        epochs=124,
        opt=dict(kind="adamw", lr=5e-5, betas=(0.9, 0.999), weight_decay=1e-4),
        make_opt=make_opt,
        # train.py:100-102 MultiStepLR([100k, 200k, 300k], 0.2).  일정 안에 드는 것은 100k 하나다
        # (200k 는 마지막 갱신 뒤, 300k 는 밖).  100k 에 가장 가까운 에폭 경계:
        #   wv3 62 -> step 100,378 · qb 35 -> 99,995 · gf2 30 -> 99,060
        # StepLR(step_size) 로 두면 gf2(61 epoch, step_size 30)에 epoch 60 = step 198,120 의 두 번째
        # 감쇠가 생긴다 -- 그래서 저장소와 같은 multistep 으로 적는다.
        sched=dict(kind="multistep", milestones=[62], gamma=0.2),
        per_dataset={"qb":  dict(epochs=70, sched=dict(milestones=[35])),
                     "gf2": dict(epochs=61, sched=dict(milestones=[30]))},
        grad_clip=1.0,                      # train.py:159, utils/misc.py:25-34 (norm, 1.0) -- 저장소 값
        val_cap=64,                         # DDIM-25 는 표본당 25 forward -- 체크포인트 선택용 부분집합
        # 첫 valid 평가를 세 데이터셋 모두 step 20,000 에 둔다 (int(steps*0.099) = 19,874 / 19,799 /
        # 19,940 -> 다음 eval_every(2,000) 배수).  저장소의 첫 검증도 iteration 20,000 이고
        # (save_per_iter, train.py:275) EMA 도 20k 에서 시작한다 (train.py:95).
        eval_after_frac=0.099,
        # 시험 집합의 출력은 DDIM 초기 잡음(전역 난수)에 따라 달라진다 -- 표의 값은 표본 하나다.
        # dump_outputs 는 이 표시로 seed-0 규칙(bs 1, 집합마다 한 번 seed 0)을 쓴다.
        stochastic=True,
        # 평가는 학습 크기(64)의 겹치지 않는 타일로 한다 (2026-09-30; UTeRM 같은 규칙 -- 저자는 한 크기(128)
        # 타일로만 학습 · 추론한다).  256 · 512 를 통째로 넣으면 크기가 박힌 LayerNorm · 패치 격자 · 위치 인코딩 밖이라
        # qb RR 32.68 -> 64 타일 37.10, gf2 39.16 -> 42.00 (같은 best.pt).  가중치는 그대로라 평가만 다시 낸다.
        eval_tile=64,
        # 잡음 규칙은 입력 크기로 정한다: 64x64 = valid 분할(체크포인트 선택) -> 고정 seed,
        # 그 밖(RR 256 · FR 512 · zero-shot · 덤프 · 그림) -> 전역 난수.  sample() docstring 참조.
        forward=lambda mm, ms_up, pan, lrms, ctx: mm.sample(
            ms_up, pan, fixed_noise=int(ms_up.shape[-1]) == _TRAIN_SIZE),
        train_step=lambda model, batch, ctx, lossfn: model.train_step(batch, ctx),
        diff=(
            "저장소 수정은 한 줄뿐이다: models/PanVIT_modules.py:7 의 `from torchsummary import "
            "summary` 를 주석 처리했다 (저장소 어디서도 쓰이지 않는 죽은 import 인데 env 에 없어 "
            "import 자체가 죽는다).  손실·옵티마이저·스케줄·배치·확산 스케줄은 전부 train.py "
            "기본값 그대로다.\n"
            "harness 계약에 맞추느라 한 것들 (연산·목적함수 불변):\n"
            " (1) 웨이블릿을 pywt+PIL 대신 동일한 torch Haar 로 계산한다 -- pywt db1 level-1 과 "
            "1e-13 이내 일치를 확인했고, 저장소처럼 **0-255 스케일**에서 계산해 cond 의 255배 "
            "비대칭을 그대로 보존한다.  여기서 [0,1] 로 '고치면' 조건 신호의 2/3 이 바뀐다.\n"
            " (2) PanDiT 은 생성 시 공간 크기가 박히지만 학습 텐서는 크기 독립이다(64/256/512 의 "
            "state_dict 키·shape 동일).  그래서 학습 모듈은 H=64 로 두고, 평가 H 마다 파라미터를 "
            "**공유하는** 사본을 만들어 forward 한 텐서를 돌려준다.  가중치 복사도 재학습도 없다.  "
            "사본의 (곧 버려지는) 무작위 초기화는 fork_rng 안에서 해 전역 난수를 소비하지 않는다.\n"
            " (3) 평가는 train.py:212 와 같이 EMA 가중치(decay 0.995, 20k iter 까지는 하드 카피)로 "
            "한다.  EMA 갱신은 opt.step() 직후가 아니라 **다음 train_step 맨 앞**에서 도는데, "
            "그 시점 파라미터가 정확히 직전 opt.step() 직후 값이라 저장소와 같은 수열이다.  EMA "
            "사본은 평가에 쓰이므로 체크포인트에 남도록 서브모듈로 등록하되 requires_grad=False 다.  "
            "평가 시점의 EMA 는 저장소 대비 opt.step() 한 번만큼 뒤처지는데, 그것이 harness 가 "
            "train_step 밖에서 opt.step() 을 하는 구조의 유일한 귀결이다.\n"
            " (4) 학습용과 샘플링용 GaussianDiffusion 인스턴스를 분리했다 -- ddim_sample_loop 가 "
            "space_new_betas 로 자기 betas/num_timesteps 를 덮어써서, 공유하면 검증 한 번 뒤의 "
            "학습이 500 이 아니라 25 스텝 스케줄로 돈다.  pred_mode 는 train.py:71 의 'x_start', "
            "n_timestep=500(cosine), 샘플링은 ddim25 -- predict.py 의 'noise'/n_steps=100/'ddim2' "
            "조합은 train.py 로 학습한 가중치에 맞지 않는다(저장소 내부 모순, 패치하지 않고 "
            "train.py 를 따랐다).\n"
            " (5) 일정: 저장소는 epoch 이 아니라 200,000 **iteration** 을 세고(train.py:274) "
            "MultiStepLR([100k, 200k, 300k], gamma 0.2)을 iteration 마다 step 한다(train.py:100-102, 163). "
            "harness 는 에폭 경계에서 step 하므로 200k · 100k 에 가장 가까운 에폭으로 옮겼다 (batch 6, "
            "spe=ceil(N/6)): wv3 124 x 1619 = 200,756 step, 감쇠 epoch 62 = step 100,378 / qb 70 x 2857 = "
            "199,990, 감쇠 35 = 99,995 / gf2 61 x 3302 = 201,422, 감쇠 30 = 99,060 (per_dataset).  kind 는 "
            "저장소와 같은 multistep, 마일스톤은 일정 안의 100k 하나다.  예전의 StepLR(step_size) 표기는 "
            "gf2(61 epoch, step_size 30)에서 epoch 60 = step 198,120 에 두 번째 감쇠를 만들고 규칙이 "
            "그 지점을 마지막 감쇠로 읽는다.  저장소는 200k 에서 두 번째 감쇠가 걸린 뒤 그 에폭 끝까지 "
            "돈다(while 조건을 에폭 경계에서만 본다, train.py:140-141) -- 저장소 데이터의 에폭 길이에 달린 "
            "1 에폭 미만의 꼬리라 옮기지 않았다.\n"
            " (6) 예산(RTX 4090 실측): 한가한 GPU 에서 0.19-0.23 s/step(H=64, B=6, 392.7M 파라미터, "
            "peak allocated 13.0GiB) -- 200k step 이면 11-13 시간, GPU 를 나눠 쓰면 0.30-0.46 s/step 으로 "
            "17-26 시간이다.  48h 구간 / 칸당 96h 상한 안에 저장소 일정 전체가 들어간다 (예전 5시간 "
            "상한 때의 tier='extra' 사유는 더 이상 성립하지 않는다).  평가는 DDIM 25 스텝을 표본마다 돌려 "
            "256 에서 ~9 s/장, 512 에서 ~60-114 s/장이다 (최종 FR 20장 20-38 분).  val_cap=64 는 체크포인트 "
            "선택용 부분집합일 뿐이고(전수면 valid 한 번에 2,201장 x 25 스텝), 표에 실리는 수치는 언제나 "
            "전수 평가다.  체크포인트가 크다: best.pt ~3.14 GB (디노이저 + EMA, fp32), state.pt ~9.4 GB "
            "(모델 3.14 + AdamW 모멘트 3.14 + best_ckpt 사본 3.14).\n"
            " (7) DDIM 초기 잡음: 저장소는 seed 를 걸지 않는다(train.py:212, 전역 난수).  우리는 **valid "
            "분할(64x64, 체크포인트 선택)에서만** fork_rng 안에서 고정 seed(_EVAL_SEED + 덩어리 위치)로 "
            "뽑는다 -- 고정하지 않으면 valid PSNR 이 평가마다 잡음으로 흔들려 'best' 선택이 무의미해지고, "
            "fork_rng 라 학습 난수열도 건드리지 않는다.  **시험 집합(RR · FR · zero-shot · 공개 덤프 · 그림)은 "
            "저장소처럼 전역 난수에서 뽑는다** -- 그래서 stochastic=True 이고, 표의 값은 학습기 난수 흐름의 "
            "표본 하나이며(PanFlowNet 과 같은 처지), 공개 영상은 dump_outputs 의 seed-0 규칙을 따른다.  구분은 "
            "forward 에서 입력 크기(H == 64)로 한다 -- valid 분할만 64x64 이고 시험 집합은 256/512 다.\n"
            " (8) eval_after_frac=0.099: 첫 valid 평가가 세 데이터셋 모두 step 20,000 이다.  저장소의 첫 "
            "검증도 iteration 20,000(save_per_iter, train.py:275)이고 EMA 도 20k 에서 시작한다(train.py:95) "
            "-- 그 전 가중치(EMA 가 아니라 하드 카피)는 저장소가 한 번도 평가하지 않은 것이라 best 후보에서 "
            "뺀다.\n"
            " (9) 기울기 클립 1.0 은 저장소 값이다(train.py:159, utils/misc.py:25-34, 디노이저 전체 노름) -- "
            "harness 기본 10.0 이 아니다.  EMA 사본은 requires_grad=False 라 clip_grad_norm_ 에 기울기가 없다.\n"
            "harness 공통 차이 (방법과 무관, 원장에 적는다): PanCollection 64x64 h5 로 학습한다 -- 저장소는 "
            "128px 8-bit tif 를 즉석 Wald 열화(bicubic 1/4 -> x4)로 만든다(train.py:270, data/PanDataset.py:78-92).  "
            "표본은 복원추출(DataLoader shuffle 아님), 체크포인트는 2,000 step 마다 valid PSNR 최고로 고른다 "
            "(저장소는 20k 마다 저장만 한다).\n"
            "저장소 쪽 사실 둘(우리 배선 문제가 아님): (a) 1,055 개 파라미터 텐서 중 4 개 "
            "-- final_layer.cond_inj.global_linear.{weight,bias} 와 final_layer.cond_inj.mlp.1."
            "{weight,bias} -- 는 gradient 가 None 이다.  FinalLayer 가 CondBlock 의 global_token 을 "
            "받아서 버리기 때문이다(models/PanVIT_modules.py:687 `cond, _ = self.cond_inj(fusion)`). "
            "(b) ddim_sample_loop 는 pred_x_start 를 한 번도 갱신하지 않아(diffusion_ddpm_pan.py:614-647) "
            "샘플링 중 self-conditioning 입력은 늘 None 이다 -- 학습은 50% 확률로 self-cond 를 쓴다. "
            "둘 다 공개된 코드 그대로이고, 나중에 배선 버그로 오인하지 않도록 적어 둔다."),
    )
