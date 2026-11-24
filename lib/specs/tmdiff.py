"""TMDiff(TGRS 2024): 잔차 gt-lms 위의 조건부 DDPM + 얼린 CLIP 텍스트 변조.  학습은 x0(깨끗한
잔차) 회귀인데 공개된 샘플러는 그 출력을 '잡음'으로 읽는다 -- 그대로 쓰면 학습한 것과 다른 모델을 평가한다."""
import sys
import contextlib
import importlib
import torch
import torch.nn as nn

_M = {}                      # 저장소 모듈 1회 임포트 캐시
_WARN = set()

_BY_BAND = {4: "QB", 8: "WV3"}
_SAT = {"wv3": "WV3", "qb": "QB", "gf2": "GF2", "wv2": "WV2", "wv4": "WV4"}
_EMB = ("qb", "gf2", "wv3", "wv2", "wv4")       # prompt_code 의 순서 (바꾸면 옛 체크포인트의 코드가 뒤바뀐다)

# 저장소는 epoch 이 아니라 **iteration** 을 센다: `while current_step < max_iter` (general_sharpening_joint_
# random_batch_finetune.py:161), max_iter 150,000 (config/general_finetune.json:76), batch 4 (json:19,26,33).
# 하네스는 epochs x ceil(N/batch) 이므로(train_pancoll.py:282) 데이터셋마다 **가장 가까운 정수 epoch** 을
# 쓴다 (pancrafter.py:61-66 과 같은 규칙).
#   wv3 N=9,714   spe=ceil(9714/4)=2,429   62 ep = 150,598 step (+0.40%)   (61 -> 148,169, -1.22%)
#   qb  N=17,139  spe=ceil(17139/4)=4,285  35 ep = 149,975 step (-0.02%)   (36 -> 154,260, +2.84%)
#   gf2 N=19,809  spe=ceil(19809/4)=4,953  30 ep = 148,590 step (-0.94%)   (31 -> 153,543, +2.36%)
_EPOCHS = {"wv3": 62, "qb": 35, "gf2": 30}

# 검증은 일정의 마지막 25% 에서만 한다 (학습 궤적은 바뀌지 않는다 -- 검증은 난수를 소비하지 않는다, 아래 sample).
#  (1) 저장소는 max_iter 까지 **무조건** 돌고(script:161) lr 은 max_iter 에서 정확히 0 이 되는 선형 감쇠다
#      (GeneralModel/model.py:32).  하네스의 patience(= max(20*eval_every, ...) = 40,000 step, train_pancoll.py:517)
#      가 그 전에 멈추면 감쇠 끝부분이 잘린다 -- 연장(milestone_policy.py:18)은 train_step 안의 lr 을 보지
#      못한다.  0.75 에서는 첫 검증이 114,000/114,000/112,000 step(wv3/qb/gf2, eval_every 2000)이고 +40,000 이 셋
#      모두 일정 끝을 넘으므로 patience 정지가 일어날 수 없다 = 저장소처럼 끝까지 돈다.
#  (2) 후보에는 저장소 자신의 끝점(step == steps 검증, train_pancoll.py:1253)이 언제나 들어간다.
#  (3) 1000-step 검증 76번 -> 20번 (wv3).  비용은 diff (12).
_EVAL_AFTER = 0.75


def register(SPECS, REPOS, _load, _pkg):
    ROOT = REPOS / "TMDiff"

    # ------------------------------------------------------------------ 임포트
    def _imports():
        """저장소가 `config`/`core`/`utils`/`DWT_IDWT`/`GeneralModel` 을 **절대 임포트**한다.
        전부 흔한 이름이라 sys.path 로는 다른 저장소와 섞인다(특히 utils, core, config).
        그래서 _pkg 로 각 이름을 TMDiff 밑에 직접 박는다 -- 저장소가 요구하는 chdir 도 필요 없다."""
        if _M:
            return _M
        for n in ("config", "core", "utils", "DWT_IDWT", "GeneralModel"):
            _pkg(ROOT / n, n)
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        hu = importlib.import_module("GeneralModel.Hyper_unet_general")
        dg = importlib.import_module("GeneralModel.diffusion_general")
        nw = importlib.import_module("GeneralModel.networks")
        _M.update(WavBEST=hu.WavBEST, GeneralDiffusion=dg.GeneralDiffusion,
                  init_weights=nw.init_weights)
        return _M

    # ------------------------------------------------------------------ 프롬프트
    def _prompt(ms_ch):
        """학습 때의 텍스트 프롬프트(위성 이름 문자열, Hyper_unet_general.py:566-598).

        **짓는 순간에만** 쓰인다.  학습기는 argv 에 --dataset 이 있어 정확하다.  그 밖의 도구(dump_outputs ·
        render_visual · redo_zeroshot)는 --dataset 을 넘기지 않아 밴드 수로 떨어지는데, 4밴드는 QB 와 GF2 를
        가를 수 없다 -- 예전 스펙은 GF2 가중치를 QB 문장으로 평가했을 것이다.  그래서 학습한 프롬프트를
        `prompt_code` buffer 로 체크포인트에 싣고, 적재(load_state_dict)가 그것으로 덮는다.  환경변수로
        바꾸던 손잡이는 뺐다 -- 기록되지 않는 값이 표의 수를 바꾸면 안 된다."""
        for i, a in enumerate(sys.argv):
            if a == "--dataset" and i + 1 < len(sys.argv):
                s = _SAT.get(str(sys.argv[i + 1]).lower())
                if s:
                    return s
            if a.startswith("--dataset="):
                s = _SAT.get(a.split("=", 1)[1].lower())
                if s:
                    return s
        return _BY_BAND.get(int(ms_ch), "QB")

    # ------------------------------------------------------------------ 래퍼
    class TMDiff(nn.Module):
        """GeneralDiffusion(=`net`) 하나만 들고 있는 얇은 래퍼.

        얼린 CLIP 텍스트 타워(123,060,480 params)는 **모듈 트리에서 아예 제거**했다.
        저장소는 옵티마이저 param 목록에서 이름에 "clip_text" 가 든 파라미터를 걸러 내는
        방식으로 같은 일을 한다(GeneralModel/model.py:25-27).  우리는 build 에서 5개 문장
        임베딩을 캐시한 뒤 타워를 지운다 -- 캐시된 임베딩은 nn.Parameter 도 buffer 도 아닌
        평범한 텐서라 state_dict 에 들어가지 않고(검증됨), forward 가 읽는 것은 그 캐시뿐이다
        (Hyper_unet_general.py:587-598, 602).  타워를 남기면 best.pt 마다 약 500MB 가
        얹히고, 옵티마이저가 requires_grad=False 파라미터를 들고 다니게 된다.

        state_dict 에 더한 것은 `prompt_code`(long 스칼라 buffer, 학습한 프롬프트의 _EMB 번호) 하나다.
        파라미터가 아니라 n_params 는 그대로다(30,928,513)."""

        def __init__(self, net, prompt, *, eval_seed, eval_pix, val_sampler):
            super().__init__()
            self.net = net
            self.register_buffer("prompt_code", torch.tensor(_EMB.index(prompt.lower()), dtype=torch.long))
            self.prompt = prompt                            # 파이썬 쪽 사본 -- load_state_dict 가 buffer 로 다시 맞춘다
            self.sample_steps = net.num_timesteps          # 공표값 1000 (config/general_finetune.json:52-59)
            self.eval_seed = int(eval_seed)
            # harness 의 val 은 배치 64 로 부른다.  역확산은 활성값을 통째로 들고 있으므로
            # H=256 에서 배치 64 는 곧바로 OOM 이다.  픽셀 예산으로 쪼갠다(H=64 -> 4장,
            # H=256/512 -> 1장).  청크 크기는 잡음이 뽑히는 순서를 정하므로 값이 아니라 상수로 둔다.
            self.eval_pix = int(eval_pix)
            self.val_sampler = str(val_sampler)
            self.base_lr = 1e-4
            self._opt = []                                  # make_opt 이 여기에 넣는다(리스트 = 미등록)
            # 학습 중 검증인가.  train_step 이 켜고 load_state_dict 가 끈다 (아래 sample 참고).
            self._val_crn = False

        def load_state_dict(self, state_dict, strict=True, **kw):
            """적재 = 학습 밖의 평가가 시작된다는 표시다.  학습기의 최종 평가(train_pancoll.py:1309),
            재평가 · warm 이어 돌리기(:973), exact 이어 돌리기(:851), dump_outputs · render_visual 이 모두
            이 길로 가중치를 올린다.  (1) 프롬프트를 체크포인트의 `prompt_code` 로 맞추고 (2) 검증용 고정
            시드(CRN)를 끈다 -- exact 이어 돌리기는 곧이어 train_step 이 다시 켠다."""
            r = super().load_state_dict(state_dict, strict=strict, **kw)
            self.prompt = _EMB[int(self.prompt_code)].upper()
            self._val_crn = False
            return r

        # 저장소는 forward 안에서 `.to("cuda")` 를 못 박아 두었다(Hyper_unet_general.py:602).
        # 그 "cuda" 는 현재 디폴트 디바이스로 해석되므로, 모델이 cuda:1 에 있으면 현재
        # 디바이스를 그 인덱스로 바꿔 준 채 호출해야 한다.  저장소를 고치지 않는 쪽을 택했다.
        def _ctx(self, x):
            if not x.is_cuda:
                raise RuntimeError(
                    "TMDiff 는 CUDA 전용이다: core/clip.py:23 과 Hyper_unet_general.py:562,602 가 "
                    "'cuda' 를 하드코딩한다.  CUDA_VISIBLE_DEVICES 로 카드를 고르라.")
            return torch.cuda.device(x.device)

        def _align(self, dev):
            """캐시된 프롬프트 임베딩은 buffer 가 아니라서 .to(device) 를 따라가지 않는다."""
            u = self.net.denoise_fn
            for k in _EMB:
                a = k + "_embeding"
                t = getattr(u, a)
                if t.device != dev:
                    setattr(u, a, t.to(dev))

        def denoise(self, x_noisy, t, pan, ms_up, prompt=None):
            """저장소의 학습/샘플링 호출과 같은 인자 순서: (x_t, t, PAN, MS, prompt).
            diffusion_general.py:368 과 :183.  'MS' 라는 이름은 저해상 ms 가 아니라
            업샘플된 lms 다(data/LRHR_dataset.py:127-133)."""
            self._align(ms_up.device)
            with self._ctx(ms_up):
                return self.net.denoise_fn(x_noisy, t, pan, ms_up, prompt or self.prompt)

        @torch.no_grad()
        def sample(self, ms_up, pan, prompt=None):
            """평가용 역확산.  결과는 (B,C,H,H) [0,1] 텐서 하나.

            **저장소 기본 경로를 쓰지 않는다.**  p_sample(:203-208) 은 p_mean_variance(:155-171)
            를 부르는데, 그 함수는 denoiser 출력을 predict_start_from_noise 에 '잡음'으로 넣는다.
            학습 손실은 x0(깨끗한 잔차) 회귀다(:351,:368-369).  완벽한 x0 denoiser 를 꽂고
            두 체인을 돌려 보면 x0 체인은 오차 0.00000, 잡음 체인은 0.745 -- 후자는 학습된
            모델과 무관한 결과를 낸다.  같은 파일의 형제 함수 p_mean_variance_xo(:174-190) 가
            x0 해석이고, 저자들의 DPM-Solver 래퍼도 model_type="x_start" 로 선언한다(:239).
            그래서 p_sample 의 잡음 해석만 형제 함수로 바꾼 것이고, 나머지(클램프 -> q_posterior
            -> 사후 분산 잡음 주입 -> res2img)는 p_sample_loop(:211-225) 그대로다.
            덤으로 p_sample_loop 은 궤적을 dim=0 으로 이어 붙이므로 배치 1 에서만 옳다
            (:221,:225).  여기서는 청크 단위로 배치를 그대로 돈다.

            **난수 (stochastic=True).**  두 경우로 나뉜다.
              - 학습 중 검증(train_step 이 `_val_crn` 을 켰고 적재가 아직 없다): 체크포인트 선택용.
                전역 RNG 를 fork 하고 호출마다 `eval_seed` 로 다시 뿌린다 -- 공통 난수(CRN)라 검증
                PSNR 의 차이가 표본 잡음이 아니라 가중치의 차이이고(`_plateau` 의 0.05 dB 판정도 같다),
                전역 RNG 를 건드리지 않으므로 학습 궤적은 검증 일정과 무관하다.
              - 그 밖(학습기의 최종 RR · FR · zero-shot · valid 기록, --eval_only, dump_outputs,
                render_visual): **전역 RNG 에서 차례로** 뽑는다.  PanFlowNet 과 같은 규칙이다 -- 표의 값은
                학습기 난수 흐름의 표본 하나이고, dump_outputs 의 seed-0 규칙(집합마다 한 번
                manual_seed(0), 한 장씩, 차례로 뽑기, dump_outputs.py:19-22)이 그대로 성립한다.  예전처럼
                호출마다 다시 뿌리면 bs 1 덤프에서 모든 장면이 같은 잡음을 받고 seed 0 은 무시된다.
            최종 평가는 언제나 1000-step x0 DDPM 이다.  `val_sampler` 는 검증에만 걸린다."""
            self._align(ms_up.device)
            net, prompt = self.net, (prompt or self.prompt)
            H, W = ms_up.shape[-2:]
            k = max(1, self.eval_pix // (H * W))            # 청크 크기(픽셀 예산)
            crn = bool(self._val_crn)
            sampler = self.val_sampler if crn else "ddpm"
            devs = [ms_up.device] if ms_up.is_cuda else []
            outs = []
            with self._ctx(ms_up), (torch.random.fork_rng(devices=devs) if crn
                                    else contextlib.nullcontext()):
                if crn:
                    torch.manual_seed(self.eval_seed)
                for s0 in range(0, ms_up.shape[0], k):
                    ms_c, pan_c = ms_up[s0:s0 + k], pan[s0:s0 + k]
                    x_in = {"PAN": pan_c, "MS": ms_c}
                    if sampler.startswith("dpm"):
                        # 저자들 자신의 30-step DPM-Solver++(x_start), diffusion_general.py:228-256.  기본값이
                        # 아니다 (build 의 val_sampler="ddpm").  주의: discrete 래퍼는 잡음 준위
                        # alphas_cumprod[j] 에 t 임베딩 j 를 넣는데 학습은 같은 준위에 j+1 을 넣는다(:97-98,
                        # :353-363) -- 한 칸 어긋나고, 클램프도 static 이 아니라 dynamic thresholding 이다.
                        outs.append(net.sample_by_dpmsolver(
                            {"Res": torch.zeros_like(ms_c), "MS": ms_c, "PAN": pan_c}, prompt))
                        continue
                    x = torch.randn_like(ms_c)
                    for i in reversed(range(self.sample_steps)):
                        mean, logvar = net.p_mean_variance_xo(
                            x, i, clip_denoised=True, x_in=x_in, prompt=prompt)
                        n = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
                        x = mean + n * (0.5 * logvar).exp()
                    outs.append(ms_c + x)                   # utils/util.py:135-137 res2img
            return torch.cat(outs, dim=0).clamp(0.0, 1.0)

    # ------------------------------------------------------------------ build
    def build(ms_ch=4, ratio=4):
        M = _imports()
        if not torch.cuda.is_available():
            raise RuntimeError("TMDiff: CUDA 필요 (core/clip.py:23 이 CLIP 타워를 'cuda' 에 고정)")
        # 공표 폭.  config/general_finetune.json:63-66 의 channel_multiplier.
        # 밴드 수 인자는 없다 -- 스펙트럼 축이 conv3d 의 depth 이고 채널 축은 1 이라
        # (to3D/to2D, Hyper_unet_general.py:120-127, 608-609, 636) 4밴드와 8밴드가
        # 완전히 같은 가중치 집합을 쓴다(파라미터 수 30,928,513 동일).
        unet = M["WavBEST"](channels=[32, 64, 128, 256]).cuda()
        emb = {k: getattr(unet, k + "_embeding").detach() for k in _EMB}
        del unet.clip_text_model                    # 얼린 123.06M 을 모듈 트리에서 제거
        # 저장소는 train phase 에서 netG 전체에 kaiming 초기화를 건다(networks.py:86-87).
        # nn.Linear 에만 적중한다(모델에 nn.Conv2d 가 하나도 없다) -- embed/embed2 와 각 블록의
        # Dense 변조층이 그 대상이고, 초기화도 레시피의 일부라 그대로 재현한다.  CLIP 을 먼저
        # 지웠으므로 얼린 타워를 되살려 덮어쓰는 저장소의 부작용만 빠진다(임베딩은 이미 캐시됨).
        M["init_weights"](unet, init_type="kaiming")
        for k, v in emb.items():
            setattr(unet, k + "_embeding", v)
        net = M["GeneralDiffusion"](denoise_fn=unet, loss_type="l1").cuda()
        net.set_loss("cuda")                        # nn.L1Loss  (diffusion_general.py:76-85)
        net.set_new_noise_schedule({"schedule": "cosine", "n_timestep": 1000}, "cuda")
        # 평가 상수는 여기 **글자**로 둔다 -- 값이 build 의 바이트코드 상수라 code_digest(registry.py
        # spec_code_digest)에 들어가고, 바꾸면 이전 결과가 게이트에서 낡음이 된다.  환경변수는 기록되지 않는다.
        return TMDiff(net, _prompt(ms_ch), eval_seed=20260919, eval_pix=4 * 64 * 64, val_sampler="ddpm")

    # ------------------------------------------------------------------ 옵티마이저
    def make_opt(model, o):
        """AdamW(lr=1e-4, wd=1e-4), "clip_text" 파라미터 제외 -- GeneralModel/model.py:25-31.
        (타워를 이미 지웠으므로 필터는 실제로 아무것도 걸러 내지 않지만, 저장소 규칙이라 남긴다.)
        학습률 스케줄은 아래 train_step 이 직접 건다: 저장소는 HuggingFace 의 linear 스케줄을
        **매 옵티마이저 스텝마다** 돌리는데(model.py:32,44) harness 의 sched 는 step/cosine 을
        에폭 단위로만 지원한다."""
        params = [p for n, p in model.named_parameters() if "clip_text" not in n]
        opt = torch.optim.AdamW(params, lr=o["lr"], betas=o.get("betas", (0.9, 0.999)),
                                weight_decay=o.get("weight_decay", 1e-4))
        model.base_lr = float(o["lr"])
        model._opt = [opt]                          # 리스트 = nn.Module 등록 회피
        return opt

    def _lr_factor(step, total, warmup=100):
        """transformers.get_scheduler("linear", warmup=100, num_training_steps=max_iter) 와
        같은 람다.  num_training_steps 는 저장소에서 max_iter 에 묶여 있으므로(model.py:32)
        우리가 짧게 돌리면 ctx["total"] 로 같이 줄어야 한다 -- 안 그러면 학습률이 끝까지
        안 내려간다."""
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))

    # ------------------------------------------------------------------ 학습 스텝
    def train_step(model, b, ctx, lossfn):
        """diffusion_general.py:349-370 (p_losses_dynamic) 을 그대로 옮긴 것.

            res   = gt - lms                       # [-1,1], 신경망이 맞히는 대상
            t     ~ U{1..1000}                     # 표본별
            a     = sqrt_alphas_cumprod_prev[t]
            x_t   = a*res + sqrt(1-a^2)*eps
            pred  = unet(x_t, t, pan, lms, prompt) # x0 예측(잡음 예측이 아니다)
            loss  = L1(res, pred)

        **항은 이것 하나뿐이다.**  보조 손실도, 가중치도, perceptual/spectral 항도 없다
        (loss_type "l1", config/general_finetune.json:69).  저장소 루프의 `.sum()`
        (model.py:41) 은 이미 스칼라인 mean-L1 에 걸리므로 계수 1.0, 무의미하다.
        grad clip 도 EMA 도 저장소 어디에도 없다 (harness 의 10.0 은 걸린다 -- diff (10))."""
        model._val_crn = True                            # 이 뒤의 검증은 학습 중 검증이다 (sample 참고)
        net = model.net
        lms, pan, gt = b["lms"], b["pan"], b["gt"]      # b["ms"](저해상)는 쓰이지 않는다
        res = gt - lms
        B = res.shape[0]
        # 저장소는 np.random.randint 를 쓴다(:353).  분포가 같은 torch.randint 로 바꿔
        # harness 의 torch 시드만으로 재현되게 했다 -- 이것이 학습 경로의 유일한 변경이다.
        t = torch.randint(1, net.num_timesteps + 1, (B,))
        a = torch.as_tensor(net.sqrt_alphas_cumprod_prev[t.numpy()],
                            dtype=torch.float32, device=res.device).view(B, 1)
        x_noisy = net.q_sample(res, a.view(-1, 1, 1, 1))
        pred = model.denoise(x_noisy, t.to(res.device).view(B, 1), pan, lms)
        loss = net.loss_func(res, pred)                  # nn.L1Loss, mean

        # 저장소의 학습률 스케줄(선형 warmup 100 -> 선형 감쇠 0)을 옵티마이저 스텝마다 건다.
        # HF LambdaLR 은 k 번째 스텝에 lambda(k-1) 을 쓰므로 ctx["step"]-1 을 넘긴다.
        if model._opt:
            lr = model.base_lr * _lr_factor(int(ctx["step"]) - 1, int(ctx["total"]), 100)
            for g in model._opt[0].param_groups:
                g["lr"] = lr
        elif "nosched" not in _WARN:
            _WARN.add("nosched")
            print("[TMDiff] make_opt 을 쓰지 않는 harness다 -- 저장소의 linear 학습률 스케줄이 "
                  "적용되지 않는다(lr 고정).", flush=True)
        return lms + pred, loss                          # 반환 예측은 1-step x0 추정(샘플러 출력 아님)

    SPECS["TMDiff"] = dict(
        tier="confirmed",   # (2026-09-28): 스모크 통과, 표에 올린다
        build=build, venue="TGRS 2024", loss="l1", batch=4, epochs=_EPOCHS["wv3"],
        # 데이터셋별 epoch (위 _EPOCHS 산식).  디스패처 줄은 --epochs 를 넘기지 못하므로(_logs/dispatch.py:395-405)
        # 여기 없으면 qb/gf2 가 wv3 의 62 epoch = 265,670 / 307,086 step 을 돈다 (저장소의 1.77x / 2.05x).
        per_dataset={d: dict(epochs=e) for d, e in _EPOCHS.items()},
        # 추론이 가우시안 잡음에서 출발한다 -- 표의 값은 표본 하나다.  dump_outputs · render_visual 이 읽는다
        # (bs 1, 집합마다 seed 0 한 번, 차례로 뽑기).  sample() 의 두 난수 경로 참고.
        stochastic=True,
        eval_after_frac=_EVAL_AFTER,
        # kind 는 make_opt 가 짓는 클래스와 **같아야 한다** (tests/test_optimizer_kind.py).  예전의
        # "adam" 은 kind 를 SGD 로 떨어뜨리던 옛 train_baseline 을 위한 안전값이었는데, 하네스가 이제
        # kind 를 지키므로(`lib/harness.make_optimizer`) 적힌 그대로 AdamW 다.  이 방법은 결과가 없다.
        opt=dict(kind="adamw", lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4),
        make_opt=make_opt,                               # 진짜 옵티마이저는 AdamW 다
        val_cap=16,                                      # 1000-step 샘플러 -- diff (12) 비용
        forward=lambda mm, ms_up, pan, lrms, ctx: mm.sample(ms_up, pan),
        train_step=train_step,
        diff=(
            "저장소 코드 수정 없음(감사에서 이미 적용된 한 줄 제외). "
            "(1) config/sample_config.py:30 의 죽은 절대경로 CLIP 체크포인트를 저장소 자신의 "
            "기본값 openai/clip-vit-large-patch14 로 되돌렸다(core/clip.py:23, readme 1단계와 동일). "
            "이게 없으면 WavBEST.__init__ 이 OSError 로 죽는다. 얼린 타워라 기여와 무관하다. "
            "(2) **평가 샘플러**: 저장소 기본 경로 p_sample->p_mean_variance(:155-171)는 denoiser "
            "출력을 predict_start_from_noise 에 '잡음'으로 넣는데, 학습 손실은 x0(깨끗한 잔차) "
            "회귀다(:351,368-369). 완벽한 x0 denoiser 로 두 체인을 돌리면 x0 체인 오차 0.00000, "
            "잡음 체인 0.745 -- 후자는 학습한 모델을 평가하지 않는다. 그래서 같은 파일의 형제 함수 "
            "p_mean_variance_xo(:174-190)를 쓴다(저자들의 DPM-Solver 래퍼도 model_type='x_start', "
            ":239). 그 외 체인(클램프->q_posterior->사후분산 잡음->res2img)은 p_sample_loop 그대로이고, "
            "배치 1 에서만 옳은 궤적 concat(:221,225)만 피했다. super_resolution(:339)이 prompt 를 "
            "continous 슬롯에 넘기는 버그도 이 경로를 쓰지 않으므로 함께 우회된다. 표의 모든 수는 "
            "1000-step x0 DDPM 이다(저자들의 30-step DPM-Solver++ 는 쓰지 않는다). "
            "(3) 얼린 CLIP 텍스트 타워(123,060,480 params)는 문장 5개 임베딩을 캐시한 뒤 삭제한다 "
            "-- 저장소는 옵티마이저 param 필터로 같은 일을 한다(model.py:25-27). 임베딩은 파라미터도 "
            "buffer 도 아니라 state_dict 에 없고, 재빌드 때 같은 CLIP 가중치에서 동일하게 재생성된다. "
            "state_dict 에는 학습한 프롬프트 번호 `prompt_code` 하나만 더했다(파라미터 아님). "
            "(4) 학습 t 추출을 np.random.randint -> torch.randint 로(분포 동일, harness 시드로 재현). "
            "(5) 옵티마이저는 AdamW(1e-4, wd 1e-4) + HF linear(warmup 100 -> 0) 스케줄을 매 스텝 "
            "재현한다(make_opt + train_step). harness 의 sched 는 에폭 단위 step/cosine 뿐이라 "
            "직접 걸었다. opt['kind'] 는 make_opt 과 같은 'adamw' 다 -- 두 하네스(train_pancoll · "
            "train_baseline)가 모두 lib/harness.make_optimizer 로 make_opt 를 쓴다. "
            "(6) **일정 해석**: 저장소의 150,000 iteration(config/general_finetune.json:76, "
            "script:161)을 데이터셋별 가장 가까운 정수 epoch 으로 옮겼다(per_dataset): wv3 62 ep = "
            "150,598 / qb 35 ep = 149,975 / gf2 30 ep = 148,590 step. 저장소의 150k 는 QB+GF2+WV3 "
            "**합동** 학습의 총량이고(위성 추출 확률 ∝ 밴드 x 배치 수, script:156-162 -> 위성당 약 46k-53k), "
            "배포된 config 는 비공개 I100000 에서 이어 도는 phase 'val' 파일이다(json:4,14; model.py:136-140 "
            "은 옵티마이저 · 스케줄러를 되살리지 않는다). 공개된 처음부터의 레시피가 없으므로 같은 폭 · 배치 · "
            "lr · 스케줄 · 150k 를 데이터셋마다 처음부터 돌린다. 저장소에는 epoch 개념이 없다. "
            "(7) **난수**: 출력이 확률적이다(stochastic=True). 학습 중 검증만 호출마다 고정 시드 "
            "20260919 로 공통 난수를 쓰고(체크포인트 선택, 전역 RNG 를 건드리지 않음), 최종 평가 · 재평가 · "
            "덤프 · 그림은 전역 RNG 에서 차례로 뽑는다 -- 표의 값은 표본 하나다(PanFlowNet 과 같은 규칙). "
            "공개 영상은 dump_outputs 의 seed-0 규칙(bs 1)을 따른다. 샘플러 · 시드 · 청크 크기를 바꾸던 "
            "환경변수는 뺐다(기록되지 않는 값이 표를 바꾸면 안 된다). "
            "(8) 프롬프트 문자열은 학습 때 --dataset 에서 고르고(없으면 밴드 수: 4->QB, 8->WV3), "
            "체크포인트의 prompt_code 가 적재 때 이긴다 -- 그래서 --dataset 을 모르는 도구도 GF2 가중치를 "
            "GF2 문장으로 평가한다. WV2 zero-shot 은 학습 문장(WV3)을 쓴다: 저장소의 'WV2' 문장은 "
            "GaoFen-2 · 4밴드를 서술하는 복사본이다(Hyper_unet_general.py:581-582). **주의**: 논문 설정은 "
            "합동 학습이고 텍스트 변조가 위성을 구분하는 장치다. 데이터셋별로 따로 학습하면 프롬프트가 "
            "상수가 되어 embed2 가 고정 스타일 벡터로 축퇴한다 -- 구조는 그대로지만 기여는 작동하지 않는다. "
            "표에 그 각주가 필요하다. "
            "(9) H 는 8의 배수여야 한다(Haar DWT 3단). 64/256/512 모두 충족. "
            "(10) harness 기본 grad clip 10.0 이 걸린다(저장소에는 clip 이 없다, 사용자 결정 2026-09-28). "
            "초기 grad norm 은 0.07-0.22 로 훨씬 작다 -- 걸렸는지는 결과 JSON 의 grad_clip_hits 가 답한다. "
            "(11) 파라미터 텐서 272개 중 56개(233,857개 = 0.76%)는 저장소 구조상 forward 에 "
            "쓰이지 않아 grad 가 없다(conv21/Conv_1/dense2 등). 저장소 UNet 을 직접 불러도 "
            "똑같이 56개라 우리 배선 실수가 아니라 저장소 자체 성질이다 -- 검증 스크립트가 확인한다. "
            "(12) **검증 · 비용**: 검증은 일정의 마지막 25% 에서만 한다(eval_after_frac 0.75). 저장소는 "
            "max_iter 까지 무조건 돌고 lr 이 그 지점에서 0 이 되므로(script:161, model.py:32), 그 전에 "
            "patience(40,000 step)가 멈추면 감쇠 끝이 잘린다 -- 0.75 에서는 셋 다 patience 정지가 "
            "불가능하고, 후보에 저장소의 끝점(마지막 step 검증)이 언제나 들어간다. 검증은 valid 앞 16장 "
            "(val_cap), 1000-step. 샘플러는 픽셀 예산으로 청크를 쪼갠다(H=64 -> 4장, H>=256 -> 1장; "
            "harness 의 val 배치 64 를 그대로 받으면 OOM). 비용(4090, 1000 스텝, 측정값): H=64 8밴드 "
            "8-10s/장(부하 시 26s), H=64 4밴드 부하 시 18s, H=256 8밴드 116-140s/장, H=512 8밴드 "
            "denoiser 0.7s/회 -> 약 700s/장. 학습은 batch 4 에서 0.062s/step(4밴드)·0.100s/step(8밴드). "
            "wv3 한 칸(유휴 카드): 학습 4.2h + 검증 20회 0.8h + 최종 RR 0.7h · FR 2.6-3.9h · WV2 0.7h · "
            "valid 전수 1,080장 2.4-3.0h(학습기의 선택 집합 기록, val_cap 이 걸리지 않는다) = 약 11-14h. "
            "qb(1,905장) · gf2(2,201장)는 학습 2.6h + 검증 0.5h + RR 0.4-0.6h + FR 1.6-2.7h + valid 전수 "
            "2.9-4.3h = 약 8-11h (4밴드 H>=256 은 8밴드의 0.6-0.7배로 어림, 미측정). 공유 카드는 1.5-3배. "
            "병목은 학습이 아니라 1000-step 평가다."
        ),
    )
