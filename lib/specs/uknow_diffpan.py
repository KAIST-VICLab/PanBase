"""U-Know-DiffPAN(CVPR 2025): PriorNet -> 불확실성 teacher 확산 -> teacher 출력 export -> KD student 확산의 4단계 파이프라인.  틀리기 쉬운 곳은 세 군데다 -- (1) 학생 손실은 확산이 돌려주는 L1 이 **아니라** teacher variance 로 가중한 hard/soft/feature **3항** KD 이고(저장소도 L1 은 "실제 학습에 쓰지 않음"이라 주석해 두었다), (2) ddim 샘플링이 diffusion 객체의 beta 버퍼를 **제자리에서** 갈아엎어 학습 스케줄을 파괴한다, (3) stage 1.5 export 는 **EMA teacher** 로 학습 split 을 한 번 샘플링한 고정 표적이다 -- 매 스텝 다시 뽑으면 한 칸이 약 180 GPU-시간이 된다."""
import bisect
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# 저장소의 반복 예산: PriorNet 300k -> FSA-T 400k -> (export) -> FSA-S 300k = 1,000,000 step
# (config/Prior_config.py:26, config/FSA_T_config.py:32, config/FSA_S_config.py:31).
# 한 번의 harness 실행 안에서 같은 비율로 단계를 나눈다.
_P0, _P1 = 0.30, 0.70          # 0~30% PriorNet, 30~70% teacher, 70~100% student
# EmaUpdater(decay=0.995, start_iter=20_000): S1_FSA_T_train.py:201-203 / S2_FSA_S_train.py:203-205.
# update(iterations) 는 opt.step() 직후에 **0부터 센** 반복 번호로 불리고(S1:265-270, S2:262-265)
# `iteration > start_iter` 일 때만 섞는다(utils/optim_utils.py:49-63) -- 끝난 스텝 수 c 로 쓰면 c-1 > 20000.
_EMA_DECAY, _EMA_START = 0.995, 20_000
# MultiStepLR(milestones=[100k, 200k, 350k], gamma=0.5), 반복마다 step (S1:206-209,268 / S2:207-210,264).
# 단계마다 새 옵티마이저라 반복 번호는 **단계 안에서** 센다.  PriorNet 은 매 에폭 스케줄러를 새로 만들어
# (prior_train.py:107-112) milestone 이 영영 안 걸린다 -- 상수 1e-4 가 저장소의 실제 동작이다.
_MS, _GAMMA = (100_000, 200_000, 350_000), 0.5
_DDIM = "ddim25"               # 저장소의 export/추론 샘플링 (S1_FSA_T_pretrain_save.py:117)
# 데이터셋별 epochs: 저장소 예산은 반복 수(1M)이고 데이터셋과 무관하다.  steps = epochs x ceil(N/32).
#   wv3 N=9,714  -> 304 step/epoch x 3289 =   999,856
#   qb  N=17,139 -> 536 step/epoch x 1866 = 1,000,176
#   gf2 N=19,809 -> 620 step/epoch x 1613 = 1,000,060
_EPOCHS = {"wv3": 3289, "qb": 1866, "gf2": 1613}
# export 캐시의 저장 dtype.  저장소 h5 는 float32 다(S1_FSA_T_pretrain_save.py:126-138 이 fp32 텐서를
# numpy 로 옮겨 그대로 저장).  fp16 으로 줄이면 호스트 RAM 이 절반이지만 저장소와 달라진다.
_CACHE_DTYPE = torch.float32
# stage 0/1 을 도는 프로세스는 첫 스텝에서 stage 1 최고치만큼 GPU 메모리를 할당자 캐시에 미리 잡아 둔다.
# stage 0 은 1.1 GiB 만 써서, 디스패처가 그 45 분 동안 이 카드를 비었다고 보고 옆에 작업을 붙이면 stage 1
# 진입 순간 OOM 이 난다.  student 첫 스텝에서 놓는다.  GiB = a + b*batch -- stage 1 실측 최고치(4090, H=64):
# B=2 1.53 / 4 2.48 / 8 4.37 / 16 8.15 / 32 15.72~15.81 GiB (verify 로그).  (0, 0) 이면 끈다.
_RESERVE = (0.6, 0.49)

_M = {}                        # 저장소 모듈 캐시


def _n_before(total, frac):
    """1..total 중 step/total < frac 인 step 의 개수 -- train_step 의 단계 판정(`epoch_frac < frac`)과
    **같은 부동소수 식**으로 센다(경계에서 int(total*frac) 가 한 칸 어긋날 수 있다)."""
    k = int(total * frac)
    while k > 0 and not (k / total < frac):
        k -= 1
    while k + 1 <= total and (k + 1) / total < frac:
        k += 1
    return k


def _lr_factor(it):
    """단계 안 1부터 센 반복 it 에 쓰이는 MultiStepLR 배율.  it 번째 opt.step() 앞에는 scheduler.step()
    이 it-1 번 불렸다 -> gamma ** (#milestone <= it-1)."""
    return _GAMMA ** bisect.bisect_right(_MS, it - 1)


def register(SPECS, REPOS, _load, _pkg):
    ROOT = REPOS / "U-Know-DiffPAN"

    def _mods():
        """저장소 모듈을 적재한다.

        models/ · diffusion/ · utils/ 에 __init__.py 가 없고(네임스페이스 패키지)
        diffusion/*.py 가 `from utils.loss_utils import ...` 로 **절대 임포트**를 한다.
        `utils` 는 다른 저장소도 쓰는 흔한 이름이라 sys.path 로는 가려진다 -- _pkg 로
        패키지 객체를 직접 박아야 한다."""
        if _M:
            return _M
        _pkg(ROOT / "utils", "utils")
        _pkg(ROOT / "models", "models")
        _pkg(ROOT / "diffusion", "diffusion")
        _M["prior"] = _load(ROOT / "models" / "priornet" / "pannet_variance.py", "uknow_prior")
        _M["fsa_t"] = _load(ROOT / "models" / "FSA_T.py", "uknow_fsa_t")
        _M["fsa_s"] = _load(ROOT / "models" / "FSA_S.py", "uknow_fsa_s")
        _M["d_t"] = _load(ROOT / "diffusion" / "diffusion_ddpm_pan.py", "uknow_diff_teacher")
        _M["d_s"] = _load(ROOT / "diffusion" / "diffusion_ddpm_pan_for_student.py", "uknow_diff_student")
        _M["d_d"] = _load(ROOT / "diffusion" / "diffusion_ddpm_pan_for_dist_feature.py", "uknow_diff_dist")
        _M["loss"] = _load(ROOT / "utils" / "loss_utils.py", "uknow_loss_utils")
        # 25 step 짜리 샘플링이 tqdm 막대로 로그를 덮는다.  연산은 그대로다.
        for k in ("d_t", "d_s", "d_d"):
            _M[k].tqdm = lambda it, **kw: it
        return _M

    # ------------------------------------------------------------------ 웨이블릿 조건
    def _swt1(x):
        """db1(Haar) level-1 **정지**(undecimated) 웨이블릿.

        저장소는 pywt.swt2(x, "db1", level=1) 를 원본 h5 배열에 걸고 나서 division 으로
        나눈다.  SWT 는 선형이라 [0,1] 텐서에 바로 걸어도 수치가 같다(감사 로그: 최대
        차이 6.7e-16).  pywt 와의 정렬/부호는 실측으로 맞췄다 -- 순환 패딩 후 2x2 Haar,
        cA=.5(a+b+c+d) / cH=.5(a+b-c-d) / cV=.5(a-b+c-d) / cD=.5(a-b-c+d) 가
        pywt 출력과 1e-16 안에서 일치한다(pywt 는 norm=False 라 cA 가 대략 [0,2] 다)."""
        p = F.pad(x, (0, 1, 0, 1), mode="circular")
        a, b = p[..., :-1, :-1], p[..., :-1, 1:]
        c, d = p[..., 1:, :-1], p[..., 1:, 1:]
        return (0.5 * (a + b + c + d), 0.5 * (a + b - c - d),
                0.5 * (a - b + c - d), 0.5 * (a - b - c + d))

    def _wavelets(lms, pan):
        """cond['wavelets'] = cat[lms_cA, pan_cH, pan_cD, pan_cV]  -> (B, C+3, H, W).

        저장소는 `_, (h, v, d) = pywt.swt2(pan,...)` 로 풀어 놓고 concat 은 [h, d, v] 로
        한다(dataset/pan_dataset.py:74-79,150-153).  즉 두 detail 판의 순서가 저장소
        안에서 어긋나 있는데, 그 순서 그대로 재현한다 -- 고치면 학습된 조건 배치가 달라진다."""
        ca, _, _, _ = _swt1(lms)
        _, ph, pv, pd = _swt1(pan)
        return torch.cat([ca, ph, pd, pv], dim=1)

    # ------------------------------------------------------------------ 모델
    class UKnow(nn.Module):
        """세 단계를 한 그루의 모듈 트리에 담는다.

        옵티마이저가 보는 것: prior(450,696) + teacher 디노이저(25.46M) + student
        디노이저(9.71M).  **평가에 쓰이는 것은 student 뿐이고**(teacher/prior 는 학습
        시간 전용) 표에 실려야 할 크기도 student 의 9.71M 이다 -- registry.n_params 는
        셋을 합쳐 약 35.6M 을 찍는다.

        EMA 사본 둘(teacher_ema, student_ema)은 requires_grad=False 다.  옵티마이저는 grad 가
        None 이라 건드리지 못하고 n_params 도 세지 않는다.  그러면서 등록된 서브모듈로 남겨
        state.pt/best.pt 에 실리게 했다 -- 평가는 student EMA 로, export 는 teacher EMA 로 돈다.

        export 캐시(`_uk["store"]`)는 **트리 밖** 호스트 메모리에 둔다(wv3 약 14 GB, qb/gf2
        약 23/27 GB, fp32).  state_dict 에 들어가면 체크포인트마다 그만큼 커진다."""

        def __init__(self, C, ratio=4):
            super().__init__()
            M = _mods()
            self.C = C
            betas = M["d_t"].make_beta_schedule(schedule="cosine", n_timestep=500, cosine_s=8e-3)
            self._betas = betas

            self.prior = M["prior"].PanNet_variance(spectral_num=C)

            def unet(mod):
                # S1_FSA_T_train.py:104-116 / S2_FSA_S_train.py:103-115 그대로.
                # image_size=64 는 어느 깊이에 self-attention 을 둘지만 고르고(now_res in
                # attn_res) 파라미터도 forward 도 만들지 않는다 -- H=64/256/512 에서 동작
                # 확인됨.  유일한 제약은 H 가 8 의 배수일 것(stride-2 다운샘플 3회).
                return mod.UNetSR3(in_channel=C, out_channel=C, lms_channel=C, pan_channel=1,
                                   inner_channel=32, norm_groups=1, channel_mults=(1, 2, 2, 4),
                                   attn_res=(8,), dropout=0.2, image_size=64, self_condition=True)

            t_unet, s_unet = unet(M["fsa_t"]), unet(M["fsa_s"])

            def wrap(cls, net, loss_type):
                d = cls(net, image_size=64, channels=C, pred_mode="x_start",
                        loss_type=loss_type, device="cpu", clamp_range=(0, 1))
                d.set_new_noise_schedule(betas=betas)
                return d

            self.teacher_diff = wrap(M["d_t"].GaussianDiffusion, t_unet, "beta_nll")
            # loss_type='l1' 은 저장소가 직접 "실제 학습에서는 사용하지 않은 부분"이라
            # 적어 둔 죽은 코드다(S2_FSA_S_train.py:180).  진짜 손실은 KD_Loss.
            self.student_diff = wrap(M["d_s"].GaussianDiffusion, s_unet, "l1")
            # 저장소는 teacher 에도 EMA 를 두고(S1:201-203) export 는 그 EMA 가중치로 한다
            # (config/FSA_T_config.py:53 'ema_diffusion_...', S1_FSA_T_pretrain_save.py:218).
            self.teacher_ema = copy.deepcopy(t_unet).requires_grad_(False)
            self.student_ema = copy.deepcopy(s_unet).requires_grad_(False)

            # --- 트리 밖 샘플러 두 개 ---------------------------------------
            # ddim_sample_loop 는 space_new_betas 로 자기 buffer(betas/alphas_cumprod/...)를
            # **제자리에서** 500 -> 25 로 갈아엎는다.  학습용 래퍼로 샘플링하면 그 순간
            # 학습 스케줄이 죽고 state_dict 모양까지 바뀐다.  그래서 샘플링 전용 래퍼를
            # 따로 두고(디노이저는 EMA 사본을 공유) 트리 밖 리스트에 담아 state_dict 오염을 막는다.
            self._samplers = [wrap(M["d_d"].GaussianDiffusion, self.teacher_ema, "l1"),   # export
                              wrap(M["d_s"].GaussianDiffusion, self.student_ema, "l1")]   # 평가
            self._kd = M["loss"].KD_Loss()
            self._bnll = M["loss"].Beta_nll_Loss()
            # 트리 밖 상태 (state_dict 에 없다).  opt/lr0 은 make_opt 가, n_train/seed 는 on_train_start 가 채운다.
            self._uk = dict(opt=None, lr0=None, n_train=None, seed=0, store=None, filled=None,
                            layout=None, n_filled=0, pin=None, first_call=True)

        # -------------------------------------------------------------- 도구
        def _prep(self, dev):
            for s in self._samplers:
                if s.betas.device != dev:
                    s.to(dev)

        def _reset(self, d, dev):
            """샘플링 직전마다 500-step 스케줄을 되돌린다.  space_new_betas 는 현재
            alphas_cumprod 를 다시 잘라 쓰므로 두 번 부르면 25 를 또 잘라 망가진다."""
            d.set_new_noise_schedule(betas=self._betas, device=dev)

        def diffusion_params(self):
            return list(self.teacher_diff.model.parameters()) + list(self.student_diff.model.parameters())

        @staticmethod
        @torch.no_grad()
        def _ema(src, dst, completed):
            """EmaUpdater.update(completed-1) 와 같은 갱신 (utils/optim_utils.py:49-63).
            completed = 그 단계에서 끝난 opt.step() 수.  섞기는 p_ema*decay + p*(1-decay)."""
            ps = [p.detach() for p in src.parameters()]
            es = list(dst.parameters())
            if completed - 1 > _EMA_START:
                torch._foreach_mul_(es, _EMA_DECAY)
                torch._foreach_add_(es, ps, alpha=1.0 - _EMA_DECAY)
            else:
                torch._foreach_copy_(es, ps)

        def _set_lr(self, group, it):
            """make_opt 가 만든 옵티마이저의 한 그룹 lr 을 단계 안 반복 it 의 값으로 둔다.
            매 스텝 step 에서 다시 계산하므로 exact/warm 이어 돌리기 어느 쪽에서도 같다."""
            opt, lr0 = self._uk["opt"], self._uk["lr0"]
            if opt is None:
                return
            opt.param_groups[group]["lr"] = lr0 * (1.0 if group == 0 else _lr_factor(it))

        def _reserve(self, dev, frac, batch):
            """이 프로세스의 첫 학습 스텝에서 한 번.  연산에는 영향이 없다 (빈 텐서를 잡았다 놓을 뿐)."""
            if not self._uk["first_call"]:
                return
            self._uk["first_call"] = False
            gib = _RESERVE[0] + _RESERVE[1] * batch
            if dev.type != "cuda" or frac >= _P1 or gib <= 0:
                return
            try:
                x = torch.empty(int(gib * 2 ** 30), dtype=torch.uint8, device=dev)
                del x                       # 할당자 캐시에 남는다 (nvidia-smi 에는 쓰는 것으로 보인다)
                print(f"[U-Know] stage 1 최고치 {gib:.1f} GiB 를 할당자 캐시에 잡아 두었다 (student 첫 스텝에서 놓는다)",
                      flush=True)
            except torch.OutOfMemoryError:
                print(f"[U-Know] stage 1 용 {gib:.1f} GiB 를 미리 잡지 못했다 -- 그대로 간다", flush=True)

        @staticmethod
        def _flip(hf, vf, *ts):
            """stage 2 증강: 항목별 H/V 뒤집기(p=0.5), 모든 텐서에 **같은** 변환.

            저장소는 pan/ms/lms/gt/sr/var_map + teacher feature 6장에 같은 seed 로
            건다(pan_dataset_for_dist.py:196-197,247-262): RandomHorizontalFlip(마지막 축) 다음
            RandomVerticalFlip(끝에서 둘째 축).  export 는 뒤집기 **전** 표본으로 만들어 두고
            여기서 입력과 함께 뒤집는다 -- 저장소와 같다.
            stage 0/1 에는 증강이 없다 -- 저장소의 aug 호출이 주석 처리돼 있어
            aug_prob=0.5 가 죽은 인자다(pan_dataset.py:244-245)."""
            out = []
            for t in ts:
                t = torch.where(hf.view(-1, 1, 1, 1), t.flip(-1), t)
                out.append(torch.where(vf.view(-1, 1, 1, 1), t.flip(-2), t))
            return out

        # -------------------------------------------------------------- stage 1.5: export
        @staticmethod
        @torch.no_grad()
        def _ddim_from(d, cond, img):
            """diffusion_ddpm_pan_for_dist_feature.py:652-713 의 조건부 분기 그대로, 초기 잡음만 받는다.
            (그 함수는 x_T 를 전역 난수로 뽑는다.)  self_cond 는 저장소에서도 늘 None 이다 --
            pred_x_start 가 루프 안에서 갱신되지 않는다.  eta=0 이라 초기 잡음 외에는 결정적이다."""
            d.space_new_betas(d.space_timesteps(d.num_timesteps, _DDIM))
            b = img.shape[0]
            dev = cond["lms"].device
            variance = feats = None
            for i in reversed(range(0, len(d.betas))):
                img, variance, feats = d.ddim_sample(
                    img, torch.full((b,), i, device=dev, dtype=torch.long),
                    condition_x=cond, self_cond=None, eta=0.0)
            return img, variance, feats

        def _noise(self, idx_cpu, shape, dev):
            """표본 인덱스마다 고정된 초기 잡음 x_T.  저장소 export 는 표본마다 잡음을 한 번 뽑아
            h5 에 고정한다 -- 여기서는 (run seed, 표본 인덱스)로 정해 두어, 캐시가 프로세스와 함께
            사라져도(이어 돌리기) **같은** export 를 다시 만든다."""
            # CPU 생성기(mt19937)는 씨앗의 **하위 32비트만** 쓴다 (seed 3 과 999*2**32+3 이 같은 흐름).
            # 그래서 32비트 안에서 섞는다: 한 런 안에서는 인덱스마다 다른 씨앗(i 의 계수 1, 충돌 없음)이고
            # run seed 가 다르면 홀수 곱으로 전체가 옮겨진다.
            base = int(self._uk["seed"]) * 1_000_003
            zs = [torch.randn(shape, generator=torch.Generator().manual_seed((base + int(i)) % 2 ** 32))
                  for i in idx_cpu.tolist()]
            return torch.stack(zs).to(dev)

        @torch.no_grad()
        def _export(self, lms, pan, idx_cpu):
            """S1_FSA_T_pretrain_save.py:98-138 -- EMA teacher 로 ddim25 한 번.  (sr, var_map, feat 6장).
            전역 난수를 쓰지 않는다(fork_rng): ddim_sample 은 eta=0 이어도 매 스텝 randn 을 소비하는데,
            그 소비가 '처음 보는 표본이 있는 스텝'에만 생기면 이어 돌린 런과 끊김 없는 런의 난수 흐름이
            갈라진다."""
            dev = lms.device
            self._prep(dev)
            d = self._samplers[0]
            self.prior.eval(); self.teacher_ema.eval()
            x_t = self._noise(idx_cpu, tuple(lms.shape[1:]), dev)
            devs = [dev.index if dev.index is not None else torch.cuda.current_device()] \
                if dev.type == "cuda" else []
            with torch.random.fork_rng(devices=devs):
                pre = self.prior(lms, pan)
                cond = {"lms": lms, "pan": pan,
                        "pre_out": pre["out"], "pre_variance": pre["variance"],
                        "wavelets": _wavelets(lms, pan)}
                self._reset(d, dev)
                sr_res, var_map, feats = self._ddim_from(d, cond, x_t)
                self._reset(d, dev)
            sr = (sr_res + lms).clip(0, 1)                   # S1_FSA_T_pretrain_save.py:119-120,129-130
            # h5 의 dist_feat0..5 = dist_feature_map[1..6] (:132-137).  [0] 은 3C+1 채널 입력 concat.
            # var_map 은 저장소에서도 division 으로 나누지 않는다(pan_dataset_for_dist.py:178).
            return sr, var_map, list(feats[1:7])

        def _layout_of(self, parts):
            return [tuple(p.shape[1:]) for p in parts]

        def _teacher_targets(self, lms, pan, idx):
            """학습 표본 인덱스로 export 를 찾는다.  처음 보는 인덱스만 그 자리에서 export 해 채운다.

            저장소는 학습 split 전체를 stage 2 전에 한 번 export 한다.  teacher 는 stage 2 내내
            얼어 있고 잡음은 표본마다 고정이므로, '처음 뽑힐 때 채우기'는 한 번에 미리 채우는 것과
            **같은 표적**을 준다(계산 시점만 다르다).  표본 추출이 복원추출이라 N 표본을 다 채우는 데
            wv3 약 2.8k / qb 5.6k / gf2 6.9k 스텝이 걸리고, export 가 도는 스텝은 그중 1.2k/2.2k/2.6k 다."""
            U = self._uk
            N = U["n_train"]
            assert N is not None, "on_train_start 가 불리지 않았다 -- export 캐시 크기를 모른다"
            idx_cpu = idx.detach().to("cpu", torch.long).reshape(-1)
            assert int(idx_cpu.min()) >= 0 and int(idx_cpu.max()) < N, (int(idx_cpu.max()), N)
            if U["filled"] is None:
                U["filled"] = torch.zeros(N, dtype=torch.bool)
            miss = ~U["filled"][idx_cpu]
            if bool(miss.any()):
                first = {}
                for j, i in enumerate(idx_cpu.tolist()):
                    if not bool(U["filled"][i]) and i not in first:
                        first[i] = j
                ids = torch.tensor(list(first.keys()), dtype=torch.long)
                js = torch.tensor(list(first.values()), dtype=torch.long, device=lms.device)
                parts = [*self._export(lms[js], pan[js], ids)]
                parts = [parts[0], parts[1], *parts[2]]
                lay = self._layout_of(parts)
                if U["store"] is None:
                    U["layout"] = lay
                    D = sum(int(torch.Size(s).numel()) for s in lay)
                    U["store"] = torch.empty((N, D), dtype=_CACHE_DTYPE)
                    print(f"[U-Know] export 캐시: {N:,} 표본 x {D:,} 값 = "
                          f"{N * D * U['store'].element_size() / 2 ** 30:.2f} GiB (호스트, {_CACHE_DTYPE})",
                          flush=True)
                assert lay == U["layout"], (lay, U["layout"])
                rows = torch.cat([p.reshape(p.shape[0], -1) for p in parts], dim=1)
                U["store"][ids] = rows.to("cpu", _CACHE_DTYPE)
                U["filled"][ids] = True
                U["n_filled"] += len(first)
            # 읽기: 방금 채운 표본도 저장소에서 읽는다 -- 채운 스텝과 이후 스텝이 같은 값을 본다.
            dev = lms.device
            B = idx_cpu.shape[0]
            if dev.type == "cuda":
                pin = U["pin"]
                if pin is None or pin.shape[0] < B:
                    pin = U["pin"] = torch.empty((B, U["store"].shape[1]), dtype=_CACHE_DTYPE,
                                                 pin_memory=True)
                torch.index_select(U["store"], 0, idx_cpu, out=pin[:B])
                rows = pin[:B].to(dev, dtype=torch.float32)       # 동기 복사 -- pin 을 다음 스텝에 다시 써도 안전
            else:
                rows = U["store"].index_select(0, idx_cpu).to(torch.float32)
            out, o = [], 0
            for s in U["layout"]:
                n = int(torch.Size(s).numel())
                out.append(rows[:, o:o + n].reshape(B, *s))
                o += n
            return out[0], out[1], out[2:]

        # -------------------------------------------------------------- 단계별 스텝
        def prior_step(self, b):
            """stage 0 -- prior_train.py:120-123.  beta=0 의 beta-NLL.  lr 상수(위 _MS 주석)."""
            self._set_lr(0, 1)
            o = self.prior(b["lms"], b["pan"])
            loss = self._bnll(b["gt"], o["out"], o["variance"], 0.0)
            return o["out"], loss

        def teacher_step(self, b, ctx):
            """stage 1 -- S1_FSA_T_train.py:225-244.  잔차 gt-lms 에 대한 beta=0.5 beta-NLL.

            p2_loss_weight_gamma=0 이라 p2 가중은 1, pred_var=False 라 decoder_nll 은
            0 이다.  즉 유효 손실은 beta-NLL 항 하나뿐이지만, 저장소 p_losses 를 그대로
            호출하므로 self-conditioning(확률 0.5)까지 함께 재현된다."""
            it = ctx["step"] - _n_before(ctx["total"], _P0)          # 단계 안 반복 (1부터)
            # EMA: 저장소는 opt.step() 직후에 갱신한다.  harness 는 train_step 이 끝난 뒤에
            # step 하므로, 다음 스텝 맨 앞에서 '끝난 it-1 스텝'으로 갱신하면 같은 시점이다.
            self._ema(self.teacher_diff.model, self.teacher_ema, it - 1)
            self._set_lr(1, it)
            self.prior.eval()
            with torch.no_grad():
                pre = self.prior(b["lms"], b["pan"])
            cond = {"lms": b["lms"], "pan": b["pan"],
                    "pre_out": pre["out"], "pre_variance": pre["variance"],
                    "wavelets": _wavelets(b["lms"], b["pan"])}
            # FSA_T.forward 가 cond['prior_z'] 를 제자리에 써 넣는다(models/FSA_T.py:161).
            # 스텝마다 새 dict 를 만드는 이유다.
            loss, recon = self.teacher_diff(b["gt"] - b["lms"], cond=cond)
            return recon + b["lms"], loss

        def student_step(self, b, ctx):
            """stage 2 -- S2_FSA_S_train.py:226-245.  teacher 표적은 export 캐시에서 읽는다."""
            dev = b["gt"].device
            self._prep(dev)
            n0, n1 = _n_before(ctx["total"], _P0), _n_before(ctx["total"], _P1)
            it = ctx["step"] - n1                                     # 단계 안 반복 (1부터)
            if it == 1:
                # teacher 의 마지막 스텝(n1-n0 번째) 뒤 EMA 갱신은 '다음 teacher 스텝'이 없어 아직
                # 안 됐다 -- export 전에 한 번 한다.  이어 돌리기가 stage 2 중간에서 시작하면 이 갱신은
                # 이미 state_dict(teacher_ema)에 들어 있다.
                self._ema(self.teacher_diff.model, self.teacher_ema, n1 - n0)
                if dev.type == "cuda":
                    torch.cuda.empty_cache()                          # stage 1 용 예약을 놓는다
            self._ema(self.student_diff.model, self.student_ema, it - 1)
            self._set_lr(2, it)

            idx = ctx.get("sample_idx")
            assert idx is not None, "stage 2 는 학습 ctx 의 sample_idx 로 export 캐시를 찾는다"
            t_sr, t_var, t_feat = self._teacher_targets(b["lms"], b["pan"], idx)

            B = b["gt"].shape[0]
            hf = torch.rand(B, device=dev) < 0.5
            vf = torch.rand(B, device=dev) < 0.5
            gt, lms, pan, t_sr, t_var, *t_feat = self._flip(hf, vf, b["gt"], b["lms"], b["pan"],
                                                            t_sr, t_var, *t_feat)
            # --- student ---------------------------------------------------
            res, teacher_res = gt - lms, t_sr - lms
            # 저장소는 여기서 확산을 두 번 부르고 첫 결과를 버린다(S2_FSA_S_train.py:241).
            # 손실에 쓰이지 않는 순수 낭비라 한 번만 부른다(감사의 cleanup (1)).
            _, sr, s_feat = self.student_diff(res, cond={"lms": lms, "pan": pan})
            s_feat = s_feat[1:]           # [0] 은 3C+1 채널 원본 concat -- teacher 에 짝이 없다
            # feature 짝이 어긋나면 KD 의 feat 항이 조용히 다른 것을 비교하게 된다.
            assert len(t_feat) == 6 and len(s_feat) == 6, (len(t_feat), len(s_feat))
            for a_, b_ in zip(t_feat, s_feat):
                assert a_.shape == b_.shape, (tuple(a_.shape), tuple(b_.shape))
            # hard(1.0) + soft(0.1) + feat(0.001).  세 항 전부가 논문의 KD 기여다.
            # var_map 을 division 으로 나누지 않아 soft 가중 (1 - var) 이 음수가 될 수 있는데
            # 저장소와 같게 그대로 둔다.
            loss = self._kd(res, teacher_res, sr, t_feat, s_feat, t_var)
            return sr + lms, loss

        # -------------------------------------------------------------- 평가
        @torch.no_grad()
        def sample(self, lms, pan):
            """S2_FSA_S_train.py:283-287 -- EMA student 를 ddim25 로 돌려 잔차를 얻고
            lms 를 더해 [0,1] 로 자른다.  이미지 한 장에 디노이저 forward 25 회.
            초기 잡음이 전역 난수라 출력이 확률적이다(스펙 stochastic=True)."""
            dev = lms.device
            self._prep(dev)
            s = self._samplers[1]
            self._reset(s, dev)
            self.student_ema.eval()
            out = s({"lms": lms, "pan": pan}, mode="ddim_sample", section_counts=_DDIM)
            self._reset(s, dev)          # 학습/저장 상태를 원래 스케줄로 돌려놓는다
            return (out[0] + lms).clamp(0, 1)

        def forward(self, lms, pan):
            return self.sample(lms, pan)

    # ------------------------------------------------------------------ spec
    def build(ms_ch=4, ratio=4):
        return UKnow(ms_ch, ratio)

    def train_step(model, b, ctx, lossfn):
        f = ctx["epoch_frac"]
        model._reserve(b["gt"].device, f, b["gt"].shape[0])
        if f < _P0:
            return model.prior_step(b)
        if f < _P1:
            return model.teacher_step(b, ctx)
        return model.student_step(b, ctx)

    def on_train_start(model, n_train, device):
        """export 캐시를 이 런의 학습 split 크기로 비워 두고, 표본별 잡음 씨앗을 run seed 로 정한다.
        harness 는 torch.manual_seed(--seed) 직후, 이어 돌리기가 난수 상태를 되살리기 **전에** 이것을
        부른다 -- initial_seed() 가 프로세스마다 같은 값(--seed)이다."""
        U = model._uk
        U.update(n_train=int(n_train), seed=int(torch.initial_seed()), store=None, filled=None,
                 layout=None, n_filled=0, pin=None)

    def make_opt(model, o):
        """저장소의 세 옵티마이저를 파라미터 그룹 셋으로 합친다 (순서: prior, teacher, student).

        PriorNet 은 Adam(1e-4), weight decay 없음(prior_train.py:31).
        두 디노이저는 AdamW(1e-4, wd=1e-4)(S1:204, S2:206).
        weight_decay=0 인 AdamW 는 Adam 과 같은 갱신이라 prior 도 정확히 재현된다.
        grad 가 None 인 파라미터는 AdamW 가 건너뛰므로(모멘트 · decay 모두) 단계 밖 그룹은 멈춰 있다 --
        저장소의 '단계마다 새 옵티마이저'와 같다.  lr 은 train_step 이 그룹별로 매 스텝 정한다.
        EMA 사본은 requires_grad=False 라 여기서 걸러진다."""
        wd = o.get("weight_decay", 1e-4)
        pri = [p for p in model.prior.parameters() if p.requires_grad]
        tea = [p for p in model.teacher_diff.model.parameters() if p.requires_grad]
        stu = [p for p in model.student_diff.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            [dict(params=pri, weight_decay=0.0),
             dict(params=tea, weight_decay=wd),
             dict(params=stu, weight_decay=wd)],
            lr=o["lr"], betas=o.get("betas", (0.9, 0.999)))
        model._uk.update(opt=opt, lr0=float(o["lr"]))
        return opt

    SPECS["U-Know-DiffPAN"] = dict(
        tier="confirmed",   # (2026-09-28): 스모크 통과, 표에 올린다
        build=build, venue="CVPR 2025", loss="l1", batch=32, epochs=_EPOCHS["wv3"],
        per_dataset={d: dict(epochs=e) for d, e in _EPOCHS.items() if d != "wv3"},
        opt=dict(kind="adamw", lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4),
        make_opt=make_opt,
        grad_clip=0.003,                  # S1:262-263 / S2:259-260  norm clip, 저장소 값 (prior 는 저장소에 클립 없음 -- diff (7))
        eval_after_frac=_P1,              # student 가 학습되기 전 평가는 의미가 없다
        val_cap=256,                      # 평가 1장 = 디노이저 25회
        # 평가가 가우시안 잡음에서 출발한다(ddim eta=0 이지만 x_T 가 전역 난수) -- 부를 때마다 다른 출력.
        stochastic=True,
        # 평가는 학습 크기(64)의 겹치지 않는 타일 (2026-09-30).  student 디노이저(UNetSR3, attn_res=(8,),
        # norm_groups=1)를 64 패치로만 학습했는데, 256 영상을 통째로 넣으면 wv3 RR 이 31.31 dB 로 무너진다
        # (valid 37.69 · 64 타일 37.44 · 128 타일 36.27, 같은 best.pt · seed 0).  저자 시험 스크립트는 통째다
        # (S2_FSA_S_test.py:40 image_size=256) -- 우리 쪽이 방법에 유리하게 벗어난 것이고 harness.TILE_WHY 에 적는다.
        # 샘플링 잡음은 타일마다 전역 난수에서 차례로 뽑는다 (seed 0 규칙 그대로; 고정 잡음 경로가 없다).
        eval_tile=64,
        forward=lambda mm, ms_up, pan, lrms, ctx: mm.sample(ms_up, pan),
        train_step=train_step,
        on_train_start=on_train_start,
        diff="저장소 파일은 한 줄도 고치지 않았다(git clean).  네 번의 실행을 한 번의 "
             "harness 실행으로 접었고, 접는 방식은 다음과 같다.\n"
             "(1) 단계 분할과 예산: 저장소 예산 300k(PriorNet) / 400k(FSA-T) / 300k(FSA-S) 는 반복 수이고 "
             "데이터셋과 무관하다.  그래서 데이터셋마다 epochs 를 1M 반복에 맞췄다 -- wv3 3289 x 304 = "
             "999,856 / qb 1866 x 536 = 1,000,176 / gf2 1613 x 620 = 1,000,060 step.  단계 경계는 "
             "ctx.epoch_frac 의 0.30 / 0.70 이다 (wv3 기준 299,956 / 399,943 / 299,957).  저장소 while "
             "루프는 에폭을 끝까지 돌아 단계마다 수십~수백 반복을 더 도는데(예: wv3 300,048 / 400,064 / "
             "300,048) 그 반올림은 따르지 않는다.\n"
             "(2) stage 1.5(export): 저장소는 EMA teacher 로 학습 split 전체를 ddim25 로 한 번 샘플링해 "
             "h5(sr, var_map, dist_feat0..5)에 저장하고 stage 2 는 그것을 읽는다.  우리는 학습 ctx 의 "
             "sample_idx 로 같은 표적을 캐시한다: stage 2 에서 처음 뽑힌 표본만 그 자리에서 export 하고 "
             "(얼어 있는 EMA teacher, eval 모드, 표본 인덱스와 run seed 로 고정한 초기 잡음, ddim25, "
             "전역 난수 미사용) 이후에는 캐시를 읽는다.  teacher 가 stage 2 내내 고정이라 미리 한 번에 "
             "export 한 것과 같은 표적이다.  캐시는 호스트 메모리 fp32(wv3 약 14 GB, qb 23 GB, gf2 27 GB)이고 "
             "체크포인트에 넣지 않는다 -- 이어 돌리면 같은 잡음으로 다시 채운다(수치는 GPU 비결정성 안에서 "
             "같다).  채우는 비용은 스텝 약 1.2k(wv3)~2.6k(gf2) 번의 부분 배치 ddim25 다.\n"
             "(3) 손실 3종을 전부 살렸다.  stage 0 beta-NLL(beta=0), stage 1 잔차 "
             "beta-NLL(beta=0.5), stage 2 KD = 1.0*mean((1+var_T)|res-sr|) + "
             "0.1*mean((1-var_T)|res_T-sr|) + 0.001*sum_{i=0..5} charbonnier(feat_T_i, feat_S_i). "
             "확산이 돌려주는 L1 은 저장소 스스로 죽은 코드라 주석해 둔 것이라 버린다. "
             "var_T 는 저장소와 똑같이 division 으로 나누지 않아 soft 가중이 음수가 될 수 "
             "있고 그대로 둔다.\n"
             "(4) 웨이블릿 조건은 pywt.swt2(db1,level=1) 대신 같은 값을 내는 torch 2x2 "
             "Haar(순환 패딩)로 계산한다 -- pywt 와 1e-16 안에서 일치함을 실측했다. "
             "저장소가 detail 판을 [h, d, v] 순으로 붙이는(자기 unpack 순서와 어긋나는) "
             "것까지 그대로 둔다.\n"
             "(5) stage 2 의 H/V 뒤집기(p=0.5, 항목별 공유 seed)는 train_step 안에서 gt/lms/pan 과 "
             "캐시에서 읽은 teacher sr/var_map/feature 6장에 함께 건다 -- 저장소와 같다. harness 의 aug 는 "
             "D4(회전 포함)뿐이고 stage 0/1 은 저장소에서 증강이 없다(aug 호출이 주석 처리된 죽은 인자).\n"
             "(6) S2 의 중복 forward 한 번을 부르지 않는다(감사 cleanup (1)) -- 결과를 "
             "버리는 호출이라 손실은 불변이고, 난수 소비와 student forward 한 벌만 줄어든다.\n"
             "(7) 옵티마이저: 파라미터 그룹 셋(prior wd=0 / teacher wd=1e-4 / student wd=1e-4)으로 저장소의 "
             "Adam·AdamW 를 그대로 재현한다.  다만 harness 의 norm clip 0.003 이 "
             "PriorNet 에도 걸린다(저장소는 prior 를 클리핑하지 않는다; prior 의 기울기 노름은 1e3~2e4 라 "
             "매 스텝 걸린다 -- 방향은 같고 스텝별 크기가 정규화된 Adam 이 된다).  편차로 기록한다.\n"
             "(8) MultiStepLR: teacher 와 student 의 [100k, 200k, 350k] x 0.5(반복마다)를 train_step 안에서 "
             "**단계 안 반복 번호**로 재현한다(teacher 는 세 지점 모두, student 는 100k/200k 가 걸린다). "
             "PriorNet 은 저장소에서 매 에폭 스케줄러를 새로 만들어 milestone 이 안 걸리므로 상수 1e-4 다 "
             "(prior_train.py:107-112).  harness 의 sched 는 None 이라(마지막 lr 감쇠 전 patience "
             "정지 연장) 규칙은 이 감쇠를 보지 못한다.\n"
             "(9) EMA(decay 0.995, 단계 안 20k 반복 뒤부터)는 teacher 와 student 모두 재현했다. "
             "저장소가 opt.step() 직후에 update(0부터 센 반복) 하는 것을 다음 스텝 맨 앞에서 같은 번호로 "
             "갱신해 시점을 맞춘다(평가 · best.pt 의 student EMA 는 그 스텝의 갱신 하나가 늦다).  teacher 의 "
             "마지막 갱신은 student 첫 스텝에서 export 전에 한다.  EMA 사본은 requires_grad=False 라 "
             "옵티마이저와 n_params 가 건드리지 않고, 등록된 서브모듈로 남아 state.pt/best.pt 에 실린다.\n"
             "(10) ddim_sample_loop 가 diffusion 객체의 beta buffer 를 제자리에서 500->25 "
             "로 갈아엎으므로 샘플링 전용 래퍼 둘을 트리 **밖** 리스트에 두고(디노이저는 "
             "EMA 사본 공유) 호출마다 500-step 스케줄을 되돌린다.  export 는 그 루프를 초기 잡음만 받도록 "
             "옮겨 적었다(연산 동일).\n"
             "(11) 표에 실릴 크기는 **student 9,709,988(C=4) / 9,714,600(C=8)** 이다. "
             "registry.n_params 는 prior+teacher+student 를 합쳐 ~35.6M 을 찍지만 teacher 와 "
             "prior 는 학습 시간 전용이고 평가에는 student 만 쓰인다.\n"
             "(12) lrms(ms) 는 어떤 망에도 들어가지 않는다 -- 저장소도 로더가 싣기만 하고 "
             "쓰지 않는다(prior_train.py:117, S2:279).  tqdm 만 무음 처리했다(연산 불변).\n"
             "(13) 저장소 안의 어긋남: FSA_T_config 는 prior_train.py 가 쓰지 않는 'best_checkpoint.pth' 를 "
             "읽고(prior_train.py:139-146 은 checkpoint_epoch_* 만 쓴다), export 스크립트의 __main__ 은 "
             "400k 가 아니라 iter 300000 의 EMA teacher 를 싣는다(S1_FSA_T_pretrain_save.py:218).  우리는 "
             "stage 0 끝의 PriorNet 과 stage 1 끝(설정값 400k)의 EMA teacher 를 쓴다.\n"
             "(14) 운영: stage 0/1 을 도는 프로세스는 첫 스텝에서 stage 1 최고치(B=32 에서 16.3 GiB)를 "
             "할당자 캐시에 미리 잡고 student 첫 스텝에서 놓는다(연산 불변 -- 옆 작업이 stage 0 동안 붙어 "
             "stage 1 에서 OOM 나는 것을 막는다).  비어 있는 4090 실측: stage 0 약 0.009 s/step, stage 1 약 "
             "0.28~0.29 s/step (B=32, 15.7~15.8 GiB) -- stage 0+1 이 약 32~34 시간이라 **첫 48h 구간 안에** "
             "student 가 시작되어야 평가가 생긴다(eval_after_frac 0.7).  카드를 다른 학습과 나누면 이 전제가 "
             "깨진다.  캐시를 쓰는 stage 2 는 FLOP 비로 약 0.06~0.12 s/step 로 추정(미측정).")
