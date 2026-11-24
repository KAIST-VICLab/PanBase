"""BNNPan (CVPR 2025) -- 이진화 웨이블릿 본체 + 64차원 잠재 prior 를 4스텝 DDPM 으로 뽑는 팬샤프닝.
가장 틀리기 쉬운 곳: `gt` 가 손실 타깃이 아니라 **학습 시 모델 입력**(z = LE(gt))이고, 학습이
S1->S2->S3 3단계 커리큘럼이라 S3 만 돌리면 논문의 기여(사전 증류)가 통째로 사라진다."""
import torch

_EP_PER_STAGE = 1000           # configs/*.yaml epoch_num -- 단계마다 1000 에폭
_STEP_SIZE, _GAMMA = 250, 0.1  # models/pipeline.py:104 StepLR(step_size=250, gamma=0.1), 단계마다 새로 시작
# S3 가 시작하는 진행도.  이 앞(S1: DM 미학습, S2: DM 증류 중)에서는 평가하지 않는다 -- 아래 eval_after_frac.
# steps = 3000 * spe 라 2/3 지점은 정확히 2000 * spe 번째 step 이고, 그 step 이 첫 S3 step 이다
# (_stage_of 에서 f*3 == 2.0).  2000 * spe 는 eval_every(2000) 의 배수라 첫 평가가 바로 거기서 난다.
_S3_FRAC = 2.0 / 3.0


def register(SPECS, REPOS, _load, _pkg):
    # ---------------------------------------------------------------- build
    def build(ms_ch=4, ratio=4):
        root = REPOS / "BNNPan"
        # models/BiWaveDiffSNTBBFDN.py:5 가 `from ldm.ddpm import DDPM` 로 **절대 임포트**한다.
        # `ldm` 은 확산 계열 저장소들이 즐겨 쓰는 이름이라 sys.path 로는 서로를 가린다.
        _pkg(root / "ldm", "ldm")
        M = _load(root / "models" / "BiWaveDiffSNTBBFDN.py", "bnnpan_model")
        # 배포 모델 = S3 클래스.  S2 를 상속하므로 LE / DM / net 을 모두 갖는다 -- 즉 이 한
        # 인스턴스로 S1(LE+net), S2(DM), S3(전체) 세 단계를 전부 실행할 수 있다.
        # 인자는 configs/config_wv3.yaml 그대로 (qb/gf2 config 도 lms_channels 외에는 같다).
        net = M.BiWaveDiffSNTBBFDNS3(pan_channels=1, lms_channels=ms_ch, hid_channels=64,
                                     layers=4, n_denoise_res=1, linear_start=0.1,
                                     linear_end=0.99, timesteps=4)
        # nn.Module 트리 밖에 두는 상태.  dict 은 Parameter/Module 이 아니라 등록되지 않고
        # state_dict 에도 들어가지 않는다.  옵티마이저 참조를 여기 담아야 train_step 이
        # 단계별 StepLR 재시작과 Adam 모멘트 초기화를 할 수 있다(하네스는 opt 를 넘겨주지 않는다).
        # base_lr 은 make_opt 가 (per_dataset 이 병합된) opt.lr 로 채운다 -- 레시피 스냅샷의
        # opt.lr 이 실제로 쓴 값과 같아야 한다 (예전에는 밴드 수로 골라 wv3 스냅샷이 1e-4 로 적혔다).
        net._bnnpan = {"bands": int(ms_ch), "stage": None, "opt": None, "base_lr": None}
        return net

    # ------------------------------------------------------------ 커리큘럼
    def _stage_of(ctx):
        """진행도 -> (단계 0/1/2, 그 단계 안에서의 진행도 0~1).  전체 예산을 3등분한다."""
        f = ctx.get("epoch_frac")
        if f is None:
            f = ctx["step"] / max(1, ctx["total"])
        f = min(max(float(f), 0.0), 1.0)
        s = f * 3.0
        st = min(int(s), 2)
        return st, min(s - st, 1.0)

    def _apply_stage(model, stage):
        """단계가 바뀌면 Adam 모멘트를 비운다 (저장소에서 각 단계는 **별개의 프로세스**라 Adam 이
        새로 만들어진다).

        동결은 requires_grad 를 바꾸지 않고 **그래프에서 빼는 것**으로 한다 -- 결과는 같다:
          S1  DM 은 forward 에 없다 -> grad None -> Adam 이 건너뛴다 (S1 클래스에는 DM 자체가 없다)
          S2  net 은 forward 에 없고 LE 는 no_grad 로 부른다 -> 둘 다 grad None
              (models/pipeline.py:95-99 는 LE/net 을 requires_grad=False 로 두고 Adam 을 DM 에만 건다)
          S3  전부 학습 (pipeline.py:103)
        requires_grad 를 바꾸면 하네스의 `R.n_params`(requires_grad 인 것만 센다)가 단계에 따라 달라져,
        S1/S2 에서 멈춘 구간(천장)의 결과 JSON `params` 가 DM 또는 LE+net 을 빠뜨린다."""
        c = model._bnnpan
        if c["stage"] == stage:
            return
        c["stage"] = stage
        opt = c.get("opt")
        if opt is not None:
            opt.state.clear()

    def _set_lr(model, stage, local):
        """단계마다 StepLR(250, 0.1) 을 처음부터 다시 돌린다.  하네스의 sched 를 쓰면
        3000 에폭 동안 연속 감쇠해 S2/S3 가 lr 1e-8 로 죽는다 -- 그래서 sched 는 선언하지 않고
        여기서 직접 건다."""
        c = model._bnnpan
        opt = c.get("opt")
        if opt is None:                            # make_opt 를 모르는 하네스: 옵티마이저 lr 고정
            return
        # 마지막 스텝에서 local==1.0 이 되면 에폭 1000(범위 밖)이 되므로 999 로 자른다.
        # 저장소에서 네 번째 감쇠는 마지막 에폭이 끝난 뒤에 걸려 학습에 쓰이지 않는다.
        ep = min(int(local * _EP_PER_STAGE), _EP_PER_STAGE - 1)
        lr = c["base_lr"] * (_GAMMA ** (ep // _STEP_SIZE))
        for g in opt.param_groups:
            g["lr"] = lr

    def make_opt(model, o):
        c = model._bnnpan
        # models/pipeline.py:95/103 -- Adam(params, lr, betas=(0.9,0.999)).
        # config 의 weight_decay: 1e-6 은 pipeline.py:72 에서 파싱만 되고 Adam 에 **전달되지 않는다**.
        # 코드를 재현한다 -> weight_decay=0.
        # lr 은 데이터셋별 config 값이다 (per_dataset): WV3 3e-4, QB/GF2 1e-4.
        c["base_lr"] = float(o["lr"])
        opt = torch.optim.Adam(model.parameters(), lr=c["base_lr"],
                               betas=tuple(o.get("betas", (0.9, 0.999))))
        c["opt"] = opt
        return opt

    # ------------------------------------------------------------ 학습 스텝
    def train_step(model, batch, ctx, lossfn):
        """공식 손실은 세 단계 모두 **단일 항 L1** 이다 (models/pipeline.py:128/131/134).
        단, 무엇에 거는 L1 인지가 단계마다 다르다:
          S1  L1(net(lms,pan,LE(gt))+lms, gt)   -- 오라클 prior 로 본체를 먼저 세운다
          S2  L1(z_hat, z)                      -- LE/net 동결, DM 만 prior 를 증류한다
          S3  L1(out, gt)                       -- 전체 미세조정.  S3 는 z/z_hat 도 돌려주지만
                                                   저장소는 그 항을 **일부러 쓰지 않는다**
        SSIM/웨이블릿 항은 저장소에 존재하지만 전부 주석이거나 config 가 False 라 죽은 코드다."""
        lms, pan, gt = batch["lms"], batch["pan"], batch["gt"]
        stage, local = _stage_of(ctx)
        _apply_stage(model, stage)
        _set_lr(model, stage, local)
        model.train()                               # DM 의 train/eval 분기가 여기에 걸린다
        if stage == 0:                              # --- S1 (BiWaveDiffSNTBBFDNS1.forward 와 동일)
            z = model.LE(gt)
            out = model.net(lms, pan, z) + lms
            return out, lossfn(out, gt)
        if stage == 1:                              # --- S2 (BiWaveDiffSNTBBFDNS2.forward, train 분기)
            with torch.no_grad():                   # LE 동결 (pipeline.py:96-97) -> z 는 상수 타깃
                z = model.LE(gt)
            z_hat, _ = model.DM(pan, lms, z)
            # S2 는 이미지 출력이 아예 없는 단계다(저장소도 (z, z_hat) 만 돌려준다).
            # 하네스는 train_step 분기에서 예측을 쓰지 않으므로 자리표시자로 lms 를 돌려준다.
            return lms.detach(), lossfn(z_hat, z)
        out, _z, _z_hat = model(lms, batch["ms"], pan, gt)   # --- S3
        return out, lossfn(out, gt)

    # ------------------------------------------------------------ 평가 forward
    def forward(mm, ms_up, pan, lrms, ctx):
        if mm.training:
            # 학습 경로로 잘못 불린 것이다.  여기서 조용히 eval 로 바꿔 주면 "샘플러를 통과해
            # 학습하는" 다른 목적함수가 되어 버린다 -- 조용한 오답보다 즉시 실패가 낫다.
            raise RuntimeError(
                "BNNPan: forward() 는 평가 전용이다. 학습은 train_step 을 쓰는 하네스가 필요하고, "
                "평가 직전에 model.eval() 이 호출돼야 한다 (train 모드에서는 z=LE(gt) 라 gt 없이 죽는다).")
        # eval 경로의 prior 는 ldm/ddpm.py:219 의 torch.randn 에서 나온다 (x_T ~ N(0,I), 64차원).
        # **확률적 방법**이다 (stochastic=True).  시드를 여기서 걸지 않는다 -- 호출자의 전역 난수를 쓴다.
        # 공개 영상의 추론 seed 0 규칙(dump_outputs 규칙 3: bs 1, 집합마다 한 번 seed 0, 장면마다 차례로
        # 뽑기)이 실제로 이 난수를 정해야 하기 때문이다.  예전처럼 호출마다 고정 시드로 fork 하면
        # dump 의 seed 0 이 무시되고(기록은 'seed 0' 인데 실제는 다른 시드), bs 1 에서 모든 장면이
        # 같은 x_T 를 받는다.  학습 중 valid 도 전역 난수를 쓴다 -- state.pt 가 평가 뒤의 CPU/CUDA
        # 난수 상태를 저장하므로 exact 이어 돌리기는 그대로 성립한다 (train_pancoll.py:588-589, 859-861).
        return mm(ms_up, lrms, pan)                 # (lms, ms, pan) -- ms 는 저장소가 읽지 않는다

    # ------------------------------------------------------------ 이어 돌리기
    def on_resume(model, ctx):
        """exact 이어 돌리기 직후 (하네스가 옵티마이저 상태를 되살린 뒤) 한 번 부른다.

        `model._bnnpan["stage"]` 는 nn.Module 트리 밖이라 state_dict 에 안 들어가고, 모델을 새로
        지으면 None 이다.  그대로 두면 이어 돈 첫 train_step 의 `_apply_stage` 가 '단계가 바뀌었다'
        고 보고 `opt.state.clear()` 를 불러, 되살린 Adam 모멘트를 **조용히 버린다** -- 결과에는
        optimizer=restored 로 적힌 채로 (검토, 2026-09-24).  그래서 끊긴 step(ctx)의 단계로
        캐시를 맞추되, 그 한 번은 옵티마이저를 떼어 두어 비우지 않게 한다.
        단계 경계를 넘는 다음 스텝의 초기화는 끊김 없는 런과 똑같이 일어난다."""
        c = model._bnnpan
        opt, c["opt"] = c.get("opt"), None
        try:
            _apply_stage(model, _stage_of(ctx)[0])
        finally:
            c["opt"] = opt

    SPECS["BNNPan"] = dict(
        tier="confirmed",   # (2026-09-28): 스모크 통과, 표에 올린다
        build=build, venue="CVPR 2025", loss="l1", batch=32, epochs=3000,
        # opt.lr 은 QB/GF2 값(configs/config_qb.yaml, config_gf2.yaml base_lr 1e-4).
        opt=dict(kind="adam", lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0),
        # 저장소는 데이터셋마다 config 를 따로 둔다: configs/config_wv3.yaml base_lr 3e-4.
        # 나머지(epoch_num 1000, batch_size 32, step_size 250, gamma 0.1)는 세 config 가 같다.
        per_dataset={"wv3": {"opt": {"lr": 3e-4}},
                     "qb": {"opt": {"lr": 1e-4}},
                     "gf2": {"opt": {"lr": 1e-4}}},
        make_opt=make_opt,                 # 단계별 lr 재시작을 위해 옵티마이저 참조가 필요하다
        # sched 없음 -- StepLR(250,0.1) 을 단계마다 다시 시작해야 해서 train_step 이 직접 건다
        # grad_clip 없음 = 하네스 기본 10.0 (사용자 결정 2026-09-28: 저장소는 클립하지 않지만
        # (pipeline.py:138 주석) 하네스 기본값을 유지하고 원장에 편차로 밝힌다).
        eval_after_frac=_S3_FRAC,          # S1(DM 미학습)/S2 구간의 평가 · patience 정지를 막는다
        stochastic=True,                   # 평가가 x_T ~ N(0,I) 에서 prior 를 뽑는다 (ldm/ddpm.py:219)
        forward=forward,
        train_step=train_step,
        on_resume=on_resume,               # 코드 지문(_CODE_FIELDS) 밖 -- 이어 돌리기에서만 쓴다
        diff="저장소는 S1/S2/S3 를 **각각 1000 에폭짜리 별도 실행**으로 돌리고 앞 단계 체크포인트를 "
             "불러온다(models/pipeline.py:51-99).  우리 하네스는 단일 실행이므로 배포 모델인 S3 "
             "클래스 한 인스턴스로 세 단계를 순서대로 실행한다 -- S3 는 S2 를 상속해 LE/DM/net 을 "
             "모두 가지고 있고, S1/S2 의 forward 는 그 부분모듈 호출과 문자 그대로 같다(:358-387). "
             "체크포인트 저장/로드가 프로세스 안에서의 연속 학습으로 바뀔 뿐 가중치 흐름은 동일하다. "
             "단계 경계는 epochs 예산의 1/3 지점들이다(3000 = 1000x3). "
             "S2 의 LE/net 동결은 requires_grad 대신 그래프에서 빼는 것(LE 는 no_grad, net 은 호출 안 함)으로 "
             "재현한다 -- 갱신은 같고(기울기 None -> Adam 이 건너뜀), 하네스의 파라미터 수가 단계에 따라 "
             "바뀌지 않는다. "
             "단계별 Adam 재시작(opt.state.clear())과 StepLR(250,0.1) 재시작을 make_opt 로 받은 "
             "옵티마이저에 직접 걸어 재현한다. "
             "lr 은 저장소가 데이터셋별로 다르게 준다 -- WV3 3e-4(configs/config_wv3.yaml), QB/GF2 1e-4 "
             "(config_qb.yaml, config_gf2.yaml) -- per_dataset 으로 옮겼다. "
             "weight_decay 는 config 가 1e-6 을 선언하지만 pipeline.py 가 Adam 에 넘기지 않아 실효 0 이다. "
             "평가는 S3 가 시작하는 2/3 지점부터 한다(eval_after_frac) -- S1 에서는 DM 이 학습되지 않아 "
             "평가 경로의 prior 가 무작위이고, S2 는 DM 증류 중이라, 그 구간의 patience 가 커리큘럼을 "
             "S3 전에 끊을 수 있었다(PATIENCE 91,200/160,800/186,000 step < 단계 길이 304k/536k/620k). "
             "저장소의 검증(pipeline.py:156-182)은 train 모드로 도는 손실 기록일 뿐 체크포인트를 고르지 "
             "않는다. 단계 사이 인계는 마지막(epoch 1000) 가중치이고(config pre_train_model, "
             "pipeline.py:56,62) -- 우리도 프로세스 안에서 그대로 잇는다 -- 최종 시험 가중치는 config 의 "
             "test_weight_path 가 데이터셋마다 다르다(wv3 epoch1000, qb epoch925, gf2 epoch300; 고른 기준은 "
             "저장소에 없다). 우리는 하네스 규칙대로 S3 안에서 valid PSNR 최고 step 을 고른다. "
             "평가 출력이 확률적이다(x_T ~ N(0,I), 64차원, ldm/ddpm.py:219) -- 시드를 따로 걸지 않고 "
             "호출자의 전역 난수를 쓰며 stochastic=True 로 적었다(표의 값은 표본 하나다). "
             "저장소 소스는 한 줄도 고치지 않았다(선언된 의존성 pytorch_wavelets 1.3.0 + PyWavelets 설치가 전부). "
             "손실은 세 단계 모두 단일 항 L1 그대로이고 SSIM/wave_loss 는 저장소에서 죽은 코드다. "
             "클립은 하네스 기본 10.0 을 쓴다(저장소는 :138 에서 주석 처리 -- 클립 없음). 사용자 결정 "
             "2026-09-28 로 유지하고 원장에 편차로 밝힌다. "
             "배치는 하네스 공통의 복원추출이고 에폭당 step 은 ceil(N/32) 이다(저장소 drop_last 보다 "
             "0.2-0.3% 많다). "
             "측정치(4090, batch 32, H=64): S1 0.06-0.08 s/step, S2 0.005 s/step, S3 0.06-0.07 s/step "
             "-> 커리큘럼 평균 약 0.045 s/step. 공식 3000 에폭은 WV3 약 11, QB 약 21, GF2 약 24 GPU-h 다.")
