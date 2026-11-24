"""FAME-Net (AAAI 2024): 주파수 적응형 MoE 팬샤프닝.  손실의 게이트 항(CVLoss x 3)을
빠뜨리면 전문가 부하분산이 사라져 MoE 가 MoE 가 아니게 된다 -- 반드시 train_step 으로 건다."""
import torch


def register(SPECS, REPOS, _load, _pkg):

    # ---------------------------------------------------------------- 공식 상수
    # solver/moesolver.py:129-130  `idx = math.pow(0.99, self.epoch - 1)`; epoch 은 1부터.
    # option.yml `nEpochs: 1000` -- 감쇠는 1000 에폭에 걸쳐 1.0 -> 4.4e-5 로 내려간다.
    OFFICIAL_EPOCHS = 1000

    # 마스크 지도(mask_gt)를 만들어 줄 함수를 꽂는 자리.  (batch)->(B,2,H,W) 텐서.
    # 기본은 None = 그 항을 끈다.  아래 `diff` 와 주석에 이유를 전부 적어 두었다.
    # 저장소의 DCT 라벨 생성기가 공개되면 여기에 연결하면 된다 -- train_step 은 그대로 둔다.
    MASK_TARGET_FN = None

    def build(ms_ch=4, ratio=4):
        # moe.py 는 `from .modules import ...`, `from .refine import ...` 상대 임포트를 쓴다.
        # 그래서 파일 단독 적재(_load)로는 안 되고 `model` 패키지를 이 저장소로 갈아 끼워야 한다.
        _pkg(REPOS / "FAME-Net" / "model")
        from model.moe import Net
        # base_filter=32 는 solver/moesolver.py:109 와 counter_flops_params.py:41 의 값.
        # args 는 __init__ 시그니처에만 있고 본문에서 한 번도 읽지 않는다(저장소는 cfg 전체를
        # 넘긴다) -- None 으로 동일하다.
        # 파라미터: C=4 -> 1,244,228 / C=8 -> 1,246,536.
        return Net(num_channels=ms_ch, base_filter=32, args=None)

    def _cv(z):
        """solver/moesolver.py:87-96 의 CVLoss, loss_weight=1.0 그대로.

        z 는 (B, num_experts) 게이팅 행렬이다.  저장소가 릴리스한 코드는 여기에 (n,1,1,1)
        짜리 마지막 전문가 계수를 넘겨서 std(dim=1) 이 원소 1개의 불편표준편차 -> NaN 이
        되었다(감사 증거 B: B=1,2,4,8 전부 loss=nan).  같은 저자가 TS-Pan 에서 고쳐 배포한
        대로 (B,num_experts) 를 받도록 저장소를 패치해 두었다."""
        return torch.mean((torch.std(z, dim=1) / torch.mean(z, dim=1)) ** 2)

    def _step(model, b, ctx, lossfn):
        # 공식 학습 스텝: solver/moesolver.py:129-148.
        #   y, mask, [lf,hf,dec] = model(lms, lms, pan)      <- arg1 은 저해상 MS(H/4)
        #   gl   = CV(lf) + CV(hf) + CV(dec)
        #   loss = L1(y, gt) + para * L1(mask_gt, mask) + gl
        # 둘째 인자는 이름이 literally `_` 이고 본문에서 참조되지 않는다(moe.py:329).
        # 모델이 arg1 을 스스로 bicubic 업샘플하므로(moe.py:336-339) 여기에 ms_up 을 주면
        # H -> H 로 두 번 업샘플하는 셈이 된다.  반드시 b["ms"](저해상)를 준다.
        y, mask, (lf_gate, hf_gate, dec_gate) = model(b["ms"], None, b["pan"])

        gl = _cv(lf_gate) + _cv(hf_gate) + _cv(dec_gate)      # :143-145, :147  가중치 1.0
        loss = lossfn(y, b["gt"]) + gl                        # :148  본 L1 가중치 1.0

        # --- 세 번째 항: para * L1(mask_gt, mask) ------------------------------
        # para = 0.99**(epoch-1).  epoch 은 진행도로 환산한다: 스펙의 epochs 가 공식값
        # 1000 과 같으므로 그대로 두면 실제 에폭 번호와 정확히 일치하고, --epochs 로 예산을
        # 줄이면 감쇠 곡선 전체가 그 예산에 맞춰 압축된다(코사인 스케줄과 같은 취급).
        ep = min(OFFICIAL_EPOCHS, int(ctx.get("epoch_frac", 0.0) * OFFICIAL_EPOCHS) + 1)
        para = 0.99 ** (ep - 1)
        mgt = b.get("mask_gt")
        if mgt is None and MASK_TARGET_FN is not None:
            mgt = MASK_TARGET_FN(b)
        if mgt is not None:
            loss = loss + para * lossfn(mgt, mask)            # :146  L1(mask_gt, mask)
        # mgt 가 없으면 이 항은 0 이다.  저장소가 라벨 생성기를 공개하지 않았고(README 는
        # "generate the mask label by dct first" 한 줄뿐, `grep -rn dct` 0 hits, upstream
        # HEAD == 우리 커밋), 블록 크기/임계값/원본(PAN vs GT)을 전부 모른다.  추측해서
        # 채우면 그건 논문의 지도신호가 아니라 우리가 지어낸 지도신호다.  빼는 쪽이 정직하다.
        # 대신 반드시 각주로 밝힌다 -- 초반 para~1.0 이라 짧은 예산에서는 작지 않은 항이다.
        return y, loss

    SPECS["FAME-Net"] = dict(
        tier="extra",                      # 코드만 준비, 학습 대기열에는 아직 없음
        build=build, venue="AAAI 2024", loss="l1", batch=4, epochs=OFFICIAL_EPOCHS,
        opt=dict(kind="adam", lr=5e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0),
        sched=dict(kind="cosine", T_max=OFFICIAL_EPOCHS, eta_min=5e-8),
        grad_clip=4.0,                     # option.yml `gclip: 4`, moesolver.py:159-163
        # 평가: 저장소 eval 호출 moesolver.py:185 `model(lms_image, bms_image, pan_image)`
        # -- arg1 은 학습과 같은 저해상 MS 이고 arg2(=ms_up)만 바뀌는데 그 자리는 쓰이지 않는다.
        forward=lambda m, ms_up, pan, lrms, ctx: m(lrms, ms_up, pan)[0],
        train_step=_step,
        diff="저장소 4곳을 고쳤다(repos/FAME-Net/model/moe.py). 전부 '릴리스가 실행 자체를 "
             "못 하는' 버그이고, 세 개는 같은 저자가 3개월 뒤 TS-Pan 에 올린 동일 MoE 코드의 "
             "수정본이 근거다. (A) LfExpert/HfExpert/Decoder 에 `self.num_experts` 대입 누락 "
             "-> forward 즉시 AttributeError (TS-Pan tspan.py:162). (B) `nn.Conv2d(4,...)` 와 "
             "`Refine(...,4)` 의 하드코딩 밴드 4 를 이미 시그니처에 있던 num_channels 로 "
             "(8밴드 필수). (C) MoE 블록이 `return out,cof_k` -> (n,1,1,1) 를 CVLoss 에 넣어 "
             "std(dim=1) 이 원소 1개 -> **모든 배치 크기에서 loss=NaN**, 학습 불가. `cof` 를 "
             "돌려주게 고쳤다 (TS-Pan tspan.py:188-190 의 all_cof_k 와 값이 동일). "
             "(D) 라우팅 가드 `if cof[:,idx].all()==0` -> `if not torch.any(cof[:,idx]>0)`; "
             "`.all()` 은 배치 전체가 동의할 때만 참이라 한 표본만 달라도 그 전문가를 배치 "
             "통째로 지운다(실측 의도 실행의 ~9% 소실). B=1 과 B=4 의 산술이 달라지는 "
             "train/test 불일치라 반드시 고쳐야 한다 (TS-Pan tspan.py:178-179 가 원본 줄을 "
             "주석 처리하고 정확히 이 줄로 교체해 두었다). 마스크 예측기·DCT 주파수 분할·"
             "가역 pre-fusion·전문가 본체·top-2/4 라우팅·CV 부하분산 목적함수는 한 글자도 "
             "건드리지 않았다. 손실은 `L1(y,gt) + para*L1(mask_gt,mask) + (CV(lf)+CV(hf)+"
             "CV(dec))`, para=0.99**(epoch-1). "
             "**각주 필수 1**: mask_gt 는 저장소가 라벨 생성기를 공개하지 않아(README 한 줄 "
             "'generate the mask label by dct', 저장소 전체 dct 검색 0 hits) 그 항을 0 으로 "
             "두었다. 추측한 DCT 레시피를 넣으면 논문의 지도신호를 날조하는 것이라 하지 "
             "않았다. 마스크 예측기는 hard gumbel 의 straight-through 로 본 L1 에서 계속 "
             "학습된다(감사 증거 D: L1 만 backward 해도 maskp.* 가 무기울기 목록에 없다). "
             "그래도 초반 para~1.0 이므로 짧은 예산에서 작지 않은 편차다 -- 표에 각주로 남긴다. "
             "**각주 필수 2**: `F.gumbel_softmax(tau=1,hard=True)` 에 train/eval 분기가 없어 "
             "**추론이 확률적이다**(같은 입력 두 번에 max 차이 1.06). argmax 로 바꾸면 방법이 "
             "달라지므로 그대로 두었다 -- 평가는 시드 고정으로 하고 그 사실을 밝힌다. "
             "HinResBlock.conv_3/identity 는 선언만 되고 forward 에서 안 쓰이는 저장소의 사장 "
             "레이어다(무기울기 68개) -- 저장소 옵티마이저도 그대로 물고 있으므로 제거하지 "
             "않았다. ms_up 은 쓰이지 않는다(둘째 인자 이름이 `_` 다). 저장소 데이터 로더는 "
             "쓸 수 없다: dataset.py:94 가 os.path.join 에 list 를 넘겨 무조건 TypeError.")
