"""HyperTransformer (CVPR 2022): PAN 을 V/K, 업샘플 MS 를 Q 로 쓰는 3-스케일 텍스처-스펙트럼
어텐션 융합.  **틀리기 쉬운 곳**: forward 가 텐서가 아니라 dict 를 돌려주고, 공식 루프가
`pred` 말고 `x13`/`x23`/`tp_loss` 까지 **네 항**을 더한다.  `pred` 에만 L1 을 걸면 다중 스케일
심층 감독과 전이-지각 손실이 통째로 사라져 이 베이스라인만 조용히 약해진다."""
import torch
import torch.nn.functional as F


def register(SPECS, REPOS, _load, _pkg):
    ROOT = REPOS / "HyperTransformer"
    EPS = 1e-6

    # ---------------------------------------------------------------- 유사 RGB 밴드 색인
    def _rgb_idx(C):
        """train.py:178-183 의 유사 RGB 합성은 설정값 R/G/B (밴드 색인) 를 요구한다.
        저장소가 배포한 값(pavia R=10,G=30,B=60 / chikusei R=10,G=35,B=61)은 초분광 전용이고,
        그 값으로는 가운데 채널이 `x[:, 60:10]` -- **빈 슬라이스** -> NaN 이다(botswana4 만 정상).
        우리 MS 밴드 순서에 맞는 색인을 넣는다.  이건 코드 변경이 아니라 데이터셋 설정값이다.
          4밴드 (QB/GF2: B,G,R,NIR)                        -> R=2, G=1, B=0
          8밴드 (WV3: coastal,B,G,Y,R,RE,NIR1,NIR2)        -> R=4, G=2, B=1
        아래 식은 이 둘을 정확히 재현한다."""
        b = C // 8
        g = max(b + 1, C // 4)
        r = max(g + 1, C // 2)
        if not (0 <= b < r <= C and 0 < g < C):
            raise ValueError(f"HyperTransformer: {C}밴드에 쓸 유사 RGB 색인을 만들 수 없다.")
        return r, g, b

    def _prgb(x, r, g, b, C):
        """train.py:178-183 그대로.  (평균(0:G), 평균(B:R), 평균(G:bands))"""
        return torch.cat((x[:, 0:g].mean(1, keepdim=True),
                          x[:, b:r].mean(1, keepdim=True),
                          x[:, g:C].mean(1, keepdim=True)), 1)

    # ---------------------------------------------------------------- 어텐션 온도 한 줄 패치
    def _sdpa(self, v, k, q, mask=None):
        """models/HyperTransformer.py:344-366 원본에서 **한 줄**만 바꾼다.

        원본:  attn = matmul(q / self.temperature, k^T)
        여기:  attn = matmul(q / (h*w),            k^T)

        생성자가 self.temperature 에 넣는 값이 바로 LR_size**2 / (2*LR_size)**2 /
        (4*LR_size)**2 (:544-550), 즉 **학습 해상도에서의 그 레벨 h*w** 다.  따라서 학습
        크기에서는 비트 단위로 동일하고(측정: max|patched-original| = 0.000e+00),
        평가 해상도가 커질 때만 같은 scaled-dot-product 의미를 유지한다.  놔두면 H=256 에서
        로짓이 16배(std 0.062 -> 0.250) 커져 softmax 가 학습 때와 다르게 날카로워진다.
        어텐션 연산·V/K/Q 배선·3스케일 융합·전이지각 손실은 그대로다."""
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)
        attn = torch.matmul(q / (h * w), k.transpose(-2, -1))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v).view(b, c, h, w)

    # ---------------------------------------------------------------- 빌더
    def build(ms_ch=4, ratio=4, lr_size=16):
        if ratio != 4:
            # LFE 의 2단 maxpool 과 PixelShuffle(2)+PixelShuffle(2) 피라미드, up_conv13(stride 4)
            # 이 전부 배율 4 전제다.  배율 2 를 조용히 돌리면 안 된다.
            raise ValueError("HyperTransformer 의 LFE/PixelShuffle 피라미드는 x4 전용이다.")
        M = _load(ROOT / "models" / "HyperTransformer.py", "hypertransformer_model")
        M.ScaledDotProductAttentionOnly.forward = _sdpa

        cfg = {
            "is_DHP_MS": False,        # 입력은 LR MS.  모델이 내부에서 bicubic x4 한다.
            "N_modules": 1,            # 1 -> 파라미터 없는 채널축 SDPA (해상도 무관).
                                       # >=2 -> MHFA, nn.Linear(h*w) 라 크기 고정 (diff 참고).
            "train_dataset": "panbase",
            "panbase": {"spectral_bands": ms_ch, "factor": ratio,
                       "LR_size": lr_size, "HR_size": ratio * lr_size,
                       "feature_sum": False,                          # 저장소 기본값
                       "loss_type": "L1", "Normalized_L1": True,
                       "VGG_Loss": True, "VGG_Loss_F": 0.1,
                       "Transfer_Periferal_Loss": True, "Transfer_Periferal_Loss_F": 0.05,
                       "Spatial_Loss": False, "Spatial_Loss_F": 1.0,  # 배포 설정 전부 false
                       "multi_scale_loss": True, "multi_scale_loss_F": 0.2,
                       "max_value": 1.0},
        }
        net = M.HyperTransformer(cfg)

        # 옵티마이저가 보면 안 되는 것 / 모듈 트리에 올리면 안 되는 것은 __dict__ 에 직접
        # 넣는다.  nn.Module.__setattr__ 를 우회하므로 parameters()/state_dict() 에 절대
        # 잡히지 않는다.  리스트로 감싸는 이유는 나중에 동결 VGG19 를 여기에 담기 때문.
        net.__dict__["_panbase_rgb"] = _rgb_idx(ms_ch)
        net.__dict__["_panbase_vgg"] = []
        return net

    # ---------------------------------------------------------------- 동결 VGG19 (지각 손실용)
    def _vgg(model, device):
        """utils/vgg_perceptual_loss.py 의 VGG19 + VGGPerceptualLoss 를 그대로 쓴다.

        **모듈 트리 밖에 둔다.**  저장소는 `vggnet = VGG19(); DataParallel(...).cuda()` 로
        model 과 분리된 객체를 만들고 옵티마이저에는 `model.parameters()` 만 준다
        (train.py:140-142, 107-113) -- 즉 VGG 는 원래 학습되지 않는다.  우리 하네스는
        `model.parameters()` 를 그대로 옵티마이저에 넘기므로, VGG 를 속성으로 붙이면
        2천만 파라미터가 Adam 상태에 들어가 저장소와 달라진다.  그래서 __dict__ 의 평범한
        리스트에 담고 requires_grad_(False) 로 못을 박는다 (fake 특징으로의 역전파는 그대로
        흐른다 -- 손실이 예측을 통과해야 하므로)."""
        holder = model.__dict__["_panbase_vgg"]
        if not holder:
            V = _load(ROOT / "utils" / "vgg_perceptual_loss.py", "hypertransformer_vggloss")
            net = V.VGG19().to(device).eval()
            for p in net.parameters():
                p.requires_grad_(False)
            holder.append(net)
            holder.append(V.VGGPerceptualLoss)
        return holder[0], holder[1]

    # ---------------------------------------------------------------- 공식 손실 (네 항 전부)
    def train_step(model, batch, ctx, lossfn):
        """train.py:165-198 을 항 하나도 빼지 않고 재현한다.  계수는 config_HSIT.json
        pavia 블록(논문 본 설정) 값이다.

          (1) train.py:171-173  Normalized_L1=true -> L1(pred/max_ref, gt/max_ref)
          (2) train.py:176-186  + 0.1  * VGGPerceptualLoss(유사RGB(pred), 유사RGB(gt))
          (3) train.py:189-190  + 0.05 * out["tp_loss"]      (모델 내부 계산, :769-773)
          (4) train.py:197-198  + 0.2  * L1(gt, x13) + 0.4 * L1(gt, x23)
                                 -- 저장소가 문자 그대로 `+ 2*F*criterion(...)` 이라 x23 은 0.4 다
          Spatial_Loss 는 배포된 모든 설정에서 false 이므로 다섯째 항은 없다(train.py:193-194).

        lossfn 은 하네스가 spec["loss"]="l1" 에서 만든 torch.nn.L1Loss 로, train.py:131-132
        의 criterion 과 같은 객체다."""
        gt, pan, lrms = batch["gt"], batch["pan"], batch["ms"]
        out = model(lrms, pan[:, 0])          # ms_up 은 쓰지 않는다 -- 모델이 직접 만든다
        pred, C = out["pred"], gt.shape[1]

        # (1) 정규화 L1.  clamp_min 은 전채널 0 타일에서의 0 나눗셈 방지용 안전장치일 뿐,
        #     실제 데이터(amax ~ 0.1-1.0)에서는 아무 영향이 없다.
        max_ref = torch.amax(gt, dim=(2, 3)).unsqueeze(2).unsqueeze(3).expand_as(gt).clamp_min(EPS)
        loss = lossfn(pred / max_ref, gt / max_ref)

        # (2) VGG 지각 손실.  가중치 [1, 0.2, 0.04] 를 conv1_2/conv2_2/conv3_3 에 건다.
        vgg, vggloss = _vgg(model, pred.device)
        r, g, b = model.__dict__["_panbase_rgb"]
        loss = loss + 0.1 * vggloss(_prgb(pred, r, g, b, C), _prgb(gt, r, g, b, C), vgg)

        # (3) 전이-지각 손실 (모델이 이미 축약해서 돌려준 스칼라)
        loss = loss + 0.05 * out["tp_loss"]

        # (4) 다중 스케일 심층 감독
        loss = loss + 0.2 * lossfn(gt, out["x13"]) + 0.4 * lossfn(gt, out["x23"])
        return pred, loss

    SPECS["HyperTransformer"] = dict(
        tier="extra",                       # 코드만 준비.  아직 학습 큐에 넣지 않는다.
        build=build, venue="CVPR 2022", loss="l1", batch=8, epochs=150,
        opt=dict(kind="adam", lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4),
        sched=dict(kind="step", step_size=30, gamma=0.1),
        forward=lambda mm, ms_up, pan, lrms, ctx: mm(lrms, pan[:, 0])["pred"].clamp(0, 1),
        train_step=train_step,
        diff=(
            "1) 어텐션 온도 한 줄(models/HyperTransformer.py:354): self.temperature -> 런타임 h*w. "
            "생성자가 거기에 넣는 값이 곧 학습 해상도의 h*w 라 학습 크기(H=64)에서는 "
            "비트 단위로 동일하고(max|diff|=0.000e+00), 평가 해상도가 커질 때만 의미를 지킨다. "
            "저장소 소스는 손대지 않고 빌더에서 메서드 교체로 적용한다. "
            "2) N_modules=1 (저장소가 스스로 제공하는 단일헤드 SDPA 분기, :548-550). "
            "논문 기본값 4 의 MHFA 는 nn.Linear(flatten(h*w)) 라 생성 시점 해상도에 "
            "영구 고정된다 -- 64 로 학습하면 H=256 forward 가 "
            "'mat1 and mat2 shapes cannot be multiplied (512x4096 and 256x64)' 로 죽는다(4/8밴드 동일). "
            "**표에 '단일헤드 SDPA 변형'이라고 밝혀야 한다.** "
            "3) 유사 RGB 밴드 색인은 데이터셋 설정값이라 우리 센서에 맞춰 넣었다 "
            "(4밴드 R=2,G=1,B=0 / 8밴드 R=4,G=2,B=1). 저장소 배포값(R=10,G=30,B=60)은 "
            "초분광 전용이고 우리 밴드수에서 가운데 채널이 빈 슬라이스가 되어 NaN 이 된다. "
            "손실 항을 끄는 대신 색인을 고쳤으므로 **네 항 모두 살아 있다**. "
            "4) utils/helpers.py:initialize_weights_new(train.py:128) 는 쓰지 않는다. "
            "BN 없는 16단 잔차 SFE 에 MSRA fan-out 을 걸면 분산이 누적되어 step 0 에서 "
            "|pred|max=3.8e8, 손실 ~2.2e7 이 된다.  저장소는 2단계 모델을 HSIT_PRE 사전학습 "
            "가중치에서 --resume 으로 받기 때문에 문제가 드러나지 않을 뿐이다.  1단계로 "
            "처음부터 학습하는 우리 설정에서는 PyTorch 기본 초기화가 맞다. "
            "5) epochs 150 (+ StepLR step_size 30 = 전체의 20%, gamma 0.1, 5회 감쇠). "
            "저장소의 10000 epoch 은 학습셋이 17-61장일 때의 값이라 옵티마이저 스텝으로는 "
            "~30k 에 불과하다.  PanCollection 은 9,714-19,809장(배치 8 -> 1,215-2,477 step/epoch) "
            "이라 10000 을 그대로 쓰면 12-25M step 이 된다.  **스텝 수가 아니라 스케줄의 모양을 "
            "옮겼다**: 첫 감쇠를 전체의 20% 지점에, gamma 0.1 로 5회 -- 저장소와 같은 비율이다. "
            "150 epoch 은 실측 0.032-0.042 s/step(같은 노드 GPU 경합에 따라) 기준 wv3 1.7-2.1h / "
            "qb 2.9-3.8h / gf2 3.3-4.3h 로 가장 느린 경우에도 가장 큰 split 이 "
            "5 GPU-h 예산 안에서 끝나고, 스텝 수(182k-372k)는 저장소 총량(~30k)의 6-12배다. "
            "6) 죽은 VGG_LFE_HSI/VGG_LFE_PAN 가지(:637-638, 1.11M 파라미터)는 그대로 둔다. "
            "이를 쓰는 줄이 주석 처리(:763-766)되어 있어 conv_RGB 4개 파라미터의 grad 가 "
            "None 으로 돌아오는데, 이건 배선 오류가 아니라 공개 아키텍처 그대로다. "
            "지각 손실용 VGG19 는 저장소처럼 모델과 분리된 객체이고 __dict__ 리스트에 담아 "
            "옵티마이저/state_dict 에서 완전히 제외했다."
        ),
    )
