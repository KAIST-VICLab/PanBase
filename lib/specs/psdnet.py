"""PSDNet (TGRS 2023) -- 멀티패치 점진 팬샤프닝 + 교사/학생 지식증류.  틀리기 쉬운 곳:
학습이 **2단계**라 앞의 9/200 에폭은 GT 를 보는 교사만 학습하고 학생에는 그래디언트가
한 번도 가지 않는다 (optimize1.py:69 `if epoch<=9`).  이 구간을 빼면 증류항이 붙을
교사가 없어 논문의 기여가 통째로 사라진다."""
import torch


def register(SPECS, REPOS, _load, _pkg):
    import torch.nn as nn

    # 저장소 스케줄: nEpochs=200 (optimize1.py:30), 교사 구간 epoch<=9 (optimize1.py:69).
    PHASE_A_EPOCHS, TOTAL_EPOCHS = 9, 200
    PHASE_A = PHASE_A_EPOCHS / TOTAL_EPOCHS          # = 0.045.  eval_after_frac 로 쓴다.
    MSE = nn.MSELoss()                               # optimize1.py:223 -- 증류항 전용
    # 보고용 파라미터 수 = 추론 네트워크(학생 Net) 뿐이다.  하네스 `n_params`(결과 JSON `params`)는
    # 교사까지 센 합계라 표에 그대로 실으면 두 배가 된다 -- make_tables 가 이 값을 읽는다.
    # build 가 저장소 코드로 다시 세어 대조한다 (저장소가 바뀌면 조용히 틀린 수를 싣지 않게).
    REPORT_PARAMS = {4: 2_783_250, 8: 2_789_074}

    # ------------------------------------------------------------------ build
    def build(ms_ch=4, ratio=4):
        if ratio != 4:
            raise ValueError("PSDNet 은 x4 전용이다 -- Net.up 이 scale_factor=4 고정이고 "
                             "(dbpn.py:795) forward 가 PAN 패치를 H*4/2 로 자른다 (dbpn.py:803).")
        # dbpn.py 는 `from Networks.base_networks import *` 로 절대 임포트를 한다.
        # sys.path 에 얹으면 다른 저장소가 먼저 잡은 동명 패키지에 가릴 수 있으므로
        # 패키지 객체를 직접 박아 둔다.
        _pkg(REPOS / "PSDNet" / "Networks", "Networks")
        M = _load(REPOS / "PSDNet" / "Networks" / "dbpn.py", "psdnet_dbpn")
        # optimize1.py:219-220 은 `Net()` / `NetGT()` 로 전부 기본값을 쓴다.  기본값을
        # 그대로 적어 두되 num_channels 만 밴드 수로 준다 (dbpn.py:730,539 에서 이미 인자다).
        kw = dict(num_channels=ms_ch, base_filter=28, n_feat=28, num_stages=3,
                  scale_factor=4, scale_feat=28, kernel_size=3, reduction=4, bias=False)

        class PSDNet(nn.Module):
            """학생(Net)과 교사(Net_GT)를 한 모듈에 담는다.

            교사를 모듈 트리 **안**에 두는 것은 의도적이다: 교사는 단계 A 에서 실제로
            학습돼야 하므로 옵티마이저가 반드시 봐야 하고, state.pt · best.pt 에 함께
            실려야 이어 돌리기(exact · warm)가 증류 소스를 잃지 않는다.  단계 B 에서는
            no_grad 로만 호출하므로 grad 가 None 이 되고, Adam 은 grad 가 없는 파라미터를
            건너뛴다 -- 저장소가 optimizer_GT.step() 을 멈추는 것과 같다.

            보고용 파라미터 수는 학생 2,783,250 (C=4) / 2,789,074 (C=8) 이다 (REPORT_PARAMS).
            하네스가 찍는 합계는 교사를 포함해 그 두 배가 된다 (diff (6)).
            """

            def __init__(self):
                super().__init__()
                self.student = M.Net(**kw)          # 평가/보고 대상
                self.teacher = M.Net_GT(**kw)       # GT 를 보는 특권 네트워크, 증류 소스

            def forward(self, lrms, pan):
                # 평가 경로.  (img1, img2, img3, feat1, feat2) 중 img3 가 최종 예측이다
                # (optimize1.py:128-132, test.py:77-87 둘 다 prediction3 만 점수화한다).
                return self.student(lrms, pan)[2]

        m = PSDNet()
        n_st = sum(p.numel() for p in m.student.parameters() if p.requires_grad)
        if ms_ch in REPORT_PARAMS and n_st != REPORT_PARAMS[ms_ch]:
            raise AssertionError(f"PSDNet 학생 파라미터 {n_st:,} != REPORT_PARAMS[{ms_ch}] "
                                 f"{REPORT_PARAMS[ms_ch]:,} -- 저장소 코드가 바뀌었다")
        return m

    # -------------------------------------------------------------- optimizer
    def make_opt(model, o):
        """저장소는 Adam 을 둘 만든다 (optimize1.py:229-230).  하이퍼파라미터는 완전히
        같고(lr=MSlr=5e-4, betas=(0.9,0.999), eps=1e-8, wd 없음) **감쇠 시작 에폭만**
        다르다 (optimize1.py:271-276, 에폭 **머리**에서 곱한다):
            교사  epoch>1  & epoch%5==0  -> 에폭 5, 10, 15, ... 부터 0.95 배
            학생  epoch>10 & epoch%5==0  -> 에폭 15, 20, 25, ... 부터 0.95 배
        에폭 E 동안의 lr 은 교사 5e-4*0.95^floor(E/5), 학생(E>=10) 5e-4*0.95^(floor(E/5)-2).
        Adam 은 파라미터별로 독립이라 하이퍼파라미터가 같은 두 옵티마이저는 하나로
        합쳐도 수학적으로 동일하다 (grad 가 None 인 파라미터는 step 에서 건너뛰므로 학생의
        Adam step 수 · 모멘트는 저장소의 optimizer2 처럼 단계 B 첫 step 에서 0 부터 시작한다).
        남는 차이는 감쇠 시작 에폭 하나뿐이므로 파라미터 그룹을 둘로 나누고 학생 그룹의
        초기 lr 만 0.95^-2 배 해 둔다.  스케줄러(`sched`, 아래)가 두 그룹을 같은 지점에서
        함께 감쇠시키면 학생은 단계 B 전 구간(에폭 10-200)에서 저장소와 **같은** lr 을 밟고,
        교사는 단계 A(에폭 1-9)에서 5e-4 -> (에폭 5 부터) 4.75e-4 를 그대로 밟는다.
        (단계 A 의 학생 그룹 lr 값 · 단계 B 의 교사 그룹 lr 값은 grad 가 없어 쓰이지 않는다.)"""
        lr = o["lr"]
        return torch.optim.Adam(
            [{"params": list(model.teacher.parameters()), "lr": lr},
             {"params": list(model.student.parameters()), "lr": lr / 0.95 ** 2}],
            lr=lr, betas=o.get("betas", (0.9, 0.999)), eps=1e-8,
            weight_decay=o.get("weight_decay", 0.0))

    # -------------------------------------------------------------- train_step
    def train_step(model, batch, ctx, lossfn):
        """공식 손실 전부.  단계 A 는 L1 3항, 단계 B 는 L1 3항 + 1e-6 * MSE 2항."""
        gt, lrms, pan = batch["gt"], batch["ms"], batch["pan"]
        # batch["lms"](업샘플 MS)는 쓰지 않는다 -- Net 이 내부에서 bicubic x4 를 직접
        # 만든다 (dbpn.py:795,800).  저장소도 lrms 와 pan 만 먹인다.

        # 단계 경계는 **정수 step** 으로 가른다: step 1..total*9//200 (= 에폭 1-9 의 마지막 step
        # 까지) 이 교사다.  예전의 `step/total < 9/200` 은 부동소수에서 9*spe/total == 0.045 가
        # 정확히 같아져 에폭 9 의 마지막 step 하나를 학생 step 으로 돌렸다 (wv3 10,934 대 10,935).
        if ctx["step"] <= ctx["total"] * PHASE_A_EPOCHS // TOTAL_EPOCHS:
            # ---- 단계 A: 교사만 학습 (optimize1.py:82-91).  학생은 forward 조차 안 한다.
            q1, q2, q3, _, _ = model.teacher(gt, lrms, pan)
            loss = lossfn(q1, gt) + lossfn(q2, gt) + lossfn(q3, gt)
            # 반환 예측은 교사 출력이다 (GT 를 본 값이라 지표로 쓰면 안 된다).  하네스는
            # train_step 경로에서 이 텐서를 로그에도 쓰지 않는다 -- 체크포인트 선택은
            # eval_after_frac 로 이 구간 전체를 건너뛴다.
            return q3.detach(), loss

        # ---- 단계 B: 학생만 학습, 교사 동결 (optimize1.py:140,154-164).
        model.teacher.eval()                      # 저장소 optimize1.py:140.  (dbpn.py 의
                                                  # BatchNorm 23 곳은 전부 주석이라 실효는
                                                  # 없지만 저장소 그대로 둔다.)
        with torch.no_grad():
            _, _, _, GT_feat1, GT_feat2 = model.teacher(gt, lrms, pan)
        p1, p2, p3, outfeat1, outfeat2 = model.student(lrms, pan)
        loss_distill = 1 * (MSE(outfeat2, GT_feat2) + MSE(outfeat1, GT_feat1))   # :163
        loss = lossfn(p1, gt) + lossfn(p2, gt) + lossfn(p3, gt) + 1e-6 * loss_distill  # :164
        return p3, loss

    SPECS["PSDNet"] = dict(
        tier="confirmed",   # (2026-09-28): 스모크 통과, 표에 올린다
        build=build, venue="TGRS 2023", loss="l1", batch=8, epochs=200,
        opt=dict(kind="adam", lr=5e-4, betas=(0.9, 0.999), weight_decay=0.0),
        make_opt=make_opt,
        # 저장소는 에폭 **머리**에서 곱한다 (optimize1.py:271-276, 에폭 5,10,...,200).  학습기는
        # 에폭 **끝**(`step % spe == 0`)에서 sched.step() 을 부르므로, 에폭 e 끝의 감쇠 = 에폭 e+1
        # 머리의 감쇠다.  그래서 지점을 한 에폭 앞당겨 4,9,...,199 (40개) 에 둔다 -- StepLR(5)
        # 로 두면 두 그룹 모두 한 에폭 늦고, 학생의 에폭 10 이 5e-4/0.95 로 돈다.
        sched=dict(kind="multistep", milestones=list(range(4, TOTAL_EPOCHS, 5)), gamma=0.95),
        forward=lambda mm, ms_up, pan, lrms, ctx: mm(lrms, pan),
        train_step=train_step,
        eval_after_frac=PHASE_A,     # 학생이 아직 한 스텝도 안 배운 구간에서 best.pt 를
                                     # 고르면 안 된다 (그리고 평가 시간도 아낀다).
        # 데이터셋별 덮어쓰기 없음: 저장소 학습 스크립트는 하나(optimize1.py)이고 배치 8 · 200 에폭 ·
        # lr 5e-4 · 감쇠 규칙이 센서와 무관하다.  4밴드도 dbpn.py 의 Net(num_channels=4) 를 쓴다 --
        # optimize1.py:16-18 이 dbpn 을 import 하고 dbpn_QB_GF2 줄은 주석이며, dbpn_QB_GF2.py 는
        # 64x64 학습 패치에서 shape 오류로 죽는다 (diff 머리).
        per_dataset={},
        # 평가는 학생 한 번의 결정적 forward 다 (dbpn.py 에 Dropout · 난수 없음).
        stochastic=False,
        report_params=REPORT_PARAMS,
        diff="저장소 소스 수정 없음 -- num_channels 가 이미 생성자 인자라 4/8 밴드가 그대로 "
             "된다 (dbpn.py:730,539).  QB/GF2(4밴드)도 dbpn.py 의 Net/Net_GT(num_channels=4) 를 쓴다: "
             "공개 학습 스크립트가 dbpn 을 import 하고(optimize1.py:16-17) dbpn_QB_GF2 줄은 주석이다 "
             "(:18).  Networks/dbpn_QB_GF2.py 는 기본값 외에 DBPN_block1 의 이음 루프가 다르다 -- W 를 "
             "저해상 특징 폭으로 바꿔 놓고 루프는 고해상 폭 기준 인덱스(W//4-8, W-32)를 그대로 써서, "
             "64x64 학습 패치에서 shape 오류로 죽고 256/512 에서는 다른 열을 섞는다 (같은 가중치에서 "
             "최대 차 3.8e-3, CPU 확인).  그 파일로는 PanCollection 을 학습할 수 없다.\n"
             "(1) 학습 루프: optimize1.py 는 2단계다.  에폭 1-9 는 교사(Net_GT, GT 를 입력으로 "
             "받는 특권 네트워크)만 L1(q1..q3,gt) 로 학습하고, 에폭 10-200 은 교사를 eval+no_grad "
             "로 얼린 뒤 학생만 L1(p1..p3,gt) + 1e-6*(MSE(of2,GT_f2)+MSE(of1,GT_f1)) 로 학습한다. "
             "학습기 일정이 저장소와 같은 200 에폭 x ceil(N/8) step 이므로 경계는 step "
             "total*9//200 (= 에폭 9 의 마지막 step) 이다.  단계 A 를 없애고 학생만 돌리면 증류항의 "
             "교사가 랜덤 초기화 상태가 되어 논문의 기여가 사라진다.\n"
             "(2) 증류 계수 1e-6 은 수치적으로 거의 0 이지만(raw 0.018 -> 1.8e-8 대 총손실 0.95) "
             "저장소 값 그대로 둔다.  빼면 단계 A 의 존재 이유 자체가 없어진다.\n"
             "(3) optimize1.py:160-162 의 erggg() ERGAS 3항은 손실에 **들어가지 않는다** "
             "(epoch_loss 출력용).  학습항으로 넣으면 저장소보다 강한 베이스라인이 된다.\n"
             "(4) 교사 배선은 optimize1.py:155 의 실제 호출 `model_GT(target, LR, pan)` = "
             "(gt, lrms, pan) 을 따른다.  optimize1.py:75 의 3튜플 언패킹은 주석 처리된 옛 "
             "데이터셋(dataset.py:104-110) 잔재로, 현 5튜플 로더에서는 pan_label 을 lrms 자리에 "
             "넣어 채널 수부터 깨진다 -- 복사하면 안 된다.\n"
             "(5) 옵티마이저 · 스케줄: 저장소의 Adam 2개는 하이퍼파라미터가 같고 lr 감쇠 시작 에폭만 "
             "다르다.  make_opt 으로 교사/학생 파라미터 그룹을 나누고 학생 그룹 초기 lr 만 0.95^-2 "
             "배 했다.  저장소는 에폭 머리에서 감쇠하고 학습기는 에폭 끝에서 sched.step() 을 부르므로 "
             "MultiStepLR 지점을 에폭 4,9,...,199 에 두어, 교사(에폭 1-9)와 학생(에폭 10-200)이 "
             "저장소와 에폭 단위로 같은 lr 을 밟는다 (CPU 에서 200 에폭 전부 대조).  --GTlr(2e-4)는 "
             "저장소가 선언만 하고 읽지 않으므로 교사도 5e-4 다.\n"
             "(6) 하네스가 찍는 파라미터 수(결과 JSON params)는 교사+학생 합계(4밴드 5,566,500 / "
             "8밴드 5,578,148)다.  표에 싣는 값은 추론 네트워크인 학생 2,783,250(4밴드)/2,789,074(8밴드) "
             "이고 스펙의 report_params 에 있다.  같은 크기의 교사는 에폭 1-9 에만 학습되고 추론에 "
             "쓰이지 않는다.\n"
             "(7) 저장소는 증강도 그래디언트 클리핑도 없다 (dataset.py 의 augment() 는 호출부가 "
             "없고 --patch_size 도 미사용).  하네스 공통 클립 10.0 이 걸린다 (사용자 결정 "
             "2026-09-28: 하네스 기본값 유지, 저장소와 다른 점으로 공개).  표본 추출은 하네스 공통 "
             "복원추출이다 (저장소는 DataLoader shuffle=True).\n"
             "(8) Net 의 trans_enc2/3, trans_dec2/3 8개 텐서는 만들어지고 forward 에서 한 번도 "
             "쓰이지 않는다.  단일 GPU Adam 에는 무해하다 (DDP 라면 find_unused_parameters 필요).\n"
             "(9) 저장소는 8비트 WV2 tif 를 /255 로 읽어 학습했다 (dataset.py:153-154).  우리는 "
             "PanCollection 을 하네스 공통 나눗수(2047/1023)로 [0,1] 에 둔다 -- 입력 범위는 같다.")
