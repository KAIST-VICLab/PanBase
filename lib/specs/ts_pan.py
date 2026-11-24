"""TS-Pan(TGRS'26) 2단계 MoE 팬샤프닝.  함정: 첫 인자는 업샘플 lms 가 아니라 저해상 ms 이고,
손실에 MoE 부하분산 항 0.3*(CV(spa_cof)+CV(spe_cof)) 이 반드시 들어간다.

레시피 출처 (2026-09-28 재확인).  배포된 학습 진입점(main.py -> solver/solver.py)은 그대로는 돌지 않는다 --
solver/basesolver.py:3-4 가 저장소에 없는 `data.dataset` / `data.data` 를 import 한다.  그래서 값은
돌려 본 것이 아니라 코드에서 읽었다:
  - 손실  L1(mean) + 0.3*(CVLoss(spa_cof)+CVLoss(spe_cof))     solver/solver.py:25-34,85-90 · utils/utils.py:40
  - Adam  lr 1e-3 (option.yml:40) · betas (0.9,0.999) · wd 0     utils/utils.py:26 (weight_decay=0 하드코딩;
          option.yml 의 weight_dency 0.1 · decay 1000 · gamma 0.1 · gclip 0.4 는 어디서도 읽히지 않는다)
  - StepLR(step_size=100 epoch, gamma=0.5), 에폭 끝마다 한 번      solver/solver.py:54,113
  - batch 16 (option.yml:36) · 특징 폭 n_colors 32 (option.yml:37) · 증강 없음 (data/load_train_data.py)
  - 500 epoch (option.yml:4; epoch 1..500 -- basesolver.py:13 + solver.py:205 `while epoch <= nEpochs`)
  - 기울기 클립 없음 (solver.py:107-111 주석 처리)
  - drop_last=True (basesolver.py:23-24) -> 에폭당 floor(N/16)
데이터셋별 설정은 저장소에 없다: option.yml 하나를 경로만 바꿔 qb/gf2 에 썼다(option.yml:8,12-14 의 qb 경로 ·
:13 주석의 gf2 학습 경로 · :26 의 gf2 시험 경로).  그래서 `per_dataset` 가 없다 -- 세 칸 모두 같은 epoch 수 · lr · 스케줄이다.
option.yml:4 주석("For specific number of epochs, please refer to our paper")이 논문을 가리키지만 논문의
데이터셋별 epoch 는 대조하지 못했다 (공개 코드 값 500 을 쓴다 -- 원장에 적는다).  2026-09-28 에 찾아봤다: 논문
(DOI 10.1109/TGRS.2025.3648716, IEEE Xplore 11316500, TGRS vol. 64, 2026) 은 비공개 접근이고 (OpenAlex/Unpaywall
oa_status closed, 이 기관은 구독 없음) arXiv 등 사본이 없다.  공개된 단서는 저자의 GitHub issue #1 답변
(2025-12-29, https://github.com/lzm-01/TS-Pan/issues/1) 뿐인데 "PanCollection 의 QB 는 WV3·GF2 보다 빨리 수렴하니
과적합에 주의하라" · "QB 에 따로 트릭은 필요 없다" 는 정성적 말이고 숫자가 없다 -- 그래서 추측으로 per_dataset 을
두지 않는다 (valid PSNR 최고 선택이 과적합을 덮는다).  논문 사본을 구하면 첫 학습 전에 대조한다.
"""
import os
import time
import torch

_CACHE = {}
_LOCK_STALE_S = 600         # 확장 빌드 lock 이 이보다 오래되면 죽은 프로세스가 남긴 것으로 본다


def _clear_stale_ext_lock(name):
    """torch 의 JIT 빌드 lock (`<확장 루트>/<name>/lock`) 이 죽은 프로세스의 잔재면 치운다.

    torch 2.7.1 의 FileBaton.wait() 는 시한이 없다 (utils/file_baton.py: `while exists(lock): sleep`).
    load() 는 새 프로세스마다 lock 을 잡고 ninja 를 돌리므로 (cpp_extension._jit_compile), 그 사이에
    죽은 프로세스가 하나라도 있으면 이후 TS-Pan import 가 **전부** 영원히 멈춘다 -- 학습 칸뿐 아니라
    이 스펙을 빌드하는 reconcile/make_tables 주기까지.  그래서 import 전에 한 번 본다:
      - lock 이 없으면 그냥 간다.
      - lock 이 _LOCK_STALE_S 보다 어리면 정상 빌드 중일 수 있으니 (두 칸이 동시에 뜨면 서로 몇 초씩
        잡는다) 1초 간격으로 기다리되, 시한을 둔다.
      - 그보다 오래됐고 빌드된 .so 가 있으면 지우고 로그를 남긴다 (정상 경로의 lock 은 ninja 가
        'no work to do' 하는 몇 초, 첫 빌드도 ~15-40s 만 잡힌다).
      - 오래됐는데 .so 가 없으면 빌드가 도중에 죽은 것이라 손으로 봐야 한다 -- 기다리지 않고 멈춘다.
    경로는 load() 와 같은 함수(_get_build_directory: TORCH_EXTENSIONS_DIR 또는 기본 루트)로 푼다.
    이 함수는 code_digest 밖이다 (registry._CODE_FIELDS 는 build/forward/... 의 바이트코드만 본다)."""
    from torch.utils.cpp_extension import _get_build_directory
    d = _get_build_directory(name, False)
    lock, so = os.path.join(d, "lock"), os.path.join(d, f"{name}.so")
    while True:
        try:
            age = time.time() - os.path.getmtime(lock)
        except FileNotFoundError:
            return
        if age < _LOCK_STALE_S:
            time.sleep(1.0)
            continue
        if os.path.exists(so):
            try:
                os.remove(lock)
            except FileNotFoundError:
                pass
            print(f"[TS-Pan] 오래된 확장 빌드 lock 을 치웠다: {lock} (나이 {age / 60:.1f}분, {so} 있음)", flush=True)
            return
        raise RuntimeError(
            f"TS-Pan: 확장 빌드 lock {lock} 이 {age / 60:.1f}분째 남아 있고 {so} 가 없다 -- 빌드 도중에 죽은 "
            f"프로세스의 잔재다.  {d} 를 확인하고 lock 을 지운 뒤 다시 돌린다 (torch 는 여기서 영원히 기다린다)")


def _net_cls(REPOS, _pkg):
    """`model.tspan.Net` 을 한 번만 적재해 캐시한다.

    두 가지 이유로 캐시가 필요하다.  (1) model/rwkv.py 가 import 시점에 Bi-WKV CUDA
    확장을 빌드한다 -- 첫 빌드 ~40s, 이후는 ninja 가 넘어가지만 재적재마다 autograd
    Function 클래스가 새로 생긴다.  (2) 다른 스펙이 나중에 `_pkg("model")` 로 `model`
    이름을 자기 저장소로 갈아끼워도, 클래스 객체를 이미 들고 있으면 영향을 받지 않는다.

    시퀀스 상한은 둘이다 (2026-09-28, rwkv.py 패치).  T = H*W 다.
      - TSPAN_T_MAX = 262144 (여기서 건다): WKV.forward 의 `assert T <= T_MAX` 만 정한다.  저장소 기본
        65536 이면 평가 H=512 에서 죽는다.  kernel_forward 는 Tmax 를 쓰지 않으므로 커널과는 무관하다.
      - TSPAN_T_MAX_BWD = 4096 (기본값, 저자 학습 크기 64*64 -- rwkv.py:8 주석): 확장의 -DTmax 이자
        WKV.backward 의 assert 기준.  -DTmax 는 kernel_backward 의 스레드별 지역 배열
        `y/z/zexp[Tmax/32]` (cuda/wkv_cuda.cu:7,130) 만 키우는데, 드라이버가 첫 backward 때
        (스택 프레임 x 상주 스레드) 만큼 지역 메모리를 잡아 두고 이것은 torch.cuda.max_memory_allocated
        에 안 보인다.  ptxas: 프레임 1,536 B (Tmax 4096) / 24,576 B (65536) / 98,304 B (262144).
        4090 GPU1 실측: 첫 backward 의 장치 사용량 증가가 4096 빌드 ~116 MiB, 65536 빌드 4,416 MiB
        (262144 는 그 4배, ~17GiB 로 추정 -- 8GB 한도 때문에 재지 않았다).  예전 설명의 '262144 로
        올려도 메모리 동일' 은 torch 할당기 peak 만 본 것이라 틀렸다.  두 빌드의 y/gw/gu/gk/gv 는
        T=4096 에서, T=262144 forward 출력은 비트 단위로 같다 (수학 불변).  확장 이름이
        wkv_bwd<Tmax> 라 예전 wkv_262144 빌드를 다시 쓰지 않는다.

    GPU 가 안 보이는 프로세스에서는 짓지 않는다 (2026-09-28).  torch 2.7.1 의 JIT 빌드는
    TORCH_CUDA_ARCH_LIST 가 없으면 **보이는 카드**에서 아키텍처를 고르는데
    (cpp_extension._get_cuda_arch_flags), 카드가 0 장이면 `arch_list[-1] += '+PTX'` 에서 IndexError 로
    죽는다 -- 그것도 공유 빌드 디렉터리(~/.cache/torch_extensions/py39_cu126/wkv_bwd4096)에 `lock` 을
    잡은 **뒤**에.  디스패처는 게이트(make_tables._recipe_now 의 파라미터 셈)를 CUDA 를 숨긴 자식
    프로세스에서 돌리므로 표 방법(make_tables.BASELINES)이 되면 매 주기 이 길을 밟는다.  그 사이에 프로세스가 죽으면 lock 이 남고,
    FileBaton.wait() 는 시한이 없어 이후 TS-Pan 학습이 전부 import 에서 영원히 멈춘다.  여기서 먼저
    멈추면 게이트 결과는 같고(파라미터 '모름' -> recipe_diff 가 그 항목을 뺀다) 공유 디렉터리를
    건드리지 않는다.  TORCH_CUDA_ARCH_LIST 를 명시하면 CPU 에서도 빌드한다 (시험용 -- 이때는
    TORCH_EXTENSIONS_DIR 도 따로 줘야 공유 .so 를 다른 플래그로 다시 빌드하지 않는다).
    이 가드는 CUDA 를 숨긴 디스패처 게이트만 막는다.  GPU 가 보이는 프로세스(학습 · reconcile/plan.sh ·
    make_tables/autotable.sh 주기)는 load() 에서 lock 을 잡으므로, import 직전에 _clear_stale_ext_lock 이
    죽은 lock 을 치우거나 (10분 넘고 .so 있음) 분명한 오류로 멈춘다 -- 영원히 기다리지 않는다."""
    if "Net" not in _CACHE:
        if not torch.cuda.is_available() and not os.environ.get("TORCH_CUDA_ARCH_LIST"):
            raise RuntimeError(
                "TS-Pan: Bi-WKV CUDA 확장(repos/TS-Pan/model/rwkv.py:21)은 import 때 JIT 빌드되는데 "
                "이 프로세스에는 보이는 GPU 가 없다 -- 짓지 않는다 (CPU 빌드는 TORCH_CUDA_ARCH_LIST 와 "
                "TORCH_EXTENSIONS_DIR 를 함께 준다)")
        want = 262144                       # forward assert: 512*512 까지 커버
        cur = int(os.environ.get("TSPAN_T_MAX", 0) or 0)
        os.environ["TSPAN_T_MAX"] = str(max(cur, want))
        # backward 컴파일 상한 (= 확장 이름).  rwkv.py 와 같은 기본값을 명시해 둘이 어긋나지 않게 한다.
        os.environ.setdefault("TSPAN_T_MAX_BWD", str(64 * 64))
        # GPU 가 보이는 프로세스(학습 · reconcile · make_tables 주기)도 죽은 lock 에 영원히 걸리지 않게.
        _clear_stale_ext_lock(f"wkv_bwd{int(os.environ['TSPAN_T_MAX_BWD'])}")
        _pkg(REPOS / "TS-Pan" / "model")    # tspan.py 가 `from model.HFG_S2EN import ...` 를 쓴다
        from model.tspan import Net
        _CACHE["Net"] = Net
    return _CACHE["Net"]


def _cv(logits):
    """저장소의 CVLoss (solver/solver.py:25-34).  loss_weight=1.0 기본값 그대로.
    std 는 expert 축(dim=1, num_experts=8) 위에서 torch 기본 unbiased=True 로 잡는다."""
    cv = torch.std(logits, dim=1) / torch.mean(logits, dim=1)
    return torch.mean(cv ** 2)


def register(SPECS, REPOS, _load, _pkg):
    def build(ms_ch=4, ratio=4):
        Net = _net_cls(REPOS, _pkg)
        # dim=32 는 밴드 수가 아니라 **특징 폭**이다.  solver/solver.py:47 이
        # cfg['data']['n_colors'] 를 넘기고 option.yml:37 이 n_colors: 32 로 둔다.
        # 밴드 수는 별도 인자 ms_dim (저장소는 8 하드코딩, 패치로 인자화).
        # num_experts=8 / top_k=4 / num_heads=2 / win_size=4 는 전부 공표 기본값.
        return Net(dim=32, ms_dim=ms_ch)

    # 평가 1청크당 픽셀 예산.  256*256 에서 B=4, 512*512 에서 B=1 이 된다 (~1.5GB).
    _EVAL_PIX = 4 * 256 * 256

    def _forward(mm, ms_up, pan, lrms, ctx):
        # 첫 인자는 **저해상 ms** 다.  solver/solver.py:79 의 지역변수 이름이 `lms` 라
        # 헷갈리지만 load_train_data.py:44-49 의 반환 순서상 batch[4] 는 ms 이고,
        # tspan.py:261 이 그 텐서에 x4 PixelShuffle 업샘플을 건다.  ms_up 을 넣으면
        # 출력이 4배 커진다.  ms_up 은 쓰이지 않는다(모델이 업샘플러를 학습한다).
        #
        # 하니스의 평가 배치는 스펙 eval_bs(없으면 64, train_pancoll.py:390)이고 FR 은 4장씩인데,
        # 이 모델은 RWKV 시퀀스 길이가 H*W 라 256x256 을 64장 한 번에 넣으면 24GB 가 터진다.
        # 그래서 여기서 쪼갠다.  배치 결합은 없다: 게이트는 샘플별 GAP -> top-k 이고
        # (tspan.py:133-155, dim=1 은 expert 축), 전문가 디스패치도 샘플별 마스크이며
        # (tspan.py:167-182), BatchNorm 6개는 track_running_stats=True 라 eval 에서
        # running stat 을 쓴다.  실측으로도 확인했다 -- TF32 를 끄면 옆자리에 0 텐서를
        # 넣든 자기 복사본을 넣든 차이가 6.02e-06 으로 같다.  청크 크기에 따라 남는 1.2e-4
        # (H=256) 는 cudnn conv 알고리즘/TF32 반올림이고, 저장소 추론도 한 장씩 돈다.
        # 청크는 바깥 배치와 무관하게 픽셀 예산으로만 정해진다 -- 학습기 · dump_outputs 가
        # 같은 영상에 같은 청크 경계를 쓴다 (RR 20장 -> 4장씩, FR -> 1장씩).
        n, H, W = lrms.shape[0], pan.shape[-2], pan.shape[-1]
        c = max(1, _EVAL_PIX // max(1, H * W))
        if n <= c:
            return mm(lrms, pan)[0].clamp(0, 1)    # 저장소도 추론에서만 clamp (testsolver_1.py:66)
        return torch.cat([mm(lrms[i:i + c], pan[i:i + c])[0].clamp(0, 1)
                          for i in range(0, n, c)], dim=0)

    def _step(model, b, ctx, lossfn):
        # solver/solver.py:85-90 그대로.
        #   loss = L1(sr, gt) + (CV(spa_cof) + CV(spe_cof)) * 0.3
        # 뒤 항은 MoE 부하분산 정규화라 sr 에 직접 닿지 않지만, 1스텝부터 ~0.686 을
        # 차지하고 거의 움직이지 않는다.  빼면 8개 전문가가 붕괴한다 -- 반드시 살린다.
        # gradient clipping 은 저장소에서 주석 처리돼 있다(solver.py:107-111).
        sr, spa_cof, spe_cof = model(b["ms"], b["pan"])
        loss = lossfn(sr, b["gt"]) + (_cv(spa_cof) + _cv(spe_cof)) * 0.3
        return sr, loss

    SPECS["TS-Pan"] = dict(
        tier="confirmed",   # (2026-09-28): 스모크 통과, 표에 올린다
        build=build, venue="TGRS 2026", loss="l1", batch=16, epochs=500,
        opt=dict(kind="adam", lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0),
        sched=dict(kind="step", step_size=100, gamma=0.5),   # solver.py:54, 에폭당 1회
        forward=_forward,
        train_step=_step,
        # 결정적 방법이다 (dump_outputs 의 seed-0 규칙 대상 아님).  게이트의 '잡음'(tspan.py:141-148)은
        # 난수가 아니라 fc0(입력) 의 표준화라 입력의 결정적 함수이고, 확률 요소는 SSAIB 의
        # Dropout2d(0.1)(my_modules.py:99,111) 하나인데 eval() 에서 꺼진다.
        stochastic=False,
        # 일부러 두지 않는 키 (2026-09-28):
        #  - grad_clip: 저장소는 클립하지 않지만 하네스 기본 10.0 을 그대로 쓴다 (사용자 결정 2026-09-28).
        #    키를 적으면 make_ledger.CLIP_SPECS 가 '저장소 클립 값을 스펙에 옮긴 방법' 으로 싣는다 -- 틀린 공개다.
        #    레시피 스냅샷의 grad_clip 은 어느 쪽이든 10.0 (train_pancoll.py `CLIP`, make_tables._recipe_now).
        #  - per_dataset: 저장소에 데이터셋별 값이 없다 (모듈 머리말).  epoch 기반이라 반복 수 보정도 필요 없다.
        #  - eval_after_frac: 학습 중에 굳는 구조가 없다 (ARConv 류가 아니다) -- 첫 평가부터 best 후보다.
        diff="저장소 4곳을 고쳤고 기여(2단계 MoE·HFG-LGEB·Bi-WKV)는 바이트 단위로 불변이다 (patches/TS-Pan.patch). "
             "(1) model/tspan.py:4 의 깨진 import (`model.HF_guided` -> 실제 파일명 "
             "`model.HFG_S2EN`) -- 배포본 그대로는 ImportError 라 실행 자체가 안 된다. "
             "(2) WV3 8밴드가 tspan.py 와 HFG_S2EN.py 에 하드코딩돼 있어 생성자 인자 "
             "ms_dim 으로 빼냈다(기본값 8 유지 -> 원래 호출부는 비트 동일). "
             "(3) model/rwkv.py 의 시퀀스 상한(T=H*W)을 둘로 나누고 CUDA 소스 경로를 절대경로로 바꿨다(빌드 배관만, "
             "커널 소스 불변).  forward 는 파이썬 assert 만 TSPAN_T_MAX=262144 로 둔다 -- 저장소 기본 65536 이면 "
             "평가 H=512 에서 죽고, kernel_forward 는 Tmax 를 쓰지 않는다.  확장은 -DTmax=TSPAN_T_MAX_BWD=4096 "
             "(저자 학습 크기 64x64, rwkv.py:8 주석) 로 wkv_bwd4096 이라는 이름으로 짓고 backward 가 T<=4096 을 "
             "assert 한다.  -DTmax 는 kernel_backward 의 스레드별 지역 배열만 키우는데 드라이버가 첫 backward 때 "
             "(프레임 x 상주 스레드) 만큼 장치 메모리를 잡아 torch 통계에 안 보인다 -- 프레임이 1,536 B (4096) 대 "
             "98,304 B (262144) 이고, 4090 실측 증가분이 ~116 MiB (4096) 대 4,416 MiB (65536) 다.  두 빌드의 출력과 "
             "기울기(y/gw/gu/gk/gv, T=4096)와 T=262144 forward 출력은 비트 단위로 같다. "
             "손실은 `L1(sr,gt) + 0.3*(CVLoss(spa_cof)+CVLoss(spe_cof))` 전부 재현했다. "
             "(4) model/rwkv.py 가 WKV 버퍼를 device='cuda'(=항상 cuda:0)로 잡고 확장 "
             "모듈에 CUDAGuard 가 없어 --device cuda:1 에서 죽었다.  버퍼를 입력 텐서의 "
             "디바이스로 잡고 커널 호출을 torch.cuda.device 로 감쌌다 -- cuda:0 결과와 "
             "max|diff|=0 으로 동일함을 확인했다(수학 불변, 배관만). "
             "**레시피**: 배포된 학습 진입점은 solver/basesolver.py:3-4 가 없는 data.dataset/data.data 를 import 해 "
             "그대로는 돌지 않아, 값은 solver/solver.py · utils/utils.py · option.yml 에서 읽었다 -- Adam lr 1e-3, "
             "weight decay 0 (utils/utils.py:26; option.yml 의 weight_dency 0.1 은 읽히지 않는다), StepLR(100 epoch, 0.5) "
             "에폭당 한 번 (solver.py:54,113), batch 16, 500 epoch (option.yml:4,36), 증강 없음.  세 데이터셋 모두 같은 값이다 "
             "-- 저장소에 데이터셋별 설정이 없다.  option.yml:4 주석은 데이터셋별 epoch 를 논문에 미루지만 논문 값과는 "
             "대조하지 못했다 (공개 코드의 500 을 쓴다) -- 논문(DOI 10.1109/TGRS.2025.3648716)은 비공개 접근이고 사본이 없으며, "
             "저자의 GitHub issue #1 답변은 'QB 가 WV3·GF2 보다 빨리 수렴하니 과적합에 주의' 라는 정성적 말뿐이다 (2026-09-28 확인). "
             "**저장소와 다른 점**: (a) 기울기 클립 -- 저장소는 클립하지 않는다 (solver.py:107-111 주석 처리, option.yml "
             "gclip 0.4 는 쓰이지 않는다).  하네스 기본 10.0 을 건다 (사용자 결정 2026-09-28) -- 걸린 step 수는 결과 JSON 의 "
             "grad_clip_hits 에 있다. (b) 에폭 길이 -- 저장소는 drop_last=True (basesolver.py:23-24) 라 에폭당 floor(N/16) = "
             "607/1071/1238 step (wv3/qb/gf2), 하네스는 ceil = 608/1072/1239 step 이고 복원추출이다 (하네스 공통); lr 감쇠는 "
             "하네스의 에폭 경계(100·200·300·400 epoch)에서 걸린다. (c) 체크포인트 선택 -- 저장소는 고르지 않는다 (10 epoch "
             "마다 저장, solver.py:208-209; option.yml:30 의 150.pth 는 QB 로 학습한 체크포인트를 GF2 에 교차 시험하는 설정이다, option.yml:12,26,31).  우리는 valid PSNR 최고를 "
             "싣는다 (하네스 공통). (d) 저장소는 cudnn.deterministic=True · benchmark=False 로 돈다 (solver.py:163-164); "
             "하네스는 torch 기본값이다 (수치 결정성만 다르다). (e) 하네스는 zero_grad(set_to_none=True) 다 -- torch>=2.0 의 "
             "기본값이고 저장소의 optimizer.zero_grad() (solver.py:82) 도 torch>=2.0 이면 같다 (저장소는 torch 버전을 "
             "고정하지 않는다; torch<2.0 이면 뽑히지 않은 전문가도 Adam 모멘텀으로 계속 움직였을 것이다). "
             "노이지 top-4/8 MoE 특성상 매 스텝 전문가 4개와 게이트 잡음 분기 fc0 가 "
             "grad=None 이다(저장소 설계 그대로, 배선 오류 아님).  게이트 '잡음' 은 입력의 결정적 함수라 "
             "추론은 결정적이다 (stochastic 아님).  모델은 CUDA 전용이다 "
             "-- WKV 연산에 CPU 경로가 없다. "
             "평가 forward 는 하니스의 평가 배치를 픽셀 예산으로 쪼갠다(256px 에서 4장씩, "
             "512px 에서 1장씩; 64px 에서는 64 그대로) -- RWKV 시퀀스가 H*W 라 안 쪼개면 "
             "256px x64 장에서 24GB 가 터진다.  배치 결합이 없음을 확인했다: TF32 를 끄고 "
             "cudnn.deterministic 을 켠 상태에서 sample0 옆에 0 텐서를 넣든 sample0 복사본을 "
             "넣든 출력 차이가 6.020069e-06 으로 **완전히 같다**(=옆자리 내용과 무관). "
             "게이트 계수는 비트 단위로 같고 BatchNorm 6개는 track_running_stats=True 다. "
             "청크 크기에 따라 남는 1.2e-4(H=256) 는 배치 크기가 바뀌면 cudnn 이 다른 conv "
             "알고리즘/TF32 를 고르는 데서 오는 반올림이다(TF32 를 끄면 1.3e-3 -> 1.2e-4). "
             "저장소 추론도 한 장씩 돌린다(testsolver_1.py:60-65).")
