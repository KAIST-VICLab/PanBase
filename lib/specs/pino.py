"""PINO (NeurIPS 2025) -- LIIF 좌표 디코더 + 양방향 트랜스포머 인코더.

틀리기 쉬운 지점: 모델은 **[-1,1] 공간**에서 돌고(입력·GT·손실 전부), 예측은 질의 좌표에서
뽑은 ms_up 값 위에 얹는 **잔차**이며, 공식 손실은 `L1 + 0.01 * HistogramLoss('emd', 64)` 이다.
[0,1] 을 그대로 먹이면 L1 스케일이 절반이 되고 히스토그램 bin([-0.05,1.05])이 엉뚱한 곳에
놓인다.  히스토그램 항을 빼면 겉보기엔 잘 돌지만 다른 목적함수를 학습한다.

**저자의 [0,1] 은 언제나 DN/2047 이다 -- GF2 도** (datasets/image_folder.py:49, 데이터셋 구분 없음;
models/histogram_loss.py:212 에 `1023.0 if 'GF2'` 줄이 주석 처리돼 있다).  하네스 데이터는 DN/나눗수
(wv3 · qb 2047, gf2 1023) 라 gf2 만 저자 공간과 2배 어긋난다.  그래서 모델에 `pino_in_scale`
(= 하네스 나눗수 / 2047; wv3 · qb 1.0, gf2 1023/2047) 버퍼를 두고 입력 · GT 를 저자 공간으로 옮긴 뒤
[-1,1] 로 정규화한다.  값은 per_dataset 이 `opt["pino_data_rescale"]` 로 주고 `make_opt` 가 학습 시작 때
버퍼에 쓴다 -- 스펙 함수가 데이터셋 이름을 받는 길이 이것뿐이다.  버퍼는 state_dict 에 실려 best.pt ·
state.pt 와 함께 다니므로 가중치를 읽는 모든 도구(dump_outputs · render_visual · redo_zeroshot ·
make_checkpoints)가 같은 값을 쓴다.  wv3 · qb 는 1.0 을 곱하고 나눌 뿐이라 예전 스펙과 비트 단위로 같다.
"""
from __future__ import annotations
import contextlib
import sys
import types

import numpy as np
import torch


def register(SPECS, REPOS, _load, _pkg):
    ROOT = REPOS / "PINO"

    # ------------------------------------------------------------------ 상수
    # 저장소가 함께 배포하는 센서별 파장 구간(nm).  학습되지 않는 데이터 상수다.
    #
    # 8밴드: make_band_wavelengths.py:24 의 WV3 배열은 Coastal, Red, Blue, RedEdge,
    #   Green, NIR1, Yellow, NIR2 순서다 (specdecoder.py:294 의 `names` 로 확인).
    #   PanCollection h5 는 표준 센서 순서(Coastal, Blue, Green, Yellow, Red,
    #   RedEdge, NIR1, NIR2)이므로 **행을 그 순서로 재배열**했다.  min-max 정규화는
    #   전역 min/max 로 하므로 순서와 무관 -- 값은 그대로고 밴드-행 대응만 바로잡는다.
    #   이걸 안 하면 학습되는 분광 응답(ResponseFunctionEstimator)이 엉뚱한 밴드에 묶인다.
    #   재배열의 근거는 데이터다 (train_wv3.h5 gt 앞 3,000장, 채널 상관): ch1-ch2 r=0.973 (Blue-Green),
    #   ch6-ch7 r=0.912 (NIR1-NIR2), ch5 는 ch6/ch7 과 0.85/0.87 (RedEdge).  저장소 이름표대로라면 ch6 은
    #   Yellow 인데 이웃이어야 할 Red(ch1)/Green(ch4) 과 0.56/0.69 뿐이다 -- h5 는 표준 순서다.
    # 4밴드: make_band_wavelengths.py:33 의 GF2 배열 그대로.
    WAVES = {
        8: [[397, 454], [445, 517], [507, 586], [580, 629],
            [626, 696], [698, 749], [765, 899], [857, 1039]],
        4: [[450, 520], [520, 590], [620, 690], [770, 890]],
    }
    SAMPLE_Q = 2048          # wrappers.py:193 train sample_q
    HIST_W = 0.01            # train_liif.py:158-159 (어느 yaml 에도 없는 기본값)
    HIST_BINS = 64           # train_liif.py:92  HistogramLoss('emd', num_bins=64)
    EVAL_BSUB = 1            # configs/test-wv3/test_bdt-liif.yaml: test batch_size 1
    EVAL_XSUB = 8192         # eval_bsize.  아래 diff 참고 -- 40000 은 H=256 에서 14.9GiB 를 쓴다.
    AUTHOR_DIV = 2047.0      # datasets/image_folder.py:49  `data[i] / 2047` (wv3 · gf2 공통)
    # 하네스 나눗수 / 저자 나눗수.  per_dataset 이 opt 에 실어 make_opt 로 넘긴다 (위 머리말).
    # pancoll.SPEC 과 대조는 CPU 점검에서 한다 -- 여기 숫자를 바꾸면 gf2 입력 범위가 바뀐다.
    RESCALE = {"wv3": 2047.0 / AUTHOR_DIV, "qb": 2047.0 / AUTHOR_DIV, "gf2": 1023.0 / AUTHOR_DIV}

    _CACHE: dict = {}

    # ------------------------------------------------------------------ 적재
    def _repo():
        """PINO 저장소를 import 한다.  한 번만 수행하고 캐시한다."""
        if "pkg" in _CACHE:
            return _CACHE

        # (1) `utils` 셔틀.  liif.py:10 / misc.py:7 이 `from utils import make_coord, to_cuda`
        #     를 한다.  저장소의 utils.py 는 tensorboardX/matplotlib/`from eval import *` 를
        #     끌고 오는 200줄짜리 학습 유틸이라 통째로 import 할 이유가 없고, 게다가 거기
        #     to_cuda 는 `.cuda()` 를 박아 두어 cuda:1 에서 장치가 어긋난다.  그래서
        #     make_coord 는 정의에서 새로 쓰고 (공개본) to_cuda 만 장치 인지로 바꾼다.
        dev_box = [None]

        def make_coord(shape, ranges=None, flatten=True):
            """격자 칸의 중심 좌표 (공개본: 정의에서 새로 썼다 -- 저장소 코드를 옮기지 않는다).

            축마다 [lo, hi] (기본 [-1, 1]) 를 n 칸으로 나눈 칸의 중심 lo + (2k+1)h, h = (hi - lo) / (2n).
            축들을 ij 순서로 엮고 flatten 이면 (칸 수, 축 수) 로 편다.  PINO 의 utils.make_coord 와 호출 모양이
            같고 결과는 torch.equal 로 같다."""
            centres = []
            for axis, n in enumerate(shape):
                lo, hi = (-1, 1) if ranges is None else ranges[axis]
                h = (hi - lo) / (2 * n)                          # 반 칸
                centres.append(lo + h + (2 * h) * torch.arange(n).float())
            grid = torch.stack(torch.meshgrid(*centres, indexing="ij"), dim=-1)
            return grid.view(-1, grid.shape[-1]) if flatten else grid

        def to_cuda(obj):
            d = dev_box[0]
            if d is not None:
                return obj.to(d)
            return obj.cuda() if torch.cuda.is_available() else obj

        shim = types.ModuleType("utils")
        shim.make_coord = make_coord
        shim.to_cuda = to_cuda

        prev_utils = sys.modules.get("utils")
        prev_models = sys.modules.get("models")
        sys.modules["utils"] = shim
        try:
            # (2) `models` 패키지.  models/__init__.py 가 상대 import 를 쓰므로 __path__ 가
            #     박힌 패키지 객체 안에서 실행해야 한다 (_pkg 가 그 객체를 만들어 준다).
            pkg = _pkg(ROOT / "models", "models")
            init = ROOT / "models" / "__init__.py"
            exec(compile(init.read_text(), str(init), "exec"), pkg.__dict__)
        finally:
            # liif/misc 는 import 시점에 make_coord/to_cuda 를 직접 바인딩했으므로 이제
            # 전역 이름을 원상복구해도 된다.  `utils`/`models` 는 다른 저장소와 부딪히기
            # 쉬운 이름이라 남겨 두지 않는다.
            for name, prev in (("utils", prev_utils), ("models", prev_models)):
                if prev is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = prev

        # (3) 손실.  histogram_loss.py 는 저장소 내부 import 가 없어 파일째로 적재된다.
        HL = _load(ROOT / "models" / "histogram_loss.py", "pino_histogram_loss")

        _CACHE.update(pkg=pkg, HL=HL, dev_box=dev_box, make_coord=make_coord)
        return _CACHE

    # ------------------------------------------------------------------ 도우미
    def _dev_ctx(dev):
        """specdecoder.py:87,459 가 `.cuda()` 를 하드코딩해 두어 (band_coord_samples 와
        freq_mat) cuda:1 에서 장치가 갈린다.  저장소를 고치는 대신 현재 CUDA 장치를
        그 범위 동안만 바꾼다 -- 연산은 한 글자도 달라지지 않는다."""
        return torch.cuda.device(dev) if dev.type == "cuda" else contextlib.nullcontext()

    def _norm(m, x):
        """하네스 [0,1] -> 저자 [0,1](DN/2047) -> [-1,1].  utils.py:293-301, data_norm sub=[0.5] div=[0.5].
        wv3 · qb 는 pino_in_scale == 1.0 이라 곱셈이 값을 바꾸지 않는다 (비트 동일)."""
        return (x * m.pino_in_scale - 0.5) / 0.5

    def _denorm(m, y):
        """test.py:138-139 (저자 공간에서 역정규화 후 clamp(0,1)) -> 하네스 [0,1]."""
        return ((y * 0.5 + 0.5).clamp(0, 1) / m.pino_in_scale).clamp(0, 1)

    def _coord(m, H, W, dev):
        key = ("c", H, W, dev)
        if key not in m._pino_cache:
            m._pino_cache[key] = _CACHE["make_coord"]([H, W]).to(dev)
        return m._pino_cache[key]

    def _bandcoord(m, B, dev):
        key = ("b", dev)
        if key not in m._pino_cache:
            s = np.asarray(WAVES[m._pino_C], dtype=np.float64)
            # utils.make_band_coords:152-162 -- 전역 min/max 로 [0,1] 후 [-1,1]
            s = ((s - s.min()) / (s.max() - s.min()) * 2 - 1).astype(np.float32)
            m._pino_cache[key] = torch.from_numpy(s).to(dev)[None]
        return m._pino_cache[key].expand(B, -1, -1)

    def _hist(m, dev):
        """HistogramLoss 를 **모듈 트리 밖**(평범한 list 속성)에 둔다.  HistLayer 는
        requires_grad=False 인 Conv2d 두 개(bin 중심/폭)를 들고 있는데, 그래도
        optimizer 의 param group 에 들어가면 안 되고 state_dict 도 오염시키면 안 된다.
        nn.Module.__setattr__ 은 list 를 등록하지 않으므로 이렇게 보관한다.

        만들 때 전역 CPU 난수를 **쓰지 않는다** (fork_rng).  HistLayer 의 Conv2d 두 개는 생성자에서
        kaiming 초기화로 CPU 난수를 뽑고 곧바로 상수(fill_)로 덮는다 -- 값은 난수와 무관한데 난수 흐름만
        밀린다.  이것은 첫 train_step 에서 생기므로, exact 이어 돌리기(난수를 state.pt 에서 되살린 뒤
        첫 step 에 다시 만든다)와 끊김 없는 런의 CPU 난수 흐름이 한 번 어긋났다 (CPU 시험: 이어 돈 두 번째
        step 부터 좌표 표집이 달라짐).  GPU 학습의 좌표 표집은 CUDA 난수라 원래도 영향이 없었다."""
        for d, mod in m._pino_hist:
            if d == dev:
                return mod
        with torch.random.fork_rng(devices=[]):
            mod = _CACHE["HL"].HistogramLoss("emd", num_bins=HIST_BINS)
        mod = mod.to(dev)
        m._pino_hist.append((dev, mod))
        return mod

    # ------------------------------------------------------------------ build
    def build(ms_ch=4, ratio=4):
        assert ratio == 4, f"PINO 인코더 Direction2 는 ms 를 H/4 로 가정한다 (ratio={ratio})"
        if ms_ch not in WAVES:
            raise ValueError(f"PINO 는 센서 파장표가 필요하다 -- {ms_ch}밴드 표가 없다")
        R = _repo()
        C = int(ms_ch)
        spec = {"name": "liif", "args": {
            # configs/train-wv3/train_bdt-liif.yaml (8밴드) / train-gf2(4밴드) 그대로.
            "with_no": True, "local_ensemble": False,
            "encoder_spec": {"name": "bdt-ori", "args": {
                "num_heads": [8, 8, 8],
                "image_size": 64,     # 죽은 인자 -- Stage/PatchMerging 이 H,W 를 forward 로 받는다
                "embed_dim": 64, "out_dim": C, "hsi_dim": C, "msi_dim": 1}},
            "imnet_spec": {"name": "banddec", "args": {
                "fedec_spec": {"name": "mlp", "args": {
                    "out_dim": 256, "hidden_list": [256, 256, 256]}},
                "bandposenc_spec": {"name": "bandposenc", "args": {
                    "bandposenc_type": "band_rb_mlp", "freq": 32, "max_radius": 1,
                    "min_radius": 0.0001, "freq_init": "geometric", "hidden_list": [512]}},
                "bandnerf_spec": {"name": "bandnerf", "args": {
                    "out_dim": 1, "bandnerf_type": "img_band_dot",
                    "hidden_list": [], "act": "identity"}},
                "num_band_int_sample": 16, "resp_func_type": "rf_est",
                "band_int_sample_type": "fix", "hsi_dim": C,
                "with_sr": True, "with_wm": True}}}}
        m = R["pkg"].make(spec)
        # 모듈 트리를 저장소와 동일하게 유지하려고 래퍼를 쓰지 않는다.  부속물은
        # nn.Module 이 등록하지 않는 평범한 파이썬 컨테이너로 붙인다.
        m._pino_C = C
        m._pino_cache = {}
        m._pino_hist = []
        # 입력 범위 버퍼 (머리말).  파라미터가 아니라 학습 · 파라미터 수(R.n_params)에 들어가지 않는다.
        # 기본 1.0 (wv3 · qb).  gf2 는 make_opt 가 1023/2047 로 쓰고, 가중치를 읽는 쪽은 best.pt 에서 받는다.
        m.register_buffer("pino_in_scale", torch.ones(()))
        return m

    def make_opt(model, o):
        """하네스 기본 Adam 과 **같은** 옵티마이저 (harness.make_optimizer 의 기본 경로와 같은 인자).
        하는 일은 하나 더: per_dataset 이 opt 에 실은 입력 범위 배율을 모델 버퍼에 쓴다.
        학습기(처음부터 · 이어 돌리기 · 재평가)는 가중치를 싣기 **전에** 이것을 부르므로, 이어 돌리기 ·
        재평가에서는 state.pt/best.pt 의 버퍼가 이 값을 다시 덮는다 (값이 같다 -- opt 는 게이트가 대조한다)."""
        s = float(o.get("pino_data_rescale", 1.0))
        assert 0.0 < s <= 1.0, f"pino_data_rescale {s} -- 하네스 나눗수 / 2047 이어야 한다"
        model.pino_in_scale.fill_(s)
        return torch.optim.Adam(model.parameters(), lr=o["lr"], betas=tuple(o.get("betas", (0.9, 0.999))),
                                eps=o.get("eps", 1e-8), weight_decay=o.get("weight_decay", 0.0))

    # ------------------------------------------------------------------ 학습 한 스텝
    def train_step(model, batch, ctx, lossfn):
        """train_liif.py:95-111 그대로.

        pred = model(pan, ms_up, lrms, res_value, coord, band_coord)  (전부 [-1,1])
        loss = L1(pred, gt_q) + 0.01 * HistogramLoss('emd',64)(pred^T, gt_q^T)
        """
        R = _repo()
        gt, lms, pan, ms = batch["gt"], batch["lms"], batch["pan"], batch["ms"]
        B, C, H, W = gt.shape
        assert C == model._pino_C, (
            f"band_coord 는 {model._pino_C}밴드로 만들어졌는데 배치는 {C}밴드다 -- "
            "분광 응답이 엉뚱한 밴드에 묶인다")
        dev = gt.device
        R["dev_box"][0] = dev

        pan_n, lms_n, ms_n, gt_n = _norm(model, pan), _norm(model, lms), _norm(model, ms), _norm(model, gt)

        # 좌표 부분표집: wrappers.py:193-197 은 **표본마다 독립적으로**
        # np.random.choice(H*W, 2048, replace=False) 를 한다 (배치 공용 인덱스가 아니다).
        # argsort(rand) 의 앞 2048 개는 비복원 균등 표집과 분포가 같고 GPU 에서 끝난다.
        X = min(SAMPLE_Q, H * W)
        idx = torch.rand(B, H * W, device=dev).argsort(dim=1)[:, :X]

        flat_lms = lms_n.reshape(B, C, -1).permute(0, 2, 1)      # (B, HW, C)
        flat_gt = gt_n.reshape(B, C, -1).permute(0, 2, 1)
        gc = idx.unsqueeze(-1).expand(-1, -1, C)
        res = torch.gather(flat_lms, 1, gc)                      # liif.py:210 `ret += res_value`
        gt_q = torch.gather(flat_gt, 1, gc)
        crd = torch.gather(_coord(model, H, W, dev)[None].expand(B, -1, -1),
                           1, idx.unsqueeze(-1).expand(-1, -1, 2))
        bc = _bandcoord(model, B, dev)

        with _dev_ctx(dev):
            pred = model(pan_n, lms_n, ms_n, res, crd, bc)       # (B, X, C), [-1,1]
            recon = lossfn(pred, gt_q)                           # yaml loss_fn: L1
            h = _hist(model, dev)(pred.permute(0, 2, 1).unsqueeze(-1),
                                  gt_q.permute(0, 2, 1).unsqueeze(-1))
            loss = recon + HIST_W * h
        # pred 는 2048 개 질의 좌표의 값 (B,X,C) 이다.  학습 루프는 train_step 분기에서
        # 이 텐서를 쓰지 않는다(손실만 쓴다).  이미지로 복원할 수 있는 예측이 아니므로
        # 억지로 (B,C,H,W) 를 만들어 돌려주지 않는다 -- 평가는 항상 forward 쪽이다.
        return pred, loss

    # ------------------------------------------------------------------ 평가 forward
    @torch.no_grad()
    def _predict(model, lms, pan, ms):
        R = _repo()
        B, C, H, W = lms.shape
        assert C == model._pino_C, (
            f"band_coord 는 {model._pino_C}밴드로 만들어졌는데 입력은 {C}밴드다")
        dev = lms.device
        R["dev_box"][0] = dev

        pan_n, lms_n, ms_n = _norm(model, pan), _norm(model, lms), _norm(model, ms)
        coord = _coord(model, H, W, dev)[None]                   # (1, HW, 2)
        flat_lms = lms_n.reshape(B, C, -1).permute(0, 2, 1)      # (B, HW, C)
        out = torch.empty(B, H * W, C, device=dev, dtype=lms.dtype)

        with _dev_ctx(dev):
            for b0 in range(0, B, EVAL_BSUB):
                b1 = min(B, b0 + EVAL_BSUB)
                bc = _bandcoord(model, b1 - b0, dev)
                model.gen_feat(pan_n[b0:b1], lms_n[b0:b1], ms_n[b0:b1])
                for q0 in range(0, H * W, EVAL_XSUB):
                    q1 = min(H * W, q0 + EVAL_XSUB)
                    out[b0:b1, q0:q1] = model.query_rgb(
                        flat_lms[b0:b1, q0:q1].contiguous(),
                        coord[:, q0:q1].expand(b1 - b0, -1, -1).contiguous(), bc)
        # test.py:138-139 -- 역정규화 후 clamp (저자 공간), 그리고 하네스 [0,1] 로
        return _denorm(model, out.reshape(B, H, W, C).permute(0, 3, 1, 2))

    SPECS["PINO"] = dict(
        tier="confirmed",   # (2026-09-28): 스모크 통과, 표에 올린다
        build=build,
        venue="NeurIPS 2025",
        loss="l1",                       # recon 항.  전체 손실은 train_step 이 맡는다.
        batch=8,                         # configs/train-{wv3,gf2}/train_bdt-liif.yaml:17-18  batch_size 8
        epochs=200,                      # 같은 yaml epoch_max: 200 (wv3 :91, gf2 :92)
        # yaml optimizer: adam, lr 1.e-4 (utils.make_optimizer:97-103 -> torch Adam 기본 betas/eps, wd 0).
        # `make_opt` 가 같은 Adam 을 짓는다 (입력 범위 버퍼를 쓰려고 거친다 -- 머리말).
        opt=dict(kind="adam", lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0),
        make_opt=make_opt,
        # yaml multi_step_lr: milestones [70, 140], gamma 0.7 -> train_liif.py:73 MultiStepLR, 에폭마다 step (:162-163).
        # 하네스도 에폭 경계마다 한 번 부른다.  감쇠는 에폭 70 · 140 (milestone_policy: 마지막 감쇠 step = 140*spe).
        sched=dict(kind="multistep", milestones=[70, 140], gamma=0.7),
        aug="d4",                        # wrappers.py:151-166 augment: true
        # 입력 범위: 저자는 모든 데이터셋을 DN/2047 로 읽는다 (datasets/image_folder.py:49).  하네스 gf2 는
        # DN/1023 이므로 gf2 만 1023/2047 을 곱해 저자 공간으로 옮긴다 (머리말).  세 데이터셋 모두 적어
        # 두어 결과 JSON 의 레시피(opt)에 값이 남게 한다.
        per_dataset={d: {"opt": {"pino_data_rescale": s}} for d, s in RESCALE.items()},
        # 추론은 결정적이다: band_int_sample_type 'fix'(specdecoder.py:75-88, linspace), dropout · drop_path 0
        # (bidirection.py:215 기본값), 좌표 부분표집(torch.rand)은 train_step 에만 있다.  같은 가중치 · 같은 입력이면
        # 같은 출력 -- seed-0 공개 규칙(bs 1, 집합마다 reseed)이 필요 없다.
        stochastic=False,
        # eval_after_frac 은 두지 않는다 (0).  저장소는 2 에폭마다(epoch_val: 2) 처음부터 검증해 PSNR 최고
        # 체크포인트를 고르고(train_liif.py:188-218), 구조가 중간에 바뀌는 단계도 없다.  valid 한 번은
        # val_cap 256 에서 약 4초라 앞부분을 건너뛸 이유도 없다.
        #
        # 평가가 이미지 1장씩 도는 표본별 루프라(아래 diff (5)) valid 전수(1,080~2,201행)를
        # 매번 도는 것은 예산 낭비다.  학습 중 valid 는 체크포인트 선택에만 쓰이므로
        # 부분집합으로 충분하다.  표에 실리는 RR/FR/zero-shot 수치는 언제나 전수다
        # (train_pancoll.py:410-411 -- val_cap 은 step 이 주어진 호출에만 걸린다).
        val_cap=256,
        forward=lambda mm, ms_up, pan, lrms, ctx: _predict(mm, ms_up, pan, lrms),
        train_step=train_step,
        diff=(
            "손실(L1 + 0.01*HistogramLoss('emd',64), 2048 좌표/표본) · 옵티마이저(Adam lr 1e-4) · 배치 8 · "
            "200 에폭 · MultiStepLR([70,140], 0.7) · 증강은 저장소 값 그대로 (configs/train-{wv3,gf2}/"
            "train_bdt-liif.yaml, train_liif.py:92-163). 바꾼 것과 이유: "
            "(1) 2단계(LIIF)만 스크래치로 학습한다 -- 1단계 인코더 사전학습(train_bdt.py, L1+0.1*(1-SSIM), "
            "Adam 2e-4, 배치 32, 1000 에폭, configs/train-*/train_bdt_ori.yaml)은 하지 않았다. README 는 "
            "1단계 -> 2단계 순서를 적지만, 공식 WV3 2단계 yaml 에는 encoder_spec.weights 가 없어 README 대로 "
            "두 명령을 돌려도 WV3 2단계는 스크래치로 시작한다. GF2 2단계 yaml 은 weights 를 적지만 "
            "배포된 그대로는 돌지 않는다 (`resume: \"\"` 가 None 이 아니라 torch.load('') 에서 멈추고, weights "
            "경로 save/gf2/train_bdt_ori/ 는 train_bdt.py 가 저장하는 ./save/_train_bdt_ori/<시각>/ 과 다르다). "
            "QB 설정은 저장소에 없다. 그래서 돌아가는 공식 설정(WV3)을 세 데이터셋에 똑같이 쓴다. README 의 두 단계 "
            "절차는 발표 수치(WV3 · GF2)가 사전학습 인코더로 나왔음을 시사하므로, 1단계 생략은 **세 데이터셋 모두에서 "
            "PINO 에 불리할 수 있는 이탈**이다 (크기 미측정). "
            "(2) 입력 범위: 저자는 모든 데이터셋을 DN/2047 로 읽고 (datasets/image_folder.py:49) [-1,1] 로 "
            "옮긴다. 하네스 gf2 는 DN/1023 이라 스펙이 gf2 입력 · GT 에 1023/2047 을 곱해 저자 공간으로 "
            "옮기고, 예측은 저자 공간에서 clamp(0,1) 한 뒤(test.py:138-139) 되돌린다. 그래서 gf2 의 [-1,1] "
            "값은 저자와 같이 [-1, 약 0] 에 있고 히스토그램 bin([-0.05,1.05])에 드는 값이 0.02% 미만(학습 패치 실측 0.008~0.016%)이라 히스토그램 "
            "항이 사실상 꺼진다 -- 저자 학습에서도 그렇다. 배율은 per_dataset 의 opt.pino_data_rescale "
            "(wv3 · qb 1.0, gf2 1023/2047)이고 make_opt 가 모델 버퍼 pino_in_scale 에 쓴다 (state_dict 에 실려 "
            "가중치와 함께 다닌다; 파라미터가 아니다). "
            "(3) 8밴드 파장표를 저장소 순서(Coastal,Red,Blue,RedEdge,Green,NIR1,Yellow,NIR2 -- "
            "make_band_wavelengths.py:24, specdecoder.py:294 의 이름표)에서 PanCollection h5 채널 순서(표준 "
            "Coastal,Blue,Green,Yellow,Red,RedEdge,NIR1,NIR2)로 재배열했다. 값이 아니라 행 순서만 바꾼 것이다. "
            "h5 가 표준 순서라는 근거는 데이터다 (채널 상관: ch1-ch2 0.973, ch6-ch7 0.912; 저장소 이름표대로면 "
            "Yellow 인 ch6 이 Red/Green 과 0.56/0.69). 저자 학습은 이 어긋난 표로 돌았으므로 **저장소와 다르다** "
            "-- 논문이 말하는 센서 파장 기반 분광 응답을 맞는 밴드에 묶으려는 수정이다. 규칙: **센서 상수**(파장표)는 "
            "데이터에 맞게 바로잡고, **학습 파이프라인의 동작**(입력 나눗수 (2) 등)은 저자 코드 그대로 따른다. "
            "(4) 4밴드는 저장소가 배포한 GF2 표만 있어 QB 도 같은 표를 쓴다 (QB 실제 구간은 "
            "450-520/520-600/630-690/760-900, 정규화 후 band_coord 차이 <0.05). 센서 상수라 "
            "네트워크 용량이나 손실에는 영향이 없다. "
            "(5) 평가 forward 는 이미지 1장씩(bsub=1), 좌표 8192개씩 질의한다. Galerkin 커널"
            "(bidirection.py:759)은 행마다 k^T v 를 만들어 **호출 한 번의 행 수**로 나누므로 출력이 질의 "
            "조각 크기에 (약하게) 의존한다. 학습은 8x2048=16384 행이다. 저자 테스트는 eval_bsize 40000"
            "(WV3; H=256 에서 40000+25536 조각) · 8000(GF2 · WV2; 8x8000+1536) 이다. 우리는 크기와 무관하게 "
            "8192 로 고정한다 -- 64/256/512 모두 나누어 떨어져 짧은 조각이 없고, H=64 valid(4096 행)는 저자 "
            "설정과 같다. 40000 대비 실측 차이 max 4e-4 · mean 1e-4 (40000 은 8밴드 H=256 에서 14.9GiB). "
            "(6) 좌표 부분표집을 np.random.choice 대신 argsort(torch.rand) 로 했다 -- "
            "표본별 비복원 균등 표집으로 분포가 같고 GPU 에서 끝난다. "
            "(7) 증강은 하네스의 aug='d4'. 저장소는 hflip/vflip/dflip 을 각각 p=0.5 로 "
            "독립 적용하는데(wrappers.py:151-166) 그 결과 분포가 이면체군 균등분포로 같다. "
            "(8) 저장소 파일은 한 줄도 손대지 않았다. `utils` 는 make_coord 를 정의(격자 칸의 중심 좌표)에서 새로 쓰고 (공개본 -- 저장소 코드를 옮기지 않았고, 결과는 torch.equal 로 동일) to_cuda 만 장치 인지로 바꾼 셔틀을 쓰며(원본 utils.py 는 tensorboardX/"
            "matplotlib/`from eval import *` 를 끌고 온다), specdecoder.py:87,459 의 "
            "하드코딩 `.cuda()` 는 forward 동안 torch.cuda.device(dev) 로 감싸 해결한다. "
            "(9) HistogramLoss 는 list 속성(m._pino_hist)에 보관해 모듈 트리/optimizer/"
            "state_dict 에 들어가지 않게 했다. 학습 파라미터가 아니라 bin 상수다. 만들 때 전역 CPU 난수를 쓰지 "
            "않게(fork_rng) 해서 exact 이어 돌리기의 난수 흐름이 끊김 없는 런과 같다. "
            "(10) HistLayer 의 bin 은 [-0.05,1.05] 인데 데이터는 [-1,1] 이다 "
            "(histogram_loss.py:38,168 -- (image+1)/2 재매핑이 주석 처리돼 있다). "
            "공식 학습 루프가 그대로 돌리므로 우리도 그대로 둔다. "
            "(11) encoder.t.upsample_layers.0 (82,944 파라미터)은 num_heads 길이가 3일 때 "
            "도달하지 않는 죽은 가지다(bidirection.py:530-537). 공표 파라미터 수의 일부라 "
            "건드리지 않는다 -- grad 가 None 으로 남는 것이 정상이다. "
            "(12) 하네스 전역 정책인 grad_clip 10.0 을 그대로 받는다 (저장소는 클리핑 없음 -- train_liif.py:117-119). "
            "(13) 체크포인트는 valid PSNR 로 고른다. 저장소는 2 에폭마다 **test** reduced 집합(무작위 20000 좌표)의 "
            "PSNR 로 고른다(train_liif.py:188-218) -- 시험셋으로 고르지 않는다. "
            "(14) 1단계를 빼고 2단계만 도는 WV3 공식 설정에는 GF2 yaml 의 with_no/with_sr/with_wm 이 없는데, "
            "셋 다 저장소 기본값이 True 라(liif.py:29, specdecoder.py:166) 두 yaml 의 모델은 같다."
        ),
    )

# ---------------------------------------------------------------------------------------------
# NOTICE (release copy): `make_coord` inside register() above is written for this release from its
# definition (the cell centres of a regular grid on [-1, 1] or given ranges); no code of the PINO
# repository is copied.  It keeps the call signature of the `make_coord` helper in PINO's utils.py,
# which PINO's modules import, and returns identical tensors.  PINO is an extra method: it is not in
# the paper tables.
