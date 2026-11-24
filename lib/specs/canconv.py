"""CANConv -- Content-Adaptive Non-Local Convolution (CVPR 2024).

저장소: repos/CANConv (github.com/duanyll/CANConv).  핵심은 파이썬이 아니라 CUDA 에
있다 -- 화소를 k-means 로 군집한 뒤 군집마다 다른 커널을 쓰는 연산자를 직접 구현했고,
그래서 `pwacnative.so` 와 `libKMCUDA.so` 를 빌드해야 모델이 돌아간다.  그 빌드가
오래 막혀 있었기 때문에 이 방법은 26개 감사 목록에 들어가지 못했다.  지금은 된다.

빌드 메모 (재현용): torch 2.7.1+cu126 와 시스템 nvcc 11.8 이 어긋나 격리 CUDA 12.6 을
.toolchain/cuda_home 에 세웠고, torch 의 cuda.cmake 가 NVTX3 를 먼저 찾는데
`CUDA_INCLUDE_DIRS` 가 비어 탐색이 안 되던 것이 마지막 원인이었다.  `-Dnvtx3_dir` 과
`-DCUDA_INCLUDE_DIRS` 를 캐시 변수로 직접 박아 해결했다.  scripts/build_canconv.sh 참고.
"""
import importlib.util
import sys
from pathlib import Path


def register(SPECS, REPOS, _load, _pkg):
    ROOT = Path(REPOS) / "CANConv"
    _sched = {}          # 저장소의 KMeansCacheScheduler 인스턴스를 담아 둔다
    _seen = set()        # 군집 캐시가 이미 초기화된 표본 인덱스

    def _on_train_start(model, n_train, device):
        from canconv.layers.kmeans import reset_cache, KMeansCacheScheduler
        reset_cache(n_train)
        _seen.clear()                       # default.json 의 학습셋 크기 자리
        _sched["s"] = KMeansCacheScheduler([[10, 1], [100, 10], [300, 20], 50])

    def _on_epoch(model, epoch):
        # step() 이 전역 cache_mode 를 init -> update/ready 로 옮긴다.  epoch 1 이
        # init 이어야 하므로 매 에폭 시작에 정확히 한 번만 부른다.
        _sched["s"].step()
        _sched["mode"] = None       # 아래 forward 가 다시 읽도록 비워 둔다

    def _forward(m, ms_up, pan, lrms, ctx):
        """캐시가 **아직 초기화되지 않은** 표본을 만나면 그 스텝만 init 으로 돌린다.

        저장소는 `shuffle=True` 인 DataLoader 라 epoch 1 에 모든 표본이 정확히 한 번씩
        지나가고, 그때 전부 init 된다.  우리 트레이너는 `torch.randint` -- 복원추출이라
        한 에폭이 인덱스의 63.2% 에만 닿는다.  나머지 36.8% 는 init 을 한 번도 못 받고,
        epoch 2 부터 `ready` 모드에서 **0 으로 채워진 군집 배정**을 읽는다.  그러면 그
        표본들에 대해 CANConv 이 군집 하나짜리 -- 즉 보통의 컨볼루션으로 퇴화한다.
        논문의 기여가 학습 데이터의 3분의 1 에서 꺼지는 셈이다."""
        idx = ctx.get("sample_idx")
        if idx is not None and _sched.get("s") is not None:
            from canconv.layers.kmeans import cache as KC
            fresh = [int(i) for i in idx.tolist() if int(i) not in _seen]
            if fresh:
                if _sched.get("mode") is None:
                    _sched["mode"] = KC.cache_mode
                KC.cache_mode = "init"
                _seen.update(fresh)
            elif _sched.get("mode") is not None:
                KC.cache_mode = _sched["mode"]
        return m(pan, ms_up, idx)

    def build(ms_ch=4, ratio=4):
        if ratio != 4:
            raise ValueError(f"CANConv 저장소 설정은 x4 뿐이다 (ratio={ratio})")
        sys.path.insert(0, str(ROOT))
        # `canconv.models.cannet.__init__` 은 학습기를 끌어오고, 그 경로의
        # `util/log.py:65` 가 `torch.Tensor | np.ndarray` 를 쓴다 -- PEP 604 라
        # 파이썬 3.9 에서 import 자체가 터진다.  우리는 학습기를 쓰지 않으므로
        # 패키지 __init__ 을 건너뛰고 모델 파일만 직접 적재한다.  네트워크 코드는
        # 손대지 않는다.
        from canconv.layers.canconv import CANConv          # noqa: F401 (CUDA 확장 적재)
        spec = importlib.util.spec_from_file_location(
            "canconv_cannet_model", ROOT / "canconv" / "models" / "cannet" / "model.py")
        M = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(M)
        # default.json 의 channels 32 / cluster_num 32 / filter_threshold 0.005.
        # spectral_num 은 원래부터 생성자 인자다 -- 밴드 하드코딩이 없어 패치가 없다.
        return M.CANNet(spectral_num=ms_ch, channels=32, cluster_num=32,
                        filter_threshold=0.005)

    SPECS["CANConv"] = dict(
        tier="confirmed",
        build=build, venue="CVPR 2024", loss="l1", batch=32, epochs=500,
        # default.json: WV3 기준 lr 1e-3, StepLR(step_size=250).  gamma 는 넘기지
        # 않으므로 torch 기본값 0.1 이다 (config.py:25-26).
        opt=dict(kind="adam", lr=1e-3, weight_decay=0.0),
        sched=dict(kind="step", step_size=250, gamma=0.1),
        # presets.json 이 GF2·QB 에만 lr 5e-4 / lr_step_size 500 을 덮어쓴다.
        per_dataset={
            "gf2": dict(opt=dict(lr=5e-4), sched=dict(step_size=500)),
            "qb":  dict(opt=dict(lr=5e-4), sched=dict(step_size=500)),
        },
        forward=_forward,
        on_train_start=_on_train_start, on_epoch=_on_epoch,
        diff="네트워크 코드 수정 없음 (밴드 수가 원래 생성자 인자다). 두 가지만 기록한다. "
             "(1) 모델 파일을 패키지 __init__ 대신 직접 적재한다 -- 그 __init__ 이 "
             "끌어오는 학습기에 파이썬 3.10 전용 문법이 있어서이고, 모델은 그대로다. "
             "(2) 저장소는 k-means 군집 배정을 표본 인덱스로 캐시하고 "
             "`kmeans_cache_update` 일정([[10,1],[100,10],[300,20],50])에 따라 "
             "갱신한다. 우리도 저장소의 `KMeansCacheScheduler` 를 그대로 써서 같은 "
             "일정을 돈다 -- 캐시를 끄면 11배 느려져 같은 시간 상한에서 일정의 5% 만 "
             "돌게 되고, 그건 비교가 아니다. 캐시는 학습 표본 19,809개 기준 852 MB "
             "(모듈 3개 x 5,376 int64/표본) 로 GPU 에 올라간다. 검증 순전파는 "
             "`cache_indice=None` 이라 매번 새로 군집한다.")
