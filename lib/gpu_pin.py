"""CUDA 장치를 **저장소 코드가 건드리기 전에** 고정한다.

왜 필요한가.  저장소 일부가 import 시점에 `os.environ['CUDA_VISIBLE_DEVICES']` 를
덮어쓴다 -- `repos/PanTCR-GF2/net.py:11` 이 모듈 맨 위에서 '0' 으로 박는다.  우리
하네스는 CUDA 를 `model.to(dev)` 에서야 초기화했고 그 직전이 `registry.build()` 라,
저장소가 바꿔 놓은 값으로 장치가 잡혔다.  2026-09-23 에 원격 워커가 GPU 5·6 을
배정한 PanTCR 두 런이 **물리 GPU 0** 에 올라가 있었다.  원격의 GPU 0~3 은 다른
사용자의 카드다.

어떻게 막나.  CUDA 드라이버는 `cuInit` 때 **한 번만** `CUDA_VISIBLE_DEVICES` 와
`CUDA_DEVICE_ORDER` 를 읽는다.  그러니 저장소 import 전에 초기화해 두면 이후 누가
환경변수를 바꿔도 이 프로세스의 장치는 움직이지 않는다.  `CUDA_DEVICE_ORDER` 를
`PCI_BUS_ID` 로 두는 것은 번호를 `nvidia-smi` 와 같게 하려는 것이다 -- 기본값
`FASTEST_FIRST` 는 같은 카드 여러 장 사이의 순서를 정의하지 않는다.

`registry.build()` 도 import 전후로 환경변수를 되돌린다 (`guard_env`).  장치는 이미
위에서 고정되므로 그쪽은 **환경을 진실하게 유지**하고 누가 바꿨는지 로그에 남기는
역할이다 -- 자식 프로세스가 틀린 값을 물려받지 않게.
"""
from __future__ import annotations

import contextlib
import os
import socket

_KEY = "CUDA_VISIBLE_DEVICES"

#: 호스트별로 **우리가 쓸 수 있는 카드** -- 물리 uuid -> nvidia-smi 번호.  한 곳에만 둔다:
#: 디스패처(`_logs/dispatch.py`)가 여기서 카드를 고르고, `pin()` 이 여기에 없는 카드면
#: 멈춘다.  원격(GPU: RTX A6000)의 GPU 0~3 은 다른 사용자의 카드라 **넣지 않는다.**
ALLOWED = {
    # "<hostname>": {"GPU-<uuid>": <nvidia-smi 번호>, ...},   공개본은 비워 둔다 -- 표에 없는 호스트는 검사하지 않는다
}


def allowed_here():
    """이 호스트에서 쓸 수 있는 {uuid: 번호}.  표에 없는 호스트면 None (검사하지 않는다)."""
    return ALLOWED.get(socket.gethostname())


def check_uuid(uuid, what=""):
    """uuid 가 이 호스트의 허용 카드가 아니면 멈춘다."""
    allow = allowed_here()
    if allow is not None and uuid not in allow:
        raise SystemExit(f"[gpu 거부] {what}{uuid} 는 이 호스트({socket.gethostname()})에서 "
                         f"우리가 쓸 수 있는 카드가 아니다 (허용: GPU "
                         f"{sorted(allow.values())}).  시작하지 않는다.")


def pin(log=print):
    """저장소 import 전에 부른다.  (의도한 CVD, 실제로 잡힌 GPU uuid) 를 돌려준다."""
    import torch
    already = torch.cuda.is_initialized()
    if not already:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    want = os.environ.get(_KEY)
    # 허용 표가 있는 호스트에서는 카드를 **반드시** 지정해야 한다.  비워 두면 CUDA 는
    # 물리 GPU 0 을 쓰는데, 원격에서 그것은 다른 사용자의 카드다.
    if allowed_here() is not None and not want and os.environ.get("PANBASE_ALLOW_CPU") == "1":
        os.environ[_KEY] = ""          # 카드를 하나도 보이지 않게 한다 -- CPU 로만 돈다
    elif allowed_here() is not None and not want:
        raise SystemExit(f"[gpu 거부] {_KEY} 가 비어 있다 -- 이 호스트({socket.gethostname()})"
                         f"에서는 카드를 지정해야 한다 (허용: GPU {sorted(allowed_here().values())}).")
    if not torch.cuda.is_available():
        return want, None
    torch.cuda.init()
    uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
    log(f"[gpu] {_KEY}={want} order={os.environ.get('CUDA_DEVICE_ORDER')} -> {uuid}"
        + ("  (경고: CUDA 가 이미 초기화돼 있었다 -- 순서 설정이 안 먹었을 수 있다)"
           if already else ""))
    # 번호가 아니라 **물리 카드**를 대조한다.  순서 설정이 어긋나든 저장소가 환경을
    # 바꾸든, 이 호스트에서 우리 것이 아닌 카드에 올라가면 학습 전에 멈춘다 -- 손으로
    # 띄운 런도 마찬가지다.  디스패처는 배정한 카드의 uuid 를 따로 넘기고, 그것과도 맞춘다.
    check_uuid(uuid, f"{_KEY}={want} 로 잡힌 카드 ")
    if torch.cuda.device_count() != 1 and os.environ.get("PANBASE_MULTI_GPU") != "1":
        raise SystemExit(f"[gpu 거부] 카드가 {torch.cuda.device_count()} 장 보인다 ({_KEY}={want}).  "
                         f"한 장만 보이게 하라 (여러 장이 의도라면 PANBASE_MULTI_GPU=1).")
    expect = os.environ.get(EXPECT_KEY)
    if expect and uuid != expect:
        raise SystemExit(f"[gpu 거부] 배정된 카드는 {expect} 인데 {uuid} 에 올라갔다 "
                         f"({_KEY}={want}).  다른 사용자의 카드일 수 있으므로 시작하지 않는다.")
    return want, uuid


#: 디스패처가 배정한 물리 카드.  `pin()` 이 대조한다.
EXPECT_KEY = "PANBASE_EXPECT_GPU_UUID"


@contextlib.contextmanager
def guard_env(what: str, log=print):
    """이 블록 안에서 누가 `CUDA_VISIBLE_DEVICES` 를 바꾸면 되돌리고 알린다."""
    before = os.environ.get(_KEY)
    try:
        yield
    finally:
        after = os.environ.get(_KEY)
        if after != before:
            if before is None:
                os.environ.pop(_KEY, None)
            else:
                os.environ[_KEY] = before
            log(f"[gpu 경고] {what} 가 import 중 {_KEY} 를 {before!r} -> {after!r} 로 "
                f"바꿨다.  되돌렸다 (장치는 gpu_pin.pin() 이 이미 고정했다).")
