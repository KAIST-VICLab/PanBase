"""Global-skip 변형 연구의 in 팔 (lib/skipvar.py) -- registry.py 에 있는 YES 방법 넷: PanTCR · ARConv · AWFLN · WFANet.

YES 방법은 하네스 `lms` 를 global skip 으로 더하고 본체도 같은 `lms` 를 읽는다.  sk 팔은 데이터만 바꾼 본래 스펙이다
(`--skip`, lms = S).  in 팔은 그 위에서 본체가 lms 를 처음 읽는 PAN 해상도 conv 가 LRMS 를 추가 채널로 더 받는다.
파일 이름이 zz_ 로 시작하는 까닭: registry 가 specs/*.py 를 이름 순으로 읽으므로 본래 스펙이 모두 등록된 뒤에 돈다.
"""
import os as _os  # 공개본 (make_release): 뿌리 = $PANBASE_ROOT (기본: 이 저장소)
_GB = _os.environ.get("PANBASE_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
import sys

sys.path.insert(0, (_GB + "/lib"))
import skipvar as SV                                                          # noqa: E402


def register(SPECS, REPOS, _load, _pkg):
    # PanTCR-GF2/net.py:181-182  x = shallow_conv(cat[pan, lms])
    SV.make_in(SPECS, "PanTCR", "PanTCR-in", ["shallow_conv"], "L",
               note="자리: net.py:181-182 shallow_conv(cat[pan, lms]).  prompt 블록의 ms_deg(lms 마지막 밴드)는 sk 와 같이 S 다.")
    # ARConv/models/models.py:37-38  x1 = head_conv(cat[pan, lms])
    SV.make_in(SPECS, "ARConv", "ARConv-in", ["head_conv"], "L",
               note="자리: models.py:37-38 head_conv(cat[pan, lms]).")
    # AWFLN/model_{4,8}b.py:54  ms_up_conv = blk_{C}_30_3(ms_up)  (data_cat 은 저장소에서 쓰이지 않는다)
    SV.make_in(SPECS, "AWFLN", "AWFLN-in",
               lambda m: ["blk_8_30_3.0"] if hasattr(m, "blk_8_30_3") else ["blk_4_30_3.0"], "L",
               note="자리: model_{4,8}b.py:51/54 blk_{C}_30_3(ms_up) 의 conv.")
    # WFANet/net_torch.py:404-405  lms_1 = act(lms(ms) + lms);  lms_2 = lms_raise_channel(lms_1)
    SV.make_in(SPECS, "WFANet", "WFANet-in", ["lms_raise_channel.raise_conv.0"], "L",
               note="자리: net_torch.py:405 lms_raise_channel 의 첫 conv (입력 lms_1 = PReLU(업샘플(ms) + lms)).  "
                    "WFANet 은 lms 를 concat 하지 않고 업샘플 MS 에 더해 본체에 넣는다 -- 그 합을 처음 읽는 conv 다.")
