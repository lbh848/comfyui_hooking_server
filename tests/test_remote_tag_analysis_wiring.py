"""원격 태그 분석이 실제로 결과를 돌려주기 위한 배선 두 가지.

둘 다 **조용히 실패한다.** 어긋나면 실행은 정상으로 보이는데 결과만 비어 있다.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_remote_input_goes_to_the_comfy_input_root():
    """SoyaRefImageLoader 의 콤보 목록은 input 최상위 파일로만 만들어진다.

    하위 폴더에 두면 목록에 없어 ComfyUI 가 제출 단계에서 거부한다
    (value_not_in_list). 파일명에 난수 접미사가 있어 루트에 둬도 충돌하지 않는다.
    """
    source = (ROOT / "modes" / "asset_tool_mode.py").read_text(encoding="utf-8")
    assert 'remote_input_folder = ""' in source
    assert f'remote_input_folder = f"{{execution_target}}_tag_analysis"' not in source


def test_worker_listens_on_the_port_the_node_hardcodes():
    """SoyaTextSender 는 수신 주소를 모듈 상수로 하드코딩하고 실패를 삼킨다.

    server_url 입력 자체가 없어 워크플로우로 바꿀 수 없다. 워커가 임의 포트로
    열면 아무도 받지 못하고 text_outputs 가 빈 채로 끝난다.
    """
    source = (ROOT / "modal_backend" / "modal_app.py").read_text(encoding="utf-8")
    match = re.search(r"^SOYA_TEXT_SENDER_PORT = (\d+)$", source, re.MULTILINE)
    assert match, "수신 포트가 상수로 고정돼 있어야 한다"
    port = int(match.group(1))
    assert port == 8189, "노드가 하드코딩한 포트와 같아야 한다"
    assert port != 8188, "컨테이너 안에서 ComfyUI 가 쓰는 포트와 겹치면 안 된다"
    assert re.search(
        r'ThreadingHTTPServer\(\s*\("127\.0\.0\.1", SOYA_TEXT_SENDER_PORT\)', source
    ), "수신 서버가 그 상수로 바인딩해야 한다"
