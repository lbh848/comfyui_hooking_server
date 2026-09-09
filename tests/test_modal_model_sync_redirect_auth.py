"""저장소→Volume 직접 다운로드가 리다이렉트에서 토큰을 흘리지 않아야 한다.

실제로 밟았다: civitai 는 큰 파일을 S3 호환 스토리지의 presigned URL 로 넘기는데,
`Authorization: Bearer …` 를 그대로 들고 가면 S3 가 그 헤더를 서명으로 해석해
`Missing x-amz-content-sha256` 400 으로 거절한다. 6.46 GiB 체크포인트 하나만
계속 실패하고, civitai 가 직접 내려주는 작은 LoRA 는 통과해서 인증 문제로 보이지
않았다.

토큰을 남의 호스트로 보내지 않는 것이 원래 옳기도 하다.
"""

from __future__ import annotations

import ast
from pathlib import Path

MODAL_APP = Path(__file__).resolve().parents[1] / "modal_backend" / "modal_app.py"
SOURCE = MODAL_APP.read_text(encoding="utf-8")


def _sync_function() -> ast.FunctionDef:
    tree = ast.parse(SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "sync_models_from_source":
            return node
    raise AssertionError("sync_models_from_source 를 찾지 못했습니다.")


def test_download_uses_the_redirect_stripping_opener() -> None:
    """urlopen 을 직접 부르면 헤더가 리다이렉트 대상까지 따라간다."""
    body = ast.get_source_segment(SOURCE, _sync_function()) or ""
    assert "opener.open(request, timeout=300)" in body, (
        "다운로드는 Authorization 을 떼는 opener 를 거쳐야 합니다."
    )
    assert "urllib.request.urlopen(request" not in body, (
        "urlopen 을 직접 쓰면 리다이렉트에서 토큰이 새어 나갑니다."
    )


def test_redirect_handler_strips_authorization_across_hosts() -> None:
    body = ast.get_source_segment(SOURCE, _sync_function()) or ""
    assert "_StripAuthOnCrossHostRedirect" in body
    assert "unredirected_hdrs.pop(\"Authorization\", None)" in body, (
        "urllib 은 Authorization 을 unredirected_hdrs 에 넣으므로 그쪽도 지워야 합니다."
    )


def test_same_host_redirect_keeps_authorization() -> None:
    """같은 호스트 안에서의 리다이렉트까지 인증을 버리면 정상 경로가 깨진다."""
    body = ast.get_source_segment(SOURCE, _sync_function()) or ""
    assert "same_host" in body
    assert "if not same_host:" in body
