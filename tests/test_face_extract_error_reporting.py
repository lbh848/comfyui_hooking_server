"""얼굴 추출 실패 진단이 원인을 가리지 않는지.

결과 검증은 로컬·원격 분기가 합류한 뒤에 있는데, 진단 메시지가 **로컬 분기
안에서만** 만들어지는 변수를 참조했다. 원격에서 결과가 비면 의도한 ValueError
대신 NameError 가 나서, 하필 원인을 봐야 할 순간에 원인이 가려진다.

그래서 '초기화가 어딘가에 있다' 로는 부족하다. 조건 분기 밖, 즉 **모든 경로가
지나는 자리**에 있어야 한다.
"""

import ast
from pathlib import Path

FUNCTION = "_handle_instance_lora_face_extract"
VARIABLE = "real_outputs"
SOURCE = (Path(__file__).resolve().parents[1] / "queue_manager.py").read_text(
    encoding="utf-8"
)


def _function() -> ast.AST:
    for node in ast.walk(ast.parse(SOURCE)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == FUNCTION:
                return node
    raise AssertionError(f"함수를 찾지 못했습니다: {FUNCTION}")


def _binds(statement: ast.stmt) -> bool:
    if isinstance(statement, ast.AnnAssign):
        return getattr(statement.target, "id", "") == VARIABLE
    if isinstance(statement, ast.Assign):
        return any(getattr(t, "id", "") == VARIABLE for t in statement.targets)
    return False


def test_diagnostic_variable_is_bound_outside_every_branch():
    func = _function()
    top_level = [s.lineno for s in func.body if _binds(s)]
    uses = [
        node.lineno
        for node in ast.walk(func)
        if isinstance(node, ast.Name)
        and node.id == VARIABLE
        and isinstance(node.ctx, ast.Load)
    ]
    assert uses, f"{VARIABLE} 를 쓰는 곳이 없습니다"
    assert top_level, (
        f"{VARIABLE} 가 조건 분기 안에서만 초기화됩니다 — "
        "원격 분기의 실패가 NameError 로 가려진다"
    )
    assert min(top_level) < min(uses), "초기화가 첫 사용보다 뒤에 있습니다"
