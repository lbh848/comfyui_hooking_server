"""LoRA 로드 경로를 조립할 때 구분자는 입력 경로를 따라야 한다.

역슬래시로 고정하면 POSIX 에서 `loras\\SOYA_CHAR_LORA` 라는 폴더 이름 하나가
만들어진다. 읽는 쪽(stripManagedLoraPath)이 두 구분자를 모두 받아 주기 때문에
UI 왕복은 멀쩡해 보이고, 실제 LoRA 폴더와만 조용히 어긋난다.
"""

from pathlib import Path


FRONTEND_HTML = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


def _combine_source() -> str:
    source = FRONTEND_HTML.read_text(encoding="utf-8")
    return source.split("function combineLoraLoadPath", 1)[1].split(
        "function stripManagedLoraPath", 1
    )[0]


def test_separator_follows_the_input_path_instead_of_being_hardcoded():
    body = _combine_source()

    assert "join(separator)" in body, "구분자를 변수로 골라 join 해야 한다"
    assert "join('\\\\')" not in body, "역슬래시 고정은 POSIX 에서 폴더 이름을 망가뜨린다"


def test_windows_style_base_keeps_backslash():
    # 기존 Windows 설정을 뒤섞인 구분자로 바꾸지 않는다.
    body = _combine_source()

    assert "includes('\\\\')" in body and "includes('/')" in body, (
        "역슬래시만 있고 슬래시가 없을 때만 역슬래시를 쓴다는 판정이 있어야 한다"
    )


def test_reader_still_accepts_both_separators():
    # 판정이 한쪽만 보게 되면 기존에 저장된 값이 벗겨지지 않는다.
    source = FRONTEND_HTML.read_text(encoding="utf-8")
    reader = source.split("function stripManagedLoraPath", 1)[1].split("}", 1)[0]

    assert "[\\\\/]SOYA_CHAR_LORA" in reader
