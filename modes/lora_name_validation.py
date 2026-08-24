def validate_lora_project_name(name: str) -> str:
    """Return an error message when a project name cannot be used as-is as a directory."""
    cleaned = name.strip()
    if not cleaned:
        return "LoRA 프로젝트명을 입력하세요"
    invalid = list(dict.fromkeys(
        c for c in cleaned
        if not (c.isalnum() or c in (' ', '_', '-', '.'))
    ))
    if invalid:
        invalid_text = ", ".join(repr(c) for c in invalid)
        return (
            f"LoRA 프로젝트명에 사용할 수 없는 문자: {invalid_text}. "
            "허용 문자: 한글/영문/숫자, 공백, _, -, ."
        )
    if cleaned in ('.', '..'):
        return "LoRA 프로젝트명으로 '.' 또는 '..'을 사용할 수 없습니다"
    return ""
