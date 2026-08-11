"""LoRA 학습 이미지 내보내기에 공통으로 사용하는 파일명 유틸리티."""


MIN_EXPORT_INDEX_WIDTH = 5


def format_lora_export_filename(index: int, total: int, extension: str) -> str:
    """문자열 정렬에서도 숫자 순서가 유지되는 학습 이미지 파일명을 만든다."""
    index_width = max(MIN_EXPORT_INDEX_WIDTH, len(str(max(index, total))))
    return f"[{index:0{index_width}d}]{extension}"
