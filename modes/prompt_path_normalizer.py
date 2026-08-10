"""Modal ComfyUI로 보내는 프롬프트의 기계용 경로만 이식 가능하게 만든다.

일반 프롬프트 텍스트의 역슬래시는 작품명/아티스트 이스케이프일 수 있으므로
절대 전역 치환하지 않는다. 백엔드와 Soya 노드가 계약으로 소비하는 JSON 제어
블록의 명시적인 경로 필드만 다룬다.
"""

from __future__ import annotations

import json
import re
import traceback
from typing import Any


_CONTROL_BLOCK_HEADER_RE = re.compile(
    r"(?m)^[ \t]*\[(?P<name>[A-Z][A-Z0-9_]*)\][ \t]*(?:\r?\n|$)"
)

# 자연어 키워드가 아니라 ComfyUI 커스텀 노드가 직접 소비하는 제어 스키마다.
_PATH_FIELDS_BY_BLOCK = {
    "CACHE_PATH": frozenset({"emb_path"}),
    "FACE_ID_DIR": frozenset({"ipa_path"}),
    "LORA_DATA": frozenset({"lora_path"}),
    "FACE_LORA_DATA": frozenset({"lora_path"}),
    "STYLE_LORA_DATA": frozenset({"lora_path"}),
}


def _normalize_payload_paths(
    value: Any,
    *,
    path_fields: frozenset[str],
    block_name: str,
) -> int:
    """JSON 트리 안에서 지정된 경로 필드만 제자리 정규화한다."""
    changed = 0
    if isinstance(value, dict):
        for key, child in value.items():
            if key in path_fields:
                if not isinstance(child, str):
                    raise TypeError(
                        f"[{block_name}] {key} 값이 문자열이 아닙니다: "
                        f"type={type(child).__name__}, value={child!r}"
                    )
                normalized = child.replace("\\", "/")
                if normalized != child:
                    value[key] = normalized
                    changed += 1
                continue
            changed += _normalize_payload_paths(
                child,
                path_fields=path_fields,
                block_name=block_name,
            )
    elif isinstance(value, list):
        for child in value:
            changed += _normalize_payload_paths(
                child,
                path_fields=path_fields,
                block_name=block_name,
            )
    return changed


def normalize_modal_prompt_paths(prompt: str) -> tuple[str, int]:
    """Modal용 positive 프롬프트의 제어 JSON 경로를 ``/``로 통일한다.

    반환값은 ``(정규화된 프롬프트, 변경된 경로 필드 수)``다. 대상 블록 밖의
    텍스트는 그대로 보존한다.
    """
    if not isinstance(prompt, str):
        message = (
            "[PROMPT_PATH_NORMALIZE] 입력 형식 오류: "
            f"type={type(prompt).__name__}"
        )
        print(message)
        raise TypeError(message)

    headers = list(_CONTROL_BLOCK_HEADER_RE.finditer(prompt))
    replacements: list[tuple[int, int, str]] = []
    total_changed = 0

    for index, header in enumerate(headers):
        block_name = header.group("name")
        path_fields = _PATH_FIELDS_BY_BLOCK.get(block_name)
        if path_fields is None:
            continue

        content_start = header.end()
        content_end = (
            headers[index + 1].start()
            if index + 1 < len(headers)
            else len(prompt)
        )
        raw_content = prompt[content_start:content_end]
        json_text = raw_content.strip()
        if not json_text:
            message = f"[{block_name}] JSON 제어 블록이 비어 있습니다"
            print(f"[PROMPT_PATH_NORMALIZE] 실패: {message}")
            raise ValueError(message)

        try:
            payload = json.loads(json_text)
            changed = _normalize_payload_paths(
                payload,
                path_fields=path_fields,
                block_name=block_name,
            )
        except Exception as exc:
            print(
                f"[PROMPT_PATH_NORMALIZE] [{block_name}] 처리 실패: "
                f"error={type(exc).__name__}: {exc}, payload={json_text!r}"
            )
            traceback.print_exc()
            raise

        if changed == 0:
            continue

        leading_length = len(raw_content) - len(raw_content.lstrip())
        trailing_length = len(raw_content) - len(raw_content.rstrip())
        leading = raw_content[:leading_length]
        trailing = raw_content[len(raw_content) - trailing_length :] if trailing_length else ""
        normalized_content = (
            leading
            + json.dumps(payload, ensure_ascii=False)
            + trailing
        )
        replacements.append((content_start, content_end, normalized_content))
        total_changed += changed

    normalized_prompt = prompt
    for start, end, replacement in reversed(replacements):
        normalized_prompt = normalized_prompt[:start] + replacement + normalized_prompt[end:]
    return normalized_prompt, total_changed

