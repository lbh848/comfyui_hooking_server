"""
PromptEnhanceMode - 배치 프롬프트 강화 모드

동작 방식:
1. 배치 요청이 들어오면 [UPSCALE] 섹션에서 {char} 블럭을 파싱
2. | 로 구분된 각 캐릭터 블럭에서 이름 추출
3. outfit_mode의 이전 복장 추출 결과를 가져옴
4. LLM으로 각 캐릭터 복장 묘사를 강화/검증
5. 강화된 프롬프트로 [Positive], [ILXL], [UPSCALE] 섹션 교체
6. 원본 프롬프트를 추적하여 재전송 예약에서 문제 방지
"""

import asyncio
import json
import os
import re
import copy
import importlib.util
import traceback
from typing import Optional, Callable, Awaitable


# ─── 상수 ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOMPROMPT_DIR = os.path.join(BASE_DIR, "customprompt")


class PromptEnhanceMode:
    """프롬프트 강화 모드 매니저"""

    def __init__(self):
        self.enabled: bool = False
        self.enhance_prompt_file: str = ""
        self._lock = asyncio.Lock()
        self.mode_log_func: Optional[Callable] = None
        self.notify_frontend_func: Optional[Callable] = None
        # outfit_mode 참조 (server.py에서 설정)
        self.outfit_mode_ref = None
        # 원본 프롬프트 추적: request_id -> original_positive
        self._original_prompts: dict[str, str] = {}
        # 강화 프롬프트 캐시: normalized_prompt -> enhanced_prompt
        self._enhance_cache: dict[str, str] = {}
        # 캐릭터별 마지막 강화 추적 (복장만 저장, 표정 제외)
        self._last_enhanced_block: dict[str, str] = {}  # char_name -> outfit_only_block

    def _log(self, action: str, data: dict = None):
        if self.mode_log_func:
            self.mode_log_func("enhance_mode", action, data)

    # ─── 프롬프트 파싱 ──────────────────────────────────────

    @staticmethod
    def _extract_section(text: str, section_name: str) -> str:
        """프롬프트에서 특정 섹션 내용 추출.
        [SECTION_NAME]\n...content... 이후 다음 [SECTION] 또는 끝까지.
        """
        pattern = rf'\[{section_name}\]\s*(.*?)(?=\n\s*\[|\Z)'
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
        return ""

    @staticmethod
    def _split_char_blocks(text: str) -> list[str]:
        """| 로 구분된 캐릭터 블럭 분리.
        (weight:val) 내부의 | 는 무시.
        """
        if not text:
            return []

        blocks = []
        current = []
        paren_depth = 0

        for char in text:
            if char == '(':
                paren_depth += 1
                current.append(char)
            elif char == ')':
                paren_depth = max(0, paren_depth - 1)
                current.append(char)
            elif char == '|' and paren_depth == 0:
                block = ''.join(current).strip()
                if block:
                    blocks.append(block)
                current = []
            else:
                current.append(char)

        block = ''.join(current).strip()
        if block:
            blocks.append(block)

        return blocks

    def _extract_character_name(self, block: str) -> str:
        """캐릭터 블럭에서 풀네임 추출.
        1. outfit_mode에 추적된 캐릭터 이름이 block에 있으면 사용 (가장 신뢰도 높음)
        2. 콤마로 분리된 토큰 중 2단어 이름 패턴 찾기 (일반 태그 단어 제외)
        3. 기존 정규식 fallback
        """
        # 1. outfit_mode의 추적된 캐릭터 이름 중 block에 있는 것 찾기
        if self.outfit_mode_ref:
            block_lower = block.lower()
            for name in self.outfit_mode_ref.character_results:
                if name.lower() in block_lower:
                    return name.lower()

        # 2. 콤마로 분리된 토큰 중 캐릭터 이름 패턴 찾기
        # 일반 태그에 자주 쓰이는 단어 (이 단어가 포함된 토큰은 이름이 아님)
        COMMON_TAG_WORDS = {
            'aged', 'mature', 'adolescent', 'girl', 'boy', 'man', 'woman',
            'hair', 'eyes', 'skin', 'breasts', 'chest', 'face',
            'shirt', 'skirt', 'dress', 'pants', 'jacket', 'coat', 'uniform',
            'cotton', 'wool', 'silk', 'leather', 'denim', 'chiffon', 'lace',
            'pleated', 'mini', 'collared', 'fitted', 'oversized', 'loose',
            'short', 'long', 'medium', 'large', 'small', 'tall',
            'blonde', 'black', 'white', 'red', 'blue', 'brown', 'green',
            'purple', 'aqua', 'pink', 'silver', 'grey', 'gray', 'golden',
            'school', 'standing', 'sitting', 'lying', 'walking', 'running',
            'looking', 'pointing', 'holding', 'gripping', 'shouting',
            'angry', 'happy', 'sad', 'flustered', 'blushing', 'smiling',
            'left', 'right', 'center', 'top', 'bottom', 'front', 'back',
            'ornament', 'glasses', 'socks', 'stockings', 'shoes', 'boots',
            'tie', 'necktie', 'ribbon', 'bow', 'scarf', 'glove',
            'over', 'under', 'open', 'closed', 'up', 'down',
            'old', 'young', 'years', 'kg', 'cm', 'tall',
        }

        tokens = [t.strip() for t in block.split(',')]
        for token in tokens:
            token = token.strip()
            # 2단어 이름만 매칭 (각 단어 2글자 이상, 숫자/괄호 없음)
            if re.match(r'^[a-z][a-z]+\s[a-z][a-z]+$', token.lower()):
                words = token.lower().split()
                if not any(w in COMMON_TAG_WORDS for w in words):
                    return token.lower()

        # 3. 기존 패턴: name (series) 형식
        m = re.search(r'([a-zA-Z][a-zA-Z\s\-]+?)\s*\(([^)]+)\)', block)
        if m:
            return m.group(1).strip().lower()

        return ""

    def _extract_all_character_names(self, text: str) -> list[str]:
        """전체 캐릭터 블럭에서 모든 캐릭터 이름 추출"""
        blocks = self._split_char_blocks(text)
        names = []
        for block in blocks:
            name = self._extract_character_name(block)
            if name:
                names.append(name)
        return names

    # ─── 복장 결과 조회 ─────────────────────────────────────

    def _get_outfit_result(self, character_name: str) -> Optional[str]:
        """outfit_mode에서 캐릭터의 LLM 복장 통합 결과 가져오기"""
        if not self.outfit_mode_ref:
            return None

        char_name_lower = character_name.lower()
        for name, char_data in self.outfit_mode_ref.character_results.items():
            if name.lower() == char_name_lower:
                return char_data.llm_result
        return None

    def _get_outfit_entries(self, character_name: str) -> list[dict]:
        """outfit_mode에서 캐릭터의 복장 추출 엔트리 목록 가져오기.
        entries는 [0]=가장 오래됨, [-1]=가장 최신.
        """
        if not self.outfit_mode_ref:
            return []

        char_name_lower = character_name.lower()
        for name, char_data in self.outfit_mode_ref.character_results.items():
            if name.lower() == char_name_lower:
                return [
                    {
                        "outfit_prompt": e.outfit_prompt,
                        "positive_prompt": e.positive_prompt,
                    }
                    for e in char_data.entries
                ]
        return []

    def _get_previous_chat(self, character_name: str) -> str:
        """outfit_mode에서 캐릭터의 가장 최근 엔트리의 chat_content 가져오기.
        이전 복장 통합 결과를 만들 때 사용했던 chat을 previous chat으로 사용.
        entries[-1]이 가장 최신 데이터.
        """
        if not self.outfit_mode_ref:
            return ""

        char_name_lower = character_name.lower()
        for name, char_data in self.outfit_mode_ref.character_results.items():
            if name.lower() == char_name_lower:
                if char_data.entries:
                    # 가장 최신 엔트리의 채팅 내용 사용
                    return char_data.entries[-1].chat_content or ""
        return ""

    # ─── 프롬프트 강화 ─────────────────────────────────────

    async def enhance_prompt(self, positive: str, chat_content: str = "") -> tuple[str, str]:
        """
        프롬프트의 캐릭터 블럭을 강화.

        Args:
            positive: 원본 긍정 프롬프트 ([CHAT] 제거됨)
            chat_content: 분리된 채팅 내용

        Returns:
            (enhanced_positive, original_positive)
            강화 실패 또는 비활성화 시 (original, original) 반환
        """
        if not self.enabled:
            return positive, positive

        if not self.enhance_prompt_file:
            self._log("skip_no_prompt_file", {})
            return positive, positive

        # [UPSCALE] 섹션에서 char 블럭 추출
        upscale_content = self._extract_section(positive, "UPSCALE")
        if not upscale_content:
            self._log("skip_no_upscale", {})
            return positive, positive

        char_blocks = self._split_char_blocks(upscale_content)
        if not char_blocks:
            self._log("skip_no_chars", {})
            return positive, positive

        self._log("enhance_start", {
            "char_blocks": len(char_blocks),
            "chat_length": len(chat_content) if chat_content else 0,
        })

        # 프론트엔드 토스트 알림
        if self.notify_frontend_func:
            await self.notify_frontend_func("enhance_started", {
                "char_count": len(char_blocks),
            })

        # 캐릭터별 강화 수행
        enhanced_blocks = {}  # original_block -> enhanced_block
        for block in char_blocks:
            char_name = self._extract_character_name(block)
            if not char_name:
                self._log("skip_no_name", {"block_preview": block[:50]})
                continue

            outfit_result = self._get_outfit_result(char_name)
            previous_chat = self._get_previous_chat(char_name)
            # 이전 강화 결과를 항상 전달 (LLM이 Case A/B로 판단하여 일관성 유지)
            previous_enhanced = self._last_enhanced_block.get(char_name, "")

            # ─── 일관성 추적 디버그 로그 ───
            self._log("consistency_tracking", {
                "char_name": char_name,
                "has_outfit_result": outfit_result is not None,
                "outfit_result_preview": (outfit_result[:200] if outfit_result else None),
                "has_previous_enhanced": bool(previous_enhanced),
                "previous_enhanced_length": len(previous_enhanced) if previous_enhanced else 0,
                "previous_enhanced_preview": (previous_enhanced[:300] if previous_enhanced else None),
                "tracked_characters": list(self._last_enhanced_block.keys()),
            })

            try:
                combined, outfit_only = await self._run_enhance_prompt(
                    char_name, block, outfit_result, chat_content,
                    previous_chat=previous_chat,
                    previous_enhanced=previous_enhanced,
                )
                if combined and combined != block:
                    enhanced_blocks[block.strip()] = combined
                    self._log("block_enhanced", {
                        "name": char_name,
                        "original_length": len(block),
                        "combined_length": len(combined),
                        "outfit_only_length": len(outfit_only),
                        "has_expression": combined != outfit_only,
                    })
                else:
                    self._log("block_unchanged", {"name": char_name})
                # 강화 일관성 추적: OUTFIT ONLY 저장 (표정은 제외, 매번 새로 생성)
                if outfit_only:
                    self._last_enhanced_block[char_name] = outfit_only
            except Exception as e:
                self._log("block_enhance_error", {"name": char_name, "error": str(e)})
                traceback.print_exc()

        if not enhanced_blocks:
            self._log("enhance_no_changes", {})
            return positive, positive

        # 프롬프트의 모든 섹션에서 char 블럭 교체
        enhanced_positive = self._replace_blocks_in_prompt(positive, enhanced_blocks)

        # 구조 준수 검증
        if not self._validate_structure(positive, enhanced_positive):
            self._log("structure_validation_failed", {})
            return positive, positive

        self._log("enhance_complete", {
            "blocks_changed": len(enhanced_blocks),
            "original_length": len(positive),
            "enhanced_length": len(enhanced_positive),
        })

        # 프론트엔드 토스트 알림
        if self.notify_frontend_func:
            await self.notify_frontend_func("enhance_completed", {
                "blocks_changed": len(enhanced_blocks),
            })

        return enhanced_positive, positive

    async def _run_enhance_prompt(
        self,
        character_name: str,
        char_block: str,
        outfit_result: Optional[str],
        chat_content: str = "",
        previous_chat: str = "",
        previous_enhanced: str = "",
    ) -> tuple[str, str]:
        """customprompt/ 의 강화 스크립트를 로드하여 실행.
        Returns: (combined_block, outfit_only_block)
        """
        filepath = os.path.join(CUSTOMPROMPT_DIR, self.enhance_prompt_file)
        if not os.path.isfile(filepath):
            self._log("prompt_file_not_found", {"file": self.enhance_prompt_file})
            return char_block, char_block

        spec = importlib.util.spec_from_file_location("enhance_prompt", filepath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        if not hasattr(mod, "run"):
            self._log("no_run_function", {"file": self.enhance_prompt_file})
            return char_block, char_block

        result = await mod.run(
            character_name=character_name,
            char_block=char_block,
            outfit_result=outfit_result,
            chat_content=chat_content,
            previous_chat=previous_chat,
            previous_enhanced=previous_enhanced,
        )

        # 하위 호환: str 반환 구형 스크립트 지원
        if isinstance(result, str):
            result = (result, result)

        combined, outfit_only = result

        if not combined or (isinstance(combined, str) and combined.startswith("[LLM 실패]")):
            self._log("llm_failed", {"name": character_name, "result": (combined or "")[:100]})
            return char_block, char_block

        # 검증
        validated_combined = self._validate_enhanced_block(char_block, combined, character_name)
        validated_outfit = self._validate_enhanced_block(char_block, outfit_only, character_name) if outfit_only else validated_combined
        return validated_combined, validated_outfit

    # ─── 검증 ───────────────────────────────────────────────

    def _validate_enhanced_block(self, original: str, enhanced: str, char_name: str) -> str:
        """강화된 블럭 검증. 하드 코딩으로 고칠 수 있는 것은 고치고,
        그렇지 않으면 LLM 재실행.
        """
        # 1. 캐릭터 이름 누락 검사
        if char_name.lower() not in enhanced.lower():
            self._log("validation_name_missing", {"name": char_name})
            # 원본에서 풀네임 패턴 찾아서 삽입 시도
            name_pattern = re.search(
                re.escape(char_name) + r'\s*\([^)]+\)', original, re.IGNORECASE
            )
            if name_pattern:
                full_name = name_pattern.group(0)
                # 프롬프트 시작 부분에 이름 삽입
                if enhanced and not enhanced.endswith(','):
                    enhanced += ','
                enhanced = f"{full_name}, {enhanced}"
                self._log("validation_name_fixed", {"name": char_name})

        # 2. | 구분자 손상 검사 (강화된 블럭 자체에 | 가 포함되면 안됨)
        if '|' in enhanced:
            self._log("validation_pipe_in_block", {"name": char_name})
            # | 제거 (캐릭터 구분자로 오인될 수 있음)
            enhanced = enhanced.replace('|', ',')

        # 3. 빈 블럭 검사
        if not enhanced.strip():
            self._log("validation_empty_block", {"name": char_name})
            return original

        return enhanced

    # ─── 프롬프트 교체 ──────────────────────────────────────

    def _replace_blocks_in_prompt(self, prompt: str, replacements: dict[str, str]) -> str:
        """프롬프트의 모든 섹션에서 원본 블럭을 강화된 블럭으로 교체"""
        result = prompt
        for original, enhanced in replacements.items():
            result = result.replace(original, enhanced)
        return result

    def _validate_structure(self, original: str, enhanced: str) -> bool:
        """강화 후 프롬프트 구조가 올바른지 검증.
        [Positive], [ILXL], [UPSCALE] 섹션이 모두 존재하는지 확인.
        """
        for section in ["Positive", "ILXL", "UPSCALE"]:
            orig_has = f"[{section}]" in original
            enh_has = f"[{section}]" in enhanced
            if orig_has and not enh_has:
                self._log("structure_broken", {"section": section})
                return False
        return True

    # ─── 원본 추적 ──────────────────────────────────────────

    def track_original(self, request_id: str, original_positive: str):
        """원본 프롬프트 추적 등록"""
        if original_positive:
            self._original_prompts[request_id] = original_positive

    def get_original(self, request_id: str) -> Optional[str]:
        """추적된 원본 프롬프트 반환"""
        return self._original_prompts.get(request_id)

    def clear_tracking(self, request_id: str):
        """추적 정보 삭제"""
        self._original_prompts.pop(request_id, None)

    def clear_all_tracking(self):
        """모든 추적 정보 삭제"""
        self._original_prompts.clear()

    def get_original_for_resend(self, enhanced_positive: str, request_id: str = "") -> str:
        """재전송 예약용 원본 프롬프트 반환.
        request_id가 있으면 추적된 원본 반환, 없으면 enhanced_positive 그대로 반환.
        """
        if request_id and request_id in self._original_prompts:
            return self._original_prompts[request_id]
        return enhanced_positive

    # ─── 상태 ────────────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "enabled": self.enabled,
            "prompt_file": self.enhance_prompt_file,
            "tracked_prompts": len(self._original_prompts),
        }


# 싱글톤 인스턴스
enhance_mode = PromptEnhanceMode()
