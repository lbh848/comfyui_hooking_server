"""
PromptEnhanceMode - 배치 프롬프트 강화 모드

동작 순서:
1단계: 프롬프트 파싱
  순차 파싱으로 6개 변수 추출:
  - [SLOT] → slot
  - [CHAT] → chat
  - [UPSCALE] → char
  - [ILXL]에서 char 제거 → setup
  - 상단 섹션에서 char 이후 → supplement
  - 상단 섹션에서 char 이전 → prefix (퀄리티 태그 등)

2단계: characters.txt 기반 캐릭터 이름 추출
  - 스토리지 디렉토리의 characters.txt에서 이름 목록 로드
  - char 블럭에서 이름 매칭 (가장 긴 이름 우선)

3단계: 파싱된 데이터를 강화 스크립트에 전달하여 프롬프트 강화
  - char, setup, supplement 각각 강화
  - LLM이 [CHAR]/[SETUP]/[SUPPLEMENT] 구분 출력
  - 강화 결과로 전체 프롬프트 재조립
"""

import asyncio
import json
import os
import re
import datetime
import importlib.util
import traceback
from typing import Optional, Callable


# ─── 상수 ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOMPROMPT_DIR = os.path.join(BASE_DIR, "customprompt")
STORAGE_DIR = os.path.join(CUSTOMPROMPT_DIR, "enhance_outfit_prompt_v4_storage")
CHARACTERS_FILE = os.path.join(STORAGE_DIR, "characters.txt")


class PromptEnhanceMode:
    """프롬프트 강화 모드 매니저"""

    def __init__(self):
        self.enabled: bool = False
        self.enhance_prompt_file: str = ""
        self._lock = asyncio.Lock()
        self.mode_log_func: Optional[Callable] = None
        self.notify_frontend_func: Optional[Callable] = None
        # 원본 프롬프트 추적: request_id -> original_positive
        self._original_prompts: dict[str, str] = {}
        # 캐릭터별 마지막 강화 추적 (복장만 저장, 표정 제외)
        self._last_enhanced_block: dict[str, str] = {}
        # 캐릭터 이름 캐시
        self._character_names: list[str] = []
        # 마지막 강화 시 와일드카드 정보
        self._last_wildcard_info: dict = {}
        self._load_character_names()
        # 배치별 --- 구분자 추적: 이번 배치에서 이미 구분자를 추가한 캐릭터
        self._batch_separator_chars: set[str] = set()

    def _log(self, action: str, data: dict = None):
        if self.mode_log_func:
            self.mode_log_func("enhance_mode", action, data)

    # ─── 캐릭터 이름 로드 ───────────────────────────────────

    def _load_character_names(self):
        """characters.txt에서 캐릭터 이름 목록 로드.
        # 으 시작하는 줄은 주석으로 무시.
        """
        if not os.path.exists(CHARACTERS_FILE):
            self._log("characters_file_not_found", {"path": CHARACTERS_FILE})
            return
        try:
            with open(CHARACTERS_FILE, 'r', encoding='utf-8') as f:
                names = [
                    line.strip().lower()
                    for line in f
                    if line.strip() and not line.strip().startswith('#')
                ]
                self._character_names = names
                self._log("characters_loaded", {"count": len(names), "names": names})
        except Exception as e:
            self._log("characters_load_error", {"error": str(e)})
            self._character_names = []

    # ─── 1단계: 프롬프트 파싱 ──────────────────────────────────

    @staticmethod
    def _extract_section(text: str, section_name: str) -> str:
        """프롬프트에서 특정 섹션 내용 추출.
        [SECTION_NAME]\\n...content... 이후 다음 [SECTION] 또는 끝까지.
        """
        pattern = rf'\[{section_name}\]\s*(.*?)(?=\n\s*\[|\Z)'
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _split_char_blocks(text: str) -> list[str]:
        """| 로 구분된 캐릭터 블럭 분리. 괄호 안의 | 는 무시."""
        if not text:
            return []
        blocks = []
        current = []
        paren_depth = 0
        for ch in text:
            if ch == '(':
                paren_depth += 1
                current.append(ch)
            elif ch == ')':
                paren_depth = max(0, paren_depth - 1)
                current.append(ch)
            elif ch == '|' and paren_depth == 0:
                block = ''.join(current).strip()
                if block:
                    blocks.append(block)
                current = []
            else:
                current.append(ch)
        block = ''.join(current).strip()
        if block:
            blocks.append(block)
        return blocks

    def _parse_prompt_sections(self, positive: str, chat_content: str = "") -> dict:
        """
        프롬프트를 6개 섹션으로 순차 파싱.

        구조:
          {prefix}, {char}, {supplement}
          [ILXL] {setup}, {char}
          [UPSCALE] {char}
          [CHAT] {chat}
          [SLOT] {slot}

        순차 파싱 순서:
          1. [SLOT] → slot
          2. [CHAT] → chat
          3. [UPSCALE] → char
          4. [ILXL]에서 char 제거 → setup
          5. 상단 섹션에서 char 이전 → prefix
          6. 상단 섹션에서 char 이후 → supplement
        """
        result = {
            "char": "",
            "setup": "",
            "supplement": "",
            "prefix": "",
            "chat": "",
            "slot": "",
        }

        # 1. [SLOT] → slot (positive 우선, 없으면 chat_content)
        result["slot"] = self._extract_section(positive, "SLOT")
        if not result["slot"]:
            result["slot"] = self._extract_section(chat_content, "SLOT")

        # 2. [CHAT] → chat (positive 우선, 없으면 chat_content)
        result["chat"] = self._extract_section(positive, "CHAT")
        if not result["chat"]:
            result["chat"] = self._extract_section(chat_content, "CHAT")
        # 둘 다 없으면 chat_content 원본을 chat으로 사용 ([SLOT] 부분은 제거)
        if not result["chat"] and chat_content:
            chat_clean = re.sub(
                r'\[SLOT\].*?(?=\n\s*\[|\Z)', '', chat_content,
                flags=re.DOTALL | re.IGNORECASE
            )
            result["chat"] = chat_clean.strip()

        # 3. [UPSCALE] → char
        result["char"] = self._extract_section(positive, "UPSCALE")

        if not result["char"]:
            return result

        # 4. [ILXL]에서 char 제거 → setup
        ilxl_content = self._extract_section(positive, "ILXL")
        if ilxl_content:
            setup = ilxl_content
            for block in self._split_char_blocks(result["char"]):
                block_clean = block.strip().rstrip(',').strip()
                if block_clean:
                    setup = setup.replace(block_clean, '')
            setup = re.sub(r',\s*,', ',', setup)
            setup = setup.strip().strip(',').strip()
            result["setup"] = setup

        # 5-6. 상단 섹션에서 prefix, supplement 추출
        top_section = re.split(r'\n\s*\[', positive)[0].strip()
        if top_section:
            char_text = result["char"].strip().rstrip(',').strip()
            idx = top_section.rfind(char_text)
            if idx >= 0:
                result["prefix"] = top_section[:idx].strip().rstrip(',').strip()
                supplement = top_section[idx + len(char_text):]
                supplement = supplement.strip().lstrip(',').strip()
                result["supplement"] = supplement

        self._log("parse_complete", {
            k: v[:200] if v else ""
            for k, v in result.items()
        })

        return result

    # ─── 2단계: 캐릭터 이름 추출 ───────────────────────────────────

    def _extract_character_name(self, block: str) -> str:
        """characters.txt의 이름 목록으로 블럭에서 캐릭터 이름 매칭.
        가장 긴 이름부터 매칭 (kashiwazaki sena > sena).
        """
        if not self._character_names:
            return ""
        block_lower = block.lower()
        for name in sorted(self._character_names, key=len, reverse=True):
            if name in block_lower:
                return name
        return ""

    # ─── 3단계: 프롬프트 강화 ─────────────────────────────────────

    async def enhance_prompt(self, positive: str, chat_content: str = "") -> tuple[str, str]:
        """
        프롬프트 파싱 → 캐릭터별 강화 → 재조립.

        Returns:
            (enhanced_positive, original_positive)
            강화 실패 또는 비활성화 시 (original, original) 반환
        """
        # 1단계: 파싱 (항상 실행)
        parsed = self._parse_prompt_sections(positive, chat_content)

        if not parsed["char"]:
            self._log("skip_no_chars", {})
            return positive, positive

        char_blocks = self._split_char_blocks(parsed["char"])
        if not char_blocks:
            self._log("skip_no_char_blocks", {})
            return positive, positive

        # 강화 모드 꺼짐 → 추적만 하고 종료
        if not self.enabled:
            self._track_outfits(char_blocks, parsed)
            return positive, positive

        if not self.enhance_prompt_file:
            self._log("skip_no_prompt_file", {})
            self._last_wildcard_info = {}
            return positive, positive

        self._log("enhance_start", {
            "char_blocks": len(char_blocks),
            "setup_preview": parsed["setup"][:80],
            "supplement_preview": parsed["supplement"][:80],
        })

        # 프론트엔드 토스트 알림
        if self.notify_frontend_func:
            await self.notify_frontend_func("enhance_started", {
                "char_count": len(char_blocks),
            })

        # 2단계 + 3단계: 캐릭터별 강화
        enhanced_chars = {}  # original_block -> enhanced_char (combined)
        enhanced_setup = parsed["setup"]  # 기본값: 원본
        enhanced_supplement = parsed["supplement"]  # 기본값: 원본

        # run_all로 모든 캐릭터 한번에 강화
        v5_result = await self._try_run_all(char_blocks, parsed)
        if v5_result is not None:
            enhanced_chars, enhanced_setup, enhanced_supplement = v5_result

        if not enhanced_chars:
            self._log("enhance_no_changes", {})
            return positive, positive

        # 4단계: 재조립
        enhanced_positive = self._reassemble_prompt(
            positive, parsed, enhanced_chars, enhanced_setup, enhanced_supplement
        )

        # 구조 검증: [ILXL], [UPSCALE] 섹션이 유지되었는지 확인
        for section in ["ILXL", "UPSCALE"]:
            if f"[{section}]" in positive and f"[{section}]" not in enhanced_positive:
                self._log("structure_broken", {"section": section})
                return positive, positive

        self._log("enhance_complete", {
            "blocks_changed": len(enhanced_chars),
            "original_length": len(positive),
            "enhanced_length": len(enhanced_positive),
        })

        # 프론트엔드 토스트 알림
        if self.notify_frontend_func:
            await self.notify_frontend_func("enhance_completed", {
                "blocks_changed": len(enhanced_chars),
            })

        return enhanced_positive, positive

    # ─── 재조립 ────────────────────────────────────────────────

    @staticmethod
    def _replace_section_content(text: str, section_name: str, new_content: str) -> str:
        """텍스트에서 특정 [SECTION]의 내용을 교체."""
        pattern = rf'(\[{section_name}\]\s*)(.*?)(?=\n\s*\[|\Z)'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return text[:match.start(2)] + new_content + text[match.end(2):]
        return text

    def _reassemble_prompt(
        self,
        positive: str,
        parsed: dict,
        enhanced_chars: dict,
        enhanced_setup: str,
        enhanced_supplement: str,
    ) -> str:
        """
        파싱의 역과정: 강화된 섹션으로 프롬프트 재조립.

        원본 구조:
          {prefix}, {char1 | char2 | ...}, {supplement}
          [ILXL] {setup}, {char1 | char2 | ...}
          [UPSCALE] {char1 | char2 | ...}

        강화 후:
          {prefix}, {enhanced_char1 | enhanced_char2 | ...}, {enhanced_supplement}
          [ILXL] {enhanced_setup}, {enhanced_char1 | enhanced_char2 | ...}
          [UPSCALE] {enhanced_char1 | enhanced_char2 | ...}
        """
        result = positive

        # 1. 캐릭터 블럭 교체 (원본 → 강화본, 모든 위치에서 교체됨)
        for original, enhanced in enhanced_chars.items():
            result = result.replace(original, enhanced)

        # 2. [ILXL]에서 setup 교체
        if enhanced_setup and parsed.get("setup") and enhanced_setup != parsed["setup"]:
            ilxl_content = self._extract_section(result, "ILXL")
            if ilxl_content:
                new_ilxl = ilxl_content.replace(parsed["setup"], enhanced_setup, 1)
                result = self._replace_section_content(result, "ILXL", new_ilxl)

        # 3. 상단 섹션에서 setup, supplement 교체
        #    상단 구조: {prefix(= quality_tags, setup)}, {char}, {supplement}
        #    setup과 supplement 모두 상단에 존재하므로 여기서 교체
        first_section = re.search(r'\n\s*\[', result)
        if first_section:
            top = result[:first_section.start()]
            rest = result[first_section.start():]

            if enhanced_setup and parsed.get("setup") and enhanced_setup != parsed["setup"]:
                top = top.replace(parsed["setup"], enhanced_setup, 1)

            if parsed.get("supplement") and enhanced_supplement != parsed["supplement"]:
                if enhanced_supplement:
                    top = top.replace(parsed["supplement"], enhanced_supplement, 1)
                else:
                    # LLM이 "none" → supplement 제거
                    top = top.replace(parsed["supplement"], "", 1)
                    top = re.sub(r',\s*,', ',', top)
                    top = top.strip().rstrip(',').strip()

            result = top + rest

        return result

    # ─── 복장 추적 (강화 모드 꺼짐 시에만 실행) ──────────────────────

    @staticmethod
    def _get_storage_path(character_name: str) -> str:
        safe_name = re.sub(r'[^\w\s-]', '', character_name.lower()).strip().replace(' ', '_')
        return os.path.join(STORAGE_DIR, f"{safe_name}.json")

    @staticmethod
    def _load_storage_history(character_name: str) -> list:
        path = PromptEnhanceMode._get_storage_path(character_name)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            except (json.JSONDecodeError, IOError):
                pass
        return []

    @staticmethod
    def _save_storage_history(character_name: str, history: list, max_entries: int = 15):
        os.makedirs(STORAGE_DIR, exist_ok=True)
        history = history[-max_entries:]
        path = PromptEnhanceMode._get_storage_path(character_name)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def _track_outfits(self, char_blocks: list[str], parsed: dict):
        """강화 없이 원본 복장만 스토리지에 저장.
        배치당 첫 entry 앞에 --- 구분자를 추가하여 배치를 구분.
        """
        slot = parsed.get("slot", "")
        slot_before = ""
        slot_after = ""
        if slot:
            if '||' in slot:
                sp = slot.split('||', 1)
                slot_before, slot_after = sp[0].strip(), sp[1].strip()
            else:
                slot_before, slot_after = slot.strip(), ""

        for block in char_blocks:
            char_name = self._extract_character_name(block)
            if not char_name:
                continue

            history = self._load_storage_history(char_name)

            # 배치당 첫 entry 전에 --- 구분자 추가
            if char_name not in self._batch_separator_chars:
                if history and history[-1] != "---":
                    history.append("---")
                self._batch_separator_chars.add(char_name)

            entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "char": block.strip(),
                "setup": parsed.get("setup", ""),
                "supplement": parsed.get("supplement", ""),
                "chat": parsed.get("chat", "")[:500],
                "slot_before": slot_before,
                "slot_after": slot_after,
                "reason": "tracked (enhance off)",
            }

            history.append(entry)
            self._save_storage_history(char_name, history)
            self._log("outfit_tracked", {"name": char_name})

    # ─── 스크립트 실행 ──────────────────────────────────────────

    async def _try_run_all(
        self,
        char_blocks: list[str],
        parsed: dict,
    ) -> tuple[dict, str, str] | None:
        """run_all로 모든 캐릭터를 단일 LLM 호출로 강화."""
        filepath = os.path.join(CUSTOMPROMPT_DIR, self.enhance_prompt_file)
        if not os.path.isfile(filepath):
            return None

        try:
            spec = importlib.util.spec_from_file_location("enhance_prompt", filepath)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception:
            return None

        if not hasattr(mod, "run_all"):
            return None

        # 캐릭터 목록 구성
        characters = []
        for block in char_blocks:
            name = self._extract_character_name(block)
            if not name:
                continue
            # previous_enhanced: 메모리 우선, 없으면 저장소에서 로드
            previous = self._last_enhanced_block.get(name, "")
            if not previous:
                history = self._load_storage_history(name)
                for entry in reversed(history):
                    if isinstance(entry, dict) and entry.get("char"):
                        previous = entry["char"]
                        break
            characters.append({
                "name": name,
                "char": block,
                "previous_enhanced": previous,
            })

        if not characters:
            return None

        try:
            import inspect
            run_params = inspect.signature(mod.run_all).parameters

            run_kwargs = {}
            for key, value in {
                "characters": characters,
                "setup": parsed["setup"],
                "supplement": parsed["supplement"],
                "chat": parsed["chat"],
                "slot": parsed["slot"],
            }.items():
                if key in run_params:
                    run_kwargs[key] = value

            result = await mod.run_all(**run_kwargs)
        except Exception as e:
            self._log("v5_run_all_error", {"error": str(e)})
            traceback.print_exc()
            return None

        if not result or "characters" not in result:
            self._log("v5_run_all_invalid", {})
            return None

        # 결과를 기존 구조로 변환
        enhanced_chars = {}
        wildcard_chars = []  # 와일드카드 정보 수집
        for char_result in result.get("characters", []):
            name = char_result.get("name", "")
            # 원본 블럭 찾기
            for block in char_blocks:
                block_name = self._extract_character_name(block)
                if block_name and block_name.lower() == name.lower():
                    original = block.strip()
                    combined = char_result.get("char", block)
                    if combined != block:
                        enhanced_chars[original] = combined
                    # 일관성 추적
                    if char_result.get("outfit_only"):
                        self._last_enhanced_block[name] = char_result["outfit_only"]
                    # 와일드카드 정보 수집
                    nsfw_replaced = char_result.get("nsfw_replaced", [])
                    outfit_llm_raw = char_result.get("outfit_llm_raw", "")
                    if nsfw_replaced or outfit_llm_raw:
                        wildcard_chars.append({
                            "name": name,
                            "nsfw_replaced": nsfw_replaced,
                            "outfit_llm_raw": outfit_llm_raw,
                        })
                    break

        enhanced_setup = result.get("setup", parsed["setup"])
        enhanced_supplement = result.get("supplement", parsed["supplement"])

        self._log("v5_run_all_complete", {
            "blocks_changed": len(enhanced_chars),
        })

        # 와일드카드 정보 저장
        has_wildcards = any(c["nsfw_replaced"] for c in wildcard_chars)
        self._last_wildcard_info = {
            "has_wildcards": has_wildcards,
            "characters": wildcard_chars,
        }

        return enhanced_chars, enhanced_setup, enhanced_supplement

    # ─── 검증 ───────────────────────────────────────────────

    @staticmethod
    def _validate_enhanced_block(original: str, enhanced: str, char_name: str) -> str:
        """강화된 블럭 기본 검증."""
        # 캐릭터 이름 누락 시 원본에서 찾아 삽입
        if char_name.lower() not in enhanced.lower():
            name_match = re.search(
                re.escape(char_name) + r'\s*\([^)]+\)', original, re.IGNORECASE
            )
            if name_match:
                full_name = name_match.group(0)
                if enhanced and not enhanced.endswith(','):
                    enhanced += ','
                enhanced = f"{full_name}, {enhanced}"

        # | 구분자 손상 방지
        if '|' in enhanced:
            enhanced = enhanced.replace('|', ',')

        # 빈 블럭
        if not enhanced.strip():
            return original

        return enhanced

    # ─── 원본 추적 (재전송 예약용) ─────────────────────────────

    def track_original(self, request_id: str, original_positive: str):
        if original_positive:
            self._original_prompts[request_id] = original_positive

    def get_original(self, request_id: str) -> Optional[str]:
        return self._original_prompts.get(request_id)

    def clear_tracking(self, request_id: str):
        self._original_prompts.pop(request_id, None)

    def clear_all_tracking(self):
        self._original_prompts.clear()

    def get_original_for_resend(self, enhanced_positive: str, request_id: str = "") -> str:
        if request_id and request_id in self._original_prompts:
            return self._original_prompts[request_id]
        return enhanced_positive

    def get_last_wildcard_info(self) -> dict:
        """마지막 enhance_prompt 호출에서 수집된 와일드카드 정보를 반환."""
        return self._last_wildcard_info

    # ─── 상태 ────────────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "enabled": self.enabled,
            "prompt_file": self.enhance_prompt_file,
            "tracked_prompts": len(self._original_prompts),
            "character_names": self._character_names,
        }


# 싱글톤 인스턴스
enhance_mode = PromptEnhanceMode()
