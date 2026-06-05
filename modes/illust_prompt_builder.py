"""
IllustPromptBuilder - 삽화 모드 프롬프트 빌더

8188(ComfyUI)로부터 들어온 raw 프롬프트를 파싱하여,
봇 설정(LoRA, 캐릭터 정보)과 tags.json(아티스트/품질)을 조합해
최종 프롬프트를 빌드한다.

처리 순서:
1. 파싱: raw 긍정 프롬프트 → [SETUP], [CHAR], [SUPPLEMENT] 섹션 분리
2. 단어 치환: 각 섹션에 봇의 단어 치환 규칙 적용 (server.py에서 수행)
3. 캐릭터 감지: 모든 섹션에서 bot.json의 캐릭터 이름과 매칭
4. 프롬프트 빌드: 감지된 캐릭터 정보 + 봇 설정 + tags.json → 최종 프롬프트 조립
"""

import json
import re
from collections import deque
import time


# ─── 로깅 ──────────────────────────────────────────────────
_illust_build_logs: deque = deque(maxlen=20)


def get_illust_logs() -> list:
    """삽화 모드 프롬프트 빌드 로그 목록 반환."""
    return list(_illust_build_logs)


def log_illust_build(raw_positive: str, sections: dict, detected: list,
                     word_replaced: dict, final_positive: str, final_negative: str):
    """삽화 모드 프롬프트 빌드 로그 기록."""
    log_entry = {
        "timestamp": time.time(),
        "raw_positive": raw_positive,
        "parsed_setup": sections.get("setup", ""),
        "parsed_char": sections.get("char", ""),
        "parsed_supplement": sections.get("supplement", ""),
        "word_replaced_setup": word_replaced.get("setup", ""),
        "word_replaced_char": word_replaced.get("char", ""),
        "word_replaced_supplement": word_replaced.get("supplement", ""),
        "detected_characters": detected,
        "final_positive": final_positive,
        "final_negative": final_negative,
    }
    _illust_build_logs.append(log_entry)
    print(f"[ILLUST_LOG] 빌드 로그 기록: detected={detected}, "
          f"positive_len={len(final_positive)}, negative_len={len(final_negative)}")


class IllustPromptBuilder:
    """삽화 모드 프롬프트 빌더"""

    @staticmethod
    def parse_sections(positive: str) -> dict:
        """긍정 프롬프트를 [SETUP], [CHAR], [SUPPLEMENT] 섹션으로 파싱.

        [SETUP] 앞의 텍스트는 무시한다.
        대소문자 무관하게 매칭한다.
        섹션 태그는 항상 줄의 시작에 위치해야 한다.
        섹션이 없으면 빈 문자열로 반환한다.

        Returns:
            {"setup": str, "char": str, "supplement": str}
        """
        if not positive:
            return {"setup": "", "char": "", "supplement": ""}

        sections = {"setup": "", "char": "", "supplement": ""}

        # 섹션 태그 위치 찾기 (대소문자 무관, 줄 시작 위치)
        section_names = ["SETUP", "CHAR", "SUPPLEMENT"]
        markers = []
        for name in section_names:
            for m in re.finditer(rf'^\[{name}\]', positive, re.IGNORECASE | re.MULTILINE):
                markers.append((m.start(), name, m.end()))
            # 문자열 시작 위치 (줄바꿈 없이 시작하는 경우)
            for m in re.finditer(rf'\n\[{name}\]', positive, re.IGNORECASE):
                start = m.start() + 1  # \n 이후 위치
                markers.append((start, name, start + len(f'[{name}]')))

        # 중복 제거 후 시작 위치 기준 정렬
        seen = set()
        unique_markers = []
        for pos, name, end in markers:
            if pos not in seen:
                seen.add(pos)
                unique_markers.append((pos, name, end))
        unique_markers.sort(key=lambda x: x[0])

        # 각 섹션의 내용 추출 (다음 섹션 시작 전까지)
        for i, (pos, name, tag_end) in enumerate(unique_markers):
            next_pos = unique_markers[i + 1][0] if i + 1 < len(unique_markers) else len(positive)
            content = positive[tag_end:next_pos].strip()
            key = name.lower()
            if key == "setup":
                sections["setup"] = content
            elif key == "char":
                sections["char"] = content
            elif key == "supplement":
                sections["supplement"] = content

        return sections

    @staticmethod
    def detect_characters(text_sections: list, char_names: list) -> list:
        """모든 텍스트 섹션에서 캐릭터 이름 포함 여부로 감지.

        대소문자 무관하게 검색하되, 결과는 bot.json의 원래 대소문자로 반환한다.
        감지된 순서를 유지하고 중복을 제거한다.

        Args:
            text_sections: 검색할 텍스트 리스트 ([setup, char, supplement])
            char_names: bot.json의 캐릭터 이름 리스트

        Returns:
            감지된 캐릭터 이름 리스트 (원래 대소문자)
        """
        detected = []
        # 대소문자 무관 매칭을 위해 소문자로 변환한 텍스트 합치기
        combined_lower = " ".join(s.lower() for s in text_sections if s)

        for name in char_names:
            if name.lower() in combined_lower:
                if name not in detected:
                    detected.append(name)

        return detected

    def build_positive_prompt(
        self,
        setup: str,
        char: str,
        supplement: str,
        detected_chars: list,
        bot: dict,
        tags: dict,
        settings: dict,
        bot_name: str
    ) -> str:
        """최종 긍정 프롬프트 빌드.

        ANIMA 섹션: 트리거워드 + 아티스트 + 품질 + setup + char + supplement
        SDXL 섹션: 트리거워드 + 아티스트 + 품질 + setup + char
        메타데이터: CHAR_LIST, CACHE_PATH, FACE_ID, LORA, HRF, IMG, FD/HD/ED, END
        """
        characters = bot.get("characters", [])

        # ─── 감지된 캐릭터의 LoRA/트리거워드 정보 수집 ───
        anima_triggers = []
        sdxl_triggers = []
        anima_loras = []
        sdxl_loras = []

        for char_name in detected_chars:
            char_data = next((c for c in characters if c["name"] == char_name), None)
            if not char_data:
                continue
            for lora in char_data.get("loras", []):
                trigger = lora.get("trigger", char_name)
                base = lora.get("BASE", "anima")
                lora_path = lora.get("lora_path", "")
                strength = lora.get("strength", 0.8)

                if base == "anima":
                    if trigger not in anima_triggers:
                        anima_triggers.append(trigger)
                    anima_loras.append({
                        "lora_path": lora_path,
                        "str": strength,
                        "BASE": "anima"
                    })
                elif base == "sdxl":
                    if trigger not in sdxl_triggers:
                        sdxl_triggers.append(trigger)
                    sdxl_loras.append({
                        "lora_path": lora_path,
                        "str": strength,
                        "BASE": "sdxl"
                    })

        # ─── 아티스트 프리셋 태그 ───
        artist_presets = tags.get("artist_presets", {})
        anima_artist_preset_name = settings.get("anima_artist_preset", "")
        sdxl_artist_preset_name = settings.get("sdxl_artist_preset", "")

        anima_artist_tags = artist_presets.get(anima_artist_preset_name, [])
        sdxl_artist_tags = artist_presets.get(sdxl_artist_preset_name, [])

        # ─── 품질 태그 (프리셋 우선, 없으면 직접 태그) ───
        quality_presets = tags.get("quality_presets", {})
        anima_quality_preset_name = settings.get("anima_quality_preset", "")
        sdxl_quality_preset_name = settings.get("sdxl_quality_preset", "")

        if anima_quality_preset_name and anima_quality_preset_name in quality_presets:
            anima_quality_tags = quality_presets[anima_quality_preset_name]
        else:
            anima_quality_tags = tags.get("anima_quality", [])

        if sdxl_quality_preset_name and sdxl_quality_preset_name in quality_presets:
            quality_tags = quality_presets[sdxl_quality_preset_name]
        else:
            quality_tags = tags.get("quality", [])

        # ─── ANIMA 섹션 조립 ───
        anima_parts = []
        # 1. 캐릭터 트리거워드
        for t in anima_triggers:
            if t.strip():
                anima_parts.append(t.strip())
        # 2. ANIMA 아티스트 프리셋
        for t in anima_artist_tags:
            if t.strip():
                anima_parts.append(t.strip())
        # 3. ANIMA 품질
        for t in anima_quality_tags:
            if t.strip():
                anima_parts.append(t.strip())
        # 4. setup
        if setup.strip():
            anima_parts.append(setup.strip())
        # 5. char
        if char.strip():
            anima_parts.append(char.strip())
        # 6. supplement
        if supplement.strip():
            anima_parts.append(supplement.strip())

        # ─── SDXL 섹션 조립 ───
        sdxl_parts = []
        # 1. 캐릭터 트리거워드
        for t in sdxl_triggers:
            if t.strip():
                sdxl_parts.append(t.strip())
        # 2. SDXL 아티스트 프리셋
        for t in sdxl_artist_tags:
            if t.strip():
                sdxl_parts.append(t.strip())
        # 3. SDXL 품질
        for t in quality_tags:
            if t.strip():
                sdxl_parts.append(t.strip())
        # 4. setup
        if setup.strip():
            sdxl_parts.append(setup.strip())
        # 5. char (supplement 없음)
        if char.strip():
            sdxl_parts.append(char.strip())

        # ─── 긍정 프롬프트 조합 ───
        positive = ", ".join(anima_parts)
        positive += "\n[SDXL]"
        positive += "\n" + ", ".join(sdxl_parts)

        # ─── 메타데이터 섹션 ───
        # CHAR_LIST
        positive += f"\n[CHAR_LIST]"
        positive += "\n" + ",".join(detected_chars)

        # CACHE_PATH
        cache_data = self.build_cache_path(detected_chars, bot_name)
        positive += "\n[CACHE_PATH]"
        positive += "\n" + json.dumps(cache_data, ensure_ascii=False)

        # FACE_ID
        face_id_activate = settings.get("face_id_activate", False)
        face_id_str = settings.get("face_id_str", 0.55)
        positive += f"\n[FACE_ID_ACTIVATE]"
        positive += f"\n{'true' if face_id_activate else 'false'}"
        positive += f"\n[FACE_ID_STR]"
        positive += f"\n{face_id_str}"

        face_id_data = self.build_face_id_dir(detected_chars, bot_name, settings)
        positive += "\n[FACE_ID_DIR]"
        positive += "\n" + json.dumps(face_id_data, ensure_ascii=False)

        # LORA
        has_lora = len(anima_loras) > 0 or len(sdxl_loras) > 0
        positive += f"\n[LORA_ACTIVATE]"
        positive += f"\n{'true' if has_lora else 'false'}"

        all_loras = anima_loras + sdxl_loras
        lora_data = {"list": all_loras}
        positive += "\n[LORA_DATA]"
        positive += "\n" + json.dumps(lora_data, ensure_ascii=False)

        # HRF
        hrf_activate = settings.get("hrf_activate", False)
        hrf_size = settings.get("hrf_size", 1.5)
        hrf_restore_size = settings.get("hrf_restore_size", False)
        positive += f"\n[HRF_ACTIVATE]"
        positive += f"\n{'true' if hrf_activate else 'false'}"
        positive += f"\n[HRF_SIZE]"
        positive += f"\n{hrf_size}"
        positive += f"\n[HRF_RESTORE_SIZE]"
        positive += f"\n{'true' if hrf_restore_size else 'false'}"

        # IMG size
        img_w = settings.get("img_w", 756)
        img_h = settings.get("img_h", 756)
        positive += f"\n[IMG_W]"
        positive += f"\n{img_w}"
        positive += f"\n[IMG_H]"
        positive += f"\n{img_h}"

        # Detailers
        fd_activate = settings.get("fd_activate", False)
        hd_activate = settings.get("hd_activate", False)
        ed_activate = settings.get("ed_activate", False)
        positive += f"\n[FD_ACTIVATE]"
        positive += f"\n{'true' if fd_activate else 'false'}"
        positive += f"\n[HD_ACTIVATE]"
        positive += f"\n{'true' if hd_activate else 'false'}"
        positive += f"\n[ED_ACTIVATE]"
        positive += f"\n{'true' if ed_activate else 'false'}"

        positive += "\n[END]"

        return positive

    @staticmethod
    def build_negative_prompt(tags: dict, settings: dict = None) -> str:
        """최종 부정 프롬프트 빌드.

        ANIMA 부정: character_negative_preset + anima_negative_preset + anima_negative + anima_character_negative
        SDXL 부정: character_negative_preset + negative_preset + negative + character_negative
        """
        if settings is None:
            settings = {}

        # ─── 부정 프리셋 (선택된 프리셋 우선, 없으면 직접 태그) ───
        char_neg_presets = tags.get("character_negative_presets", {})
        negative_presets = tags.get("negative_presets", {})

        # ANIMA 부정: 캐릭터 부정 프리셋 + ANIMA 부정 프리셋 + 직접 태그
        anima_char_neg_preset_name = settings.get("anima_character_negative_preset", "")
        anima_neg_preset_name = settings.get("anima_negative_preset", "")

        anima_cn_tags = char_neg_presets.get(anima_char_neg_preset_name, tags.get("character_negative", []))
        anima_n_tags = negative_presets.get(anima_neg_preset_name, tags.get("anima_negative", []))
        anima_char_neg_direct = tags.get("anima_character_negative", [])

        anima_neg_parts = []
        for t in anima_cn_tags if isinstance(anima_cn_tags, list) else []:
            if t.strip():
                anima_neg_parts.append(t.strip())
        for t in anima_n_tags if isinstance(anima_n_tags, list) else []:
            if t.strip():
                anima_neg_parts.append(t.strip())
        for t in anima_char_neg_direct:
            if t.strip():
                anima_neg_parts.append(t.strip())

        # SDXL 부정: 캐릭터 부정 프리셋 + 부정 프리셋 + 직접 태그
        sdxl_char_neg_preset_name = settings.get("sdxl_character_negative_preset", "")
        sdxl_neg_preset_name = settings.get("sdxl_negative_preset", "")

        sdxl_cn_tags = char_neg_presets.get(sdxl_char_neg_preset_name, tags.get("character_negative", []))
        sdxl_n_tags = negative_presets.get(sdxl_neg_preset_name, tags.get("negative", []))

        sdxl_neg_parts = []
        for t in sdxl_cn_tags if isinstance(sdxl_cn_tags, list) else []:
            if t.strip():
                sdxl_neg_parts.append(t.strip())
        for t in sdxl_n_tags if isinstance(sdxl_n_tags, list) else []:
            if t.strip():
                sdxl_neg_parts.append(t.strip())

        negative = ", ".join(anima_neg_parts)
        negative += "\n[SDXL]"
        negative += "\n" + ", ".join(sdxl_neg_parts)

        return negative

    @staticmethod
    def build_cache_path(detected_chars: list, bot_name: str) -> dict:
        """감지된 캐릭터의 캐시 경로 JSON 빌드.

        soya_bot/{bot_name}/{char_name}/cache.pt 형식
        """
        items = []
        for char_name in detected_chars:
            items.append({
                "emb_path": f"soya_bot/{bot_name}/{char_name}/cache.pt",
                "CHAR": char_name
            })
        return {"list": items}

    @staticmethod
    def build_face_id_dir(detected_chars: list, bot_name: str, settings: dict) -> dict:
        """감지된 캐릭터의 Face ID 디렉토리 JSON 빌드.

        soya_bot/{bot_name}/{char_name}/cache.ipadpt 형식
        extra_tag는 빈 문자열로 시작 (향후 확장)
        """
        face_id_str = settings.get("face_id_str", 0.55)
        items = []
        for char_name in detected_chars:
            items.append({
                "ipa_path": f"soya_bot/{bot_name}/{char_name}/cache.ipadpt",
                "str": face_id_str,
                "CHAR": char_name,
                "extra_tag": ""
            })
        return {"list": items}