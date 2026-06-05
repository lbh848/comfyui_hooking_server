"""
tag_classifier - KR CSV 카테고리 기반 태그 자동 분류

auto_complete/ 폴더의 KR CSV에서 [카테고리 > 서브카테고리]를 파싱하여
태그를 3그룹(외모/신체, 복장, 미분류)으로 분류한다.
"""

import csv
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTOCOMPLETE_DIR = os.path.join(BASE_DIR, "auto_complete")

# 태그명(소문자, 공백) → (category, subcategory, description)
_tag_info: dict[str, tuple[str, str, str]] = {}
_loaded = False

# 그룹 분류 규칙
GROUP_RULES = {
    ("인물", "눈"): "외모/신체",
    ("인물", "종족"): "외모/신체",
    ("신체", "*"): "외모/신체",
    ("패션", "헤어스타일"): "외모/신체",
    ("패션", "헤어컬러"): "외모/신체",
    ("패션", "상의"): "복장",
    ("패션", "하의"): "복장",
    ("패션", "의상"): "복장",
    ("패션", "액세서리"): "복장",
    ("패션", "디테일"): "복장",
    ("패션", "양말"): "복장",
}


def _extract_description(desc_field: str) -> str:
    """KR CSV 설명 필드에서 카테고리 태그와 키워드를 제거한 순수 설명 추출."""
    # [카테고리 > 서브] 제거
    text = re.sub(r'\[([^\]]+)\]', '', desc_field).strip()
    # . 키워드: ... 제거
    text = re.sub(r'\.\s*키워드:.*$', '', text).strip()
    return text


def _load():
    global _tag_info, _loaded
    _tag_info = {}

    if not os.path.isdir(AUTOCOMPLETE_DIR):
        _loaded = True
        return

    for f in os.listdir(AUTOCOMPLETE_DIR):
        if not f.lower().endswith(".csv"):
            continue
        if not f.lower().startswith("kr"):
            continue

        fpath = os.path.join(AUTOCOMPLETE_DIR, f)
        try:
            with open(fpath, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if len(row) < 4:
                        continue
                    name = row[0].strip().lower().replace('_', ' ').replace('﻿', '')
                    desc_field = row[3].strip()
                    m = re.match(r'\[([^>]+)\s*>\s*([^\]]+)\]', desc_field)
                    if m:
                        cat = m.group(1).strip()
                        subcat = m.group(2).strip()
                        description = _extract_description(desc_field)
                        _tag_info[name] = (cat, subcat, description)
            print(f"[TAG_CLASSIFIER] KR CSV 로드 완료: {len(_tag_info):,}개 태그")
        except Exception as e:
            print(f"[TAG_CLASSIFIER] 로드 실패: {f}: {e}")

    _loaded = True


def _classify_tag(tag_name: str) -> tuple[str, str, str]:
    """단일 태그를 분류. 반환: (group, subcategory, description)"""
    if not _loaded:
        _load()

    key = tag_name.strip().lower().replace('_', ' ')
    info = _tag_info.get(key)

    if not info:
        return ("미분류", "", "")

    cat, subcat, description = info

    group = GROUP_RULES.get((cat, subcat))
    if group:
        return (group, subcat, description)

    group = GROUP_RULES.get((cat, "*"))
    if group:
        return (group, subcat, description)

    return ("미분류", subcat, description)


def classify_prompt(prompt: str) -> dict:
    """
    프롬프트 문자열을 받아 태그를 3그룹으로 분류.

    반환: {
        "외모/신체": [{"tag": "...", "sub": "눈", "desc": "..."}, ...],
        "복장": [{"tag": "...", "sub": "상의", "desc": "..."}, ...],
        "미분류": [{"tag": "...", "sub": "...", "desc": "..."}, ...],
    }
    """
    if not _loaded:
        _load()

    groups: dict[str, list[dict]] = {
        "외모/신체": [],
        "복장": [],
        "미분류": [],
    }

    raw_tags = [t.strip() for t in prompt.split(',') if t.strip()]

    for raw in raw_tags:
        inner = raw
        m = re.match(r'[\(\[{]+[\d.]*:?\s*(.+?)[\)\]}]+$', raw)
        if m:
            inner = m.group(1).strip()

        group, sub, desc = _classify_tag(inner)
        groups[group].append({"tag": raw, "sub": sub, "desc": desc})

    return {k: v for k, v in groups.items() if v}
