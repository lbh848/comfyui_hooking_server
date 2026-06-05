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

# 태그명(소문자, 공백) → (category, subcategory)
_tag_categories: dict[str, tuple[str, str]] = {}
_loaded = False

# 그룹 분류 규칙
# key: (category, subcategory) 튜플. "*" 와일드카드로 서브카테고리 전체 매칭.
GROUP_RULES = {
    # 외모/신체
    ("인물", "눈"): "외모/신체",
    ("인물", "종족"): "외모/신체",
    ("신체", "*"): "외모/신체",
    ("패션", "헤어스타일"): "외모/신체",
    ("패션", "헤어컬러"): "외모/신체",
    # 복장
    ("패션", "상의"): "복장",
    ("패션", "하의"): "복장",
    ("패션", "의상"): "복장",
    ("패션", "액세서리"): "복장",
    ("패션", "디테일"): "복장",
    ("패션", "양말"): "복장",
}


def _load():
    global _tag_categories, _loaded
    _tag_categories = {}

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
                    desc = row[3].strip() if len(row) >= 4 else ''
                    m = re.match(r'\[([^>]+)\s*>\s*([^\]]+)\]', desc)
                    if m:
                        cat = m.group(1).strip()
                        subcat = m.group(2).strip()
                        _tag_categories[name] = (cat, subcat)
            print(f"[TAG_CLASSIFIER] KR CSV 로드 완료: {len(_tag_categories):,}개 태그 카테고리")
        except Exception as e:
            print(f"[TAG_CLASSIFIER] 로드 실패: {f}: {e}")

    _loaded = True


def _classify_tag(tag_name: str) -> tuple[str, str]:
    """단일 태그를 분류. 반환: (group, subcategory)"""
    if not _loaded:
        _load()

    key = tag_name.strip().lower().replace('_', ' ')
    info = _tag_categories.get(key)

    if not info:
        return ("미분류", "")

    cat, subcat = info

    # 정확 매칭
    group = GROUP_RULES.get((cat, subcat))
    if group:
        return (group, subcat)

    # 와일드카드 매칭 (category, "*")
    group = GROUP_RULES.get((cat, "*"))
    if group:
        return (group, subcat)

    return ("미분류", subcat)


def classify_prompt(prompt: str) -> dict:
    """
    프롬프트 문자열을 받아 태그를 3그룹으로 분류.

    반환: {
        "외모/신체": [{"tag": "...", "sub": "눈"}, ...],
        "복장": [{"tag": "...", "sub": "상의"}, ...],
        "미분류": [{"tag": "...", "sub": "..."}, ...],
    }
    """
    if not _loaded:
        _load()

    groups: dict[str, list[dict]] = {
        "외모/신체": [],
        "복장": [],
        "미분류": [],
    }

    # 쉼표로 분리, 괄호 weight 보존
    raw_tags = [t.strip() for t in prompt.split(',') if t.strip()]

    for raw in raw_tags:
        # (weight:tag) 형태에서 태그명만 추출
        inner = raw
        m = re.match(r'[\(\[{]+[\d.]*:?\s*(.+?)[\)\]}]+$', raw)
        if m:
            inner = m.group(1).strip()

        group, sub = _classify_tag(inner)
        groups[group].append({"tag": raw, "sub": sub})

    # 빈 그룹 제거
    return {k: v for k, v in groups.items() if v}
