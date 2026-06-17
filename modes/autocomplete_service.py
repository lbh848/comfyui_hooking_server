"""
autocomplete_service - CSV 기반 태그 자동완성 서비스

[검색 동작]
영문 쿼리 → 태그명 prefix/substring 매치 (기존 동작 + prefix 우선 정렬)
한글 쿼리 → description 구문(phrase) 기반 역방향 검색
<...> 쿼리 → angle_kws(<> 안 키워드)에서만 exact match

[description 구조]
[대분류 > 소분류] 상세설명. 키워드: <카테고리키워드>, 일반키워드1, 일반키워드2

[점수 체계 - 한글 검색]
4점: 쿼리 == plain_kw 완전 일치
3점: 쿼리가 plain_kw에 포함 (예: "파란" → "파란 눈")
1점: 쿼리(4글자 이상)가 body에 포함 (예: "파란색 눈동자")
1점: 쿼리가 category에 포함 (예: "인원수")

[v2 변경 핵심]
- description을 category/body/angle_kws/plain_kws 4개 필드로 파싱하여 검색 정확도 향상
- <...> 쿼리는 angle_kws에서만 exact match → 단순 description 텍스트와 분리
- 역색인(토큰 단위) 방식 제거 → 구문(phrase) 매칭으로 노이즈 대폭 감소
- "kw in q" 조건 제거 → 짧은 키워드가 긴 쿼리에 걸리는 오매칭 방지
"""

import csv
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTOCOMPLETE_DIR = os.path.join(BASE_DIR, "auto_complete")

# 로드된 태그 목록
_tags: list[dict] = []
_loaded = False


def _csv_priority(filename: str) -> int:
    """KR 파일이 가장 높은 우선순위(0), 나머지는 파일 크기 역순"""
    if filename.lower().startswith("kr"):
        return 0
    return 1


def _parse_description(desc: str) -> dict:
    """description 필드를 구조화된 딕셔너리로 파싱.

    입력: "[인물 > 눈] 파란색 눈동자. 키워드: <눈색>, 파란 눈, 벽안"
    출력:
        category  = "인물 > 눈"
        body      = "파란색 눈동자"
        angle_kws = ["눈색"]          ← <> 안 키워드 (카테고리 분류자)
        plain_kws = ["파란 눈", "벽안"] ← 일반 검색 키워드
    """
    category = ''
    body = ''
    angle_kws = []
    plain_kws = []

    if not desc:
        return {'category': category, 'body': body,
                'angle_kws': angle_kws, 'plain_kws': plain_kws}

    # [분류 > 소분류] 추출
    m = re.match(r'\[([^\]]+)\]', desc)
    if m:
        category = m.group(1)

    if '키워드:' in desc:
        before_kw, kw_part = desc.split('키워드:', 1)
        # 분류 태그 제거 후 body 추출
        body = re.sub(r'\[.*?\]', '', before_kw).strip().rstrip('. ')
        # <카테고리키워드> 추출
        angle_kws = re.findall(r'<([^>]+)>', kw_part)
        # 나머지 일반 키워드
        plain_raw = re.sub(r'<[^>]+>', '', kw_part)
        plain_kws = [k.strip() for k in plain_raw.split(',') if k.strip()]
    else:
        body = re.sub(r'\[.*?\]', '', desc).strip()

    return {'category': category, 'body': body,
            'angle_kws': angle_kws, 'plain_kws': plain_kws}


def load_all_csv():
    """auto_complete/ 폴더의 모든 CSV를 로드하여 메모리에 캐시."""
    global _tags, _loaded
    _tags = []

    if not os.path.isdir(AUTOCOMPLETE_DIR):
        _loaded = True
        return

    # for_anima 제외, CSV만, KR 우선 + 크기 큰 순 정렬
    files = []
    for f in os.listdir(AUTOCOMPLETE_DIR):
        if not f.lower().endswith(".csv"):
            continue
        if "for_anima" in f.lower():
            continue
        fpath = os.path.join(AUTOCOMPLETE_DIR, f)
        files.append((f, fpath, os.path.getsize(fpath)))

    files.sort(key=lambda x: (_csv_priority(x[0]), -x[2]))

    seen_names = set()
    source_idx = 0
    for fname, fpath, fsize in files:
        count_before = len(_tags)
        try:
            with open(fpath, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if len(row) < 3:
                        continue
                    name = row[0].strip().lower().replace('_', ' ').replace('\ufeff', '')
                    if not name or name in seen_names:
                        continue
                    seen_names.add(name)

                    desc_raw = row[3].strip() if len(row) >= 4 else ''

                    # KR 파일: description 구조 파싱
                    if _csv_priority(fname) == 0:
                        parsed = _parse_description(desc_raw)
                        _tags.append({
                            'name': name,
                            'description': desc_raw,
                            'category': parsed['category'],
                            'body': parsed['body'],
                            'angle_kws': parsed['angle_kws'],
                            'plain_kws': parsed['plain_kws'],
                            'source': source_idx,
                        })
                    else:
                        # 비KR 파일: row[3]은 영문 별명
                        aliases = [a.strip().lower().replace('_', ' ')
                                   for a in desc_raw.split(',') if a.strip()]
                        _tags.append({
                            'name': name,
                            'description': '',
                            'category': '',
                            'body': '',
                            'angle_kws': [],
                            'plain_kws': aliases,
                            'source': source_idx,
                        })

            loaded = len(_tags) - count_before
            print(f"[AUTOCOMPLETE] 로드: {fname} ({loaded:,}개 태그)")
        except Exception as e:
            print(f"[AUTOCOMPLETE] 로드 실패: {fname}: {e}")
        source_idx += 1

    _loaded = True
    print(f"[AUTOCOMPLETE] 총 {len(_tags):,}개 태그 로드 완료")


def _is_korean_query(query: str) -> bool:
    """쿼리에 한글이 포함되어 있으면 True."""
    return bool(re.search(r'[가-힣]', query))


def _score_tag(tag: dict, q: str, is_angle_query: bool) -> int:
    """태그와 쿼리의 매칭 점수 반환. 0이면 미매칭.

    <...> 쿼리:
        angle_kws에서 exact match → 10점

    일반 한글 쿼리 (phrase 매칭):
        plain_kws 매칭 (최고 점수 1개만, 누적 없음):
            4점: q == plain_kw  (완전 일치)
            3점: q in plain_kw  (쿼리가 키워드 안에 포함, 예: "파란" → "파란 눈")
        body 매칭 (plain_kws 미매칭 시에만, 보조 수단):
            3점: body 공백제거 8자 이하  (핵심 설명, 예: "파란색 눈동자")
            2점: body 공백제거 15자 이하
            1점: body 공백제거 16자 이상
            단, 공백 제거 후 쿼리 길이 2자 미만이면 body 검색 제외
        category 매칭:
            1점: q in category

    [설계 의도]
    · plain_kws 누적 제거: 여러 키워드 중 가장 잘 맞는 1개만 점수 반영
      → 키워드가 많은 캐릭터 태그가 부당하게 높은 점수를 얻는 문제 방지
    · body는 plain_kws 미매칭 시에만: body와 plain_kws 중복 시 이중 점수 방지
      예) blue bodysuit: body="파란색 바디수트", plain_kws=["파란색 바디수트"]
          "파란색" 검색 시 plain_kws에서 이미 3점 → body 점수 추가 X
    · body 길이 기반 가산: 짧은 body = 핵심 설명 → 더 높은 관련도
    · "kw in q" 제외: 짧은 키워드("원")가 긴 쿼리("인원수")에 걸리는 오매칭 방지
    """
    if is_angle_query:
        return 10 if q in tag['angle_kws'] else 0

    score = 0

    # plain_kws: 최고 점수 1개만 (누적 없음)
    kw_score = 0
    for kw in tag['plain_kws']:
        if q == kw:
            kw_score = max(kw_score, 4)
        elif q in kw:
            kw_score = max(kw_score, 3)
    score += kw_score

    # body 구문 매칭: plain_kws에서 점수를 얻지 못한 경우에만 보조 수단으로 사용
    # 공백 제거 후 2자 미만 쿼리("눈" 등 단독 짧은 단어)는 노이즈 방지를 위해 제외
    if kw_score == 0 and len(q.replace(' ', '')) >= 2 and q in tag['body']:
        body_len = len(tag['body'].replace(' ', ''))
        if body_len <= 8:
            score += 3   # 핵심 설명 (예: body="파란색 눈동자")
        elif body_len <= 15:
            score += 2
        else:
            score += 1

    # category 매칭
    if q in tag['category']:
        score += 1

    return score


def _keyword_search(query: str, limit: int) -> list[dict]:
    """한글 쿼리에 대한 phrase 매칭 역방향 검색.

    <...> 쿼리이면 angle_kws exact match만 수행.
    일반 쿼리이면 plain_kws/body/category phrase 매칭 수행.

    반환: [{'name': ..., 'description': ...}, ...]
    """
    # <...> 패턴 감지
    angle_m = re.match(r'^<([^>]+)>$', query.strip())
    if angle_m:
        q = angle_m.group(1).strip()
        is_angle = True
    else:
        # 프론트엔드가 공백을 '_'로 변환해 전송하므로 다시 공백으로 복원
        q = query.strip().replace('_', ' ')
        is_angle = False

    if not q:
        return []

    results = []
    for tag in _tags:
        s = _score_tag(tag, q, is_angle)
        if s > 0:
            results.append((s, tag['source'], tag['name'], tag['description']))

    # 점수 내림차순 → source 오름차순(KR 우선)
    results.sort(key=lambda x: (-x[0], x[1]))
    return [{'name': name, 'description': desc}
            for _, _, name, desc in results[:limit]]


def search_tags(query: str, limit: int = 20) -> list[dict]:
    """태그 검색 통합 진입점.

    - 한글 포함 or <...> 형태 → 역방향 키워드 검색
    - 영문/숫자만              → 태그명 prefix 우선 + substring 검색

    반환: [{'name': ..., 'description': ...}, ...]
    """
    if not _loaded:
        load_all_csv()

    q = query.strip()
    if not q or len(q) < 1:
        return []

    # 한글 포함 or <...> → 키워드 역방향 검색
    if _is_korean_query(q) or re.match(r'^<[^>]+>$', q):
        return _keyword_search(q, limit)

    # 영문 → 태그명 prefix/substring 매치
    q_lower = q.lower().replace('_', ' ')

    prefix_match = []
    substr_match = []
    alias_match  = []

    for tag in _tags:
        if tag['name'].startswith(q_lower):
            prefix_match.append({'name': tag['name'], 'description': tag.get('description', '')})
        elif q_lower in tag['name']:
            substr_match.append({'name': tag['name'], 'description': tag.get('description', '')})
        else:
            for alias in tag['plain_kws']:
                if q_lower in alias:
                    alias_match.append({'name': tag['name'], 'description': tag.get('description', '')})
                    break

    combined = prefix_match + substr_match + alias_match
    return combined[:limit]
