"""
autocomplete_service - CSV 기반 태그 자동완성 서비스

auto_complete/ 폴더의 CSV 파일을 로드하여 태그 검색을 제공.
우선순위: KR > 파일 크기 큰 순 (for_anima 제외)
"""

import csv
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTOCOMPLETE_DIR = os.path.join(BASE_DIR, "auto_complete")

# 로드된 태그: [{name, aliases, source_index}, ...]
_tags: list[dict] = []
_loaded = False


def _csv_priority(filename: str) -> int:
    """KR 파일이 가장 높은 우선순위(0), 나머지는 파일 크기 역순"""
    if filename.lower().startswith("kr"):
        return 0
    return 1


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
                    aliases = []
                    description = ''
                    if len(row) >= 4 and row[3]:
                        if _csv_priority(fname) == 0:
                            # KR 파일: row[3]은 한국어 설명
                            description = row[3].strip()
                            # 설명에서 "키워드: ..." 부분을 별명으로 추출
                            kw_match = row[3]
                            if '키워드:' in kw_match:
                                kw_part = kw_match.split('키워드:')[-1]
                                aliases = [a.strip().lower().replace('_', ' ')
                                           for a in kw_part.split(',')
                                           if a.strip() and '<' not in a]
                        else:
                            # 다른 파일: row[3]은 영어 별명
                            aliases = [a.strip().lower().replace('_', ' ') for a in row[3].split(',') if a.strip()]
                    _tags.append({
                        'name': name,
                        'aliases': aliases,
                        'description': description,
                        'source': source_idx,
                    })
            loaded = len(_tags) - count_before
            print(f"[AUTOCOMPLETE] 로드: {fname} ({loaded:,}개 태그)")
        except Exception as e:
            print(f"[AUTOCOMPLETE] 로드 실패: {fname}: {e}")
        source_idx += 1

    _loaded = True
    print(f"[AUTOCOMPLETE] 총 {len(_tags):,}개 태그 로드 완료")


def search_tags(query: str, limit: int = 20) -> list[dict]:
    """태그명 + 별명에서 substring 매치. 결과는 source 순서(KR 우선)."""
    if not _loaded:
        load_all_csv()

    q = query.strip().lower().replace('_', ' ')
    if not q:
        return []

    results = []
    for tag in _tags:
        if len(results) >= limit:
            break
        # 태그명 매치
        if q in tag['name']:
            results.append({'name': tag['name'], 'description': tag.get('description', '')})
            continue
        # 별명 매치
        for alias in tag['aliases']:
            if q in alias:
                results.append({'name': tag['name'], 'description': tag.get('description', '')})
                break

    return results
