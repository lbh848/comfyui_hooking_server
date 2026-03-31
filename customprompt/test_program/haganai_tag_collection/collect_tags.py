"""
하가나이 캐릭터별 태그 수집 스크립트
- NSFW 태그가 파일명에 포함된 파일 제외
- 중복 제거 (대소문자 무시, 순서 유지)
- 복장 관련 태그만 필터링
- 캐릭터별 정리된 텍스트로 저장
"""

import os
import glob

DATASET_DIR = r"E:\wsl2\matrix\Packages\kohya_ss_anima\task_folder\dataset_backup\haganai"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 파일명에서 제외할 NSFW 태그
EXCLUDE_TAGS = [
    "breast caress", "cowgirl back cumshot", "cowgirl back", "cowgirl cumshot", "cowgirl",
    "doggystyle cumshot", "doggystyle", "fellatio", "fingering", "footjob cumshot", "footjob",
    "fullnelson cumshot", "fullnelson", "handjob", "masturbation", "mating press cumshot",
    "mating press", "missionary position cumshot", "missionary position",
    "paizuri", "reverse pright straddle cumshot", "reverse pright straddle",
    "reverse standing position", "showing armpit", "showing nude",
    "spooning cumshot", "spooning", "standing position",
    "suspended congress cumshot", "suspended congress",
    "upright straddle cumshot", "upright straddle",
]


def is_excluded(filename: str) -> bool:
    """파일명에 제외 태그가 포함되어 있는지 확인"""
    name_lower = filename.lower()
    for tag in EXCLUDE_TAGS:
        if tag in name_lower:
            return True
    return False


def collect_character_tags(char_dir: str) -> tuple:
    """캐릭터 디렉토리의 .txt 파일에서 태그 수집 (NSFW 제외, 중복 제거)
    Returns: (unique_tags, file_count, excluded_count)
    """
    all_tags = []
    txt_files = sorted(glob.glob(os.path.join(char_dir, "*.txt")))
    excluded_count = 0
    file_count = 0

    for txt_file in txt_files:
        fname = os.path.basename(txt_file)
        if is_excluded(fname):
            excluded_count += 1
            continue

        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                tags = [t.strip() for t in content.split(",") if t.strip()]
                all_tags.extend(tags)
                file_count += 1
        except Exception as e:
            print(f"  [WARN] 읽기 실패: {fname}: {e}")

    # 중복 제거 (대소문자 무시, 순서 유지)
    seen = set()
    unique_tags = []
    for tag in all_tags:
        key = tag.lower()
        if key not in seen:
            seen.add(key)
            unique_tags.append(tag)

    return unique_tags, file_count, excluded_count


def main():
    char_dirs = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

    print(f"=== 캐릭터 수: {len(char_dirs)} ===\n")

    for char_name in char_dirs:
        char_path = os.path.join(DATASET_DIR, char_name)
        tags, file_count, excluded = collect_character_tags(char_path)

        safe_name = char_name.replace(" ", "_")
        output_file = os.path.join(OUTPUT_DIR, f"tags_cleaned_{safe_name}.txt")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(", ".join(tags))

        print(f"[{char_name}] 읽음:{file_count} 제외:{excluded} 고유태그:{len(tags)} → {os.path.basename(output_file)}")

    print("\n완료!")


if __name__ == "__main__":
    main()
