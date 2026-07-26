"""캐릭터 메이커용 Danbooru Tag RAG 데이터 변환기.

사용자가 내려받은 한국어 설명 CSV를 서버에 내장된 ``auto_complete/danbooru.csv``와
결합해 danbooru-tag-rag가 읽을 수 있는 UTF-8 CSV로 변환한다. 입력 파일과 내장
기준표는 읽기 전용이며, 호출자가 지정한 임시 출력 파일에만 쓴다.
"""

from __future__ import annotations

import csv
import os
import traceback
from typing import Any


csv.field_size_limit(10 * 1024 * 1024)

MIN_POST_COUNT = 50
MAX_UPLOAD_BYTES = 128 * 1024 * 1024
OUTPUT_FILENAME = "danbooru-tags.csv"


class CharacterMakerRagDataError(ValueError):
    """사용자에게 전달할 수 있는 RAG 데이터 변환 오류."""


def _normalize_source_tag(value: str) -> str:
    """자동완성 표시용 태그를 Danbooru의 정식 키 형태로 복원한다."""
    return (
        str(value or "")
        .lstrip("\ufeff")
        .strip()
        .replace(r"\(", "(")
        .replace(r"\)", ")")
        .replace(" ", "_")
    )


def _load_canonical_tags(canonical_csv_path: str) -> tuple[dict[str, tuple[str, int]], int]:
    if not os.path.isfile(canonical_csv_path):
        print(
            "[CHARACTER_MAKER_RAG_DATA] 내장 기준표 없음: "
            f"path={canonical_csv_path!r}"
        )
        raise CharacterMakerRagDataError(
            "내장 Danbooru 기준표(auto_complete/danbooru.csv)를 찾을 수 없습니다."
        )

    canonical: dict[str, tuple[str, int]] = {}
    malformed = 0
    duplicate = 0
    try:
        with open(canonical_csv_path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            for line_number, row in enumerate(reader, start=1):
                if len(row) < 3:
                    malformed += 1
                    print(
                        "[CHARACTER_MAKER_RAG_DATA] 내장 기준표 행 제외: "
                        f"line={line_number}, reason=열 부족, columns={len(row)}, "
                        f"row={row[:4]!r}"
                    )
                    continue
                tag = str(row[0] or "").strip()
                try:
                    category = int(row[1])
                    int(row[2])
                except (TypeError, ValueError):
                    malformed += 1
                    print(
                        "[CHARACTER_MAKER_RAG_DATA] 내장 기준표 행 제외: "
                        f"line={line_number}, reason=숫자 형식, row={row[:4]!r}"
                    )
                    continue
                if not tag:
                    malformed += 1
                    print(
                        "[CHARACTER_MAKER_RAG_DATA] 내장 기준표 행 제외: "
                        f"line={line_number}, reason=태그명 비어 있음"
                    )
                    continue
                if tag in canonical:
                    duplicate += 1
                    continue
                canonical[tag] = (tag, category)
    except UnicodeError as exc:
        print(
            "[CHARACTER_MAKER_RAG_DATA] 내장 기준표 UTF-8 해석 실패: "
            f"path={canonical_csv_path!r}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise CharacterMakerRagDataError(
            "내장 Danbooru 기준표를 UTF-8로 읽을 수 없습니다."
        ) from exc
    except OSError as exc:
        print(
            "[CHARACTER_MAKER_RAG_DATA] 내장 기준표 읽기 실패: "
            f"path={canonical_csv_path!r}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise CharacterMakerRagDataError(
            f"내장 Danbooru 기준표를 읽지 못했습니다: {exc}"
        ) from exc

    if not canonical:
        print(
            "[CHARACTER_MAKER_RAG_DATA] 내장 기준표 유효 행 없음: "
            f"path={canonical_csv_path!r}, malformed={malformed}"
        )
        raise CharacterMakerRagDataError("내장 Danbooru 기준표에 유효한 태그가 없습니다.")

    print(
        "[CHARACTER_MAKER_RAG_DATA] 내장 기준표 로드 완료: "
        f"rows={len(canonical)}, malformed={malformed}, duplicate={duplicate}"
    )
    return canonical, len(canonical)


def convert_kr_danbooru_csv(
    source_csv_path: str,
    canonical_csv_path: str,
    output_csv_path: str,
    *,
    min_post_count: int = MIN_POST_COUNT,
) -> dict[str, Any]:
    """한국어 설명 CSV를 정식 RAG 입력 CSV로 변환한다.

    정식 태그명과 카테고리는 내장 기준표에서만 가져온다. 기준표와 매칭되지 않는
    행을 category 0으로 추측해 넣지 않으므로 일반 태그 검색 공간을 오염시키지 않는다.
    """
    if not os.path.isfile(source_csv_path):
        print(
            "[CHARACTER_MAKER_RAG_DATA] 입력 CSV 없음: "
            f"path={source_csv_path!r}"
        )
        raise CharacterMakerRagDataError("변환할 한국어 태그 CSV를 찾을 수 없습니다.")
    if min_post_count < 0:
        print(
            "[CHARACTER_MAKER_RAG_DATA] 최소 사용 횟수 오류: "
            f"min_post_count={min_post_count}"
        )
        raise CharacterMakerRagDataError("최소 사용 횟수는 0 이상이어야 합니다.")

    canonical, canonical_rows = _load_canonical_tags(canonical_csv_path)
    summary: dict[str, Any] = {
        "input_rows": 0,
        "written_rows": 0,
        "below_frequency": 0,
        "unmatched": 0,
        "malformed": 0,
        "duplicates": 0,
        "header_rows": 0,
        "canonical_rows": canonical_rows,
        "min_post_count": int(min_post_count),
    }
    unmatched_samples: list[str] = []
    seen: set[str] = set()

    try:
        with (
            open(source_csv_path, "r", encoding="utf-8-sig", newline="") as source,
            open(output_csv_path, "w", encoding="utf-8", newline="") as output,
        ):
            reader = csv.reader(source)
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(["name", "category", "post_count", "description"])

            for line_number, row in enumerate(reader, start=1):
                if (
                    line_number == 1
                    and len(row) >= 2
                    and str(row[0]).lstrip("\ufeff").strip().casefold()
                    in {"name", "tag"}
                    and str(row[1]).strip().casefold() == "category"
                ):
                    summary["header_rows"] += 1
                    continue

                summary["input_rows"] += 1
                if len(row) != 4:
                    summary["malformed"] += 1
                    print(
                        "[CHARACTER_MAKER_RAG_DATA] 입력 행 제외: "
                        f"line={line_number}, reason=4열 아님, columns={len(row)}, "
                        f"row={row[:6]!r}"
                    )
                    continue

                source_tag = _normalize_source_tag(row[0])
                try:
                    post_count = int(str(row[2]).strip())
                except (TypeError, ValueError):
                    summary["malformed"] += 1
                    print(
                        "[CHARACTER_MAKER_RAG_DATA] 입력 행 제외: "
                        f"line={line_number}, reason=사용 횟수 형식, "
                        f"tag={source_tag!r}, value={row[2]!r}"
                    )
                    continue
                if not source_tag:
                    summary["malformed"] += 1
                    print(
                        "[CHARACTER_MAKER_RAG_DATA] 입력 행 제외: "
                        f"line={line_number}, reason=태그명 비어 있음"
                    )
                    continue
                if post_count < min_post_count:
                    summary["below_frequency"] += 1
                    continue

                canonical_item = canonical.get(source_tag)
                if canonical_item is None:
                    summary["unmatched"] += 1
                    if len(unmatched_samples) < 20:
                        unmatched_samples.append(source_tag)
                    continue
                canonical_tag, category = canonical_item
                if canonical_tag in seen:
                    summary["duplicates"] += 1
                    continue
                seen.add(canonical_tag)
                writer.writerow(
                    [canonical_tag, category, post_count, str(row[3] or "").strip()]
                )
                summary["written_rows"] += 1
    except UnicodeError as exc:
        print(
            "[CHARACTER_MAKER_RAG_DATA] 입력 CSV UTF-8 해석 실패: "
            f"path={source_csv_path!r}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise CharacterMakerRagDataError(
            "한국어 태그 CSV를 UTF-8로 읽을 수 없습니다."
        ) from exc
    except csv.Error as exc:
        print(
            "[CHARACTER_MAKER_RAG_DATA] 입력 CSV 구문 오류: "
            f"path={source_csv_path!r}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise CharacterMakerRagDataError(f"CSV 구문을 해석하지 못했습니다: {exc}") from exc
    except OSError as exc:
        print(
            "[CHARACTER_MAKER_RAG_DATA] 변환 파일 처리 실패: "
            f"source={source_csv_path!r}, output={output_csv_path!r}, "
            f"error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise CharacterMakerRagDataError(f"변환 파일 처리에 실패했습니다: {exc}") from exc

    if summary["written_rows"] == 0:
        print(
            "[CHARACTER_MAKER_RAG_DATA] 변환 결과 없음: "
            f"summary={summary}, unmatched_samples={unmatched_samples}"
        )
        raise CharacterMakerRagDataError(
            "변환 가능한 태그가 없습니다. KR_danbooru_tags_with_description CSV인지 확인하세요."
        )

    summary["unmatched_samples"] = unmatched_samples
    print(
        "[CHARACTER_MAKER_RAG_DATA] 변환 완료: "
        f"summary={summary}"
    )
    return summary
