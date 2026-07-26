"""캐릭터 메이커용 Danbooru Tag RAG 데이터 변환기.

사용자가 내려받은 한국어 설명 CSV를 서버에 내장된 ``auto_complete/danbooru.csv``와
결합해 danbooru-tag-rag가 읽을 수 있는 UTF-8 CSV로 변환한다. 입력 파일과 내장
기준표는 읽기 전용이며, 호출자가 지정한 임시 출력 파일에만 쓴다.
"""

from __future__ import annotations

import csv
import datetime
import os
import shutil
import tempfile
import traceback
from typing import Any


csv.field_size_limit(10 * 1024 * 1024)

MIN_POST_COUNT = 50
MAX_UPLOAD_BYTES = 128 * 1024 * 1024
OUTPUT_FILENAME = "danbooru-tags.csv"
INDEX_VARIANT = "b"


class CharacterMakerRagDataError(ValueError):
    """사용자에게 전달할 수 있는 RAG 데이터 변환 오류."""


def validate_rag_repository(repo_path: str) -> dict[str, str]:
    """로컬 danbooru-tag-rag 저장소와 자동 설치 대상 경로를 검증한다."""
    raw_path = str(repo_path or "").strip()
    if not raw_path:
        print("[CHARACTER_MAKER_RAG_INSTALL] 저장소 경로 비어 있음")
        raise CharacterMakerRagDataError(
            "Danbooru Tag RAG 저장소 경로를 먼저 지정하세요."
        )
    expanded_path = os.path.expandvars(os.path.expanduser(raw_path))
    if not os.path.isabs(expanded_path):
        print(
            "[CHARACTER_MAKER_RAG_INSTALL] 상대 저장소 경로 거부: "
            f"path={raw_path!r}"
        )
        raise CharacterMakerRagDataError("RAG 저장소 경로는 절대 경로여야 합니다.")

    repository = os.path.realpath(expanded_path)
    if not os.path.isdir(repository):
        print(
            "[CHARACTER_MAKER_RAG_INSTALL] 저장소 폴더 없음: "
            f"path={repository!r}"
        )
        raise CharacterMakerRagDataError(
            f"Danbooru Tag RAG 저장소 폴더가 존재하지 않습니다: {repository}"
        )

    required = (
        os.path.join(repository, "core", "config.py"),
        os.path.join(repository, "core", "builder.py"),
        os.path.join(repository, "pyproject.toml"),
    )
    missing = [
        os.path.relpath(path, repository)
        for path in required
        if not os.path.isfile(path)
    ]
    if missing:
        print(
            "[CHARACTER_MAKER_RAG_INSTALL] 저장소 구조 검증 실패: "
            f"path={repository!r}, missing={missing}"
        )
        raise CharacterMakerRagDataError(
            "선택한 폴더가 danbooru-tag-rag 저장소가 아닙니다. "
            f"누락: {', '.join(missing)}"
        )

    data_dir = os.path.realpath(os.path.join(repository, "data"))
    csv_path = os.path.realpath(os.path.join(repository, OUTPUT_FILENAME))
    index_path = os.path.realpath(
        os.path.join(data_dir, f"lancedb_{INDEX_VARIANT}")
    )
    for label, path in (
        ("CSV", csv_path),
        ("데이터", data_dir),
        ("인덱스", index_path),
    ):
        try:
            if os.path.commonpath([repository, path]) != repository:
                print(
                    "[CHARACTER_MAKER_RAG_INSTALL] 설치 경로 이탈 거부: "
                    f"label={label}, root={repository!r}, path={path!r}"
                )
                raise CharacterMakerRagDataError(
                    f"RAG {label} 설치 경로가 저장소 밖을 가리킵니다."
                )
        except ValueError as exc:
            print(
                "[CHARACTER_MAKER_RAG_INSTALL] 설치 경로 검증 실패: "
                f"label={label}, root={repository!r}, path={path!r}, error={exc}"
            )
            traceback.print_exc()
            raise CharacterMakerRagDataError(
                f"RAG {label} 설치 경로를 안전하게 확인하지 못했습니다."
            ) from exc

    return {
        "repository": repository,
        "csv_path": csv_path,
        "data_dir": data_dir,
        "index_path": index_path,
    }


def prepare_rag_install(
    converted_csv_path: str,
    repo_path: str,
    backup_root: str,
) -> dict[str, Any]:
    """기존 CSV/variant-b 인덱스를 백업하고 새 CSV를 원자적으로 설치한다."""
    if not os.path.isfile(converted_csv_path):
        print(
            "[CHARACTER_MAKER_RAG_INSTALL] 변환 CSV 없음: "
            f"path={converted_csv_path!r}"
        )
        raise CharacterMakerRagDataError("설치할 변환 CSV를 찾을 수 없습니다.")

    paths = validate_rag_repository(repo_path)
    raw_backup_root = str(backup_root or "").strip()
    if not raw_backup_root:
        print("[CHARACTER_MAKER_RAG_INSTALL] 백업 루트 비어 있음")
        raise CharacterMakerRagDataError("RAG 설치 백업 폴더가 지정되지 않았습니다.")
    if not os.path.isabs(raw_backup_root):
        print(
            "[CHARACTER_MAKER_RAG_INSTALL] 상대 백업 루트 거부: "
            f"path={raw_backup_root!r}"
        )
        raise CharacterMakerRagDataError("RAG 설치 백업 폴더는 절대 경로여야 합니다.")
    backup_root = os.path.realpath(raw_backup_root)
    os.makedirs(backup_root, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_dir = os.path.join(
        backup_root,
        f"character_maker_rag_before_install_{stamp}",
    )
    os.makedirs(backup_dir, exist_ok=False)

    csv_path = paths["csv_path"]
    index_path = paths["index_path"]
    csv_backup = os.path.join(backup_dir, OUTPUT_FILENAME)
    index_backup = os.path.join(
        backup_dir,
        f"lancedb_{INDEX_VARIANT}",
    )
    context: dict[str, Any] = {
        **paths,
        "backup_dir": backup_dir,
        "csv_backup": csv_backup,
        "index_backup": index_backup,
        "had_csv": os.path.isfile(csv_path),
        "had_index": os.path.isdir(index_path),
        "installed": False,
    }

    try:
        if context["had_csv"]:
            shutil.copy2(csv_path, csv_backup)
            print(
                "[CHARACTER_MAKER_RAG_INSTALL] 기존 CSV 백업 완료: "
                f"source={csv_path!r}, backup={csv_backup!r}"
            )
        if context["had_index"]:
            shutil.copytree(index_path, index_backup)
            print(
                "[CHARACTER_MAKER_RAG_INSTALL] 기존 인덱스 백업 완료: "
                f"source={index_path!r}, backup={index_backup!r}"
            )

        temporary = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".danbooru-tags-install-",
            suffix=".csv",
            dir=paths["repository"],
            delete=False,
        )
        temporary_path = temporary.name
        temporary.close()
        try:
            shutil.copyfile(converted_csv_path, temporary_path)
            os.replace(temporary_path, csv_path)
        finally:
            if os.path.isfile(temporary_path):
                os.remove(temporary_path)
        context["installed"] = True
        print(
            "[CHARACTER_MAKER_RAG_INSTALL] 새 CSV 설치 완료: "
            f"path={csv_path!r}, bytes={os.path.getsize(csv_path)}, "
            f"backup={backup_dir!r}"
        )
        return context
    except Exception as exc:
        print(
            "[CHARACTER_MAKER_RAG_INSTALL] 설치 준비 실패: "
            f"repository={paths['repository']!r}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        if context.get("installed"):
            try:
                restore_rag_install(context)
            except Exception:
                traceback.print_exc()
        if isinstance(exc, CharacterMakerRagDataError):
            raise
        raise CharacterMakerRagDataError(
            f"RAG 설치 파일을 준비하지 못했습니다: {exc}"
        ) from exc


def restore_rag_install(context: dict[str, Any]) -> None:
    """빌드 실패 시 설치 전 CSV와 variant-b 인덱스를 복구한다."""
    raw_repository = str(context.get("repository") or "").strip()
    raw_csv_path = str(context.get("csv_path") or "").strip()
    raw_index_path = str(context.get("index_path") or "").strip()
    if not raw_repository or not raw_csv_path or not raw_index_path:
        print(
            "[CHARACTER_MAKER_RAG_INSTALL] 복구 컨텍스트 경로 누락: "
            f"repository={raw_repository!r}, csv={raw_csv_path!r}, "
            f"index={raw_index_path!r}"
        )
        raise CharacterMakerRagDataError("RAG 설치 복구 정보가 완전하지 않습니다.")
    repository = os.path.realpath(raw_repository)
    csv_path = os.path.realpath(raw_csv_path)
    index_path = os.path.realpath(raw_index_path)
    if not os.path.isdir(repository):
        print(
            "[CHARACTER_MAKER_RAG_INSTALL] 복구 저장소 검증 실패: "
            f"repository={repository!r}"
        )
        raise CharacterMakerRagDataError("RAG 설치 실패 후 저장소를 복구하지 못했습니다.")
    if (
        os.path.basename(csv_path).casefold() != OUTPUT_FILENAME.casefold()
        or os.path.basename(index_path).casefold()
        != f"lancedb_{INDEX_VARIANT}".casefold()
    ):
        print(
            "[CHARACTER_MAKER_RAG_INSTALL] 복구 대상명 검증 실패: "
            f"csv={csv_path!r}, index={index_path!r}"
        )
        raise CharacterMakerRagDataError("RAG 설치 복구 대상 파일명이 올바르지 않습니다.")
    for label, path in (("CSV", csv_path), ("인덱스", index_path)):
        try:
            if os.path.commonpath([repository, path]) != repository:
                print(
                    "[CHARACTER_MAKER_RAG_INSTALL] 복구 경로 이탈 거부: "
                    f"label={label}, repository={repository!r}, path={path!r}"
                )
                raise CharacterMakerRagDataError(
                    f"RAG {label} 복구 경로가 저장소 밖을 가리킵니다."
                )
        except ValueError as exc:
            print(
                "[CHARACTER_MAKER_RAG_INSTALL] 복구 경로 검증 실패: "
                f"label={label}, repository={repository!r}, path={path!r}, error={exc}"
            )
            traceback.print_exc()
            raise CharacterMakerRagDataError(
                f"RAG {label} 복구 경로를 확인하지 못했습니다."
            ) from exc

    csv_backup = str(context.get("csv_backup") or "")
    index_backup = str(context.get("index_backup") or "")
    try:
        if context.get("had_csv") and os.path.isfile(csv_backup):
            temporary = tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".danbooru-tags-restore-",
                suffix=".csv",
                dir=repository,
                delete=False,
            )
            temporary_path = temporary.name
            temporary.close()
            try:
                shutil.copyfile(csv_backup, temporary_path)
                os.replace(temporary_path, csv_path)
            finally:
                if os.path.isfile(temporary_path):
                    os.remove(temporary_path)
        elif not context.get("had_csv") and os.path.isfile(csv_path):
            os.remove(csv_path)

        if context.get("had_index") and os.path.isdir(index_backup):
            if os.path.isdir(index_path):
                shutil.rmtree(index_path)
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            shutil.copytree(index_backup, index_path)
        elif not context.get("had_index") and os.path.isdir(index_path):
            shutil.rmtree(index_path)
        print(
            "[CHARACTER_MAKER_RAG_INSTALL] 설치 전 상태 복구 완료: "
            f"repository={repository!r}, backup={context.get('backup_dir')!r}"
        )
    except Exception as exc:
        print(
            "[CHARACTER_MAKER_RAG_INSTALL] 설치 전 상태 복구 실패: "
            f"repository={repository!r}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        raise CharacterMakerRagDataError(
            f"RAG 설치 실패 후 기존 자료 복구에도 실패했습니다: {exc}"
        ) from exc


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
