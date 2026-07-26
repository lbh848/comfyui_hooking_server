"""캐릭터 메이커용 Danbooru Tag RAG 데이터 변환기.

사용자가 내려받은 한국어 설명 CSV를 서버에 내장된 ``auto_complete/danbooru.csv``와
결합해 danbooru-tag-rag가 읽을 수 있는 UTF-8 CSV로 변환한다. 입력 파일과 내장
기준표는 읽기 전용이며, 호출자가 지정한 임시 출력 파일에만 쓴다.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
import traceback
from typing import Any


csv.field_size_limit(10 * 1024 * 1024)

MIN_POST_COUNT = 50
MAX_UPLOAD_BYTES = 128 * 1024 * 1024
OUTPUT_FILENAME = "danbooru-tags.csv"
INDEX_VARIANT = "b"
RAG_REPOSITORY_URL = "https://github.com/joykst96/danbooru-tag-rag.git"


class CharacterMakerRagDataError(ValueError):
    """사용자에게 전달할 수 있는 RAG 데이터 변환 오류."""


def ensure_rag_repository(
    repo_path: str,
    repository_url: str = RAG_REPOSITORY_URL,
) -> dict[str, Any]:
    """고정 경로에 RAG 저장소가 없으면 임시 경로로 clone 후 원자적으로 배치한다."""
    raw_path = str(repo_path or "").strip()
    if not raw_path:
        print("[CHARACTER_MAKER_RAG_INSTALL] 자동 준비 대상 저장소 경로 비어 있음")
        raise CharacterMakerRagDataError(
            "Danbooru Tag RAG 자동 설치 경로가 지정되지 않았습니다."
        )
    expanded_path = os.path.expandvars(os.path.expanduser(raw_path))
    if not os.path.isabs(expanded_path):
        print(
            "[CHARACTER_MAKER_RAG_INSTALL] 자동 준비 상대 경로 거부: "
            f"path={raw_path!r}"
        )
        raise CharacterMakerRagDataError(
            "RAG 저장소 자동 설치 경로는 절대 경로여야 합니다."
        )

    repository = os.path.realpath(expanded_path)
    if os.path.isdir(repository):
        return {
            **validate_rag_repository(repository),
            "repository_cloned": False,
            "repository_url": repository_url,
        }
    if os.path.lexists(repository):
        print(
            "[CHARACTER_MAKER_RAG_INSTALL] 자동 clone 대상이 폴더가 아님: "
            f"path={repository!r}"
        )
        raise CharacterMakerRagDataError(
            f"RAG 자동 설치 위치가 폴더가 아닙니다: {repository}"
        )

    git_command = shutil.which("git")
    if not git_command:
        print("[CHARACTER_MAKER_RAG_INSTALL] git 실행 파일을 찾지 못함")
        raise CharacterMakerRagDataError(
            "RAG 저장소 자동 설치에 필요한 git 실행 파일을 찾을 수 없습니다."
        )

    parent = os.path.dirname(repository)
    clone_temp = ""
    try:
        os.makedirs(parent, exist_ok=True)
        clone_temp = tempfile.mkdtemp(
            prefix=".danbooru-tag-rag-clone-",
            dir=parent,
        )
        command = [
            git_command,
            "clone",
            "--depth",
            "1",
            "--",
            repository_url,
            clone_temp,
        ]
        print(
            "[CHARACTER_MAKER_RAG_INSTALL] 저장소 자동 clone 시작: "
            f"url={repository_url!r}, target={repository!r}, command={command!r}"
        )
        completed = subprocess.run(
            command,
            cwd=parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        clone_log = str(completed.stdout or "").strip()
        if clone_log:
            for line in clone_log.splitlines():
                print(f"[CHARACTER_MAKER_RAG_GIT] {line}")
        if completed.returncode != 0:
            print(
                "[CHARACTER_MAKER_RAG_INSTALL] 저장소 자동 clone 실패: "
                f"return_code={completed.returncode}, "
                f"log_tail={clone_log.splitlines()[-20:]}"
            )
            raise CharacterMakerRagDataError(
                "Danbooru Tag RAG 저장소를 자동으로 받지 못했습니다. "
                f"git 종료 코드: {completed.returncode}"
            )

        validate_rag_repository(clone_temp)
        if os.path.lexists(repository):
            print(
                "[CHARACTER_MAKER_RAG_INSTALL] clone 중 대상 경로가 생성됨: "
                f"path={repository!r}"
            )
            raise CharacterMakerRagDataError(
                "RAG 저장소를 받는 동안 설치 위치가 다른 작업에 의해 생성되었습니다."
            )
        os.replace(clone_temp, repository)
        clone_temp = ""
        paths = validate_rag_repository(repository)
        print(
            "[CHARACTER_MAKER_RAG_INSTALL] 저장소 자동 clone 완료: "
            f"url={repository_url!r}, path={repository!r}"
        )
        return {
            **paths,
            "repository_cloned": True,
            "repository_url": repository_url,
        }
    except Exception as exc:
        print(
            "[CHARACTER_MAKER_RAG_INSTALL] 저장소 자동 준비 실패: "
            f"path={repository!r}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        if isinstance(exc, CharacterMakerRagDataError):
            raise
        raise CharacterMakerRagDataError(
            f"Danbooru Tag RAG 저장소를 자동으로 준비하지 못했습니다: {exc}"
        ) from exc
    finally:
        if clone_temp and os.path.isdir(clone_temp):
            try:
                shutil.rmtree(clone_temp)
            except OSError as cleanup_exc:
                print(
                    "[CHARACTER_MAKER_RAG_INSTALL] clone 임시 폴더 정리 실패: "
                    f"path={clone_temp!r}, "
                    f"error={type(cleanup_exc).__name__}: {cleanup_exc}"
                )
                traceback.print_exc()


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
) -> dict[str, Any]:
    """새 CSV를 저장소 폴더 안에 원자적으로 덮어쓴다(외부 백업 없음).

    기존 CSV/인덱스는 저장소 폴더 안에서 그대로 교체된다. 빌드 실패 시 이전 상태로
    되돌리지 않으므로, 호출자는 사용자에게 태그 자료를 다시 설치하도록 안내해야 한다.
    """
    if not os.path.isfile(converted_csv_path):
        print(
            "[CHARACTER_MAKER_RAG_INSTALL] 변환 CSV 없음: "
            f"path={converted_csv_path!r}"
        )
        raise CharacterMakerRagDataError("설치할 변환 CSV를 찾을 수 없습니다.")

    paths = validate_rag_repository(repo_path)
    csv_path = paths["csv_path"]
    context: dict[str, Any] = {**paths}

    try:
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
        print(
            "[CHARACTER_MAKER_RAG_INSTALL] 새 CSV 설치 완료(덮어쓰기): "
            f"path={csv_path!r}, bytes={os.path.getsize(csv_path)}"
        )
        return context
    except Exception as exc:
        print(
            "[CHARACTER_MAKER_RAG_INSTALL] 설치 준비 실패: "
            f"repository={paths['repository']!r}, error={type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
        if isinstance(exc, CharacterMakerRagDataError):
            raise
        raise CharacterMakerRagDataError(
            f"RAG 설치 파일을 준비하지 못했습니다: {exc}"
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
    progress_cb=None,
) -> dict[str, Any]:
    """한국어 설명 CSV를 정식 RAG 입력 CSV로 변환한다.

    정식 태그명과 카테고리는 내장 기준표에서만 가져온다. 기준표와 매칭되지 않는
    행을 category 0으로 추측해 넣지 않으므로 일반 태그 검색 공간을 오염시키지 않는다.

    ``progress_cb(current, total)`` 를 주면 행 단위로 진행을 알린다. 동기 콜백이며
    스레드 안전 여부는 호출자가 보장한다.
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

    total_rows = 0
    if progress_cb is not None:
        try:
            with open(source_csv_path, "rb") as _count_f:
                total_rows = sum(
                    chunk.count(b"\n")
                    for chunk in iter(lambda: _count_f.read(1 << 20), b"")
                )
        except OSError as exc:
            print(
                "[CHARACTER_MAKER_RAG_DATA] \uc785\ub825 \ud589 \uc218 \uc0ac\uc804 \uc9d1\uacc4 \uc2e4\ud328: "
                f"path={source_csv_path!r}, error={type(exc).__name__}: {exc}"
            )
            total_rows = 0
        try:
            progress_cb(0, max(total_rows, 0))
        except Exception:
            print("[CHARACTER_MAKER_RAG_DATA] progress_cb \ucd08\uae30 \ud638\ucd9c \uc2e4\ud328")
            traceback.print_exc()

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

                if (
                    progress_cb is not None
                    and total_rows > 0
                    and line_number % 2000 == 0
                ):
                    try:
                        progress_cb(line_number, total_rows)
                    except Exception:
                        print(
                            "[CHARACTER_MAKER_RAG_DATA] progress_cb 호출 실패: "
                            f"line={line_number}"
                        )
                        traceback.print_exc()

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
            if progress_cb is not None and total_rows > 0:
                try:
                    progress_cb(total_rows, total_rows)
                except Exception:
                    print("[CHARACTER_MAKER_RAG_DATA] progress_cb 종료 호출 실패")
                    traceback.print_exc()
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
