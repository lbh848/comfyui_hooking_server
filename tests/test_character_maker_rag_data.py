import csv
import subprocess
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modes.character_maker_rag_data import (
    CharacterMakerRagDataError,
    convert_kr_danbooru_csv,
    ensure_rag_repository,
    prepare_rag_install,
    restore_rag_install,
    validate_rag_repository,
)


def _create_repository_structure(path: Path) -> None:
    (path / "core").mkdir(parents=True)
    (path / "core" / "config.py").write_text("# test\n", encoding="utf-8")
    (path / "core" / "builder.py").write_text("# test\n", encoding="utf-8")
    (path / "pyproject.toml").write_text(
        "[project]\nname='rag-test'\n",
        encoding="utf-8",
    )


def _write_rows(path: Path, rows, *, encoding="utf-8"):
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(rows)


def test_converter_merges_canonical_tags_and_categories_without_touching_sources(
    tmp_path,
):
    canonical = tmp_path / "danbooru.csv"
    source = tmp_path / "KR_danbooru_tags_with_description.csv"
    output = tmp_path / "danbooru-tags.csv"
    canonical_rows = [
        ["1girl", 0, 6000000, "1girls"],
        ["long_hair", 0, 4000000, "longhair"],
        ["tatsuta_(kancolle)", 4, 5000, ""],
        ["highres", 5, 5000000, "high_resolution"],
    ]
    source_rows = [
        ["1girl", 0, 6577634, "[인물 > 인원수] 여성 한 명. 키워드: 여자 1명"],
        ["long hair", 0, 4800833, "[패션 > 헤어스타일] 긴 머리. 키워드: 장발"],
        [
            r"tatsuta \(kancolle\)",
            0,
            5545,
            "[캐릭터 > 함대 컬렉션] 타츠타. 키워드: 타츠타",
        ],
        ["low count", 0, 49, "낮은 빈도"],
        ["not in canonical", 0, 100, "미매칭"],
        ["broken", 0, 100],
        ["long hair", 0, 90, "중복"],
    ]
    _write_rows(canonical, canonical_rows)
    _write_rows(source, source_rows, encoding="utf-8-sig")
    canonical_before = canonical.read_bytes()
    source_before = source.read_bytes()

    summary = convert_kr_danbooru_csv(
        str(source),
        str(canonical),
        str(output),
    )

    with output.open("r", encoding="utf-8", newline="") as handle:
        converted = list(csv.reader(handle))
    assert converted == [
        ["name", "category", "post_count", "description"],
        ["1girl", "0", "6577634", "[인물 > 인원수] 여성 한 명. 키워드: 여자 1명"],
        ["long_hair", "0", "4800833", "[패션 > 헤어스타일] 긴 머리. 키워드: 장발"],
        [
            "tatsuta_(kancolle)",
            "4",
            "5545",
            "[캐릭터 > 함대 컬렉션] 타츠타. 키워드: 타츠타",
        ],
    ]
    assert not output.read_bytes().startswith(b"\xef\xbb\xbf")
    assert summary["input_rows"] == 7
    assert summary["written_rows"] == 3
    assert summary["below_frequency"] == 1
    assert summary["unmatched"] == 1
    assert summary["malformed"] == 1
    assert summary["duplicates"] == 1
    assert canonical.read_bytes() == canonical_before
    assert source.read_bytes() == source_before


def test_converter_accepts_an_existing_header(tmp_path):
    canonical = tmp_path / "danbooru.csv"
    source = tmp_path / "source.csv"
    output = tmp_path / "output.csv"
    _write_rows(canonical, [["blue_eyes", 0, 1000, ""]])
    _write_rows(
        source,
        [
            ["name", "category", "post_count", "description"],
            ["blue eyes", 0, 900, "[얼굴 > 눈] 파란 눈. 키워드: 벽안"],
        ],
    )

    summary = convert_kr_danbooru_csv(str(source), str(canonical), str(output))

    assert summary["header_rows"] == 1
    assert summary["input_rows"] == 1
    assert summary["written_rows"] == 1


def test_converter_rejects_a_dataset_with_no_canonical_matches(tmp_path):
    canonical = tmp_path / "danbooru.csv"
    source = tmp_path / "source.csv"
    output = tmp_path / "output.csv"
    _write_rows(canonical, [["blue_eyes", 0, 1000, ""]])
    _write_rows(source, [["invented tag", 0, 900, "설명"]])

    with pytest.raises(CharacterMakerRagDataError, match="변환 가능한 태그"):
        convert_kr_danbooru_csv(str(source), str(canonical), str(output))


def test_converter_reports_missing_builtin_canonical_file(tmp_path):
    source = tmp_path / "source.csv"
    output = tmp_path / "output.csv"
    _write_rows(source, [["blue eyes", 0, 900, "설명"]])

    with pytest.raises(CharacterMakerRagDataError, match="내장 Danbooru 기준표"):
        convert_kr_danbooru_csv(
            str(source),
            str(tmp_path / "missing.csv"),
            str(output),
        )


def test_prepare_and_restore_rag_install_preserves_existing_data(tmp_path):
    repository = tmp_path / "danbooru-tag-rag"
    _create_repository_structure(repository)
    old_csv = repository / "danbooru-tags.csv"
    old_csv.write_text("old csv\n", encoding="utf-8")
    old_index = repository / "data" / "lancedb_b"
    old_index.mkdir(parents=True)
    (old_index / "old.bin").write_bytes(b"old-index")
    converted = tmp_path / "converted.csv"
    converted.write_text(
        "name,category,post_count,description\nnew_tag,0,100,new\n",
        encoding="utf-8",
    )

    paths = validate_rag_repository(str(repository))
    context = prepare_rag_install(
        str(converted),
        str(repository),
        str(tmp_path / "요구사항"),
    )

    assert paths["repository"] == str(repository.resolve())
    assert old_csv.read_text(encoding="utf-8") == converted.read_text(encoding="utf-8")
    assert Path(context["csv_backup"]).read_text(encoding="utf-8") == "old csv\n"
    assert Path(context["index_backup"], "old.bin").read_bytes() == b"old-index"

    (old_index / "damaged.bin").write_bytes(b"damaged")
    restore_rag_install(context)

    assert old_csv.read_text(encoding="utf-8") == "old csv\n"
    assert (old_index / "old.bin").read_bytes() == b"old-index"
    assert not (old_index / "damaged.bin").exists()


def test_validate_rag_repository_rejects_an_unrelated_folder(tmp_path):
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()

    with pytest.raises(CharacterMakerRagDataError, match="저장소가 아닙니다"):
        validate_rag_repository(str(unrelated))


def test_ensure_rag_repository_reuses_an_existing_repository(
    monkeypatch,
    tmp_path,
):
    repository = tmp_path / "danbooru-tag-rag"
    _create_repository_structure(repository)

    def fail_if_git_is_checked(_name):
        raise AssertionError("기존 저장소가 있으면 git을 찾으면 안 됩니다.")

    monkeypatch.setattr("modes.character_maker_rag_data.shutil.which", fail_if_git_is_checked)

    result = ensure_rag_repository(str(repository))

    assert result["repository"] == str(repository.resolve())
    assert result["repository_cloned"] is False


def test_ensure_rag_repository_clones_to_the_fixed_location(
    monkeypatch,
    tmp_path,
):
    repository = tmp_path / "auto_complete" / "danbooru-tag-rag"
    seen_command = []

    monkeypatch.setattr(
        "modes.character_maker_rag_data.shutil.which",
        lambda name: "git-test" if name == "git" else None,
    )

    def fake_run(command, **kwargs):
        seen_command.append(command)
        clone_target = Path(command[-1])
        _create_repository_structure(clone_target)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="Cloning into temporary directory...\n",
        )

    monkeypatch.setattr(
        "modes.character_maker_rag_data.subprocess.run",
        fake_run,
    )

    result = ensure_rag_repository(str(repository))

    assert result["repository"] == str(repository.resolve())
    assert result["repository_cloned"] is True
    assert seen_command[0][:-1] == [
        "git-test",
        "clone",
        "--depth",
        "1",
        "--",
        "https://github.com/joykst96/danbooru-tag-rag.git",
    ]
    assert Path(seen_command[0][-1]).parent == repository.parent
    assert repository.is_dir()
    assert not list(repository.parent.glob(".danbooru-tag-rag-clone-*"))
