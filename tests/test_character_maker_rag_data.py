import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modes.danbooru_rag import (  # noqa: E402
    DanbooruRagIndexNotInstalledError,
    DanbooruRagIndexInstaller,
    DanbooruRagInstallError,
    DanbooruRagService,
)
from modes.danbooru_rag.installer import (  # noqa: E402
    HF_ARCHIVE_PATH,
    HF_MANIFEST_PATH,
)


def _artifact(tmp_path: Path, *, member: str | None = None):
    archive = tmp_path / "lancedb_b.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
        bundle.writestr(
            member
            or "lancedb_b/danbooru_tags_b.lance/data/0000000000000000.lance",
            b"test-index",
        )
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_version": "test-1",
                "archive": "lancedb_b.zip",
                "archive_size": archive.stat().st_size,
                "archive_sha256": digest,
                "archive_root": "lancedb_b",
                "table_name": "danbooru_tags_b",
                "row_count": 12,
                "embedding_model": "intfloat/multilingual-e5-large",
                "embedding_dimension": 1024,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifest, archive


def _use_local_artifact(monkeypatch, installer, manifest: Path, archive: Path):
    paths = {
        HF_MANIFEST_PATH: manifest,
        HF_ARCHIVE_PATH: archive,
    }
    monkeypatch.setattr(installer, "_download", lambda filename: paths[filename])


def test_huggingface_index_installs_and_records_manifest(monkeypatch, tmp_path):
    manifest, archive = _artifact(tmp_path)
    installer = DanbooruRagIndexInstaller(project_root=tmp_path)
    _use_local_artifact(monkeypatch, installer, manifest, archive)

    progress = []
    result = installer.install(
        progress_callback=lambda phase, percent, detail: progress.append(
            (phase, percent, detail)
        )
    )

    assert result["success"] is True
    assert result["artifact_version"] == "test-1"
    assert result["row_count"] == 12
    assert result["archive_sha256"] == hashlib.sha256(
        archive.read_bytes()
    ).hexdigest()
    assert (
        installer.index_path
        / "danbooru_tags_b.lance"
        / "data"
        / "0000000000000000.lance"
    ).read_bytes() == b"test-index"
    installed_manifest = json.loads(
        (installer.data_root / "install_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert installed_manifest["huggingface_revision"] == installer.revision
    assert progress[-1][0:2] == ("완료", 100)
    assert installer.status()["archive_size"] == archive.stat().st_size


def test_existing_index_is_backed_up_before_replacement(monkeypatch, tmp_path):
    manifest, archive = _artifact(tmp_path)
    installer = DanbooruRagIndexInstaller(project_root=tmp_path)
    installer.data_root.mkdir(parents=True)
    (installer.data_root / "old.txt").write_text("기존 데이터", encoding="utf-8")
    _use_local_artifact(monkeypatch, installer, manifest, archive)

    result = installer.install()

    backup = Path(result["backup_path"])
    assert backup.is_relative_to(tmp_path / "요구사항")
    assert (backup / "old.txt").read_text(encoding="utf-8") == "기존 데이터"
    assert not (installer.data_root / "old.txt").exists()


def test_zip_path_traversal_is_rejected(monkeypatch, tmp_path):
    manifest, archive = _artifact(
        tmp_path,
        member="lancedb_b/../escaped.lance",
    )
    installer = DanbooruRagIndexInstaller(project_root=tmp_path)
    _use_local_artifact(monkeypatch, installer, manifest, archive)

    with pytest.raises(DanbooruRagInstallError, match="위험한 경로"):
        installer.install()

    assert not (tmp_path / "escaped.lance").exists()


def test_embedded_search_uses_variant_b_rows(tmp_path):
    index_path = tmp_path / "lancedb_b"
    (index_path / "danbooru_tags_b.lance").mkdir(parents=True)

    class Vector(list):
        def tolist(self):
            return list(self)

    class FakeModel:
        def encode(self, text, *, normalize_embeddings):
            assert text == "query: 긴 머리"
            assert normalize_embeddings is True
            return Vector([0.0] * 1024)

    class FakeSearch:
        def __init__(self, rows):
            self.rows = rows
            self.limit_value = 0

        def limit(self, value):
            self.limit_value = value
            return self

        def to_list(self):
            return self.rows[: self.limit_value]

    class FakeTable:
        def search(self, vector):
            assert len(vector) == 1024
            return FakeSearch(
                [
                    {
                        "tag": "long_hair",
                        "category": 0,
                        "frequency": 4800833,
                        "major": "appearance",
                        "minor": "hair",
                        "definition": "긴 머리",
                        "aliases": ["longhair"],
                        "_distance": 0.2,
                    },
                    {
                        "tag": "some_artist",
                        "category": 1,
                        "frequency": 100,
                        "_distance": 0.1,
                    },
                ]
            )

    service = DanbooruRagService(index_path=index_path)
    service._model = FakeModel()
    service._table = FakeTable()

    result = service.search("긴 머리", top_k=5, categories={0})

    assert result == [
        {
            "tag": "long_hair",
            "score": 0.9,
            "category": 0,
            "frequency": 4800833,
            "major": "appearance",
            "minor": "hair",
            "definition": "긴 머리",
            "aliases": ["longhair"],
        }
    ]


def test_missing_index_is_reported_before_model_load(monkeypatch, tmp_path):
    service = DanbooruRagService(index_path=tmp_path / "missing")
    monkeypatch.setattr(
        service,
        "_get_model",
        lambda: pytest.fail("인덱스 없이 모델을 로드하면 안 됩니다."),
    )

    with pytest.raises(DanbooruRagIndexNotInstalledError, match="설치"):
        service.search("긴 머리")
