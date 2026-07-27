"""Download and atomically install the prebuilt Danbooru RAG index."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import traceback
from typing import Any, Callable
import uuid
import zipfile

from .service import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_ID,
    PROJECT_ROOT,
    TABLE_NAME,
)


HF_REPO_ID = "byung-hyun/eye_segmentation_model"
HF_REVISION = "eac8bbddb65447dfe07bbf09d79035c466d20849"
HF_MANIFEST_PATH = "comfyui_hooking_server/danbooru_rag/manifest.json"
HF_ARCHIVE_PATH = "comfyui_hooking_server/danbooru_rag/lancedb_b.zip"
INSTALL_MANIFEST_NAME = "install_manifest.json"
ProgressCallback = Callable[[str, int, str], None]


class DanbooruRagInstallError(RuntimeError):
    """Raised when the downloadable index cannot be installed safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


class DanbooruRagIndexInstaller:
    """Install one pinned Hugging Face index artifact into ``auto_complete``."""

    def __init__(
        self,
        *,
        project_root: str | os.PathLike[str] | None = None,
        data_root: str | os.PathLike[str] | None = None,
        revision: str = HF_REVISION,
    ) -> None:
        self.project_root = Path(project_root or PROJECT_ROOT).resolve()
        self.auto_complete_root = (
            self.project_root / "auto_complete"
        ).resolve()
        default_data_root = self.project_root / "auto_complete" / "danbooru_rag_data"
        self.data_root = Path(
            data_root
            or os.environ.get("DANBOORU_RAG_DATA_ROOT")
            or default_data_root
        ).resolve()
        self.index_path = self.data_root / "lancedb_b"
        self.revision = str(revision)
        if not _is_within(self.data_root, self.auto_complete_root):
            print(
                "[DANBOORU_RAG_INSTALL] 설치 경로 이탈 거부: "
                f"auto_complete={str(self.auto_complete_root)!r}, "
                f"data_root={str(self.data_root)!r}"
            )
            raise DanbooruRagInstallError(
                "Danbooru RAG 데이터는 프로젝트 auto_complete 안에만 설치할 수 있습니다."
            )

    def _progress(
        self,
        callback: ProgressCallback | None,
        phase: str,
        percent: int,
        detail: str,
    ) -> None:
        print(
            "[DANBOORU_RAG_INSTALL] "
            f"phase={phase!r}, progress={percent}, detail={detail!r}"
        )
        if callback is None:
            return
        try:
            callback(phase, max(0, min(100, int(percent))), detail)
        except Exception as exc:
            print(
                "[DANBOORU_RAG_INSTALL] 진행 콜백 실패: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()

    def _download(self, filename: str) -> Path:
        if not self.revision or self.revision.startswith("PENDING_"):
            print(
                "[DANBOORU_RAG_INSTALL] 고정 Hugging Face revision 미설정: "
                f"revision={self.revision!r}"
            )
            raise DanbooruRagInstallError(
                "Danbooru RAG 다운로드 revision이 설정되지 않았습니다."
            )
        try:
            from huggingface_hub import hf_hub_download

            path = Path(
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=filename,
                    revision=self.revision,
                )
            )
            return path
        except Exception as exc:
            print(
                "[DANBOORU_RAG_INSTALL] Hugging Face 다운로드 실패: "
                f"repo={HF_REPO_ID!r}, revision={self.revision!r}, "
                f"filename={filename!r}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise DanbooruRagInstallError(
                f"Danbooru RAG 파일을 Hugging Face에서 받지 못했습니다: {exc}"
            ) from exc

    def _load_manifest(self, path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(
                "[DANBOORU_RAG_INSTALL] 매니페스트 해석 실패: "
                f"path={str(path)!r}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise DanbooruRagInstallError(
                f"RAG 인덱스 매니페스트를 읽지 못했습니다: {exc}"
            ) from exc
        required = {
            "schema_version",
            "artifact_version",
            "archive",
            "archive_size",
            "archive_sha256",
            "archive_root",
            "table_name",
            "row_count",
            "embedding_model",
            "embedding_dimension",
        }
        missing = sorted(required.difference(payload))
        if missing:
            print(
                "[DANBOORU_RAG_INSTALL] 매니페스트 필드 누락: "
                f"missing={missing!r}, path={str(path)!r}"
            )
            raise DanbooruRagInstallError(
                f"RAG 인덱스 매니페스트 필드가 누락됐습니다: {', '.join(missing)}"
            )
        if payload["archive"] != Path(HF_ARCHIVE_PATH).name:
            raise DanbooruRagInstallError("RAG 인덱스 아카이브 이름이 일치하지 않습니다.")
        if payload["archive_root"] != "lancedb_b":
            raise DanbooruRagInstallError("RAG 인덱스 루트 이름이 올바르지 않습니다.")
        if payload["table_name"] != TABLE_NAME:
            raise DanbooruRagInstallError("RAG 인덱스 테이블 이름이 올바르지 않습니다.")
        if payload["embedding_model"] != EMBEDDING_MODEL_ID:
            raise DanbooruRagInstallError("RAG 인덱스 임베딩 모델이 올바르지 않습니다.")
        if int(payload["embedding_dimension"]) != EMBEDDING_DIMENSION:
            raise DanbooruRagInstallError("RAG 인덱스 임베딩 차원이 올바르지 않습니다.")
        return payload

    def _verify_archive(self, path: Path, manifest: dict[str, Any]) -> None:
        actual_size = path.stat().st_size
        expected_size = int(manifest["archive_size"])
        if actual_size != expected_size:
            print(
                "[DANBOORU_RAG_INSTALL] 아카이브 크기 검증 실패: "
                f"expected={expected_size}, actual={actual_size}, path={str(path)!r}"
            )
            raise DanbooruRagInstallError("다운로드한 RAG 인덱스 크기가 다릅니다.")
        actual_hash = _sha256(path)
        expected_hash = str(manifest["archive_sha256"]).lower()
        if actual_hash.lower() != expected_hash:
            print(
                "[DANBOORU_RAG_INSTALL] 아카이브 SHA-256 검증 실패: "
                f"expected={expected_hash}, actual={actual_hash}, path={str(path)!r}"
            )
            raise DanbooruRagInstallError("다운로드한 RAG 인덱스 해시가 다릅니다.")

    def _extract_safely(self, archive_path: Path, staging_root: Path) -> Path:
        try:
            with zipfile.ZipFile(archive_path, "r") as archive:
                for item in archive.infolist():
                    member = PurePosixPath(item.filename)
                    if (
                        member.is_absolute()
                        or not member.parts
                        or member.parts[0] != "lancedb_b"
                        or any(part in ("", ".", "..") for part in member.parts)
                    ):
                        print(
                            "[DANBOORU_RAG_INSTALL] 위험한 ZIP 경로 거부: "
                            f"member={item.filename!r}"
                        )
                        raise DanbooruRagInstallError(
                            f"RAG 인덱스 압축 파일에 위험한 경로가 있습니다: {item.filename}"
                        )
                    unix_mode = item.external_attr >> 16
                    if stat.S_ISLNK(unix_mode):
                        print(
                            "[DANBOORU_RAG_INSTALL] ZIP 심볼릭 링크 거부: "
                            f"member={item.filename!r}"
                        )
                        raise DanbooruRagInstallError(
                            "RAG 인덱스 압축 파일에 심볼릭 링크가 있습니다."
                        )
                    destination = staging_root.joinpath(*member.parts)
                    if not _is_within(destination, staging_root):
                        raise DanbooruRagInstallError(
                            "RAG 인덱스 압축 해제 경로가 임시 폴더를 벗어납니다."
                        )
                    if item.is_dir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with (
                        archive.open(item, "r") as source,
                        destination.open("wb") as target,
                    ):
                        shutil.copyfileobj(source, target, length=1024 * 1024)
        except DanbooruRagInstallError as exc:
            print(
                "[DANBOORU_RAG_INSTALL] 검증 또는 설치 거부: "
                f"target={str(self.data_root)!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            raise
        except Exception as exc:
            print(
                "[DANBOORU_RAG_INSTALL] 아카이브 압축 해제 실패: "
                f"archive={str(archive_path)!r}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise DanbooruRagInstallError(
                f"RAG 인덱스 압축을 풀지 못했습니다: {exc}"
            ) from exc
        return staging_root / "lancedb_b"

    def _validate_extracted(
        self,
        index_path: Path,
        manifest: dict[str, Any],
    ) -> None:
        table_root = index_path / f"{TABLE_NAME}.lance"
        data_root = table_root / "data"
        data_files = list(data_root.glob("*.lance")) if data_root.is_dir() else []
        if not index_path.is_dir() or not table_root.is_dir() or not data_files:
            print(
                "[DANBOORU_RAG_INSTALL] 압축 해제 인덱스 구조 검증 실패: "
                f"index={str(index_path)!r}, table={str(table_root)!r}, "
                f"data_files={len(data_files)}"
            )
            raise DanbooruRagInstallError(
                "압축 해제된 Danbooru RAG 인덱스 구조가 올바르지 않습니다."
            )
        if int(manifest["row_count"]) <= 0:
            raise DanbooruRagInstallError("RAG 인덱스 행 수가 올바르지 않습니다.")

    def _backup_existing(self) -> Path | None:
        if not self.data_root.exists():
            return None
        backup_parent = self.project_root / "요구사항"
        backup_parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_path = backup_parent / f"danbooru_rag_before_install_{stamp}"
        try:
            shutil.copytree(self.data_root, backup_path)
            print(
                "[DANBOORU_RAG_INSTALL] 기존 데이터 백업 완료: "
                f"source={str(self.data_root)!r}, backup={str(backup_path)!r}"
            )
            return backup_path
        except Exception as exc:
            print(
                "[DANBOORU_RAG_INSTALL] 기존 데이터 백업 실패: "
                f"source={str(self.data_root)!r}, backup={str(backup_path)!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise DanbooruRagInstallError(
                f"기존 RAG 데이터를 백업하지 못해 설치를 중단했습니다: {exc}"
            ) from exc

    def status(self) -> dict[str, Any]:
        manifest_path = self.data_root / INSTALL_MANIFEST_NAME
        installed = (
            self.index_path.is_dir()
            and (self.index_path / f"{TABLE_NAME}.lance").is_dir()
        )
        manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(
                    "[DANBOORU_RAG_INSTALL] 설치 매니페스트 읽기 실패: "
                    f"path={str(manifest_path)!r}, error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
        return {
            "installed": bool(installed),
            "data_root": str(self.data_root),
            "index_path": str(self.index_path),
            "artifact_version": str(manifest.get("artifact_version") or ""),
            "revision": str(manifest.get("huggingface_revision") or ""),
            "row_count": int(manifest.get("row_count") or 0),
            "archive_size": int(manifest.get("archive_size") or 0),
            "archive_sha256": str(manifest.get("archive_sha256") or ""),
            "source": f"{HF_REPO_ID}/{HF_ARCHIVE_PATH}",
        }

    def install(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        self.auto_complete_root.mkdir(parents=True, exist_ok=True)
        staging_root = Path(
            tempfile.mkdtemp(
                prefix=".danbooru-rag-install-",
                dir=self.auto_complete_root,
            )
        ).resolve()
        previous_path: Path | None = None
        backup_path: Path | None = None
        try:
            self._progress(
                progress_callback,
                "매니페스트 다운로드",
                5,
                "Hugging Face 배포 정보를 확인합니다.",
            )
            manifest_path = self._download(HF_MANIFEST_PATH)
            manifest = self._load_manifest(manifest_path)

            self._progress(
                progress_callback,
                "인덱스 다운로드",
                15,
                "Hugging Face에서 variant-b 인덱스를 받습니다.",
            )
            archive_path = self._download(HF_ARCHIVE_PATH)
            self._progress(
                progress_callback,
                "무결성 검증",
                65,
                "파일 크기와 SHA-256을 검증합니다.",
            )
            self._verify_archive(archive_path, manifest)

            extraction_root = staging_root / "extract"
            extraction_root.mkdir(parents=True, exist_ok=False)
            self._progress(
                progress_callback,
                "압축 해제",
                72,
                "검증된 인덱스 압축을 풉니다.",
            )
            extracted_index = self._extract_safely(archive_path, extraction_root)
            self._validate_extracted(extracted_index, manifest)

            payload_root = staging_root / "payload"
            payload_root.mkdir(parents=True, exist_ok=False)
            os.replace(extracted_index, payload_root / "lancedb_b")
            installed_manifest = {
                **manifest,
                "huggingface_repo": HF_REPO_ID,
                "huggingface_revision": self.revision,
                "huggingface_archive_path": HF_ARCHIVE_PATH,
                "installed_at": datetime.now().astimezone().isoformat(),
            }
            (payload_root / INSTALL_MANIFEST_NAME).write_text(
                json.dumps(installed_manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            self._progress(
                progress_callback,
                "기존 자료 백업",
                88,
                "기존 설치가 있으면 요구사항 폴더에 백업합니다.",
            )
            backup_path = self._backup_existing()
            self.data_root.parent.mkdir(parents=True, exist_ok=True)
            if self.data_root.exists():
                previous_path = self.data_root.with_name(
                    f".danbooru-rag-previous-{uuid.uuid4().hex}"
                )
                if not _is_within(previous_path, self.auto_complete_root):
                    raise DanbooruRagInstallError(
                        "기존 RAG 임시 이동 경로가 안전하지 않습니다."
                    )
                os.replace(self.data_root, previous_path)

            self._progress(
                progress_callback,
                "설치",
                94,
                "검증된 인덱스를 원자적으로 교체합니다.",
            )
            try:
                os.replace(payload_root, self.data_root)
            except Exception:
                if previous_path is not None and previous_path.exists():
                    os.replace(previous_path, self.data_root)
                    previous_path = None
                raise

            if previous_path is not None and previous_path.exists():
                if not _is_within(previous_path, self.auto_complete_root):
                    raise DanbooruRagInstallError(
                        "기존 RAG 정리 경로가 안전하지 않습니다."
                    )
                shutil.rmtree(previous_path)
                print(
                    "[DANBOORU_RAG_INSTALL] 교체된 이전 데이터 제거 완료: "
                    f"path={str(previous_path)!r}, backup={str(backup_path)!r}"
                )
                previous_path = None

            self._progress(
                progress_callback,
                "완료",
                100,
                "내장 Danbooru RAG 인덱스 설치가 완료됐습니다.",
            )
            return {
                "success": True,
                **self.status(),
                "archive_size": int(manifest["archive_size"]),
                "archive_sha256": str(manifest["archive_sha256"]),
                "backup_path": str(backup_path) if backup_path else "",
            }
        except DanbooruRagInstallError:
            raise
        except Exception as exc:
            print(
                "[DANBOORU_RAG_INSTALL] 설치 실패: "
                f"target={str(self.data_root)!r}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise DanbooruRagInstallError(
                f"Danbooru RAG 인덱스 설치에 실패했습니다: {exc}"
            ) from exc
        finally:
            if previous_path is not None and previous_path.exists():
                print(
                    "[DANBOORU_RAG_INSTALL] 실패 후 이전 데이터 임시 폴더 잔존: "
                    f"path={str(previous_path)!r}, backup={str(backup_path)!r}"
                )
            if staging_root.exists():
                if not _is_within(staging_root, self.auto_complete_root):
                    print(
                        "[DANBOORU_RAG_INSTALL] 임시 폴더 정리 경로 이탈로 스킵: "
                        f"path={str(staging_root)!r}"
                    )
                else:
                    try:
                        shutil.rmtree(staging_root)
                    except Exception as exc:
                        print(
                            "[DANBOORU_RAG_INSTALL] 임시 폴더 정리 실패: "
                            f"path={str(staging_root)!r}, "
                            f"error={type(exc).__name__}: {exc}"
                        )
                        traceback.print_exc()
