from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import struct
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from argon2.low_level import Type, hash_secret_raw
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


PACK_MAGIC = b"SOYAWFP1"
PACK_VERSION = 1
MAX_HEADER_BYTES = 64 * 1024
ARGON2_TIME_COST = 3
ARGON2_MEMORY_KIB = 64 * 1024
ARGON2_PARALLELISM = 2
ARGON2_HASH_LEN = 32


class WorkflowPackError(RuntimeError):
    """워크플로우 팩 생성·복호화·검증 실패."""


@dataclass(frozen=True)
class ExtractedWorkflowPack:
    target_dir: Path
    workflow_bindings: dict[str, str]
    workflow_hashes: dict[str, str]
    pack_sha256: str


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _derive_key(passphrase: str, salt: bytes, kdf: Mapping[str, int]) -> bytes:
    if not isinstance(passphrase, str) or not passphrase:
        raise WorkflowPackError("워크플로우 팩 키가 비어 있습니다.")
    return hash_secret_raw(
        secret=passphrase.encode("utf-8"),
        salt=salt,
        time_cost=int(kdf["time_cost"]),
        memory_cost=int(kdf["memory_cost_kib"]),
        parallelism=int(kdf["parallelism"]),
        hash_len=ARGON2_HASH_LEN,
        type=Type.ID,
    )


def _validate_archive_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    path = Path(normalized)
    if (
        not normalized
        or normalized.startswith("/")
        or path.is_absolute()
        or ".." in path.parts
    ):
        raise WorkflowPackError(f"안전하지 않은 워크플로우 팩 경로입니다: {name!r}")
    return normalized


def create_workflow_pack(
    workflow_bindings: Mapping[str, str | os.PathLike[str]],
    destination: str | os.PathLike[str],
    passphrase: str,
) -> dict:
    """워크플로우 파일을 Argon2id + AES-256-GCM 팩으로 묶는다."""

    try:
        if not workflow_bindings:
            raise WorkflowPackError("암호화할 워크플로우가 없습니다.")

        files: dict[str, Path] = {}
        bindings: dict[str, str] = {}
        used_names: dict[str, Path] = {}
        for config_key, source_value in workflow_bindings.items():
            source = Path(source_value).resolve()
            if not source.is_file():
                raise WorkflowPackError(
                    f"워크플로우 원본 파일이 없습니다: key={config_key!r}, path={source}"
                )
            try:
                parsed = json.loads(source.read_text(encoding="utf-8"))
            except Exception as exc:
                print(
                    "[COMFY_INSTALL][PACK] 워크플로우 JSON 검증 실패: "
                    f"key={config_key!r}, path={source}, error={exc}"
                )
                traceback.print_exc()
                raise WorkflowPackError(
                    f"워크플로우 JSON을 읽을 수 없습니다: {source.name}"
                ) from exc
            if not isinstance(parsed, dict):
                raise WorkflowPackError(
                    f"워크플로우 최상위 값이 객체가 아닙니다: {source.name}"
                )

            archive_name = _validate_archive_name(f"workflows/{source.name}")
            prior = used_names.get(archive_name.casefold())
            if prior is not None and prior != source:
                raise WorkflowPackError(
                    "서로 다른 워크플로우의 파일명이 충돌합니다: "
                    f"{prior} / {source}"
                )
            used_names[archive_name.casefold()] = source
            files[archive_name] = source
            bindings[str(config_key)] = archive_name

        file_manifest = {}
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(
            zip_buffer,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for archive_name, source in sorted(files.items()):
                payload = source.read_bytes()
                file_manifest[archive_name] = {
                    "sha256": _sha256_bytes(payload),
                    "size": len(payload),
                }
                archive.writestr(archive_name, payload)
            inner_manifest = {
                "schema_version": 1,
                "workflow_bindings": bindings,
                "files": file_manifest,
            }
            archive.writestr("manifest.json", _canonical_json(inner_manifest))

        plaintext = zip_buffer.getvalue()
        salt = os.urandom(16)
        nonce = os.urandom(12)
        kdf = {
            "name": "argon2id",
            "time_cost": ARGON2_TIME_COST,
            "memory_cost_kib": ARGON2_MEMORY_KIB,
            "parallelism": ARGON2_PARALLELISM,
        }
        header = {
            "version": PACK_VERSION,
            "kdf": {
                **kdf,
                "salt": base64.b64encode(salt).decode("ascii"),
            },
            "cipher": {
                "name": "aes-256-gcm",
                "nonce": base64.b64encode(nonce).decode("ascii"),
            },
            "plaintext_sha256": _sha256_bytes(plaintext),
        }
        header_bytes = _canonical_json(header)
        key = _derive_key(passphrase, salt, kdf)
        ciphertext = AESGCM(key).encrypt(nonce, plaintext, header_bytes)

        output = Path(destination).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        part_path = output.with_name(f"{output.name}.part")
        with part_path.open("wb") as stream:
            stream.write(PACK_MAGIC)
            stream.write(struct.pack(">I", len(header_bytes)))
            stream.write(header_bytes)
            stream.write(ciphertext)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(part_path, output)

        return {
            "path": str(output),
            "sha256": _sha256_file(output),
            "workflow_count": len(files),
            "binding_count": len(bindings),
        }
    except WorkflowPackError:
        raise
    except Exception as exc:
        print(f"[COMFY_INSTALL][PACK] 워크플로우 팩 생성 실패: {exc}")
        traceback.print_exc()
        raise WorkflowPackError(f"워크플로우 팩 생성 실패: {exc}") from exc


def _read_encrypted_pack(pack_path: Path) -> tuple[dict, bytes, bytes]:
    with pack_path.open("rb") as stream:
        magic = stream.read(len(PACK_MAGIC))
        if magic != PACK_MAGIC:
            raise WorkflowPackError("지원하지 않는 워크플로우 팩 형식입니다.")
        raw_header_size = stream.read(4)
        if len(raw_header_size) != 4:
            raise WorkflowPackError("워크플로우 팩 헤더 길이가 손상되었습니다.")
        header_size = struct.unpack(">I", raw_header_size)[0]
        if not 1 <= header_size <= MAX_HEADER_BYTES:
            raise WorkflowPackError(
                f"워크플로우 팩 헤더 크기가 유효하지 않습니다: {header_size}"
            )
        header_bytes = stream.read(header_size)
        if len(header_bytes) != header_size:
            raise WorkflowPackError("워크플로우 팩 헤더가 잘렸습니다.")
        ciphertext = stream.read()
    if len(ciphertext) < 16:
        raise WorkflowPackError("워크플로우 팩 암호문이 비어 있거나 손상되었습니다.")
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except Exception as exc:
        print(f"[COMFY_INSTALL][PACK] 워크플로우 팩 헤더 JSON 오류: {exc}")
        traceback.print_exc()
        raise WorkflowPackError("워크플로우 팩 헤더를 해석할 수 없습니다.") from exc
    return header, header_bytes, ciphertext


def _validate_header(header: dict) -> tuple[bytes, bytes, dict]:
    if not isinstance(header, dict) or header.get("version") != PACK_VERSION:
        raise WorkflowPackError(
            f"지원하지 않는 워크플로우 팩 버전입니다: {header.get('version')!r}"
        )
    kdf = header.get("kdf")
    cipher = header.get("cipher")
    if not isinstance(kdf, dict) or kdf.get("name") != "argon2id":
        raise WorkflowPackError("워크플로우 팩 KDF가 Argon2id가 아닙니다.")
    if not isinstance(cipher, dict) or cipher.get("name") != "aes-256-gcm":
        raise WorkflowPackError("워크플로우 팩 암호가 AES-256-GCM이 아닙니다.")
    try:
        salt = base64.b64decode(kdf["salt"], validate=True)
        nonce = base64.b64decode(cipher["nonce"], validate=True)
    except Exception as exc:
        print(f"[COMFY_INSTALL][PACK] salt/nonce 디코딩 실패: {exc}")
        traceback.print_exc()
        raise WorkflowPackError("워크플로우 팩 salt 또는 nonce가 손상되었습니다.") from exc
    if len(salt) < 16 or len(nonce) != 12:
        raise WorkflowPackError("워크플로우 팩 salt 또는 nonce 길이가 유효하지 않습니다.")
    for key in ("time_cost", "memory_cost_kib", "parallelism"):
        value = kdf.get(key)
        if not isinstance(value, int) or value <= 0:
            raise WorkflowPackError(f"워크플로우 팩 KDF 값이 유효하지 않습니다: {key}")
    return salt, nonce, kdf


def extract_workflow_pack(
    pack_path: str | os.PathLike[str],
    target_dir: str | os.PathLike[str],
    passphrase: str,
) -> ExtractedWorkflowPack:
    """암호화 팩을 검증하고 지정 폴더에 원자적으로 복원한다."""

    source = Path(pack_path).resolve()
    target = Path(target_dir).resolve()
    try:
        if not source.is_file():
            raise WorkflowPackError(f"워크플로우 팩 파일이 없습니다: {source}")
        header, header_bytes, ciphertext = _read_encrypted_pack(source)
        salt, nonce, kdf = _validate_header(header)
        key = _derive_key(passphrase, salt, kdf)
        try:
            plaintext = AESGCM(key).decrypt(nonce, ciphertext, header_bytes)
        except InvalidTag as exc:
            print(
                "[COMFY_INSTALL][PACK] 워크플로우 팩 인증 실패: "
                "키가 틀렸거나 파일이 변조되었습니다."
            )
            raise WorkflowPackError(
                "워크플로우 팩 키가 틀렸거나 파일이 변조되었습니다."
            ) from exc

        actual_plaintext_hash = _sha256_bytes(plaintext)
        if actual_plaintext_hash != header.get("plaintext_sha256"):
            raise WorkflowPackError("워크플로우 팩 평문 SHA-256 검증에 실패했습니다.")

        with zipfile.ZipFile(io.BytesIO(plaintext), mode="r") as archive:
            names = archive.namelist()
            for name in names:
                _validate_archive_name(name)
            if "manifest.json" not in names:
                raise WorkflowPackError("워크플로우 팩 내부 manifest.json이 없습니다.")
            try:
                inner_manifest = json.loads(
                    archive.read("manifest.json").decode("utf-8")
                )
            except Exception as exc:
                print(f"[COMFY_INSTALL][PACK] 내부 manifest 읽기 실패: {exc}")
                traceback.print_exc()
                raise WorkflowPackError(
                    "워크플로우 팩 내부 manifest를 읽을 수 없습니다."
                ) from exc

            if (
                not isinstance(inner_manifest, dict)
                or inner_manifest.get("schema_version") != 1
            ):
                raise WorkflowPackError("워크플로우 팩 내부 manifest 버전이 잘못되었습니다.")
            bindings = inner_manifest.get("workflow_bindings")
            files = inner_manifest.get("files")
            if not isinstance(bindings, dict) or not isinstance(files, dict):
                raise WorkflowPackError("워크플로우 팩 내부 manifest 형식이 잘못되었습니다.")

            verified_payloads: dict[str, bytes] = {}
            hashes: dict[str, str] = {}
            for archive_name, file_info in files.items():
                safe_name = _validate_archive_name(str(archive_name))
                if not safe_name.startswith("workflows/"):
                    raise WorkflowPackError(
                        f"허용되지 않은 워크플로우 팩 파일입니다: {safe_name}"
                    )
                if not isinstance(file_info, dict):
                    raise WorkflowPackError(
                        f"워크플로우 파일 manifest가 객체가 아닙니다: {safe_name}"
                    )
                payload = archive.read(safe_name)
                actual_hash = _sha256_bytes(payload)
                if actual_hash != file_info.get("sha256"):
                    raise WorkflowPackError(
                        f"워크플로우 SHA-256 검증 실패: {safe_name}"
                    )
                if len(payload) != file_info.get("size"):
                    raise WorkflowPackError(f"워크플로우 크기 검증 실패: {safe_name}")
                try:
                    workflow_json = json.loads(payload.decode("utf-8"))
                except Exception as exc:
                    print(
                        "[COMFY_INSTALL][PACK] 복호화 워크플로우 JSON 오류: "
                        f"file={safe_name}, error={exc}"
                    )
                    traceback.print_exc()
                    raise WorkflowPackError(
                        f"복호화된 워크플로우 JSON이 잘못되었습니다: {safe_name}"
                    ) from exc
                if not isinstance(workflow_json, dict):
                    raise WorkflowPackError(
                        f"복호화된 워크플로우가 JSON 객체가 아닙니다: {safe_name}"
                    )
                verified_payloads[safe_name] = payload
                hashes[safe_name] = actual_hash

            clean_bindings: dict[str, str] = {}
            for config_key, archive_name in bindings.items():
                safe_name = _validate_archive_name(str(archive_name))
                if safe_name not in verified_payloads:
                    raise WorkflowPackError(
                        f"워크플로우 바인딩 대상이 없습니다: {config_key!r}"
                    )
                clean_bindings[str(config_key)] = safe_name

        target.mkdir(parents=True, exist_ok=True)
        for archive_name, payload in verified_payloads.items():
            filename = Path(archive_name).name
            destination = target / filename
            if destination.exists():
                if not destination.is_file():
                    raise WorkflowPackError(
                        "워크플로우 설치 대상에 파일이 아닌 경로가 있습니다: "
                        f"{destination}"
                    )
                existing = destination.read_bytes()
                if existing == payload:
                    print(
                        "[COMFY_INSTALL][PACK] 동일한 기존 워크플로우 재사용: "
                        f"{destination}"
                    )
                    continue
                raise WorkflowPackError(
                    "기존 워크플로우 JSON이 팩 내용과 달라 덮어쓰지 않습니다. "
                    f"수동 백업/이동 후 다시 시도하세요: {destination}"
                )
            part_path = destination.with_name(f"{destination.name}.part")
            try:
                with part_path.open("wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(part_path, destination)
            except Exception as exc:
                print(
                    "[COMFY_INSTALL][PACK] 워크플로우 원자적 저장 실패: "
                    f"destination={destination}, part={part_path}, error={exc}"
                )
                traceback.print_exc()
                raise

        return ExtractedWorkflowPack(
            target_dir=target,
            workflow_bindings={
                key: str(target / Path(archive_name).name)
                for key, archive_name in clean_bindings.items()
            },
            workflow_hashes={
                str(target / Path(name).name): digest
                for name, digest in hashes.items()
            },
            pack_sha256=_sha256_file(source),
        )
    except WorkflowPackError:
        raise
    except Exception as exc:
        print(f"[COMFY_INSTALL][PACK] 워크플로우 팩 복원 실패: {exc}")
        traceback.print_exc()
        raise WorkflowPackError(f"워크플로우 팩 복원 실패: {exc}") from exc
