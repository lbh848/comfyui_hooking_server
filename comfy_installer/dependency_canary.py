from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import sys
import time
import traceback
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Iterator


PUBLIC_PYPI_SIMPLE = "https://pypi.org/simple"


class DependencyDriftCanaryError(RuntimeError):
    """The live-index dependency canary could not establish a clean install."""


def _configure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception as exc:
            print(
                "[DEPENDENCY_CANARY] UTF-8 console configuration failed: "
                f"stream={stream_name}, error={exc}"
            )
            traceback.print_exc()
            raise DependencyDriftCanaryError(
                f"Could not configure {stream_name} for UTF-8: {exc}"
            ) from exc


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _prepare_test_root(*, candidate_root: Path, test_root: Path) -> Path:
    candidate = candidate_root.resolve()
    target = test_root.resolve()
    if target == candidate or _is_within(target, candidate) or _is_within(
        candidate, target
    ):
        message = (
            "Dependency canary test root must be separate from the candidate: "
            f"candidate={candidate}, test_root={target}"
        )
        print(f"[DEPENDENCY_CANARY] {message}")
        raise DependencyDriftCanaryError(message)
    if target.exists():
        if not target.is_dir():
            message = f"Dependency canary test root is not a directory: {target}"
            print(f"[DEPENDENCY_CANARY] {message}")
            raise DependencyDriftCanaryError(message)
        existing = list(target.iterdir())
        if existing:
            message = (
                "Dependency canary requires a new or empty test root: "
                f"test_root={target}, existing={[item.name for item in existing[:10]]}"
            )
            print(f"[DEPENDENCY_CANARY] {message}")
            raise DependencyDriftCanaryError(message)
    else:
        target.mkdir(parents=True)
    return target


@contextlib.contextmanager
def _live_public_index_environment(test_root: Path) -> Iterator[dict[str, str]]:
    cache_root = test_root / "cache" / "uv"
    temp_root = test_root / "temp"
    cache_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    managed_keys = {
        "PIP_CONFIG_FILE",
        "PIP_EXTRA_INDEX_URL",
        "PIP_FIND_LINKS",
        "PIP_INDEX_URL",
        "PIP_NO_CACHE_DIR",
        "PIP_NO_INDEX",
        "PYTHONIOENCODING",
        "PYTHONUTF8",
        "TEMP",
        "TMP",
        "TMPDIR",
        "UV_CACHE_DIR",
        "UV_DEFAULT_INDEX",
        "UV_EXCLUDE_NEWER",
        "UV_EXTRA_INDEX_URL",
        "UV_FIND_LINKS",
        "UV_INDEX",
        "UV_INDEX_URL",
        "UV_NO_CONFIG",
        "UV_NO_INDEX",
        "UV_OFFLINE",
    }
    previous = {key: os.environ.get(key) for key in managed_keys}
    try:
        for key in managed_keys:
            os.environ.pop(key, None)
        os.environ.update(
            {
                "PIP_CONFIG_FILE": os.devnull,
                "PIP_INDEX_URL": PUBLIC_PYPI_SIMPLE,
                "PIP_NO_CACHE_DIR": "1",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
                "TEMP": str(temp_root),
                "TMP": str(temp_root),
                "TMPDIR": str(temp_root),
                "UV_CACHE_DIR": str(cache_root),
                "UV_DEFAULT_INDEX": PUBLIC_PYPI_SIMPLE,
                "UV_NO_CONFIG": "1",
            }
        )
        yield {
            "public_index": PUBLIC_PYPI_SIMPLE,
            "uv_cache_dir": str(cache_root),
            "temp_dir": str(temp_root),
        }
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_candidate_api(candidate_root: Path) -> SimpleNamespace:
    root = candidate_root.resolve()
    sys.path.insert(0, str(root))
    try:
        import comfy_installer
        from comfy_installer.dependency_installer import (
            create_comfy_venv,
            install_node_dependencies,
            install_python_dependencies,
        )
        from comfy_installer.downloader import ResumableDownloader
        from comfy_installer.execution_profile import custom_nodes_for_profile
        from comfy_installer.install_modes import effective_gpu_profile
        from comfy_installer.manager_dependencies import install_manager_dependencies
        from comfy_installer.manifest import load_install_manifest
        from comfy_installer.node_installer import install_custom_nodes
        from comfy_installer.operations import run_command
        from comfy_installer.source_installer import install_comfy_source
    except Exception as exc:
        print(
            "[DEPENDENCY_CANARY] Candidate installer import failed: "
            f"candidate={root}, error={exc}"
        )
        traceback.print_exc()
        raise DependencyDriftCanaryError(
            f"Could not import candidate installer from {root}: {exc}"
        ) from exc

    imported = Path(comfy_installer.__file__).resolve()
    if not _is_within(imported, root):
        message = (
            "Dependency canary imported installer code from the wrong candidate: "
            f"expected_root={root}, imported={imported}"
        )
        print(f"[DEPENDENCY_CANARY] {message}")
        raise DependencyDriftCanaryError(message)
    return SimpleNamespace(
        ResumableDownloader=ResumableDownloader,
        create_comfy_venv=create_comfy_venv,
        custom_nodes_for_profile=custom_nodes_for_profile,
        effective_gpu_profile=effective_gpu_profile,
        install_comfy_source=install_comfy_source,
        install_custom_nodes=install_custom_nodes,
        install_manager_dependencies=install_manager_dependencies,
        install_node_dependencies=install_node_dependencies,
        install_python_dependencies=install_python_dependencies,
        load_install_manifest=load_install_manifest,
        run_command=run_command,
    )


def _diagnostic_command(
    *, api: SimpleNamespace, command: list[str], cwd: Path, label: str
) -> dict[str, object]:
    try:
        lines = api.run_command(
            command,
            cwd=cwd,
            cancel_event=Event(),
            log=None,
            timeout=300,
        )
        return {"ok": True, "lines": lines}
    except Exception as exc:
        print(
            f"[DEPENDENCY_CANARY] Diagnostic command failed: label={label}, "
            f"command={command!r}, error={exc}"
        )
        traceback.print_exc()
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _write_report(path: Path, report: dict[str, object]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(
            "[DEPENDENCY_CANARY] Report write failed: "
            f"path={path}, error={exc}"
        )
        traceback.print_exc()
        raise DependencyDriftCanaryError(
            f"Could not write dependency canary report: {path}: {exc}"
        ) from exc


def run_dependency_drift_canary(
    *,
    candidate_root: Path,
    manifest_path: Path,
    test_root: Path,
    profile_id: str,
    install_mode: str,
    report_path: Path | None = None,
    api: SimpleNamespace | None = None,
) -> dict[str, object]:
    candidate = candidate_root.resolve()
    target = _prepare_test_root(
        candidate_root=candidate,
        test_root=test_root,
    )
    report_file = (
        report_path.resolve()
        if report_path is not None
        else target / "dependency-canary-report.json"
    )
    manifest_file = manifest_path.resolve()
    candidate_api = api or _load_candidate_api(candidate)
    started_at = dt.datetime.now(dt.timezone.utc)
    started_clock = time.monotonic()
    report: dict[str, object] = {
        "status": "running",
        "started_at": started_at.isoformat(),
        "candidate_root": str(candidate),
        "manifest_path": str(manifest_file),
        "test_root": str(target),
        "profile_id": profile_id,
        "install_mode": install_mode,
    }
    primary_error: BaseException | None = None
    python: Path | None = None
    comfy_root = target / "comfy"
    log_lines: list[str] = []

    def log(message: str) -> None:
        text = str(message)
        log_lines.append(text)
        print(f"[DEPENDENCY_CANARY] {text}")

    try:
        with _live_public_index_environment(target) as index_evidence:
            report["index"] = index_evidence
            manifest = candidate_api.load_install_manifest(manifest_file)
            report["manifest_sha256"] = manifest.sha256
            profiles = {
                str(profile["id"]): profile
                for profile in manifest.python["gpu_profiles"]
            }
            if profile_id not in profiles:
                message = (
                    "Unknown dependency canary profile: "
                    f"profile_id={profile_id}, available={sorted(profiles)}"
                )
                print(f"[DEPENDENCY_CANARY] {message}")
                raise DependencyDriftCanaryError(message)
            profile = candidate_api.effective_gpu_profile(
                profiles[profile_id], install_mode
            )
            report["profile_kind"] = str(profile.get("kind"))

            cancel_event = Event()
            backup_root = target / "backups"
            downloader = candidate_api.ResumableDownloader()
            candidate_api.install_comfy_source(
                destination=comfy_root,
                repository=str(manifest.comfy["repository"]),
                ref=str(manifest.comfy["ref"]),
                cancel_event=cancel_event,
                log=log,
                requirements_dir=backup_root,
            )
            python = candidate_api.create_comfy_venv(
                comfy_root=comfy_root,
                python_version=str(manifest.python["version"]),
                cancel_event=cancel_event,
                log=log,
                requirements_dir=backup_root,
            )
            candidate_api.install_python_dependencies(
                comfy_root=comfy_root,
                python=python,
                python_manifest=manifest.python,
                gpu_profile=profile,
                downloader=downloader,
                cancel_event=cancel_event,
                cache_root=comfy_root / ".installer-cache",
                log=log,
                progress=None,
            )
            candidate_api.install_manager_dependencies(
                comfy_root=comfy_root,
                python=python,
                cancel_event=cancel_event,
                log=log,
            )
            selected_nodes = candidate_api.custom_nodes_for_profile(
                manifest.custom_nodes,
                cpu_only=profile.get("kind") == "cpu",
            )
            report["custom_node_count"] = len(selected_nodes)
            node_paths = candidate_api.install_custom_nodes(
                nodes=selected_nodes,
                comfy_root=comfy_root,
                downloader=downloader,
                cancel_event=cancel_event,
                log=log,
                progress=None,
                requirements_dir=backup_root,
            )
            installed_requirements = candidate_api.install_node_dependencies(
                comfy_root=comfy_root,
                python=python,
                node_paths=node_paths,
                compatibility_packages=list(
                    manifest.python["compatibility_packages"]
                ),
                cancel_event=cancel_event,
                log=log,
            )
            report["node_requirements"] = installed_requirements
            report["status"] = "PASS"
    except BaseException as exc:
        primary_error = exc
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(
            "[DEPENDENCY_CANARY] Live dependency installation failed: "
            f"profile={profile_id}, install_mode={install_mode}, error={exc}"
        )
        traceback.print_exc()
    finally:
        report["log_tail"] = log_lines[-200:]
        if python is not None and python.is_file() and comfy_root.is_dir():
            report["pip_freeze"] = _diagnostic_command(
                api=candidate_api,
                command=["uv", "pip", "freeze", "--python", str(python)],
                cwd=comfy_root,
                label="pip_freeze",
            )
            report["pip_tree"] = _diagnostic_command(
                api=candidate_api,
                command=["uv", "pip", "tree", "--python", str(python)],
                cwd=comfy_root,
                label="pip_tree",
            )
            report["pip_check"] = _diagnostic_command(
                api=candidate_api,
                command=["uv", "pip", "check", "--python", str(python)],
                cwd=comfy_root,
                label="pip_check",
            )
        else:
            print(
                "[DEPENDENCY_CANARY] Package diagnostics skipped: "
                f"python={python}, comfy_root={comfy_root}"
            )
            report["package_diagnostics"] = {
                "ok": False,
                "reason": "target Python was not created",
            }
        finished_at = dt.datetime.now(dt.timezone.utc)
        report["finished_at"] = finished_at.isoformat()
        report["elapsed_seconds"] = round(time.monotonic() - started_clock, 3)
        _write_report(report_file, report)
        print(
            "[DEPENDENCY_CANARY] Report written: "
            f"status={report['status']}, path={report_file}"
        )

    if primary_error is not None:
        raise DependencyDriftCanaryError(
            "Live dependency canary failed; inspect the report and command output: "
            f"{report_file}"
        ) from primary_error
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_candidate = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Install the candidate's live Python dependencies in production order "
            "inside a new disposable Comfy target."
        )
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=default_candidate,
        help="Candidate project root containing comfy_installer/.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Candidate or extracted pack install_manifest.json.",
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        required=True,
        help="A new or empty disposable directory outside the candidate.",
    )
    parser.add_argument("--profile-id", required=True)
    parser.add_argument(
        "--install-mode",
        choices=("standard", "nvidia_compatibility", "cloud_only"),
        default="standard",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="UTF-8 JSON report path; defaults inside --test-root.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _configure_utf8_stdio()
    args = _parse_args(argv)
    candidate = args.candidate_root.resolve()
    manifest = (
        args.manifest.resolve()
        if args.manifest is not None
        else candidate
        / "comfy_installer"
        / "resources"
        / "install_manifest.json"
    )
    try:
        run_dependency_drift_canary(
            candidate_root=candidate,
            manifest_path=manifest,
            test_root=args.test_root,
            profile_id=args.profile_id,
            install_mode=args.install_mode,
            report_path=args.report,
        )
        return 0
    except Exception as exc:
        print(f"[DEPENDENCY_CANARY] FAILED: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
