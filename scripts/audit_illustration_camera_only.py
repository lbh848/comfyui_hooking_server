"""Render a camera-only A/B for the recorded slot-3 DETAIL descriptor.

The source descriptor is reused without an LLM call.  For each fixed seed this
script renders the original ``upper body, straight-on`` request and a copy in
which only the camera prefix of the RAW ``[SETUP]`` line is changed to
``upper body, from side``.  The wardrobe audit's check-only path prepares the
same isolated history, logs, workflow cache, and usage-accounting environment;
the camera-pair audit's renderer supplies and verifies the real Comfy seed.

All outputs are written below a new ``.work/illustration_camera_only`` run
directory unless ``--output-dir`` selects another new directory.  Importing
this module makes no external calls.  Running with ``--check-only`` makes no
LLM or image call; a normal run makes six local Comfy image calls and no LLM
calls.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import re
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

import audit_illustration_camera_pair as camera_pair  # noqa: E402
import audit_illustration_wardrobe_e2e as wardrobe  # noqa: E402


DEFAULT_SOURCE = (
    PROJECT_ROOT
    / ".work"
    / "illustration_camera_pair"
    / "20260905_235635_496617"
    / "repeat_01"
    / "built_result.json"
)
SOURCE_SLOT = 3
CAMERA_A = "upper body, straight-on"
CAMERA_B = "upper body, from side"
PAIR_SEEDS = (17426002, 17426003, 17426004)
RAW_SECTION_NAMES = ("CHAT", "SLOT", "SPEAK", "NAME", "SETUP", "CHAR", "SUPPLEMENT")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _replace_setup_camera_prefix(
    raw_positive: str, *, old_camera: str, new_camera: str
) -> str:
    marker = "[SETUP]\n"
    marker_count = raw_positive.count(marker)
    if marker_count != 1:
        raise RuntimeError(
            f"expected exactly one {marker.strip()} marker, found {marker_count}"
        )
    line_start = raw_positive.index(marker) + len(marker)
    line_end = raw_positive.find("\n", line_start)
    if line_end < 0:
        line_end = len(raw_positive)
    setup_line = raw_positive[line_start:line_end]
    expected_prefix = f"{old_camera},"
    if not setup_line.startswith(expected_prefix):
        raise RuntimeError(
            "slot-3 SETUP does not begin with the expected camera prefix: "
            f"expected={expected_prefix!r}, actual={setup_line!r}"
        )
    changed_line = f"{new_camera}{setup_line[len(old_camera):]}"
    changed = raw_positive[:line_start] + changed_line + raw_positive[line_end:]
    if changed == raw_positive:
        raise RuntimeError("camera-prefix replacement made no RAW positive change")
    return changed


def _raw_sections(raw_positive: str) -> dict[str, str]:
    marker_pattern = re.compile(
        r"(?mi)^\[(CHAT|SLOT|SPEAK|NAME|SETUP|CHAR|SUPPLEMENT)\](?:\r?\n)?"
    )
    matches = list(marker_pattern.finditer(raw_positive))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group(1).upper()
        if name in sections:
            raise RuntimeError(f"duplicate RAW section marker: {name}")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_positive)
        sections[name] = raw_positive[match.end():end]
    missing = [name for name in RAW_SECTION_NAMES if name not in sections]
    if missing:
        raise RuntimeError(f"missing RAW sections: {missing}")
    return sections


def _descriptor_without_camera_fields(descriptor: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(descriptor)
    result.pop("camera", None)
    result.pop("raw_positive", None)
    return result


def _build_variants(
    source_descriptor: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if str(source_descriptor.get("camera") or "") != CAMERA_A:
        raise RuntimeError(
            "slot-3 descriptor camera differs from the recorded A camera: "
            f"{source_descriptor.get('camera')!r}"
        )
    raw_a = str(source_descriptor.get("raw_positive") or "")
    if not raw_a:
        raise RuntimeError("slot-3 descriptor has an empty raw_positive")
    raw_b = _replace_setup_camera_prefix(
        raw_a, old_camera=CAMERA_A, new_camera=CAMERA_B
    )

    variant_a = copy.deepcopy(source_descriptor)
    variant_b = copy.deepcopy(source_descriptor)
    variant_b["camera"] = CAMERA_B
    variant_b["raw_positive"] = raw_b

    sections_a = _raw_sections(raw_a)
    sections_b = _raw_sections(raw_b)
    non_setup_sections_equal = all(
        sections_a[name] == sections_b[name]
        for name in RAW_SECTION_NAMES
        if name != "SETUP"
    )
    setup_a = sections_a["SETUP"]
    setup_b = sections_b["SETUP"]
    expected_raw_b = _replace_setup_camera_prefix(
        raw_a, old_camera=CAMERA_A, new_camera=CAMERA_B
    )
    verification = {
        "source_variant_a_exact": variant_a == source_descriptor,
        "descriptor_equal_except_camera_and_raw_positive": (
            _descriptor_without_camera_fields(variant_a)
            == _descriptor_without_camera_fields(variant_b)
        ),
        "characters_equal": variant_a.get("characters") == variant_b.get("characters"),
        "scene_equal": variant_a.get("scene") == variant_b.get("scene"),
        "supplement_equal": (
            variant_a.get("supplement") == variant_b.get("supplement")
        ),
        "speak_equal": variant_a.get("speak") == variant_b.get("speak"),
        "raw_negative_equal": (
            variant_a.get("raw_negative") == variant_b.get("raw_negative")
        ),
        "raw_positive_camera_only": raw_b == expected_raw_b,
        "raw_non_setup_sections_exact": non_setup_sections_equal,
        "setup_a_has_expected_prefix": setup_a.startswith(f"{CAMERA_A},"),
        "setup_b_has_expected_prefix": setup_b.startswith(f"{CAMERA_B},"),
        "setup_suffix_exact": setup_a[len(CAMERA_A):] == setup_b[len(CAMERA_B):],
        "raw_positive_a_sha256": _sha256_text(raw_a),
        "raw_positive_b_sha256": _sha256_text(raw_b),
        "raw_negative_sha256": _sha256_text(
            str(variant_a.get("raw_negative") or "")
        ),
    }
    required = [
        value
        for key, value in verification.items()
        if not key.endswith("_sha256")
    ]
    verification["passed"] = all(required)
    if not verification["passed"]:
        raise RuntimeError(
            f"camera-only descriptor construction verification failed: {verification}"
        )
    return variant_a, variant_b, verification


def _submitted_seed_values(render: dict[str, Any]) -> list[list[int]]:
    submissions = render.get("captured_comfy_submissions") or []
    if not submissions:
        print(
            f"[CAMERA_ONLY] no captured Comfy submissions: label={render.get('label')!r}, "
            f"status={render.get('status')!r}",
            flush=True,
        )
        return []
    return [list(submission.get("seed_values") or []) for submission in submissions]


def _submitted_workflows(render_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(render_dir.glob("submitted_workflow_*.json"))
    if not paths:
        print(
            f"[CAMERA_ONLY] no submitted workflow artifacts: directory={render_dir}",
            flush=True,
        )
        return []
    return [wardrobe._read_json(path) for path in paths]


def _replace_strings(value: Any, replacements: tuple[tuple[str, str], ...]) -> Any:
    if isinstance(value, str):
        result = value
        for old, new in replacements:
            result = result.replace(old, new)
        return result
    if isinstance(value, list):
        return [_replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_strings(item, replacements)
            for key, item in value.items()
        }
    return value


def _normalize_workflow(
    workflow: dict[str, Any], *, normalize_camera: bool, normalize_seed: bool
) -> dict[str, Any]:
    replacements: list[tuple[str, str]] = []
    if normalize_camera:
        replacements.extend(((CAMERA_A, "<CAMERA>"), (CAMERA_B, "<CAMERA>")))
    normalized = _replace_strings(workflow, tuple(replacements))
    if normalize_seed:
        def replace_seed(value: Any) -> Any:
            if isinstance(value, str):
                return re.sub(
                    r"(?m)(^\[SEED\]\s*\n)\d+(\s*$)",
                    r"\g<1><SEED>\g<2>",
                    value,
                )
            if isinstance(value, list):
                return [replace_seed(item) for item in value]
            if isinstance(value, dict):
                return {key: replace_seed(item) for key, item in value.items()}
            return value

        normalized = replace_seed(normalized)
    return normalized


def _safe_loras(render: dict[str, Any]) -> list[dict[str, Any]]:
    if render.get("status") != "ok":
        print(
            f"[CAMERA_ONLY] LoRA evidence unavailable for failed render: "
            f"label={render.get('label')!r}, error={render.get('error')!r}",
            flush=True,
        )
        return []
    return camera_pair._submitted_loras(render)


def _has_point_nine_lora(loras: list[dict[str, Any]]) -> bool:
    for item in loras:
        strength = item.get("str")
        if strength is None:
            continue
        try:
            if float(strength) == 0.9:
                return True
        except Exception:
            print(f"[CAMERA_ONLY] invalid submitted LoRA strength: {strength!r}")
            traceback.print_exc()
            return False
    return False


async def _render_safely(
    *, server: Any, run_dir: Path, render_dir: Path,
    descriptor: dict[str, Any], runtime: dict[str, Any], seed: int, label: str,
) -> dict[str, Any]:
    try:
        return await camera_pair._render_one(
            server=server,
            run_dir=run_dir,
            pair_dir=render_dir,
            descriptor=descriptor,
            runtime=runtime,
            seed=seed,
            label=label,
        )
    except Exception as exc:
        print(
            f"[CAMERA_ONLY] render raised: label={label}, seed={seed}, "
            f"error={type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()
        return {
            "label": label,
            "requested_seed": seed,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _pair_verification(
    *, seed: int, a: dict[str, Any], b: dict[str, Any],
    a_dir: Path, b_dir: Path,
) -> dict[str, Any]:
    a_seed_values = _submitted_seed_values(a)
    b_seed_values = _submitted_seed_values(b)
    a_loras = _safe_loras(a)
    b_loras = _safe_loras(b)
    a_models = (a.get("generation_params") or {}).get("submitted_models") or []
    b_models = (b.get("generation_params") or {}).get("submitted_models") or []
    a_workflows = _submitted_workflows(a_dir)
    b_workflows = _submitted_workflows(b_dir)
    workflows_camera_only = bool(a_workflows and b_workflows) and (
        len(a_workflows) == len(b_workflows)
        and [
            _normalize_workflow(item, normalize_camera=True, normalize_seed=False)
            for item in a_workflows
        ]
        == [
            _normalize_workflow(item, normalize_camera=True, normalize_seed=False)
            for item in b_workflows
        ]
    )
    final_a = str(a.get("final_positive") or "")
    final_b = str(b.get("final_positive") or "")
    final_positive_camera_only = bool(final_a and final_b and final_a != final_b) and (
        final_a.replace(CAMERA_A, "<CAMERA>")
        == final_b.replace(CAMERA_B, "<CAMERA>")
    )
    verification = {
        "both_images_ok": a.get("status") == b.get("status") == "ok",
        "requested_seed_equal": (
            a.get("requested_seed") == b.get("requested_seed") == seed
        ),
        "a_submitted_seed_values": a_seed_values,
        "b_submitted_seed_values": b_seed_values,
        "actual_seed_equal_and_exact": bool(a_seed_values and b_seed_values)
        and all(values == [seed] for values in [*a_seed_values, *b_seed_values]),
        "submitted_models_nonempty": bool(a_models and b_models),
        "submitted_models_equal": a_models == b_models,
        "a_submitted_loras": a_loras,
        "b_submitted_loras": b_loras,
        "submitted_loras_equal": bool(a_loras and b_loras) and a_loras == b_loras,
        "a_has_character_lora_strength_0_9": _has_point_nine_lora(a_loras),
        "b_has_character_lora_strength_0_9": _has_point_nine_lora(b_loras),
        "final_negative_equal": bool(a.get("final_negative"))
        and a.get("final_negative") == b.get("final_negative"),
        "final_positive_camera_only": final_positive_camera_only,
        "submitted_workflow_counts": {"a": len(a_workflows), "b": len(b_workflows)},
        "submitted_workflows_camera_only": workflows_camera_only,
    }
    required_keys = (
        "both_images_ok",
        "requested_seed_equal",
        "actual_seed_equal_and_exact",
        "submitted_models_nonempty",
        "submitted_models_equal",
        "submitted_loras_equal",
        "a_has_character_lora_strength_0_9",
        "b_has_character_lora_strength_0_9",
        "final_negative_equal",
        "final_positive_camera_only",
        "submitted_workflows_camera_only",
    )
    verification["passed"] = all(verification[key] for key in required_keys)
    return verification


async def _main_async(args: argparse.Namespace) -> int:
    source_path = Path(args.source).resolve()
    source = wardrobe._read_json(source_path)
    source_items = list(source.get("items") or [])
    source_index, source_descriptor = camera_pair._item_for_slot(
        source_items, SOURCE_SLOT
    )
    variant_a, variant_b, raw_verification = _build_variants(source_descriptor)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else PROJECT_ROOT / ".work" / "illustration_camera_only" / timestamp
    )
    setup_args = argparse.Namespace(
        output_dir=str(run_dir),
        images=0,
        repeat=1,
        render_from="",
        baseline_backup="",
        story_file="",
        check_only=True,
    )
    setup_code = await wardrobe._main_async(setup_args)
    if setup_code != 0:
        raise RuntimeError(f"wardrobe isolation setup failed with exit code {setup_code}")
    setup_report_path = run_dir / "report.json"
    setup_report = wardrobe._read_json(setup_report_path)
    if setup_report.get("external_calls_made") is not False:
        raise RuntimeError(
            "wardrobe isolation setup did not attest external_calls_made=false"
        )

    shutil.copy2(source_path, run_dir / "source_built_result.json")
    requests = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_built_result": str(source_path),
        "source_descriptor_index": source_index,
        "source_slot": SOURCE_SLOT,
        "external_llm_calls": False,
        "seeds": list(PAIR_SEEDS),
        "variants": {
            "a_straight_on": variant_a,
            "b_from_side": variant_b,
        },
        "render_requests": [
            {"seed": seed, "variant": label, "camera": camera}
            for seed in PAIR_SEEDS
            for label, camera in (
                ("a_straight_on", CAMERA_A),
                ("b_from_side", CAMERA_B),
            )
        ],
        "raw_verification": raw_verification,
    }
    wardrobe._write_json(run_dir / "requests.json", requests)

    import server

    config = wardrobe._read_json(PROJECT_ROOT / "config.json")
    runtime = server._capture_illustration_runtime_snapshot(config)
    configured_target = wardrobe._configured_execution_target(config)
    prerequisites = {
        "wardrobe_isolation_setup_pass": True,
        "wardrobe_setup_external_calls_made": setup_report.get(
            "external_calls_made"
        ),
        "provider": runtime.get("provider"),
        "illustration_workflow_type": runtime.get("illustration_workflow_type"),
        "configured_execution_target": configured_target,
        "renderer_execution_target": "local",
        "provider_is_comfy": runtime.get("provider") == "comfy",
        "workflow_is_v3_anima": (
            runtime.get("illustration_workflow_type") == "v3_anima"
        ),
    }
    prerequisites["passed"] = bool(
        prerequisites["provider_is_comfy"]
        and prerequisites["workflow_is_v3_anima"]
    )
    base_report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "slot-3 camera-only real Anima A/B with three matched seeds",
        "status": "ready" if args.check_only else "running",
        "production_data_modified": False,
        "external_llm_calls": False,
        "external_image_calls": False,
        "source_built_result": str(source_path),
        "source_descriptor_index": source_index,
        "source_slot": SOURCE_SLOT,
        "isolation_setup_report": str(setup_report_path),
        "requests_file": str(run_dir / "requests.json"),
        "cameras": {"a": CAMERA_A, "b": CAMERA_B},
        "seeds": list(PAIR_SEEDS),
        "raw_verification": raw_verification,
        "prerequisites": prerequisites,
        "pairs": [],
    }
    if not prerequisites["passed"]:
        base_report["status"] = "blocked_prerequisite"
        wardrobe._write_json(run_dir / "camera_only_report.json", base_report)
        print(
            "[CAMERA_ONLY] prerequisite failed: expected current "
            f"provider='comfy' and workflow='v3_anima', got "
            f"provider={runtime.get('provider')!r}, "
            f"workflow={runtime.get('illustration_workflow_type')!r}; "
            f"report={run_dir / 'camera_only_report.json'}",
            flush=True,
        )
        return 1

    if args.check_only:
        base_report["verification_pass"] = True
        wardrobe._write_json(run_dir / "camera_only_report.json", base_report)
        print(
            "[CAMERA_ONLY] render skipped: --check-only requested; "
            f"setup ready at {run_dir}",
            flush=True,
        )
        return 0

    pairs: list[dict[str, Any]] = []
    all_normalized_workflows: list[dict[str, Any]] = []
    for pair_number, seed in enumerate(PAIR_SEEDS, start=1):
        pair_dir = run_dir / f"pair_{pair_number:02d}_seed_{seed}"
        a_dir = pair_dir / "a_straight_on"
        b_dir = pair_dir / "b_from_side"
        a_dir.mkdir(parents=True, exist_ok=False)
        b_dir.mkdir(parents=True, exist_ok=False)
        wardrobe._write_json(a_dir / "request.json", {
            "label": "a_straight_on", "seed": seed, "descriptor": variant_a,
        })
        wardrobe._write_json(b_dir / "request.json", {
            "label": "b_from_side", "seed": seed, "descriptor": variant_b,
        })
        print(
            f"[CAMERA_ONLY] pair {pair_number}/{len(PAIR_SEEDS)} seed={seed} started",
            flush=True,
        )
        result_a = await _render_safely(
            server=server,
            run_dir=run_dir,
            render_dir=a_dir,
            descriptor=variant_a,
            runtime=runtime,
            seed=seed,
            label="a_straight_on",
        )
        result_b = await _render_safely(
            server=server,
            run_dir=run_dir,
            render_dir=b_dir,
            descriptor=variant_b,
            runtime=runtime,
            seed=seed,
            label="b_from_side",
        )
        verification = _pair_verification(
            seed=seed,
            a=result_a,
            b=result_b,
            a_dir=a_dir,
            b_dir=b_dir,
        )
        pair_workflows = [
            *_submitted_workflows(a_dir),
            *_submitted_workflows(b_dir),
        ]
        all_normalized_workflows.extend(
            _normalize_workflow(
                workflow, normalize_camera=True, normalize_seed=True
            )
            for workflow in pair_workflows
        )
        pairs.append({
            "pair_number": pair_number,
            "seed": seed,
            "a": result_a,
            "b": result_b,
            "verification": verification,
        })
        print(
            f"[CAMERA_ONLY] pair seed={seed} verification_pass={verification['passed']}",
            flush=True,
        )

    successful_images = sum(
        result.get("status") == "ok"
        for pair in pairs
        for result in (pair["a"], pair["b"])
    )
    workflows_equal_across_all_requests = bool(all_normalized_workflows) and all(
        workflow == all_normalized_workflows[0]
        for workflow in all_normalized_workflows[1:]
    )
    verification_pass = bool(
        raw_verification["passed"]
        and prerequisites["passed"]
        and successful_images == 6
        and len(all_normalized_workflows) == 6
        and workflows_equal_across_all_requests
        and all(pair["verification"]["passed"] for pair in pairs)
    )
    report = {
        **base_report,
        "status": "passed" if verification_pass else "failed",
        "external_image_calls": True,
        "expected_image_count": 6,
        "successful_image_count": successful_images,
        "submitted_workflow_count": len(all_normalized_workflows),
        "submitted_workflows_equal_after_camera_and_seed_normalization": (
            workflows_equal_across_all_requests
        ),
        "pairs": pairs,
        "verification_pass": verification_pass,
        "comparison_limits": [
            "A and B reuse the exact recorded post-DETAIL slot-3 descriptor; only the camera prefix of the RAW SETUP line changes.",
            "Each A/B pair uses one matched submitted seed; the three pairs intentionally use different seeds.",
            "The current process_prompt/card/LoRA/config runtime is used for all six requests.",
            "Local Comfy custom nodes and hardware may retain nondeterminism beyond the submitted workflow seed.",
        ],
    }
    report_path = run_dir / "camera_only_report.json"
    wardrobe._write_json(report_path, report)
    print(
        f"[CAMERA_ONLY] complete: verification_pass={verification_pass}, "
        f"report={report_path}",
        flush=True,
    )
    return 0 if verification_pass else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Prepare and verify isolation/requests without calling Comfy.",
    )
    args = parser.parse_args()
    try:
        return asyncio.run(_main_async(args))
    except Exception as exc:
        print(
            f"[CAMERA_ONLY] failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
