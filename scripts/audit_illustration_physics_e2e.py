"""Live, text-only E2E audit for illustration body visibility and composition.

The audit calls the production illustration planner/detail routes with synthetic
adult scenes. It writes only beneath ``.work/illustration_physics_e2e`` unless an
explicit output path is supplied. Semantic pass/fail remains a human judgment:
the script validates pipeline completion and preserves the full descriptors,
raw prompts, and LLM history needed for review without keyword matching.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

from modes import illustration_context_pipeline as pipeline
from modes import lighbd_service, llm_service


@dataclass(frozen=True)
class AuditCase:
    case_id: str
    subject_name: str
    narrative: str
    nsfw: bool
    expected_semantics: str


CASES = (
    AuditCase(
        case_id="uncovered_anonymous_male_pelvis",
        subject_name="Mira",
        nsfw=True,
        narrative=(
            "Mira, a thirty-two-year-old woman, and her consenting thirty-five-year-old "
            "male partner are both fully nude during vaginal intercourse on a bed. Show "
            "one physically possible instant from a close oblique side view: Mira and the "
            "anonymous man's connected lower bodies, his uncovered pelvis, and the point "
            "of penetration are all inside the frame. No clothing, hand, sheet, thigh, or "
            "foreground object blocks his pelvis or their contact. His face and most of his "
            "upper body may remain outside the frame."
        ),
        expected_semantics=(
            "The anonymous partner remains a connected, physically sufficient body region. "
            "Because his uncovered adult pelvis and contact are explicitly inside an "
            "unobstructed view, the final visible anatomy includes his penis and makes the "
            "contact unambiguous; cropping his face does not erase in-frame anatomy."
        ),
    ),
    AuditCase(
        case_id="renamed_isomorphic_uncovered_pelvis",
        subject_name="Rena",
        nsfw=True,
        narrative=(
            "Rena, a twenty-nine-year-old woman, and her consenting adult male lover are "
            "fully nude together on a sofa. Use a low three-quarter view of one coherent "
            "instant of vaginal intercourse. The lover's uncovered pelvic region and their "
            "genital contact are within the image boundaries and are not hidden by cushions, "
            "limbs, hands, clothing, or Rena's body. His head can be beyond the upper crop."
        ),
        expected_semantics=(
            "The same visibility result holds after names and wording change: the visible "
            "anonymous adult male pelvis includes the expected external genital anatomy and "
            "action-defining contact, without requiring his face or full body."
        ),
    ),
    AuditCase(
        case_id="opaque_coverage_control",
        subject_name="Nari",
        nsfw=True,
        narrative=(
            "Nari, a thirty-year-old woman in a tied white robe, embraces her adult male "
            "partner beside a pool. Frame both torsos and hips. He wears opaque, securely "
            "fastened swim shorts that completely cover his pelvis; neither garment is moved, "
            "opened, transparent, or removed. Their hands rest on each other's shoulders."
        ),
        expected_semantics=(
            "The male pelvis may be in frame, but opaque clothing remains a real cover. The "
            "output does not invent genital exposure, nudity, displaced clothing, or sexual "
            "contact merely because the physical visibility audit exists."
        ),
    ),
    AuditCase(
        case_id="physical_occlusion_control",
        subject_name="Sora",
        nsfw=True,
        narrative=(
            "Sora, a thirty-four-year-old woman, and her consenting adult male partner are "
            "nude while reclining together. Compose a side view in which a thick opaque "
            "blanket and Sora's nearer thigh fully block the man's pelvic anatomy from the "
            "camera, while his connected waist, abdomen, and legs make his position readable. "
            "Do not change the viewpoint or remove the occluders."
        ),
        expected_semantics=(
            "The connected body remains plausible, but anatomy hidden by the specified solid "
            "occluders is not forced into view. The image description respects depth order and "
            "does not silently move the camera or blanket to expose him."
        ),
    ),
    AuditCase(
        case_id="anonymous_forearms_only_control",
        subject_name="Hana",
        nsfw=False,
        narrative=(
            "Hana, a forty-year-old watchmaker, adjusts a tiny mechanism at her workbench. "
            "An anonymous adult assistant reaches in from outside the right edge and steadies "
            "the same watch with two bare hands. Only the assistant's hands and connected "
            "forearms are needed in frame; their elbows, torso, pelvis, and face stay outside "
            "the image. Keep each hand attached to the correct forearm and keep the tool clear."
        ),
        expected_semantics=(
            "The anonymous fragment is the smallest sufficient connected region: two hands "
            "with their own forearms, no floating or duplicated limbs, and no unnecessary "
            "expansion to the assistant's torso or full body."
        ),
    ),
    AuditCase(
        case_id="joint_ownership_and_depth_control",
        subject_name="Yuna",
        nsfw=False,
        narrative=(
            "Yuna, a thirty-six-year-old physiotherapist, guides an anonymous adult client's "
            "right arm through a shoulder stretch. Use a medium side view showing both upper "
            "bodies. Yuna's left hand supports the client's right elbow and her right hand "
            "supports his right wrist. Preserve continuous shoulders, elbows, wrists, and "
            "depth order so all four visible arms have one clear owner and no joint duplicates."
        ),
        expected_semantics=(
            "The pose is one physically possible contact graph: each arm has a unique owner, "
            "all joints connect in the stated order, and overlap does not create extra hands, "
            "detached limbs, or an impossible support relationship."
        ),
    ),
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except Exception:
        print(f"[ILLUST_PHYSICS_E2E] JSON read failed: path={path}", flush=True)
        traceback.print_exc()
        raise
    if not isinstance(value, dict):
        print(
            "[ILLUST_PHYSICS_E2E] JSON root must be an object: "
            f"path={path}, type={type(value).__name__}",
            flush=True,
        )
        raise ValueError(f"JSON object required: {path}")
    return value


def _profile_result(name: str) -> dict[str, Any]:
    result = pipeline._empty_profile_result()
    result["characters"] = [{
        "name": name,
        "in_history": False,
        "profile_timeline": [],
    }]
    result["current_characters"] = [{"name": name, "confidence": 1.0}]
    return result


def _reference(name: str) -> str:
    return (
        f"### {name}\n"
        f"-Name\n{name}\n"
        "-Appearance\n"
        "1woman, adult female, black hair, brown eyes\n"
        "-default_outfit\n"
        "dark blue long-sleeved dress"
    )


def _toggles(case: AuditCase) -> dict[str, Any]:
    return {
        "call1_backtranslate_enabled": False,
        "call1_enabled": False,
        "call2_parallel_enabled": True,
        "call2_parallel_max_concurrency": 1,
        "call2_parallel_slow_retry_enabled": False,
        "output_count_min": 1,
        "output_count_max": 1,
        "key_visual": False,
        "call3_enabled": False,
        "speak_enabled": False,
        "nsfw": case.nsfw,
        "supplement": True,
        "minimal_background_description": True,
        "character_limit": 1,
        "scene_mode": "manual",
        "prompt_format": "v3",
    }


async def _run_case(case: AuditCase, repeat: int) -> dict[str, Any]:
    run_id = f"{case.case_id}:r{repeat}"
    print(f"[ILLUST_PHYSICS_E2E] START {run_id}", flush=True)
    started = time.time()
    try:
        reference = _reference(case.subject_name)
        result = await pipeline.build_from_context(
            {
                "session_id": f"illustration_physics_e2e_{case.case_id}_{repeat}",
                "target_slotted": case.narrative + "\n\n[Slot 0]",
                "chats": [
                    {"role": "user", "data": "Illustrate this adult scene accurately."},
                    {"role": "char", "data": case.narrative},
                ],
            },
            _toggles(case),
            reference,
            extra_costume=(
                f"### {case.subject_name}\n-default_outfit\n"
                "dark blue long-sleeved dress"
            ),
            extra_names=case.subject_name,
            backtranslate_names=case.subject_name,
            pre_resolved_profile_result=_profile_result(case.subject_name),
            enable_multi_char_layout=False,
        )
        items = list(result.get("items") or [])
        if not items:
            raise RuntimeError("illustration pipeline completed without any item")
        elapsed = time.time() - started
        print(
            f"[ILLUST_PHYSICS_E2E] COMPLETE {run_id} "
            f"items={len(items)} elapsed={elapsed:.2f}s",
            flush=True,
        )
        return {
            "case": asdict(case),
            "repeat": repeat,
            "pipeline_pass": True,
            "manual_review_required": True,
            "items": items,
            "call2_output": result.get("call2_output", ""),
            "call2_plan_output": result.get("call2_plan_output", ""),
            "call2_detail_outputs": result.get("call2_detail_outputs", []),
            "call2_authority_audit": result.get("call2_authority_audit", {}),
            "call2_authority_audit_output": result.get(
                "call2_authority_audit_output",
                "",
            ),
            "call2_authority_audit_status": result.get(
                "call2_authority_audit_status",
                "",
            ),
            "call2_fallback_stage": result.get("call2_fallback_stage", ""),
            "call2_fallback_reason": result.get("call2_fallback_reason", ""),
            "llm_trace": result.get("llm_trace", []),
            "timing_seconds": round(elapsed, 3),
            "error": "",
        }
    except Exception as exc:
        print(
            f"[ILLUST_PHYSICS_E2E] CASE failed: run={run_id}, error={exc}",
            flush=True,
        )
        traceback.print_exc()
        return {
            "case": asdict(case),
            "repeat": repeat,
            "pipeline_pass": False,
            "manual_review_required": True,
            "items": [],
            "llm_trace": [],
            "timing_seconds": round(time.time() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _select_cases(case_ids: list[str]) -> list[AuditCase]:
    if not case_ids:
        return list(CASES)
    available = {case.case_id: case for case in CASES}
    unknown = sorted(set(case_ids) - set(available))
    if unknown:
        print(f"[ILLUST_PHYSICS_E2E] unknown cases: {unknown}", flush=True)
        raise ValueError(f"unknown cases: {unknown}")
    return [available[case_id] for case_id in case_ids]


def _redirect_audit_logs(run_dir: Path) -> None:
    """Keep both generic and LB-detailed LLM histories inside this audit run."""

    llm_log_dir = run_dir / "llm_logs"
    llm_service.LOG_DIR = str(llm_log_dir)
    llm_service.HISTORY_PATH = str(llm_log_dir / "llm_history.jsonl")
    llm_service.HISTORY_BACKUP_DIR = str(llm_log_dir / "backups")
    llm_service.HISTORY_BACKUP_PATH = str(
        llm_log_dir / "backups" / "llm_history.jsonl.bak"
    )

    lighbd_log_dir = run_dir / "lighbd_logs"
    lighbd_service.LOG_DIR = str(lighbd_log_dir)
    lighbd_service.LIGHBD_HISTORY_PATH = str(
        lighbd_log_dir / "lighbd_history.jsonl"
    )


def _load_process_local_llm_config(config: dict[str, Any], keys: dict[str, Any]) -> None:
    """Load deployed routing without echoing the complete secret-bearing config."""

    original_log = llm_service._llm_log

    def audit_safe_log(message: str) -> None:
        if str(message).startswith("설정 업데이트:"):
            print(
                "[ILLUST_PHYSICS_E2E] process-local LLM routing loaded",
                flush=True,
            )
            return
        original_log(message)

    llm_service._llm_log = audit_safe_log
    try:
        llm_service.update_config({**config, **keys})
    except Exception:
        print("[ILLUST_PHYSICS_E2E] LLM configuration load failed", flush=True)
        traceback.print_exc()
        raise
    finally:
        llm_service._llm_log = original_log


async def _main_async(args: argparse.Namespace) -> int:
    selected = _select_cases(args.cases)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = (
        Path(args.output).resolve()
        if args.output
        else PROJECT_ROOT
        / ".work"
        / "illustration_physics_e2e"
        / f"audit_{timestamp}"
        / "report.json"
    )
    if output_path.exists():
        print(
            f"[ILLUST_PHYSICS_E2E] refusing to overwrite report: path={output_path}",
            flush=True,
        )
        return 2
    run_dir = output_path.parent
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        print(f"[ILLUST_PHYSICS_E2E] run directory creation failed: {run_dir}", flush=True)
        traceback.print_exc()
        return 2

    _redirect_audit_logs(run_dir)
    config = _read_json(PROJECT_ROOT / "config.json")
    keys = _read_json(PROJECT_ROOT / "key" / "llm_keys.json")
    _load_process_local_llm_config(config, keys)

    jobs = [
        (case, repeat)
        for case in selected
        for repeat in range(1, args.repeats + 1)
    ]
    semaphore = asyncio.Semaphore(args.concurrency)

    async def run_limited(case: AuditCase, repeat: int) -> dict[str, Any]:
        async with semaphore:
            return await _run_case(case, repeat)

    print(
        f"[ILLUST_PHYSICS_E2E] cases={len(selected)} repeats={args.repeats} "
        f"jobs={len(jobs)} concurrency={args.concurrency}",
        flush=True,
    )
    results = await asyncio.gather(
        *(run_limited(case, repeat) for case, repeat in jobs)
    )
    passed = sum(result.get("pipeline_pass") is True for result in results)
    failed = len(results) - passed
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "production illustration CALL2 plan/detail text pipeline",
        "production_files_modified": False,
        "semantic_verdict_source": "manual review; no keyword-matching oracle",
        "prompt_contract": (
            "skeleton and joints -> continuous volumes -> coverage -> contact and "
            "occlusion -> camera crop -> final visible anatomy"
        ),
        "routes": {
            "plan": (config.get("llm_routing") or {}).get(
                "illustration_call2_plan",
                {},
            ),
            "detail": (config.get("llm_routing") or {}).get(
                "illustration_call2",
                {},
            ),
        },
        "summary": {"total": len(results), "passed": passed, "failed": failed},
        "results": results,
    }
    try:
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        print(f"[ILLUST_PHYSICS_E2E] report write failed: path={output_path}", flush=True)
        traceback.print_exc()
        return 2
    print(
        f"[ILLUST_PHYSICS_E2E] FINISHED pass={passed}/{len(results)} "
        f"failed={failed} report={output_path}",
        flush=True,
    )
    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="*",
        default=[],
        help="Run only listed case IDs; omit for the complete matrix.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Runs per selected case.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Maximum concurrent case pipelines in this process.",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.repeats < 1 or args.concurrency < 1:
        print(
            "[ILLUST_PHYSICS_E2E] repeats and concurrency must both be >= 1: "
            f"repeats={args.repeats}, concurrency={args.concurrency}",
            flush=True,
        )
        return 2
    try:
        return asyncio.run(_main_async(args))
    except Exception:
        print("[ILLUST_PHYSICS_E2E] unhandled execution failure", flush=True)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
