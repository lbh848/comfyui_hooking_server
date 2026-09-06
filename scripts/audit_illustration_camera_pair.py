"""Live paired Anima audit with a replayed scene plan and fixed seeds.

Builds the current DETAIL prompts from the original interaction-direction case
while replaying the recorded baseline CALL2-PLAN.  It then renders baseline and
current descriptors for slots 0 and 3 through the real process_prompt/Anima
path, using the same seed within each pair.  All writes are isolated below the
chosen output directory.

Importing this module makes no external calls.  Running it calls configured LLM
DETAIL routes and the configured local Comfy ``v3_anima`` workflow.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import re
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

import audit_illustration_wardrobe_e2e as wardrobe  # noqa: E402
from modes import illustration_context_pipeline  # noqa: E402
from modes.illust_prompt_builder import IllustPromptBuilder  # noqa: E402


DEFAULT_BASELINE = (
    PROJECT_ROOT
    / ".work"
    / "illustration_wardrobe_e2e"
    / "20260905_225819_413331"
    / "repeat_01"
    / "built_result.json"
)
DEFAULT_STORY = PROJECT_ROOT / ".work" / "wardrobe_controls" / "interaction_direction.json"
PAIR_SLOTS = (0, 3)
PAIR_SEEDS = (17426001, 17426002)


def _item_for_slot(items: list[dict[str, Any]], slot: int) -> tuple[int, dict[str, Any]]:
    matches = [(index, item) for index, item in enumerate(items) if item.get("slot") == slot]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one descriptor for slot {slot}, found {len(matches)}")
    return matches[0]


def _submitted_loras(render: dict[str, Any]) -> list[dict[str, Any]]:
    submissions = render.get("captured_comfy_submissions") or []
    result: list[dict[str, Any]] = []
    for submission in submissions:
        for raw_payload in submission.get("lora_payloads") or []:
            try:
                payload = json.loads(raw_payload)
                result.extend(
                    copy.deepcopy(item)
                    for item in payload.get("list") or []
                    if isinstance(item, dict)
                )
            except Exception:
                print(f"[CAMERA_PAIR] submitted LoRA payload parse failed: {raw_payload!r}")
                traceback.print_exc()
                raise
    return result


async def _render_one(
    *, server: Any, run_dir: Path, pair_dir: Path, descriptor: dict[str, Any],
    runtime: dict[str, Any], seed: int, label: str,
) -> dict[str, Any]:
    captured: list[dict[str, Any]] = []
    original_positive_builder = IllustPromptBuilder.build_positive_prompt
    original_submit = server.submit_to_real_comfy

    def fixed_positive_builder(self, *builder_args, **builder_kwargs) -> str:
        positive = original_positive_builder(self, *builder_args, **builder_kwargs)
        replaced, count = re.subn(
            r"(?m)(^\[SEED\]\s*\n)\d+\s*$",
            rf"\g<1>{seed}",
            positive,
            count=1,
        )
        if count != 1:
            raise RuntimeError(f"{label} could not replace the generated [SEED] value")
        return replaced

    async def capture_submit(prompt: dict, *submit_args, **submit_kwargs):
        prompt_copy = copy.deepcopy(prompt)
        submission_number = len(captured) + 1
        wardrobe._write_json(
            pair_dir / f"submitted_workflow_{submission_number:02d}.json",
            prompt_copy,
        )
        seed_values: list[int] = []
        models: list[dict[str, Any]] = []
        lora_payloads: list[str] = []
        for node_id, node in prompt_copy.items():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs") or {}
            value = str(inputs.get("value") or "")
            match = re.search(r"(?m)^\[SEED\]\s*\n(\d+)\s*$", value)
            if match:
                seed_values.append(int(match.group(1)))
                lora_match = re.search(
                    r"(?ms)^\[LORA_DATA\]\s*\n(.*?)\n\[FACE_LORA_ACTIVATE\]",
                    value,
                )
                if lora_match:
                    lora_payloads.append(lora_match.group(1).strip())
            class_type = str(node.get("class_type") or "")
            for input_name in ("unet_name", "ckpt_name", "model_name"):
                if inputs.get(input_name):
                    models.append({
                        "node_id": str(node_id), "class_type": class_type,
                        "input": input_name, "value": inputs[input_name],
                    })
        captured.append({
            "seed_values": seed_values,
            "models": models,
            "lora_payloads": lora_payloads,
            "submitted_node_count": len(prompt_copy),
        })
        return await original_submit(prompt, *submit_args, **submit_kwargs)

    IllustPromptBuilder.build_positive_prompt = fixed_positive_builder
    server.submit_to_real_comfy = capture_submit
    try:
        _indices, generated = await wardrobe._render_items(
            server=server,
            run_dir=run_dir,
            repeat_dir=pair_dir,
            items=[descriptor],
            runtime=runtime,
            image_count=-1,
            execution_target="local",
        )
    finally:
        IllustPromptBuilder.build_positive_prompt = original_positive_builder
        server.submit_to_real_comfy = original_submit
    if len(generated) != 1:
        raise RuntimeError(f"{label} produced {len(generated)} render records")
    result = generated[0]
    result["label"] = label
    result["requested_seed"] = seed
    result["captured_comfy_submissions"] = captured
    if result.get("status") == "ok":
        if not captured or any(
            request.get("seed_values") != [seed]
            for request in captured
        ):
            raise RuntimeError(f"{label} did not submit exactly one fixed Comfy seed {seed}")
        params = copy.deepcopy(result.get("generation_params") or {})
        params.update({
            "seed": seed,
            "submitted_models": captured[-1]["models"],
            "submitted_lora_payloads": captured[-1]["lora_payloads"],
        })
        result["generation_params"] = params
    return result


async def _main_async(args: argparse.Namespace) -> int:
    baseline_path = Path(args.baseline).resolve()
    story_path = Path(args.story).resolve()
    baseline = wardrobe._read_json(baseline_path)
    baseline_items = list(baseline.get("items") or [])
    recorded_plan = str(baseline.get("call2_plan_output") or "").strip()
    if not recorded_plan:
        raise RuntimeError(f"baseline has no call2_plan_output: {baseline_path}")
    for slot in PAIR_SLOTS:
        _item_for_slot(baseline_items, slot)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else PROJECT_ROOT / ".work" / "illustration_camera_pair" / timestamp
    )

    original_call = illustration_context_pipeline._call_pipeline_llm
    replay_events: list[dict[str, Any]] = []

    async def replay_plan(call_name: str, messages: list[dict], *call_args, **call_kwargs):
        if call_name == "CALL2-PLAN":
            event = {
                "call_name": call_name,
                "task_key": "illustration_call2_plan",
                "state": "skipped_replayed",
                "source": str(baseline_path),
                "external_call_made": False,
                "raw_response": recorded_plan,
                "input_message_count": len(messages),
                "note": "Process-local deterministic PLAN replay; not an API/LB Details record.",
            }
            replay_events.append(event)
            print(f"[CAMERA_PAIR] skipped live CALL2-PLAN; replay source={baseline_path}", flush=True)
            return recorded_plan
        return await original_call(call_name, messages, *call_args, **call_kwargs)

    illustration_context_pipeline._call_pipeline_llm = replay_plan
    build_args = argparse.Namespace(
        output_dir=str(run_dir), images=0, repeat=1, render_from="",
        baseline_backup="", story_file=str(story_path), check_only=False,
    )
    try:
        build_code = await wardrobe._main_async(build_args)
    finally:
        illustration_context_pipeline._call_pipeline_llm = original_call
    if build_code != 0:
        raise RuntimeError(f"current DETAIL build failed with exit code {build_code}")
    wardrobe._write_json(run_dir / "plan_replay_provenance.json", replay_events)

    after_path = run_dir / "repeat_01" / "built_result.json"
    after = wardrobe._read_json(after_path)
    after_items = list(after.get("items") or [])
    if str(after.get("call2_plan_output") or "").strip() != recorded_plan:
        raise RuntimeError("after build did not preserve the recorded CALL2-PLAN response")

    import server
    config = wardrobe._read_json(PROJECT_ROOT / "config.json")
    runtime = server._capture_illustration_runtime_snapshot(config)
    if runtime.get("provider") != "comfy" or runtime.get("illustration_workflow_type") != "v3_anima":
        raise RuntimeError(
            "paired evaluation requires configured provider=comfy and "
            "illustration_workflow_type=v3_anima; got "
            f"provider={runtime.get('provider')!r}, "
            f"workflow={runtime.get('illustration_workflow_type')!r}"
        )

    results: list[dict[str, Any]] = []
    for pair_number, (slot, seed) in enumerate(zip(PAIR_SLOTS, PAIR_SEEDS), start=1):
        before_index, before_item = _item_for_slot(baseline_items, slot)
        after_index, after_item = _item_for_slot(after_items, slot)
        pair_dir = run_dir / f"pair_{pair_number:02d}_slot_{slot}"
        pair_dir.mkdir(parents=True, exist_ok=False)
        before_dir = pair_dir / "before"
        after_dir = pair_dir / "after"
        before_dir.mkdir()
        after_dir.mkdir()
        before_result = await _render_one(
            server=server, run_dir=run_dir, pair_dir=before_dir,
            descriptor=before_item, runtime=runtime, seed=seed, label="before",
        )
        after_result = await _render_one(
            server=server, run_dir=run_dir, pair_dir=after_dir,
            descriptor=after_item, runtime=runtime, seed=seed, label="after",
        )
        before_loras = _submitted_loras(before_result)
        after_loras = _submitted_loras(after_result)
        if before_result.get("status") == "ok" and not any(
            float(item.get("str")) == 0.9 for item in before_loras if item.get("str") is not None
        ):
            raise RuntimeError(f"slot {slot} baseline submission lacks the expected 0.9 character LoRA")
        if after_result.get("status") == "ok" and not any(
            float(item.get("str")) == 0.9 for item in after_loras if item.get("str") is not None
        ):
            raise RuntimeError(f"slot {slot} after submission lacks the expected 0.9 character LoRA")
        before_models = before_result.get("generation_params", {}).get("submitted_models")
        after_models = after_result.get("generation_params", {}).get("submitted_models")
        if before_result.get("status") == after_result.get("status") == "ok":
            if not before_models or before_models != after_models:
                raise RuntimeError(f"slot {slot} before/after submitted model evidence differs")
            if before_loras != after_loras:
                raise RuntimeError(f"slot {slot} before/after submitted LoRA evidence differs")
        results.append({
            "slot": slot,
            "seed": seed,
            "baseline_descriptor_index": before_index,
            "after_descriptor_index": after_index,
            "scene_plan_equal": before_item.get("scene_brief") == after_item.get("scene_brief"),
            "submitted_models_equal": before_models == after_models,
            "submitted_loras_equal": before_loras == after_loras,
            "before_submitted_loras": before_loras,
            "after_submitted_loras": after_loras,
            "before": before_result,
            "after": after_result,
        })

    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "real Anima before/after camera prompt evaluation",
        "production_data_modified": False,
        "baseline_built_result": str(baseline_path),
        "after_built_result": str(after_path),
        "story_file": str(story_path),
        "plan_replay": replay_events,
        "configured_provider": runtime["provider"],
        "configured_workflow": runtime["illustration_workflow_type"],
        "comparison_limits": [
            "The recorded PLAN is identical, while current DETAIL and downstream prompt assembly may differ.",
            "Each before/after pair shares its seed; the two semantic cases intentionally use different seeds.",
            "Baseline descriptors were built under the earlier code/config snapshot, but rendering uses the current process_prompt/card/LoRA/config runtime for both sides.",
            "Comfy custom nodes and hardware may retain nondeterminism beyond the submitted workflow seed.",
        ],
        "pairs": results,
    }
    wardrobe._write_json(run_dir / "paired_report.json", report)
    passed = all(
        pair[side].get("status") == "ok"
        for pair in results for side in ("before", "after")
    )
    print(f"[CAMERA_PAIR] complete: report={run_dir / 'paired_report.json'}", flush=True)
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--story", default=str(DEFAULT_STORY))
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    try:
        return asyncio.run(_main_async(args))
    except Exception as exc:
        print(f"[CAMERA_PAIR] failed: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
