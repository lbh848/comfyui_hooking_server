"""Render a supplied supplement against recorded camera-only controls, without LLM calls."""
from __future__ import annotations

import argparse
import asyncio
import copy
from datetime import datetime
from pathlib import Path
import traceback

import audit_illustration_camera_only as camera
import audit_illustration_camera_pair as pair
import audit_illustration_wardrobe_e2e as wardrobe


async def run(args: argparse.Namespace) -> int:
    source_dir = Path(args.source_run).resolve()
    controls = wardrobe._read_json(source_dir / "camera_only_report.json")
    requests = wardrobe._read_json(source_dir / "requests.json")
    descriptor = copy.deepcopy(requests["variants"]["b_from_side"])
    old = descriptor["supplement"]
    new = args.supplement
    descriptor["supplement"] = new
    marker = "[SUPPLEMENT]\n"
    prefix, previous = descriptor["raw_positive"].rsplit(marker, 1)
    if previous != old:
        raise RuntimeError("RAW supplement differs from descriptor; cannot isolate substitution")
    descriptor["raw_positive"] = prefix + marker + new
    run_dir = wardrobe.PROJECT_ROOT / ".work" / "illustration_supplement_only" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    setup = argparse.Namespace(output_dir=str(run_dir), images=0, repeat=1,
        render_from="", baseline_backup="", story_file="", check_only=True)
    if await wardrobe._main_async(setup) != 0:
        raise RuntimeError("Isolated runtime setup failed")
    import server
    config = wardrobe._read_json(wardrobe.PROJECT_ROOT / "config.json")
    runtime = server._capture_illustration_runtime_snapshot(config)
    wardrobe._write_json(run_dir / "request.json", descriptor)
    results = []
    for control in controls["pairs"]:
        seed = control["seed"]
        target = run_dir / f"seed_{seed}"
        target.mkdir()
        print(f"[SUPPLEMENT_ONLY] generating seed={seed}", flush=True)
        generated = await pair._render_one(server=server, run_dir=run_dir,
            pair_dir=target, descriptor=descriptor, runtime=runtime, seed=seed,
            label="supplement_only")
        before = control["b"]
        before_dir = (source_dir / before["path"]).parent
        before_workflows = camera._submitted_workflows(before_dir)
        after_workflows = camera._submitted_workflows(target)
        normalize = lambda value: camera._replace_strings(value, ((old, "<SUPPLEMENT>"), (new, "<SUPPLEMENT>")))
        matched = (bool(before_workflows and after_workflows)
            and normalize(before_workflows) == normalize(after_workflows)
            and before["final_negative"] == generated.get("final_negative")
            and generated.get("status") == "ok")
        results.append({"seed": seed, "control_image": str(source_dir / before["path"]),
            "after": generated, "workflow_equal_except_supplement": matched})
        print(f"[SUPPLEMENT_ONLY] seed={seed} condition_match={matched}", flush=True)
    report = {"source_run": str(source_dir), "old_supplement": old,
        "new_supplement": new, "external_llm_calls": False,
        "production_data_modified": False, "pairs": results,
        "verification_pass": all(p["workflow_equal_except_supplement"] for p in results),
        "limits": "Recorded controls reused; workflow equality does not exclude hardware nondeterminism. This is a renderer probe, not validation of LLM instruction improvements."}
    wardrobe._write_json(run_dir / "supplement_only_report.json", report)
    print(f"[SUPPLEMENT_ONLY] complete: {run_dir}; pass={report['verification_pass']}", flush=True)
    return 0 if report["verification_pass"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--supplement", required=True)
    try:
        raise SystemExit(asyncio.run(run(parser.parse_args())))
    except Exception as exc:
        print(f"[SUPPLEMENT_ONLY] failed: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise SystemExit(1)
