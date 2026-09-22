from __future__ import annotations

import copy
import hashlib
import io
import json
import zipfile
from pathlib import Path
from threading import Event

from PIL import Image

from comfy_installer.image_comparison_diagnostic import (
    _inject_prompt_and_seed,
    _normalize_prompt_seed,
    run_image_comparison_diagnostic,
)


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(output, format="PNG")
    return output.getvalue()


def _api_workflow() -> dict:
    return {
        "1": {
            "class_type": "PrimitiveStringMultiline",
            "inputs": {"value": "[ANIMA_ARTIST]\ncomfy artist\n[SEED]\n-1\n[END]"},
            "_meta": {"title": "긍정프롬프트"},
        },
        "2": {
            "class_type": "PrimitiveStringMultiline",
            "inputs": {"value": "low quality"},
            "_meta": {"title": "부정프롬프트"},
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {"seed": 1, "noise_seed": 1},
            "_meta": {"title": "Sampler"},
        },
    }


def test_prompt_seed_and_api_seed_are_fixed_without_mutating_source() -> None:
    source = _api_workflow()
    source_before = copy.deepcopy(source)
    prompt = "[ANIMA_ARTIST]\nartist\n[SEED]\n-1\n[END]"

    normalized = _normalize_prompt_seed(prompt, 1234)
    submitted = _inject_prompt_and_seed(
        source,
        positive=normalized,
        negative="bad",
        seed=1234,
    )

    assert source == source_before
    assert "[SEED]\n1234" in submitted["1"]["inputs"]["value"]
    assert submitted["2"]["inputs"]["value"] == "bad"
    assert submitted["3"]["inputs"]["seed"] == 1234
    assert submitted["3"]["inputs"]["noise_seed"] == 1234


def test_comparison_diagnostic_runs_four_cases_and_keeps_source_unchanged(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "current-character-maker-workflow.json"
    source_payload = {
        "nodes": [
            {
                "id": 1,
                "type": "PrimitiveStringMultiline",
                "properties": {"cnr_id": "comfy-core", "ver": "1.0.0"},
            }
        ],
        "links": [],
    }
    source_bytes = json.dumps(source_payload, ensure_ascii=False).encode("utf-8")
    source_path.write_bytes(source_bytes)
    base_workflow = _api_workflow()
    calls: list[dict] = []
    colors = {
        "direct_comfy_prompt": (1, 2, 3),
        "program_comfy_prompt": (1, 2, 3),
        "direct_character_maker_prompt": (4, 5, 6),
        "program_character_maker_prompt": (4, 5, 6),
    }

    def production_call(request: dict) -> dict:
        calls.append(copy.deepcopy(request))
        if request["action"] == "comparison_snapshot":
            return {
                "source_path": str(source_path),
                "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "generation_workflow": "asset",
                "workflow_profile": "anima_only",
                "converted_workflow": copy.deepcopy(base_workflow),
                "direct_positive": base_workflow["1"]["inputs"]["value"],
                "direct_negative": "low quality",
                "direct_runtime": {"execution_target": "local", "port": 8188},
                "program_runtime": {"execution_target": "local", "port": 8188},
                "dependency_sets": {
                    "direct": {"missing_node_classes": [], "nodes": []},
                    "program": {"missing_node_classes": [], "nodes": []},
                },
                "summary": {"source_path": str(source_path)},
            }
        if request["action"] == "comparison_generate":
            submitted = request.get("workflow")
            if request["route"] == "program":
                submitted = _inject_prompt_and_seed(
                    base_workflow,
                    positive=request["positive"],
                    negative=request["negative"],
                    seed=request["seed"],
                )
            return {
                "image_bytes": _png_bytes(colors[request["case"]]),
                "submitted_workflow": submitted,
                "queue_item_id": f"queue-{request['case']}",
                "comfy_log": "",
                "runtime": {"port": 8188},
            }
        if request["action"] == "comparison_cleanup":
            return {"success": True, "removed": False}
        raise AssertionError(request)

    result = run_image_comparison_diagnostic(
        project_root=tmp_path,
        cancel_event=Event(),
        request={
            "positive": "[ANIMA_ARTIST]\ncharacter maker artist\n[SEED]\n-1\n[END]",
            "negative": "low quality",
            "seed": 9876,
            "width": 700,
            "height": 1024,
        },
        production_call=production_call,
        log=lambda _message, _level: None,
        progress=lambda _payload: None,
    )

    assert source_path.read_bytes() == source_bytes
    assert result["source_modified"] is False
    assert result["incomplete"] is False
    assert [case["status"] for case in result["cases"]] == ["success"] * 4
    assert len([call for call in calls if call["action"] == "comparison_generate"]) == 4
    assert calls[-1]["action"] == "comparison_cleanup"
    archive = Path(result["archive_path"])
    assert archive.is_file()
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        assert source_path.name in {Path(name).name for name in names if name.startswith("source/")}
        assert "report.md" in names
        assert "analysis.json" in names
        assert len([name for name in names if name.endswith("/image.png")]) == 4
        summary = json.loads(bundle.read("summary.json").decode("utf-8"))
    assert len(summary["cases"]) == 4
    assert "ANIMA_ARTIST" in summary["conclusions"][0]


def test_comparison_diagnostic_records_one_failure_and_continues(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "workflow.json"
    source_path.write_text("{}\n", encoding="utf-8")
    base_workflow = _api_workflow()
    generated_cases: list[str] = []

    def production_call(request: dict) -> dict:
        if request["action"] == "comparison_snapshot":
            return {
                "source_path": str(source_path),
                "generation_workflow": "asset",
                "workflow_profile": "ilxl",
                "converted_workflow": copy.deepcopy(base_workflow),
                "direct_positive": base_workflow["1"]["inputs"]["value"],
                "direct_negative": "bad",
                "direct_runtime": {"execution_target": "local", "port": 8188},
                "program_runtime": {"execution_target": "local", "port": 8188},
                "dependency_sets": {},
                "summary": {},
            }
        if request["action"] == "comparison_generate":
            generated_cases.append(request["case"])
            if request["case"] == "program_comfy_prompt":
                raise RuntimeError("representative program failure")
            submitted = request.get("workflow") or _inject_prompt_and_seed(
                base_workflow,
                positive=request["positive"],
                negative=request["negative"],
                seed=request["seed"],
            )
            return {
                "image_bytes": _png_bytes((8, 9, 10)),
                "submitted_workflow": submitted,
                "queue_item_id": request["case"],
            }
        return {"success": True}

    result = run_image_comparison_diagnostic(
        project_root=tmp_path,
        cancel_event=Event(),
        request={
            "positive": "[SEED]\n12\n[END]",
            "negative": "bad",
            "seed": 12,
            "width": 700,
            "height": 1024,
        },
        production_call=production_call,
        log=lambda _message, _level: None,
        progress=lambda _payload: None,
    )

    assert len(generated_cases) == 4
    assert result["incomplete"] is True
    assert [case["status"] for case in result["cases"]].count("failed") == 1
    assert Path(result["archive_path"]).is_file()


def test_comparison_diagnostic_cancellation_keeps_partial_archive(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "workflow.json"
    source_path.write_text("{}\n", encoding="utf-8")
    base_workflow = _api_workflow()
    cleanup_calls: list[dict] = []

    def production_call(request: dict) -> dict:
        if request["action"] == "comparison_snapshot":
            return {
                "source_path": str(source_path),
                "generation_workflow": "asset",
                "workflow_profile": "ilxl",
                "converted_workflow": copy.deepcopy(base_workflow),
                "direct_positive": base_workflow["1"]["inputs"]["value"],
                "direct_negative": "bad",
                "direct_runtime": {"execution_target": "local", "port": 8188},
                "program_runtime": {"execution_target": "local", "port": 8188},
                "dependency_sets": {},
                "summary": {},
            }
        if request["action"] == "comparison_cleanup":
            cleanup_calls.append(request)
            return {"success": True}
        raise AssertionError("cancelled diagnostics must not begin a generation case")

    cancelled = Event()
    cancelled.set()
    result = run_image_comparison_diagnostic(
        project_root=tmp_path,
        cancel_event=cancelled,
        request={
            "positive": "[SEED]\n12\n[END]",
            "negative": "bad",
            "seed": 12,
            "width": 700,
            "height": 1024,
        },
        production_call=production_call,
        log=lambda _message, _level: None,
        progress=lambda _payload: None,
    )

    assert result["cancelled"] is True
    assert result["incomplete"] is True
    assert result["cases"] == []
    assert len(cleanup_calls) == 1
    assert Path(result["archive_path"]).is_file()
