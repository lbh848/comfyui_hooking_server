import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import illustration_flow as flow


def layout_nodes(nodes, kind="illustration"):
    source = (Path(__file__).resolve().parents[1] / "frontend" / "illustration_flow.js").read_text(encoding="utf-8")
    layout_source = source[source.index("    function layoutNodes("):source.index("    function render()")]
    script = layout_source + """
const {nodes, kind} = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const layout = layoutNodes(nodes, kind);
console.log(JSON.stringify({...layout, positions: Object.fromEntries(layout.positions)}));
"""
    result = subprocess.run(
        ["node", "-e", script],
        input=json.dumps({"nodes": nodes, "kind": kind}),
        text=True, encoding="utf-8", capture_output=True, check=True,
    )
    return json.loads(result.stdout)


def layout_node(label, dependencies=(), **fields):
    return {"id": label, "label": label, "dependencies": list(dependencies), **fields}


def test_flow_dialogs_use_fixed_centering_and_ordered_layers():
    source = (Path(__file__).resolve().parents[1] / "frontend" / "illustration_flow.js").read_text(encoding="utf-8")

    def layer(selector: str) -> int:
        match = re.search(rf"{re.escape(selector)}\{{[^}}]*z-index:(\d+)", source)
        assert match, selector
        return int(match.group(1))

    assert ".if-modal{position:fixed;inset:0;" in source
    assert "margin:auto;" in source
    assert layer(".if-layer-backdrop") < layer(".if-modal")
    assert layer(".if-modal") < layer(".if-detail") < layer(".if-developer")
    assert "showModal()" not in source
    assert "flowBackdrop.hidden = false; modal.show();" in source


def test_flow_zoom_has_feedback_before_any_request():
    source = (Path(__file__).resolve().parents[1] / "frontend" / "illustration_flow.js").read_text(encoding="utf-8")
    assert "function updateZoomDisplay()" in source
    assert "현재 확대 비율 ${Math.round(scale * 100)}%" in source
    assert "updateZoomDisplay(); render();" in source


def test_flow_exposes_stop_button_and_cancel_endpoint():
    root = Path(__file__).resolve().parents[1]
    source = (root / "frontend" / "illustration_flow.js").read_text(encoding="utf-8")
    server_source = (root / "server.py").read_text(encoding="utf-8-sig")
    assert "stopButton = button('중단', cancelCurrentFlow);" in source
    assert "fetch('/api/illustration_flow/cancel'" in source
    assert "flow?.cancel_requested" in source
    assert "!terminal && cancelling ? '중단 중…' : '중단'" in source
    assert 'app.router.add_post("/api/illustration_flow/cancel", handle_illustration_flow_cancel)' in server_source

def test_flow_layout_reserves_columns_in_pipeline_order():
    columns = [
        ["CHARACTER-RESOLVE", "PROFILE-CONTEXT-CACHE", "PROFILE-RESOLVE 1/2", "PROFILE-RESOLVE 2/2"],
        ["CALL1", "CALL2-SELECT"],
        ["CALL2-PLAN 1/3", "CALL2-PLAN 2/3", "CALL2-PLAN 3/3"],
        ["CALL2-DETAIL", "CALL2-KEYVIS", "CALL3"],
        ["MULTI-CHAR_MASK", "ORIGINAL-ASSET", "ORIGINAL-RECOVERY"],
        ["SLOT-이미지 생성"],
    ]
    labels = [label for column in columns for label in column]
    # A serial graph is also displayed in the requested semantic columns.
    nodes = [layout_node(label, [labels[index - 1]] if index else []) for index, label in enumerate(labels)]
    for supplied in (nodes, list(reversed(nodes))):
        positions = layout_nodes(supplied)["positions"]
        for step, column in enumerate(columns, 1):
            assert [positions[label]["depth"] for label in column] == [step] * len(column)
            assert [positions[label]["y"] for label in column] == sorted(positions[label]["y"] for label in column)


def test_flow_layout_does_not_merge_late_original_asset_into_call2_plan_group():
    early = [layout_node("request", kind="request"), layout_node("ORIGINAL-ASSET", ["request"])]
    later = [
        layout_node("CALL2-PLAN 1/2", ["request"]),
        layout_node("CALL2-PLAN 2/2", ["request"]),
        layout_node("ORIGINAL-ASSET-RECOVERY", ["ORIGINAL-ASSET", "CALL2-PLAN 1/2"]),
    ]
    before = layout_nodes(early)["positions"]
    after = layout_nodes(early + later)["positions"]
    assert before["ORIGINAL-ASSET"] == after["ORIGINAL-ASSET"]
    assert after["ORIGINAL-ASSET"]["depth"] == after["ORIGINAL-ASSET-RECOVERY"]["depth"] == 5
    assert after["CALL2-PLAN 1/2"]["depth"] == after["CALL2-PLAN 2/2"]["depth"] == 3


def test_flow_layout_keeps_logical_stage_retries_and_images_in_one_column():
    nodes = [
        layout_node("CALL2-PLAN 1/2"),
        layout_node("CALL2-PLAN-REPAIR", ["CALL2-PLAN 1/2"]),
        layout_node("CALL2-DETAIL", ["CALL2-PLAN-REPAIR"]),
        layout_node("삽화 1/2 · slot 1", ["CALL2-DETAIL"], kind="image", layout_group="illustration_images"),
        layout_node("이미지 생성", ["삽화 1/2 · slot 1"], kind="stage", layout_group="illustration_generation"),
        layout_node("결과 반환", ["이미지 생성"], kind="result"),
    ]
    positions = layout_nodes(nodes)["positions"]
    assert positions["CALL2-PLAN 1/2"]["depth"] == positions["CALL2-PLAN-REPAIR"]["depth"] == 3
    assert positions["삽화 1/2 · slot 1"]["depth"] == positions["이미지 생성"]["depth"] == 6
    assert positions["결과 반환"]["depth"] > 6


def test_flow_layout_classifies_reserved_nodes_and_metadata_only_nodes():
    nodes = [
        layout_node("waiting-detail", call_name="CALL2-DETAIL 2/3"),
        layout_node("renamed-mask", task_key="illustration_multi_char_mask"),
        layout_node("renamed-generation", layout_group="illustration_generation"),
        layout_node("CALL3-CORRECTION", ["waiting-detail"], layout_order=-90),
    ]
    before = layout_nodes(nodes)["positions"]
    nodes[0].update(task_key="illustration_call2", layout_group="call2_detail")
    after = layout_nodes(nodes)["positions"]
    assert before == after
    assert [before[n["id"]]["depth"] for n in nodes] == [4, 5, 6, 4]


def test_pipeline_assigns_distinct_layout_groups_to_detail_and_authority_audit():
    root = Path(__file__).resolve().parents[1]
    source = (root / "modes" / "illustration_context_pipeline.py").read_text(encoding="utf-8")
    assert '"CALL2-DETAIL": ("call2_detail",' in source
    assert '"CALL2-AUTHORITY-AUDIT": (' in source
    assert '"call2_authority_audit",' in source
    assert "layout_group=(queue_subtask_group[0] if queue_subtask_group else task_key)" in source


def test_image_generator_exposes_layout_metadata_for_queue_merge():
    root = Path(__file__).resolve().parents[1]
    flow_source = (root / "illustration_flow.py").read_text(encoding="utf-8")
    server_source = (root / "server.py").read_text(encoding="utf-8")
    assert 'layout_group=None if is_root or is_review else "illustration_images"' in flow_source
    assert 'layout_group="illustration_generation"' in server_source


def test_flow_layout_honors_semantic_order_within_a_column():
    nodes = [
        layout_node("CALL3", layout_order=-100),
        layout_node("CALL2-KEYVIS"),
        layout_node("CALL2-DETAIL 10/12"),
        layout_node("CALL2-DETAIL 2/12"),
        layout_node("CALL2-DETAIL 1/12"),
    ]
    positions = layout_nodes(nodes)["positions"]
    assert sorted(positions, key=lambda label: positions[label]["y"]) == [
        "CALL2-DETAIL 1/12", "CALL2-DETAIL 2/12", "CALL2-DETAIL 10/12", "CALL2-KEYVIS", "CALL3",
    ]


def test_flow_layout_keeps_each_column_top_aligned():
    nodes = [layout_node("CALL1"), layout_node("CALL2-SELECT"), layout_node("CALL2-PLAN", ["CALL1", "CALL2-SELECT"])]
    positions = layout_nodes(nodes)["positions"]
    assert positions["CALL1"]["y"] == positions["CALL2-PLAN"]["y"]
    assert positions["CALL2-SELECT"]["y"] > positions["CALL1"]["y"]


@pytest.mark.parametrize("batch_numbers,detail_ids", [
    ((1, 2, 3), ("llm-30", "llm-20", "llm-10")),
    ((2, 10, 12), ("zeta", "alpha", "middle")),
])
def test_flow_layout_aligns_same_named_details_with_their_plans(batch_numbers, detail_ids):
    plans = [layout_node(f"CALL2-PLAN {number}/12") for number in batch_numbers]
    details = [layout_node("CALL2-DETAIL", [plan["id"]], id=detail_id,
                           layout_group="call2_detail", layout_order=0)
               for plan, detail_id in zip(plans, detail_ids)]
    nodes = plans + list(reversed(details)) + [layout_node("CALL2-KEYVIS"), layout_node("CALL3")]
    baseline = layout_nodes(nodes)["positions"]
    for supplied in (nodes, list(reversed(nodes)), nodes[::2] + nodes[1::2]):
        positions = layout_nodes(supplied)["positions"]
        assert positions == baseline
        rows = [positions[detail["id"]]["y"] for detail in details]
        assert rows == sorted(rows)
        for plan, detail in zip(plans, details):
            assert positions[plan["id"]]["y"] == positions[detail["id"]]["y"]
        assert positions["CALL2-KEYVIS"]["y"] > max(rows)
        assert positions["CALL3"]["y"] > positions["CALL2-KEYVIS"]["y"]
        assert len({(p["x"], p["y"]) for p in positions.values()}) == len(nodes)


def test_flow_layout_reserves_plan_rows_before_details_arrive_and_groups_split_batches():
    plans = [layout_node(f"CALL2-PLAN {number}/3") for number in (1, 2, 3)]
    first = layout_node("CALL2-DETAIL", [plans[2]["id"]], id="first-registered")
    early = layout_nodes(plans + [first])["positions"]
    assert early[first["id"]]["y"] == early[plans[2]["id"]]["y"]
    details = [
        layout_node("CALL2-DETAIL 2/2", [plans[0]["id"]], id="detail-second-part"),
        layout_node("CALL2-DETAIL 1/2", [plans[0]["id"]], id="detail-first-part"),
        layout_node("CALL2-DETAIL", [plans[1]["id"]], id="last-registered"),
        first,
    ]
    nodes = plans + details
    positions = layout_nodes(nodes)["positions"]
    assert positions == layout_nodes(list(reversed(nodes)))["positions"]
    assert positions["detail-first-part"]["y"] == positions[plans[0]["id"]]["y"]
    assert positions["detail-first-part"]["y"] < positions["detail-second-part"]["y"]
    assert positions["detail-second-part"]["y"] < positions[plans[1]["id"]]["y"]
    assert positions["last-registered"]["y"] == positions[plans[1]["id"]]["y"]
    assert positions[first["id"]]["y"] == positions[plans[2]["id"]]["y"]
    assert len({(p["x"], p["y"]) for p in positions.values()}) == len(nodes)


def test_flow_layout_leaves_aggregate_details_and_video_order_independent_of_plan_rows():
    plans = [layout_node(f"CALL2-PLAN {number}/2") for number in (1, 2)]
    aggregate = layout_node("CALL2-DETAIL", [plan["id"] for plan in plans])
    nodes = plans + [aggregate]
    positions = layout_nodes(nodes)["positions"]
    assert positions[aggregate["id"]]["y"] == positions[plans[0]["id"]]["y"]
    video = layout_nodes(nodes, kind="video")["positions"]
    assert video[aggregate["id"]]["depth"] > video[plans[0]["id"]]["depth"]


@pytest.mark.parametrize("slots,prefix", [((-1, 7, 20), "삽화"), ((0, 3, 12), "다른 작업")])
def test_flow_layout_aligns_each_illustration_across_mask_image_and_followup(slots, prefix):
    nodes = [layout_node("CALL3"), layout_node("ORIGINAL-ASSET")]
    masks, images, posts, reviews = [], [], [], []
    for slot in slots:
        mask = f"MULTI-CHAR-MASK slot={slot}"
        masks.append(mask)
        nodes.append(layout_node(mask, ["CALL3", "ORIGINAL-ASSET"]))
    for index, slot in enumerate(slots):
        image = f"{prefix} {index + 1}/{len(slots)} · slot {slot}"
        images.append(image)
        nodes.append(layout_node(image, masks, kind="image", prompt_id=f"{prefix}-prompt-{slot}"))
    for slot in reversed(slots):
        post = f"{prefix}-post-{slot}"
        posts.append(post)
        nodes.append(layout_node("이미지 후처리", images + posts[:-1], id=post, kind="stage", prompt_id=f"{prefix}-prompt-{slot}"))
        review = f"{prefix}-review-{slot}"
        reviews.append(review)
        nodes.append(layout_node("개별 이미지 검사", [post], id=review, task_key="illustration_quality_inspection", inspection_scope="image"))
    nodes.append(layout_node("전체 검사", reviews, task_key="illustration_quality_inspection", inspection_scope="overall"))
    nodes.append(layout_node("결과 반환", posts, kind="result"))
    baseline = layout_nodes(nodes)["positions"]
    assert layout_nodes(list(reversed(nodes)))["positions"] == baseline
    for mask, image, slot in zip(masks, images, slots):
        assert len({baseline[node_id]["y"] for node_id in [mask, image, f"{prefix}-post-{slot}", f"{prefix}-review-{slot}"]}) == 1
    image_rows = [baseline[image]["y"] for image in images]
    assert image_rows == sorted(set(image_rows))
    assert len({baseline[post]["x"] for post in posts}) == 1
    assert len({baseline[review]["x"] for review in reviews}) == 1
    for shared in ["ORIGINAL-ASSET", "전체 검사", "결과 반환"]:
        assert baseline[shared]["y"] > max(image_rows)


def test_flow_layout_aligns_unlabelled_branches_without_assigning_shared_work():
    nodes = [layout_node("ORIGINAL-ASSET")]
    for name in ("alpha", "beta"):
        nodes += [
            layout_node(f"mask-{name}", ["ORIGINAL-ASSET"], task_key="illustration_multi_char_mask"),
            layout_node(f"image-{name}", [f"mask-{name}"], kind="image"),
            layout_node("이미지 후처리", [f"image-{name}"], id=f"post-{name}", kind="stage"),
        ]
    positions = layout_nodes(nodes)["positions"]
    for name in ("alpha", "beta"):
        assert len({positions[f"{stage}-{name}"]["y"] for stage in ("mask", "image", "post")}) == 1
    assert positions["ORIGINAL-ASSET"]["y"] > positions["image-beta"]["y"]


def test_flow_layout_reserves_missing_slots_and_retry_room_without_overlap():
    nodes = [
        layout_node("MULTI-CHAR-MASK slot=-1"),
        layout_node("MULTI-CHAR-MASK slot=7"),
        layout_node("MULTI-CHAR-MASK slot=7 · retry"),
        layout_node("image-7", kind="image", illustration_slot=7),
        layout_node("image-20", kind="image", illustration_slot=20),
        layout_node("이미지 후처리", ["image-20"], id="post-20", kind="stage"),
    ]
    positions = layout_nodes(nodes)["positions"]
    assert positions["MULTI-CHAR-MASK slot=-1"]["y"] < positions["image-7"]["y"]
    assert positions["image-7"]["y"] == positions["MULTI-CHAR-MASK slot=7"]["y"]
    assert positions["image-20"]["y"] == positions["post-20"]["y"]
    assert positions["MULTI-CHAR-MASK slot=7 · retry"]["y"] < positions["image-20"]["y"]
    assert len({(p["x"], p["y"]) for p in positions.values()}) == len(nodes)
    assert positions["image-20"]["y"] - positions["MULTI-CHAR-MASK slot=7 · retry"]["y"] >= 398


def test_video_flow_layout_keeps_dependency_columns_without_illustration_stages():
    nodes = [
        layout_node("request", kind="request"),
        layout_node("CALL1", ["request"]),
        layout_node("CALL2-PLAN", ["CALL1"]),
        layout_node("CALL3", ["CALL2-PLAN"]),
    ]
    positions = layout_nodes(nodes, kind="video")["positions"]
    assert [positions[n["id"]]["depth"] for n in nodes] == [0, 1, 2, 3]
    assert len({p["y"] for p in positions.values()}) == 1


def test_flow_nodes_use_executor_colored_backgrounds():
    source = (Path(__file__).resolve().parents[1] / "frontend" / "illustration_flow.js").read_text(encoding="utf-8")
    assert "const executorColors = {llm:" in source
    assert "background:color-mix(in srgb,var(--bg2,#172033) 68%,var(--node-tint,#94a3b8) 32%)" in source
    assert "card.style.setProperty('--node-tint', executorColors[executor] || executorColors.process)" in source
    assert "처리 주체" in source


def test_flow_detail_falls_back_to_error_and_latest_raw_response():
    source = (Path(__file__).resolve().parents[1] / "frontend" / "illustration_flow.js").read_text(encoding="utf-8")
    assert "function detailOutput(n)" in source
    assert "attempts[index].raw_response" in source
    assert "fallback.error = n.error" in source
    assert "const output = detailOutput(n);" in source


@pytest.mark.asyncio
@pytest.mark.parametrize("label,provider", [
    ("삽화 1/17 · slot -1", "comfy"), ("다른 이미지 · slot 7", "chansub"),
])
async def test_image_queue_and_generator_share_lifecycle_and_details(label, provider):
    job = item(); flow.queue_added(job)
    entered, release, generated, finish = (asyncio.Event() for _ in range(4))
    children = []

    @flow.stage(
        "이미지 생성", layout_group="illustration_generation",
        executor=lambda inputs: "comfy" if inputs["provider"] == "comfy" else "process",
    )
    async def image_stage(positive, negative, provider, progress_callback=None):
        entered.set()
        await release.wait()
        return b"png", ""

    async def work():
        await llm("plan", [])
        child = item(label, "illustration")
        child.params = {"provider": provider, "prompt_data": {"slot": -1}}
        flow.queue_added(child)
        children.append(child)
        waiting = flow.detail(flow.snapshot()["id"], child.id)
        assert waiting["status"] == "waiting"
        assert waiting["started_at"] is None
        async def child_work():
            await image_stage("scene prompt", "negative prompt", provider, lambda: None)
            generated.set()
            await finish.wait()
            return {"success": True, "prompt_id": "prompt-7"}
        child.handler = child_work
        await Runner().execute(child)
        flow.merge([child])
        return {}

    job.handler = work
    running = asyncio.create_task(Runner().execute(job))
    await entered.wait()
    child = children[0]
    run_id = flow.snapshot()["id"]
    generating = flow.detail(run_id, child.id)
    assert generating["status"] == "processing"
    assert generating["executor"] == ("comfy" if provider == "comfy" else "process")
    assert generating["label"] == label
    assert generating["input"] == {
        "provider": provider, "prompt_data": {"slot": -1},
        "positive": "scene prompt", "negative": "negative prompt",
    }
    assert child.id not in generating["dependencies"]
    child.progress = 42
    child.progress_detail = {"phase": "generating", "current": 42, "total": 100}
    flow.queue_progress(child)
    assert flow.detail(run_id, child.id)["progress"] == 42
    release.set()
    await generated.wait()
    # The generator returning must not complete the still-running queue job.
    assert flow.detail(run_id, child.id)["status"] == "processing"
    assert flow.detail(run_id, child.id)["ended_at"] is None
    preview_node = flow.detail(run_id, child.id)
    assert preview_node["image_url"].startswith(f"/api/illustration_flow?run={run_id}&node={child.id}&image=1&v=")
    assert flow.preview_image(run_id, child.id) == b"png"
    assert flow.preview_image("unrelated-run", child.id) is None
    assert "_preview_image" not in preview_node
    assert "_preview_image" not in json.dumps(flow.snapshot())
    finish.set()
    await running
    graph = flow.snapshot()
    nodes = {n["label"]: n for n in graph["nodes"]}
    assert nodes[label]["layout_group"] == "illustration_images"
    assert "이미지 생성" not in nodes
    assert nodes[label]["dependencies"] == [nodes["plan"]["id"]]
    assert nodes["결과 반환"]["dependencies"] == [child.id]
    complete = flow.detail(run_id, child.id)
    assert complete["status"] == "completed"
    assert complete["started_at"] == generating["started_at"]
    assert complete["ended_at"] >= complete["started_at"]
    assert complete["progress_detail"] == child.progress_detail
    assert complete["output"] == {
        "generation": [{"image_bytes": 3}, ""],
        "queue": {"success": True, "prompt_id": "prompt-7"},
    }
    assert [n["id"] for n in graph["nodes"] if n["kind"] == "image"] == [child.id]
    assert layout_nodes(graph["nodes"])["positions"][child.id]["depth"] == 6


@pytest.mark.asyncio
async def test_image_mask_preview_keeps_each_executed_mask_and_excludes_binary_from_graph():
    job = item(); flow.queue_added(job)
    images = []

    async def work():
        await llm("mask plan", [])
        for index, image_bytes in enumerate((b"first RGB mask", b"second RGB mask", None)):
            child = item(f"Image slot {index}", "illustration")
            flow.queue_added(child)
            images.append(child)

            async def render(mask=image_bytes):
                flow.record_image_mask(mask)
                return {"success": True}

            child.handler = render
            await Runner().execute(child)
        flow.merge(images)
        return {}

    job.handler = work
    await Runner().execute(job)
    graph = flow.snapshot()
    run_id = graph["id"]
    for child, expected in zip(images, (b"first RGB mask", b"second RGB mask", None)):
        node = flow.detail(run_id, child.id)
        assert flow.preview_mask(run_id, child.id) == expected
        assert "_preview_mask" not in node
        if expected:
            assert node["mask_url"].startswith(f"/api/illustration_flow?run={run_id}&node={child.id}&mask=1&v=")
        else:
            assert node["mask_url"] == ""
    assert "_preview_mask" not in json.dumps(graph)
    assert flow.preview_mask("unrelated-run", images[0].id) is None

    import server
    response = await server.handle_illustration_flow(SimpleNamespace(query={
        "run": run_id, "node": images[0].id, "mask": "1",
    }))
    assert response.status == 200
    assert response.body == b"first RGB mask"
    assert response.content_type == "image/png"
    assert response.headers["Cache-Control"] == "no-store"
    missing = await server.handle_illustration_flow(SimpleNamespace(query={
        "run": run_id, "node": images[2].id, "mask": "1",
    }))
    assert missing.status == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["illustration_llm_build", "video_prompt_build"])
async def test_generation_without_image_queue_keeps_its_own_record(kind):
    job = item(item_type=kind); flow.queue_added(job)

    @flow.stage("standalone image", layout_group="illustration_generation")
    async def image_stage():
        return b"image", ""

    job.handler = image_stage
    await Runner().execute(job)
    graph = flow.snapshot(kind="video" if kind == "video_prompt_build" else "illustration")
    node = next(n for n in graph["nodes"] if n["label"] == "standalone image")
    assert node["kind"] == "stage"
    assert node["layout_group"] == "illustration_generation"
    assert node["dependencies"] == [job.id]
    assert flow.detail(graph["id"], node["id"])["output"] == [{"image_bytes": 5}, ""]


@pytest.mark.asyncio
async def test_parallel_image_records_and_postprocess_dependencies_stay_separate():
    job = item(); flow.queue_added(job)
    children = []
    entered, release = asyncio.Event(), asyncio.Event()

    @flow.stage("이미지 생성", layout_group="illustration_generation")
    async def image_stage(positive):
        entered.set()
        await release.wait()
        return positive.encode(), ""

    @flow.stage("이미지 후처리")
    async def postprocess():
        return {"delivered": True}

    async def work():
        await llm("DETAIL", [])
        for label, prompt in [("삽화 1/2 · slot -1", "one"), ("삽화 2/2 · slot 8", "second")]:
            child = item(label, "illustration")
            flow.queue_added(child)
            async def child_work(prompt=prompt):
                await image_stage(prompt)
                return {"success": True, "prompt_id": prompt}
            child.handler = child_work
            children.append(child)
        await asyncio.gather(*(Runner().execute(child) for child in children))
        flow.merge(children)
        await postprocess()
        return {}

    job.handler = work
    running = asyncio.create_task(Runner().execute(job))
    await entered.wait()
    assert len([n for n in flow.snapshot()["nodes"] if n["kind"] == "image"]) == 2
    release.set()
    await running
    graph = flow.snapshot()
    by_label = {n["label"]: n for n in graph["nodes"]}
    assert "이미지 생성" not in by_label
    assert set(by_label["이미지 후처리"]["dependencies"]) == {
        by_label["DETAIL"]["id"], *(child.id for child in children),
    }
    for child, prompt in zip(children, ("one", "second")):
        node = flow.detail(graph["id"], child.id)
        assert node["dependencies"] == [by_label["DETAIL"]["id"]]
        assert node["input"]["positive"] == prompt
        assert node["output"] == {
            "generation": [{"image_bytes": len(prompt)}, ""],
            "queue": {"success": True, "prompt_id": prompt},
        }


@pytest.mark.asyncio
async def test_flow_snapshot_exposes_image_and_postprocess_identity_before_completion():
    job = item(); flow.queue_added(job)
    children = []

    @flow.stage("이미지 후처리")
    async def postprocess(prompt_id, speak_text):
        return b"image"

    async def work():
        for slot in (-1, 8):
            child = item(f"삽화 · slot {slot}", "illustration")
            child.params = {"prompt_id": f"prompt-{slot}", "prompt_data": {"private": "prompt body"}}
            flow.queue_added(child)
            children.append(child)
            waiting = next(n for n in flow.snapshot()["nodes"] if n["id"] == child.id)
            assert waiting["prompt_id"] == f"prompt-{slot}"
            assert "input" not in waiting

            async def generate(prompt_id=f"prompt-{slot}"):
                return {"success": True, "prompt_id": prompt_id}
            child.handler = generate
        await asyncio.gather(*(Runner().execute(child) for child in children))
        flow.merge(children)
        for slot in (8, -1):
            await postprocess(f"prompt-{slot}", "private caption")
        review = item("전체 품질 검사", "illustration_quality_inspection")
        review.params = {"scope": "overall", "slots": [-1, 8]}
        flow.queue_added(review)
        assert next(n for n in flow.snapshot()["nodes"] if n["id"] == review.id)["inspection_scope"] == "overall"
        return {}

    job.handler = work
    await Runner().execute(job)
    graph = flow.snapshot()
    positions = layout_nodes(graph["nodes"])["positions"]
    for child in children:
        image = next(n for n in graph["nodes"] if n["id"] == child.id)
        post = next(n for n in graph["nodes"] if n["label"] == "이미지 후처리" and n["prompt_id"] == image["prompt_id"])
        assert "input" not in post
        assert positions[image["id"]]["y"] == positions[post["id"]]["y"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["empty_result", "exception", "cancelled"])
async def test_merged_generation_failure_survives_queue_result(failure, capsys):
    job = item(); flow.queue_added(job)
    children = []

    @flow.stage("이미지 생성", layout_group="illustration_generation")
    async def image_stage(positive):
        if failure == "empty_result":
            return None, "generator unavailable"
        if failure == "cancelled":
            raise asyncio.CancelledError("generator cancelled")
        raise RuntimeError("generator unavailable")

    async def work():
        child = item("failed slot", "illustration")
        flow.queue_added(child)
        children.append(child)
        async def child_work():
            try:
                await image_stage("failure input")
            except (RuntimeError, asyncio.CancelledError):
                pass
            return {"success": True}
        child.handler = child_work
        await Runner().execute(child)
        return {}

    job.handler = work
    await Runner().execute(job)
    graph = flow.snapshot()
    node = flow.detail(graph["id"], children[0].id)
    assert node["status"] == ("cancelled" if failure == "cancelled" else "failed")
    assert node["error"] == ("generator cancelled" if failure == "cancelled" else "generator unavailable")
    assert node["input"]["positive"] == "failure input"
    assert node["output"]["generation"] == (
        [None, "generator unavailable"] if failure == "empty_result"
        else {"error": "generator cancelled" if failure == "cancelled" else "generator unavailable"}
    )
    assert node["output"]["queue"] == {"success": True}
    assert len([n for n in graph["nodes"] if n["kind"] == "image"]) == 1
    assert "[ILLUST_FLOW]" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_queue_failure_retains_generated_output():
    job = item(); flow.queue_added(job)
    children = []

    @flow.stage("이미지 생성", layout_group="illustration_generation")
    async def image_stage():
        return b"png", ""

    async def work():
        child = item("image slot", "illustration")
        flow.queue_added(child)
        children.append(child)
        async def child_work():
            await image_stage()
            return {"success": False, "error": "backup failed"}
        child.handler = child_work
        await Runner().execute(child)
        return {}

    job.handler = work
    await Runner().execute(job)
    node = flow.detail(flow.snapshot()["id"], children[0].id)
    assert node["status"] == "failed"
    assert node["error"] == "backup failed"
    assert node["output"] == {
        "generation": [{"image_bytes": 3}, ""],
        "queue": {"success": False, "error": "backup failed"},
    }


@pytest.mark.asyncio
async def test_later_generation_does_not_overwrite_a_finished_image_slot():
    job = item(); flow.queue_added(job)
    children = []

    @flow.stage("이미지 생성", layout_group="illustration_generation")
    async def image_stage(positive):
        return positive.encode(), ""

    async def work():
        child = item("finished slot", "illustration")
        flow.queue_added(child)
        children.append(child)
        async def child_work():
            await image_stage("slot prompt")
            return {"success": True, "prompt_id": "original"}
        child.handler = child_work
        await Runner().execute(child)
        flow.absorb([child])
        await image_stage("independent prompt")
        return {}

    job.handler = work
    await Runner().execute(job)
    graph = flow.snapshot()
    slot = flow.detail(graph["id"], children[0].id)
    assert slot["input"]["positive"] == "slot prompt"
    assert slot["output"]["queue"]["prompt_id"] == "original"
    independent = next(n for n in graph["nodes"] if n["label"] == "이미지 생성")
    assert independent["kind"] == "stage"
    assert independent["dependencies"] == [children[0].id]
    assert flow.detail(graph["id"], independent["id"])["input"]["positive"] == "independent prompt"


@pytest.mark.asyncio
async def test_stage_executor_can_follow_actual_provider():
    job = item(); flow.queue_added(job)

    @flow.stage(
        "provider stage",
        executor=lambda inputs: "comfy" if inputs.get("provider") == "comfy" else "process",
    )
    async def provider_stage(provider="comfy"):
        return provider

    async def work():
        await provider_stage("comfy")
        await provider_stage("chansub")
        return {}

    job.handler = work
    await Runner().execute(job)
    nodes = [n for n in flow.snapshot()["nodes"] if n["label"] == "provider stage"]
    assert [n["executor"] for n in nodes] == ["comfy", "process"]


@pytest.mark.asyncio
async def test_llm_and_image_queue_nodes_expose_executor_metadata():
    job = item(); flow.queue_added(job)

    async def work():
        await llm("plan", [])
        child = item("slot", "illustration")
        child.params = {"provider": "comfy"}
        flow.queue_added(child)
        child.handler = lambda: None
        async def child_work():
            return {}
        child.handler = child_work
        await Runner().execute(child)
        return {}

    job.handler = work
    await Runner().execute(job)
    nodes = {n["label"]: n for n in flow.snapshot()["nodes"]}
    assert nodes["plan"]["executor"] == "llm"
    assert nodes["slot"]["executor"] == "comfy"
    assert nodes[job.label]["executor"] == "process"


@pytest.fixture(autouse=True)
def isolated_flow(monkeypatch):
    monkeypatch.setattr(flow, "_latest", None)
    monkeypatch.setattr(flow, "_notify", None)


def item(name="요청", item_type="illustration_llm_build"):
    return SimpleNamespace(id=name, label=name, type=item_type, params={}, status="pending")


class Runner:
    @flow.queue_execution
    async def execute(self, job):
        return await job.handler()


@flow.llm_call
async def llm(name, messages, *, entered=None, release=None):
    flow.llm_metadata(status="processing", model="test-model")
    if entered:
        entered.set()
    if release:
        await release.wait()
    return "출력: " + name


@pytest.mark.asyncio
@pytest.mark.parametrize("names", [("CALL1", "CALL2 A", "CALL2 B", "CALL3"), ("분석", "장면 하나", "장면 둘", "병합")])
async def test_actual_fork_join_and_details(names):
    job = item()
    flow.queue_added(job)
    async def work():
        await llm(names[0], [{"content": "원문"}])
        await flow.gather(llm(names[1], []), llm(names[2], []))
        await llm(names[3], [])
        return {"success": True}
    job.handler = work
    await Runner().execute(job)
    graph = flow.snapshot()
    nodes = {n["label"]: n for n in graph["nodes"]}
    assert nodes[names[1]]["dependencies"] == nodes[names[2]]["dependencies"] == [nodes[names[0]]["id"]]
    assert set(nodes[names[3]]["dependencies"]) == {nodes[names[1]]["id"], nodes[names[2]]["id"]}
    assert graph["status"] == "completed"
    assert "input" not in nodes[names[0]]
    assert flow.detail(graph["id"], nodes[names[0]]["id"])["input"] == [{"content": "원문"}]
    json.dumps(graph)


@pytest.mark.asyncio
async def test_serial_execution_keeps_chain():
    job = item()
    flow.queue_added(job)
    async def work():
        for name in ["first", "second", "third"]:
            await llm(name, [])
        return {}
    job.handler = work
    await Runner().execute(job)
    nodes = [n for n in flow.snapshot()["nodes"] if n["kind"] == "llm"]
    assert nodes[1]["dependencies"] == [nodes[0]["id"]]
    assert nodes[2]["dependencies"] == [nodes[1]["id"]]


@pytest.mark.asyncio
async def test_parallel_cap_shows_waiting_before_worker_starts():
    job = item(); flow.queue_added(job)
    entered, release = asyncio.Event(), asyncio.Event()
    gate = asyncio.Semaphore(1)
    async def worker(name):
        async with gate:
            return await llm(name, [], entered=entered, release=release)
    async def work():
        await flow.gather(*(flow.create_task(worker(name), flow_label=name) for name in ["one", "two", "three"]))
        return {}
    job.handler = work
    running = asyncio.create_task(Runner().execute(job))
    await entered.wait()
    nodes = {n["label"]: n for n in flow.snapshot()["nodes"]}
    assert nodes[job.label]["status"] == "processing"
    assert [nodes[n]["status"] for n in ["one", "two", "three"]] == ["processing", "waiting", "waiting"]
    release.set(); await running
    assert len([n for n in flow.snapshot()["nodes"] if n["kind"] == "llm"]) == 3


@pytest.mark.asyncio
async def test_older_run_cannot_replace_latest_or_leak_context():
    old = item("old"); flow.queue_added(old)
    entered, release = asyncio.Event(), asyncio.Event()
    async def work():
        await llm("old call", [], entered=entered, release=release)
        return {}
    old.handler = work
    running = asyncio.create_task(Runner().execute(old))
    await entered.wait()
    new = item("new"); flow.queue_added(new)
    latest = flow.snapshot()["id"]
    release.set(); await running
    assert flow.snapshot()["id"] == latest
    assert [n["label"] for n in flow.snapshot()["nodes"]] == ["new"]
    assert flow._run.get() is None


@pytest.mark.asyncio
async def test_child_queue_and_postprocess_join():
    job = item(); flow.queue_added(job)
    @flow.stage("image")
    async def image():
        return b"png", ""
    @flow.stage("postprocess")
    async def postprocess():
        return b"png"
    async def work():
        await llm("plan", [])
        child = item("slot", "illustration"); flow.queue_added(child)
        child.handler = image
        task = asyncio.create_task(Runner().execute(child))
        await llm("subtitle", [])
        await task
        flow.merge([child])
        await postprocess()
        return {}
    job.handler = work
    await Runner().execute(job)
    nodes = {n["label"]: n for n in flow.snapshot()["nodes"]}
    assert set(nodes["postprocess"]["dependencies"]) == {nodes["image"]["id"], nodes["subtitle"]["id"]}
    assert flow.detail(flow.snapshot()["id"], nodes["image"]["id"])["output"] == [{"image_bytes": 3}, ""]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage,slot", [
    ("MULTI-CHAR-MASK slot=4", "llm2"),
    ("CALL2-DETAIL", "llm3"),
    ("CALL2-KEYVIS", "llm10"),
])
async def test_validation_retry_delay_does_not_count_as_processing(monkeypatch, stage, slot):
    from modes import illustration_context_pipeline as pipeline, llm_service, lighbd_service
    task_key = pipeline._CALL_TASK_KEYS.get(stage.split()[0], pipeline._CALL_TASK_KEYS["CALL2"])
    suffix = "" if slot == "llm1" else slot[3:]
    config = llm_service.get_config()
    config.update({f"llm_stream{suffix}": False, f"llm_service{suffix}": "openai",
                   f"llm_model{suffix}": "offline-model",
                   f"llm_max_concurrency{suffix}": 1,
                   "llm_routing": {task_key: {"primary": slot, "fallback": False,
                       "max_retries": 1, "retry_delay_sec": 0.1}}})
    monkeypatch.setattr(llm_service, "_current_config", config)
    monkeypatch.setattr(lighbd_service, "_log_lighbd_history", lambda record: None)
    failed, other_entered, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    attempts = 0
    original_attempt = flow.llm_attempt

    def record_attempt(event, model, service):
        original_attempt(event, model, service)
        if event.get("type") == "attempt_failure":
            failed.set()

    async def provider(messages, service, model):
        nonlocal attempts
        if messages[0]["content"] == "retry":
            attempts += 1
            return "invalid" if attempts == 1 else "accepted"
        other_entered.set()
        await release.wait()
        return "accepted"

    monkeypatch.setattr(flow, "llm_attempt", record_attempt)
    monkeypatch.setattr(llm_service, "_dispatch_unlimited", provider)
    job = item(); flow.queue_added(job)

    async def work():
        retried = flow.create_task(pipeline._call_pipeline_llm(stage,
            [{"role": "user", "content": "retry"}],
            result_validator=lambda text: (text == "accepted", "invalid response")))
        await asyncio.wait_for(failed.wait(), 2)
        other = flow.create_task(pipeline._call_pipeline_llm(stage,
            [{"role": "user", "content": "hold"}]))
        try:
            await asyncio.wait_for(other_entered.wait(), 2)
            states = [node["status"] for node in flow.snapshot()["nodes"] if node["kind"] == "llm"]
            assert states == ["waiting", "processing"]
            assert llm_service._request_gate(slot).active == 1
        finally:
            release.set()
            await flow.gather(retried, other)
        return {}

    job.handler = work
    await Runner().execute(job)
    assert attempts == 2 and llm_service._request_gate(slot).active == 0


@pytest.mark.asyncio
async def test_real_pipeline_fallback_metadata_and_raw_responses(monkeypatch):
    from modes import illustration_context_pipeline as pipeline
    from modes import llm_service, lighbd_service
    monkeypatch.setattr(lighbd_service, "_log_lighbd_history", lambda record: None)
    monkeypatch.setattr(llm_service, "_base_config_get", lambda key, default=None: {"llm_model2": "fallback-model", "llm_service2": "fallback-service"}.get(key, default))
    async def call(task_key, messages, **kwargs):
        observer = kwargs["execution_observer"]
        await observer({"type": "attempt_start", "phase": "primary", "slot": "llm1", "attempt": 1})
        flow.llm_metadata(status="processing")
        await observer({"type": "attempt_failure", "phase": "primary", "slot": "llm1", "raw_response": "bad response", "reason": "parse failed"})
        await observer({"type": "attempt_start", "phase": "fallback", "slot": "llm2", "attempt": 1})
        flow.llm_metadata(status="processing")
        await observer({"type": "attempt_success", "phase": "fallback", "slot": "llm2", "attempt": 1, "raw_response": "usable response"})
        return "usable response"
    monkeypatch.setattr(llm_service, "callLLMTask", call)
    job = item(); flow.queue_added(job)
    async def work():
        await pipeline._call_pipeline_llm("CALL1", [{"role": "user", "content": "test"}])
        return {}
    job.handler = work
    await Runner().execute(job)
    n = next(n for n in flow.snapshot()["nodes"] if n["kind"] == "llm")
    details = flow.detail(flow.snapshot()["id"], n["id"])
    assert details["model"] == "fallback-model"
    assert details["service"] == "fallback-service"
    assert details["output"] == "usable response"
    assert details["attempts"][1]["raw_response"] == "bad response"


@pytest.mark.asyncio
@pytest.mark.parametrize("name,task,slot", [
    ("CALL1", "illustration_call1", "llm1"),
    ("CALL2-PLAN 1/3", "illustration_call2_plan", "llm2"),
    ("CALL2-DETAIL 2/3", "illustration_call2", "llm10"),
])
@pytest.mark.parametrize("thinking", [False, True])
async def test_pipeline_records_first_and_last_token_times_without_wait_or_done_tail(
    monkeypatch, name, task, slot, thinking,
):
    from modes import illustration_context_pipeline as pipeline, llm_service, lighbd_service
    suffix = "" if slot == "llm1" else slot[3:]
    config = llm_service.get_config()
    config.update({f"llm_stream{suffix}": True, f"llm_model{suffix}": "test-model",
                   "llm_routing": {task: {"primary": slot, "max_retries": 0}}})
    monkeypatch.setattr(llm_service, "_current_config", config)
    clock = {"now": 1000.0}
    monkeypatch.setattr(llm_service.time, "time", lambda: clock["now"])
    records, events = [], []
    monkeypatch.setattr(lighbd_service, "_log_lighbd_history", lambda record: records.append(dict(record)))

    async def notify(event): events.append(dict(event))
    monkeypatch.setattr(llm_service, "_stream_notify_func", notify)

    async def provider(*_args):
        yield {"type": "start"}
        if thinking:
            clock["now"] = 1002.0
            yield {"type": "delta", "reasoning": "first reasoning"}
            clock["now"] = 1005.0
            yield {"type": "delta", "reasoning": "last reasoning"}
        clock["now"] = 1009.0
        yield {"type": "delta", "text": "", "reasoning": ""}
        clock["now"] = 1010.0
        yield {"type": "delta", "text": "usable "}
        clock["now"] = 1017.0
        yield {"type": "delta", "text": "response"}
        clock["now"] = 1025.0
        yield {"type": "done", "text": "usable response", "completion_tokens": 4, "elapsed": 25.0}
    monkeypatch.setattr(llm_service, "_dispatch_stream_unlimited", provider)
    job = item(); flow.queue_added(job)

    async def work():
        return await pipeline._call_pipeline_llm(name, [{"role": "user", "content": "test"}])
    job.handler = work
    assert await Runner().execute(job) == "usable response"
    node = next(node for node in flow.snapshot()["nodes"] if node["kind"] == "llm")
    expected = {"thinking_started_at": 1002.0 if thinking else None,
                "thinking_ended_at": 1005.0 if thinking else None,
                "response_started_at": 1010.0, "response_ended_at": 1017.0}
    for key, value in expected.items():
        assert node[key] == value
        assert records[-1].get(key) == value
        assert events[-1][key] == value
    assert node["response_ended_at"] - node["response_started_at"] == 7.0
    assert node["ended_at"] == 1025.0
    details = flow.detail(flow.snapshot()["id"], node["id"])
    success = next(event for event in details["attempts"] if event["type"] == "attempt_success")
    assert success["response_started_at"] == 1010.0
    if thinking: assert success["thinking_ended_at"] - success["thinking_started_at"] == 3.0


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback_streaming", [False, True])
async def test_fallback_clears_primary_thinking_times_and_keeps_failed_attempt_history(monkeypatch, fallback_streaming):
    from modes import illustration_context_pipeline as pipeline, llm_service, lighbd_service
    config = llm_service.get_config()
    config.update({"llm_stream": True, "llm_stream2": fallback_streaming,
                   "llm_model": "main-model", "llm_model2": "fallback-model",
                   "llm_routing": {"illustration_call1": {"primary": "llm1", "fallback_target": "llm2"}}})
    monkeypatch.setattr(llm_service, "_current_config", config)
    clock = {"now": 1000.0}
    monkeypatch.setattr(llm_service.time, "time", lambda: clock["now"])
    records = []
    monkeypatch.setattr(lighbd_service, "_log_lighbd_history", lambda record: records.append(dict(record)))
    monkeypatch.setattr(llm_service, "_stream_notify_func", None)

    async def provider(_messages, _service, model):
        yield {"type": "start"}
        if model == "main-model":
            clock["now"] = 1001.0
            yield {"type": "delta", "reasoning": "primary thinking"}
            clock["now"] = 1004.0
            yield {"type": "delta", "text": "invalid"}
            yield {"type": "done", "text": "invalid"}
        else:
            clock["now"] = 2000.0
            yield {"type": "delta", "text": "usable "}
            clock["now"] = 2009.0
            yield {"type": "delta", "text": "response"}
            yield {"type": "done", "text": "usable response"}

    async def nonstream(*_args): return "usable response"
    monkeypatch.setattr(llm_service, "_dispatch_stream_unlimited", provider)
    monkeypatch.setattr(llm_service, "_dispatch", nonstream)
    job = item(); flow.queue_added(job)

    async def work():
        return await pipeline._call_pipeline_llm("CALL1", [{"role": "user", "content": "test"}],
            result_validator=lambda value: value == "usable response")
    job.handler = work
    assert await Runner().execute(job) == "usable response"
    node = next(node for node in flow.snapshot()["nodes"] if node["kind"] == "llm")
    assert node["phase"] == "fallback"
    assert node["thinking_started_at"] is None and node["thinking_ended_at"] is None
    assert node["response_started_at"] == (2000.0 if fallback_streaming else None)
    assert node["response_ended_at"] == (2009.0 if fallback_streaming else None)
    assert records[0]["status"] == "error"
    assert records[0]["thinking_started_at"] == 1001.0
    assert records[-1].get("thinking_started_at") is None


def test_flow_timing_ui_formats_missing_thinking_as_zero_and_distinguishes_batch_responses():
    source = (Path(__file__).resolve().parents[1] / "frontend/illustration_flow.js").read_text(encoding="utf-8")
    helpers = source[source.index("    const phaseSeconds ="):source.index("    function button(")]
    script = "const assert=require('node:assert/strict');const nodeExecutor=n=>n.executor||(n.kind==='llm'?'llm':'process');\n" + helpers + """
const n={kind:'llm',status:'completed',thinking_started_at:1002,thinking_ended_at:1005,response_started_at:1010,response_ended_at:1017};
assert.equal(llmTiming(n),'THINK 3.0s · 응답 7.0s');
assert.equal(timingDetails(n)['THINK 시간'],'3.0s');
assert.match(timingDetails(n)['본문 첫 수신'], /^[0-9]{2}:[0-9]{2}:[0-9]{2}[.][0-9]{3}$/);
delete n.thinking_started_at;delete n.thinking_ended_at;
assert.equal(llmTiming(n),'THINK 0.0s · 응답 7.0s');
assert.equal(timingDetails(n)['추론 첫 수신'],'수신 없음');
n.response_ended_at=n.response_started_at;
assert.equal(llmTiming(n),'THINK 0.0s · 응답 0.0s');
delete n.response_started_at;delete n.response_ended_at;
assert.equal(llmTiming(n),'THINK 0.0s · 응답 일괄');
assert.equal(llmTiming({...n,status:'waiting'}),'THINK 0.0s · 응답 대기');
assert.equal(llmTiming({...n,executor:'comfy'}),'');
"""
    result = subprocess.run(["node", "-"], input=script, text=True, encoding="utf-8", capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.asyncio
async def test_failed_llm_exposes_error_and_undecoded_raw_response():
    damaged_base64 = "YWJjA"

    @flow.llm_call
    async def failing_llm(name, messages):
        flow.llm_metadata(status="processing", model="test-model")
        flow.llm_attempt(
            {"type": "attempt_failure", "phase": "primary", "slot": "llm1",
             "raw_response": damaged_base64, "reason": "response parse failed"},
            "test-model",
            "test-service",
        )
        raise RuntimeError("response parse failed")

    job = item(); flow.queue_added(job)

    async def work():
        await failing_llm("CALL2-DETAIL", [{"role": "user", "content": "test"}])

    job.handler = work
    with pytest.raises(RuntimeError, match="response parse failed"):
        await Runner().execute(job)

    graph = flow.snapshot()
    node = next(n for n in graph["nodes"] if n["label"] == "CALL2-DETAIL")
    details = flow.detail(graph["id"], node["id"])
    assert details["status"] == "failed"
    assert details["output"] == {
        "error": "response parse failed",
        "raw_response": damaged_base64,
    }


@pytest.mark.asyncio
async def test_failed_non_llm_stage_exposes_error_as_output():
    @flow.stage("different failing stage")
    async def failing_stage(value):
        raise ValueError(f"cannot process {value}")

    job = item(); flow.queue_added(job)

    async def work():
        await failing_stage("scene")

    job.handler = work
    with pytest.raises(ValueError, match="cannot process scene"):
        await Runner().execute(job)

    graph = flow.snapshot()
    node = next(n for n in graph["nodes"] if n["label"] == "different failing stage")
    details = flow.detail(graph["id"], node["id"])
    assert details["output"] == {"error": "cannot process scene"}


@pytest.mark.asyncio
@pytest.mark.parametrize("bot", ["", "   ", None])
async def test_missing_bot_emits_warning(monkeypatch, bot):
    import server
    events = []
    async def notify(kind, data):
        events.append((kind, data))
    monkeypatch.setattr(server, "notify_frontend", notify)
    with pytest.raises(RuntimeError, match="활성 봇을 선택"):
        server._require_active_illustration_bot(bot, context="test request")
    await asyncio.sleep(0)
    assert events == [("illustration_warning", {"message": "삽화를 생성하려면 먼저 활성 봇을 선택해야 합니다.", "context": "test request"})]


@pytest.mark.asyncio
async def test_configured_bot_does_not_warn(monkeypatch):
    import server
    async def notify(*args):
        pytest.fail("configured bot must not warn")
    monkeypatch.setattr(server, "notify_frontend", notify)
    assert server._require_active_illustration_bot("봇 A", context="test") == "봇 A"
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_terminal_failure_and_cancelled_queue():
    job = item(); flow.queue_added(job)
    async def fail():
        raise RuntimeError("execution failed")
    job.handler = fail
    with pytest.raises(RuntimeError):
        await Runner().execute(job)
    assert flow.snapshot()["status"] == "failed"
    failed_graph = flow.snapshot()
    assert flow.detail(failed_graph["id"], job.id)["output"] == {"error": "execution failed"}
    cancelled = item("cancelled"); flow.queue_added(cancelled)
    cancelled.status = "cancelled"
    flow.queue_sync([cancelled])
    assert flow.snapshot()["status"] == "cancelled"
    cancelled_graph = flow.snapshot()
    assert flow.detail(cancelled_graph["id"], cancelled.id)["output"] == {"error": "작업 취소"}


@pytest.mark.asyncio
async def test_unrelated_queue_work_does_not_inherit_illustration():
    job = item(); flow.queue_added(job)
    async def other_work():
        assert flow._run.get() is None
        await llm("unrelated", [])
        return {}
    async def work():
        other = item("other", "tag_analysis"); other.handler = other_work
        await Runner().execute(other)
        await llm("related", [])
        return {}
    job.handler = work
    await Runner().execute(job)
    assert [n["label"] for n in flow.snapshot()["nodes"] if n["kind"] == "llm"] == ["related"]


@pytest.mark.asyncio
async def test_cancel_before_parallel_task_starts():
    job = item(); flow.queue_added(job)
    async def work():
        task = flow.create_task(llm("never started", []), flow_label="queued")
        task.cancel()
        await flow.gather(task, return_exceptions=True)
        return {}
    job.handler = work
    await Runner().execute(job)
    node = next(n for n in flow.snapshot()["nodes"] if n["label"] == "queued")
    assert node["status"] == "cancelled"
    assert flow.detail(flow.snapshot()["id"], node["id"])["output"] == {"error": "대기 작업 취소"}


@pytest.mark.asyncio
async def test_queue_returned_failure_updates_request_node():
    job = item(); flow.queue_added(job)
    async def work():
        return {"success": False, "error": "provider unavailable"}
    job.handler = work
    await Runner().execute(job)
    graph = flow.snapshot()
    assert graph["status"] == "failed"
    request = next(n for n in graph["nodes"] if n["id"] == job.id)
    assert request["status"] == "failed"
    assert request["error"] == "provider unavailable"


@pytest.mark.asyncio
async def test_cancel_request_marks_flow_and_late_children_cancelled():
    job = item()
    flow.queue_added(job)
    graph = flow.snapshot()
    assert flow.request_cancel(graph["id"]) is True
    assert flow.snapshot()["status"] == "cancelling"
    assert flow.snapshot()["cancel_requested"] is True

    token = flow._run.set(flow._latest)
    try:
        child = item("late child", "illustration")
        flow.queue_added(child)
    finally:
        flow._run.reset(token)

    assert child.status == "cancelled"
    assert getattr(child, "_illustration_cancelled_on_add", False) is True
    detail = flow.detail(graph["id"], child.id)
    assert detail["status"] == "cancelled"
    assert "중단" in detail["error"]


@pytest.mark.asyncio
async def test_queue_manager_allows_active_image_to_finish_after_flow_cancel(monkeypatch):
    import queue_manager as queue_module

    manager = queue_module.QueueManager()
    job = queue_module.QueueItem(
        id="active-flow",
        type="illustration",
        label="active-flow",
        params={},
    )
    job.completion_future = asyncio.get_running_loop().create_future()
    job.completion_future.add_done_callback(manager._mark_completion_future_observed)
    flow.queue_added(job)
    manager.items.append(job)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def execute(_self, _item):
        entered.set()
        await release.wait()
        return {"success": True, "image": "finished"}

    wrapped = flow.queue_execution(execute).__get__(manager, type(manager))
    monkeypatch.setattr(manager, "_execute_item", wrapped)
    running = asyncio.create_task(manager._run_item_pipeline(job, is_gpu=False))
    await entered.wait()

    result = await manager.cancel_illustration_flow(flow.snapshot()["id"])

    assert result["active_finishing"] == 1
    assert result["active_cancelled"] == 0
    assert running.done() is False
    assert job.completion_future.done() is False

    release.set()
    await asyncio.wait_for(running, timeout=1)

    assert job.status == "completed"
    assert running.cancelled() is False
    assert flow.snapshot()["status"] == "cancelled"
    assert job.completion_future.result() == {"success": True, "image": "finished"}


@pytest.mark.asyncio
async def test_cancel_request_interrupts_active_llm_immediately():
    job = item("llm-root")
    flow.queue_added(job)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def work():
        await llm("CALL2-DETAIL", [], entered=entered, release=release)
        return {"success": True}

    job.handler = work
    running = asyncio.create_task(Runner().execute(job))
    await entered.wait()
    run_id = flow.snapshot()["id"]

    assert flow.request_cancel(run_id) is True
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(running, timeout=1)

    graph = flow.snapshot()
    assert graph["status"] == "cancelled"
    node = next(n for n in graph["nodes"] if n["label"] == "CALL2-DETAIL")
    assert node["status"] == "cancelled"
    assert "중단" in node["error"] or "취소" in node["error"]


@pytest.mark.asyncio
async def test_queue_manager_rejects_late_child_added_after_flow_cancel():
    import queue_manager as queue_module

    manager = queue_module.QueueManager()
    root = item("root")
    flow.queue_added(root)
    run = flow._latest
    assert flow.request_cancel(run["id"]) is True

    token = flow._run.set(run)
    try:
        child = await manager.add_item(
            "illustration",
            "late queue child",
            {"provider": "comfy"},
        )
    finally:
        flow._run.reset(token)

    assert child.status == "cancelled"
    assert child in manager.items
    assert child.completion_future.done()
    assert isinstance(child.completion_future.exception(), RuntimeError)
    detail = flow.detail(run["id"], child.id)
    assert detail["status"] == "cancelled"
    assert "중단" in detail["error"]
