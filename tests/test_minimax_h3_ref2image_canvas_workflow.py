import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_WORKFLOW = (
    ROOT
    / "comfy"
    / "user"
    / "default"
    / "workflows"
    / "SOYA_USER"
    / "배포_영상_H3_REF2V_v1.json"
)
TRACKED_WORKFLOW = (
    ROOT / "experiments" / "minimax_h3_ref2image" / "workflow_canvas.json"
)
COMFY_WORKFLOW = (
    ROOT
    / "comfy"
    / "user"
    / "default"
    / "workflows"
    / "SOYA_USER"
    / "실험_이미지_H3_REF2I_T1_v1.json"
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _nodes(workflow):
    return {node["id"]: node for node in workflow["nodes"]}


def test_ref2v_derived_copy_is_installed_in_comfy() -> None:
    assert COMFY_WORKFLOW.is_file()
    assert _load(COMFY_WORKFLOW) == _load(TRACKED_WORKFLOW)


def test_ref2v_derived_canvas_has_valid_link_endpoints() -> None:
    workflow = _load(TRACKED_WORKFLOW)
    nodes = _nodes(workflow)
    links = {link[0]: link for link in workflow["links"]}

    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(links)
    for link_id, origin_id, origin_slot, target_id, target_slot, socket_type in links.values():
        assert origin_id in nodes
        assert target_id in nodes
        assert origin_slot < len(nodes[origin_id]["outputs"])
        assert target_slot < len(nodes[target_id]["inputs"])
        assert nodes[origin_id]["outputs"][origin_slot]["type"] == socket_type
        assert nodes[target_id]["inputs"][target_slot]["type"] == socket_type
        assert link_id in (nodes[origin_id]["outputs"][origin_slot]["links"] or [])
        assert nodes[target_id]["inputs"][target_slot]["link"] == link_id


def test_ref2v_transport_and_sampling_nodes_are_preserved() -> None:
    source = _nodes(_load(SOURCE_WORKFLOW))
    derived = _nodes(_load(TRACKED_WORKFLOW))
    unchanged_node_ids = {
        123,
        124,
        125,
        126,
        127,
        128,
        129,
        144,
        146,
        147,
        148,
        151,
        152,
        153,
        155,
        156,
        157,
        158,
        159,
        160,
        161,
        162,
    }

    for node_id in unchanged_node_ids:
        source_node = source[node_id]
        derived_node = derived[node_id]
        for field in ("id", "type", "pos", "size", "flags", "mode", "widgets_values"):
            assert derived_node[field] == source_node[field], (node_id, field)

    assert derived[145]["type"] == source[145]["type"] == "PrimitiveStringMultiline"
    assert "[PATH]\nsoya_video\n" in derived[145]["widgets_values"][0]
    assert "[DURATION]" in derived[145]["widgets_values"][0]
    assert "[PROMPT]" in derived[145]["widgets_values"][0]
    assert "[W]" in derived[145]["widgets_values"][0]
    assert "[H]" in derived[145]["widgets_values"][0]
    assert "[SEED]" in derived[145]["widgets_values"][0]
    assert derived[149]["widgets_values"] == source[149]["widgets_values"]


def test_default_still_prompt_uses_the_ref2v_six_section_protocol() -> None:
    workflow = _load(TRACKED_WORKFLOW)
    nodes = _nodes(workflow)
    transport = nodes[145]["widgets_values"][0]
    prompt = transport.split("[PROMPT]\n", 1)[1].split("\n[W]\n", 1)[0]

    headings = re.findall(
        r"(?m)^(subject_definitions|summary|retention_analysis|"
        r"detailed_description|overall_soundscape|non_diegetic_music):$",
        prompt,
    )
    assert headings == [
        "subject_definitions",
        "summary",
        "retention_analysis",
        "detailed_description",
        "overall_soundscape",
        "non_diegetic_music",
    ]
    assert "<Subject 1>" in prompt
    assert "<Picture 1>" in prompt
    assert "[reference generation]" in prompt
    assert "<Subject 1> (appears in [Shot 1]): fully_preserved -" in prompt
    assert "exactly one <Subject 1>" in prompt
    assert nodes[136]["widgets_values"][0] == prompt


def test_ref2v_video_path_is_replaced_by_t1_image_path() -> None:
    workflow = _load(TRACKED_WORKFLOW)
    nodes = _nodes(workflow)
    types = {node["type"] for node in workflow["nodes"]}

    assert {120, 121, 130, 131, 150, 154}.isdisjoint(nodes)
    assert {
        "CreateVideo",
        "SaveVideo",
        "VAEDecodeAudio",
        "MiniMaxH3ReferenceToVideo",
    }.isdisjoint(types)
    assert nodes[92]["type"] == "SaveImage"
    assert nodes[119]["widgets_values"] == [
        "minimax_h3_t1_image_vae_step1597.safetensors"
    ]
    assert nodes[122]["type"] == "VAEDecode"
    assert nodes[136]["type"] == "SoyaMiniMaxH3ReferenceToImage_mdsoya"
    assert nodes[123]["widgets_values"] == ["res_multistep"]
    assert nodes[124]["widgets_values"] == ["simple", 20, 1]
    assert nodes[144]["widgets_values"] == [0.1, 2, -2, 20]


def test_ref2i_socket_mapping_matches_the_deployment_transport() -> None:
    workflow = _load(TRACKED_WORKFLOW)
    nodes = _nodes(workflow)
    ref2image = nodes[136]
    links = {link[0]: link for link in workflow["links"]}

    assert [item["name"] for item in ref2image["inputs"]] == [
        "clip",
        "vae",
        *[f"ref_image_{index}" for index in range(1, 10)],
        "prompt",
        "width",
        "height",
        "ref_image_size",
    ]
    assert links[272][3:5] == [136, 0]
    assert links[273][3:5] == [136, 1]
    assert links[311][3:5] == [136, 2]
    assert links[314][3:5] == [136, 3]
    assert links[317][3:5] == [136, 4]
    assert links[297][3:5] == [136, 11]
    assert links[300][3:5] == [136, 12]
    assert links[303][3:5] == [136, 13]
    assert links[258][1:5] == [122, 0, 92, 0]
