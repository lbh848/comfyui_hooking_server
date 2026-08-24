import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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


def test_canvas_copy_is_installed_in_the_comfy_workflow_directory() -> None:
    assert COMFY_WORKFLOW.is_file()


def test_canvas_has_valid_link_endpoints_and_socket_references() -> None:
    workflow = _load(TRACKED_WORKFLOW)
    nodes = {node["id"]: node for node in workflow["nodes"]}
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


def test_canvas_is_t1_image_only_and_uses_expected_defaults() -> None:
    workflow = _load(TRACKED_WORKFLOW)
    nodes_by_type = {node["type"]: node for node in workflow["nodes"]}

    forbidden = {
        "CreateVideo",
        "SaveVideo",
        "VAEDecodeAudio",
        "MiniMaxH3ReferenceToVideo",
    }
    assert forbidden.isdisjoint(nodes_by_type)

    ref2image = nodes_by_type["SoyaMiniMaxH3ReferenceToImage_mdsoya"]
    assert [item["name"] for item in ref2image["inputs"][:3]] == [
        "clip",
        "vae",
        "ref_image_1",
    ]
    assert [item["name"] for item in ref2image["inputs"][3:11]] == [
        f"ref_image_{index}" for index in range(2, 10)
    ]
    assert ref2image["widgets_values"][1:] == [512, 512, "match"]

    assert nodes_by_type["UNETLoader"]["widgets_values"][0] == (
        "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    )
    assert nodes_by_type["VAELoader"]["widgets_values"] == [
        "minimax_h3_t1_image_vae_step1597.safetensors"
    ]
    assert nodes_by_type["MiniMaxH3SigmaShift"]["widgets_values"] == [12.0, 3.0]
    assert nodes_by_type["BasicScheduler"]["widgets_values"] == ["simple", 12, 1.0]
    assert nodes_by_type["KSamplerSelect"]["widgets_values"] == ["res_multistep"]
    assert "SaveImage" in nodes_by_type
