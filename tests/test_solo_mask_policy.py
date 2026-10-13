"""Large solo regions bypass masking; small solo and all group regions keep it."""

from copy import deepcopy
import importlib
import json

import pytest

from modes import multi_char_mask
from queue_manager import QueueItem, QueueManager


CASES = [
    (("Mira",), .6, .9, True),
    (("하린",), .6, .9, True),
    (("Distant traveller",), .2, .9, False),
    (("Mira",), .5, .6, False),
    (("하린",), .75, .4, False),
    (("Portrait",), .7501, .4, True),
    (("Full crop",), 1., 1., True),
    (("Mira", "Harin"), .8, .9, False),
    (("Nora", "Ilan", "Cleo"), .8, .9, False),
]


def _context(names, width, height):
    layout = multi_char_mask.validate_multi_char_layout({
        "background_prompt": "courtyard, afternoon light",
        "composition_prompt": "standing figures beside a stone wall",
        "regions": [{
            "name": name, "character_prompt": "girl, silver hair, blue coat",
            "x": 0., "y": 0., "width": width, "height": height,
        } for name in names],
    }, names, require_prompt_separation=True)
    return {
        "enable": True,
        "characters": [{"name": name, "positive": "girl, silver hair, blue coat"}
                       for name in names],
        "character_order": list(names), "layout": layout,
        "mask_location": "region_mask",
    }


def _positive(snapshot, newline="\n", pretty=False):
    payload = {
        "enable": True, "char_num": snapshot["char_num"],
        "char_name_list": snapshot["character_order"],
        "char_inform": ["girl, silver hair, blue coat"] * snapshot["char_num"],
        "char_trigger_list": [["portrait trigger"]] * snapshot["char_num"],
        "background_prompt": "courtyard, afternoon light",
        "composition_prompt": "standing figures beside a stone wall",
        "mask_fingerprint": snapshot["mask_fingerprint"],
        "shared_tag": {"before_char": ["quality", "courtyard"], "after_char": []},
    }
    return newline.join([
        "[ANIMA_ALL]", "portrait trigger, courtyard, girl, silver hair, blue coat",
        "[MULTI_CHAR]", json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)
        .replace("\n", newline), "[HRF_ACTIVATE]", "false",
    ])


@pytest.mark.parametrize("name", ["Shoko", "Mira"])
@pytest.mark.parametrize("bounds,skip", [
    ((.25, .3, .5, .6), False),
    ((.32, .21, .36, .68), False),
    ((.1, .2, .7, .7), True),
])
@pytest.mark.parametrize("version", [3, 4, 5])
def test_upward_extension_keeps_solo_skip_based_on_original_body_area(name, bounds, skip, version):
    x, y, width, height = bounds
    layout = multi_char_mask.validate_multi_char_layout({"mask_render_version": version, "regions": [
        {"name": name, "x": x, "y": y, "width": width, "height": height},
    ]}, [name])
    assert width * (y + height) > multi_char_mask.SOLO_MASK_SKIP_AREA_RATIO
    assert multi_char_mask.should_skip_solo_region_mask(layout) is skip
    context = multi_char_mask.normalize_multi_char_snapshot({
        "enable": True, "layout": layout, "character_order": [name],
    })
    positive = _positive(context)
    result, snapshot = multi_char_mask.apply_solo_region_mask_policy(positive, context)
    assert (snapshot is None) is skip
    assert multi_char_mask.extract_multi_char_prompt_payload(result)["enable"] is not skip
    if not skip:
        assert result == positive
        assert multi_char_mask.render_region_mask(snapshot["layout"], width=100, height=100).getpixel(
            (round((x + width / 2) * 100), 0)) == ((255, 0, 0) if version < 5 else (0, 0, 0))


@pytest.mark.parametrize("names,width,height,skip", CASES)
@pytest.mark.parametrize("newline,pretty", [("\n", False), ("\r\n", True)])
def test_policy_preserves_scene_and_control_data_and_only_skips_large_solo(
    names, width, height, skip, newline, pretty,
):
    context = multi_char_mask.normalize_multi_char_snapshot(_context(names, width, height))
    original = deepcopy(context)
    positive = _positive(context, newline, pretty)
    before = multi_char_mask.extract_multi_char_prompt_payload(positive)
    result, snapshot = multi_char_mask.apply_solo_region_mask_policy(positive, context)
    after = multi_char_mask.extract_multi_char_prompt_payload(result)

    assert after == {**before, "enable": not skip}
    assert result.split("[MULTI_CHAR]", 1)[0] == positive.split("[MULTI_CHAR]", 1)[0]
    assert result.split("[HRF_ACTIVATE]", 1)[1] == positive.split("[HRF_ACTIVATE]", 1)[1]
    assert snapshot == (None if skip else context)
    assert context == original
    if not skip:
        assert result == positive


@pytest.mark.asyncio
@pytest.mark.parametrize("names,width,height,skip", CASES)
async def test_illustration_queue_applies_policy_to_png_prompt_and_deferred_backup(
    monkeypatch, tmp_path, names, width, height, skip,
):
    import server

    bot_mode = importlib.import_module("modes.bot_mode")
    context = _context(names, width, height)
    original_layout = deepcopy(context["layout"])
    bot = {
        "name": "mask-policy-bot",
        "characters": [{
            "name": name, "gender_tag": "1girl",
            "loras_solo": [{"source": "asset", "lora_path": "solo.safetensors",
                            "strength": .37, "trigger": "portrait trigger", "BASE": "anima"}],
            "loras_group": [{"source": "asset", "lora_path": "group.safetensors",
                             "strength": .83, "trigger": "group trigger", "BASE": "anima"}],
        } for name in names],
        "illust_settings_solo": {}, "illust_settings_group": {},
    }
    bot_data = {"bots": [bot], "positive_whitelist": [], "positive_blacklist": []}
    raw_positive = "\n".join([
        "[SETUP]", "courtyard, afternoon light", "[CHAR]",
        " | ".join("girl, silver hair, blue coat" for _ in names),
        "[SUPPLEMENT]", "standing figures beside a stone wall",
        "[NAME]", ", ".join(names),
    ])
    prompt_id = "solo-mask-policy-test"
    captured = {}

    async def generate(positive, negative, **kwargs):
        captured["positive"] = positive
        return b"generated pixels", {}

    monkeypatch.setattr(bot_mode, "_load_bot_data", lambda: bot_data)
    monkeypatch.setattr(bot_mode, "_load_lb_extra", lambda _name: [])
    monkeypatch.setattr(bot_mode, "_load_patch_settings", lambda _name: {})
    monkeypatch.setattr(server.asset_mode, "_tags", {})
    monkeypatch.setattr(server, "ws_connections", {})
    monkeypatch.setattr(server, "log_to_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "extract_prompts_by_title",
                        lambda _prompt, title: raw_positive if title == "긍정프롬프트" else "lowres")
    monkeypatch.setattr(server, "generate_image_with_prompt", generate)
    monkeypatch.setattr(server.illustration_flow, "record_image_mask",
                        lambda image_bytes: captured.update(mask_bytes=image_bytes))
    monkeypatch.setitem(server.prompts, prompt_id, {
        "status": "running", "outputs": {},
        "_illustration_runtime_snapshot": {
            "bot_name": bot["name"], "provider": "comfy", "illustration_workflow_type": "v3",
            "clamp_enabled": False, "word_rules": [],
        },
    })
    mask_dir = tmp_path / "region_mask"
    mask_dir.mkdir()
    stale = mask_dir / "old.png"
    stale.write_bytes(b"previous image mask")
    manager = QueueManager()
    manager.get_config = lambda: {"comfy_input_dir": str(tmp_path)}
    manager.process_prompt_full = server.process_prompt
    item = QueueItem(id="policy-illustration", type="illustration", label="policy", params={
        "prompt_id": prompt_id, "prompt_data": {}, "raw_body": {
            "illustration_provider": "comfy", "illustration_defer_postprocess": True,
            "illustration_multi_char": context,
        },
    })

    result = await manager._handle_illustration(item)

    blocks = server.llm_prompt_edit.parse_blocks(captured["positive"])
    control = json.loads(blocks["MULTI_CHAR"])
    assert result["success"] is True
    assert control["enable"] is not skip
    assert control["char_num"] == len(names)
    assert blocks["CHAR_LIST"] == ",".join(names)
    assert "blue coat" in blocks["ANIMA_ALL"]
    loras = json.loads(blocks["LORA_DATA"])["list"]
    assert [entry["str"] for entry in loras] == ([.37] if len(names) == 1 else [.83] * len(names))
    saved = server.prompts[prompt_id]["_deferred_finalize"]
    assert saved["positive"] == captured["positive"]
    assert (saved["illustration_multi_char"] is None) is skip
    assert (mask_dir / "region_mask.png").exists() is not skip
    assert captured["mask_bytes"] == (None if skip else (mask_dir / "region_mask.png").read_bytes())
    assert stale.exists() is skip
    assert context["layout"] == original_layout


@pytest.mark.asyncio
@pytest.mark.parametrize("names,width,height,skip", CASES)
async def test_regeneration_uses_same_policy_and_keeps_new_backup_consistent(
    tmp_path, names, width, height, skip,
):
    snapshot = multi_char_mask.normalize_multi_char_snapshot(_context(names, width, height))
    positive = _positive(snapshot)
    original = deepcopy(snapshot)
    captured = {}
    manager = QueueManager()
    manager.get_config = lambda: {"comfy_input_dir": str(tmp_path)}

    async def generate(render_positive, negative, **kwargs):
        captured["positive"] = render_positive
        return b"regenerated pixels", {}

    async def save(image, prompt_id, saved_positive, negative, **kwargs):
        captured["saved_positive"] = saved_positive
        captured["snapshot"] = kwargs["illustration_multi_char"]
        return "new-policy-backup", image

    manager.generate_image_with_prompt = generate
    manager.save_backup = save
    item = QueueItem(id="policy-regenerate", type="regenerate", label="policy", params={
        "backup_name": "old-policy-backup", "positive": positive,
        "negative": "lowres", "provider": "comfy", "illustration_multi_char": snapshot,
    })
    result = await manager._handle_regenerate(item)

    control = multi_char_mask.extract_multi_char_prompt_payload(captured["positive"])
    assert result["success"] is True
    assert control["enable"] is not skip
    assert captured["saved_positive"] == captured["positive"]
    assert captured["snapshot"] == (None if skip else snapshot)
    assert (tmp_path / "region_mask" / "region_mask.png").exists() is not skip
    assert item.params["positive"] == positive
    assert snapshot == original

    # A newly saved unmasked backup can be regenerated again without a mask snapshot.
    if skip:
        item.params["positive"] = captured["saved_positive"]
        item.params["illustration_multi_char"] = None
        assert (await manager._handle_regenerate(item))["success"] is True
        assert captured["snapshot"] is None
