"""Replay recorded SELECT through its unchanged route and isolated unified queue."""
import asyncio
import copy
from datetime import datetime
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


async def main():
    from modes import illustration_context_pipeline as pipeline, lighbd_service, llm_service
    from modes import illustration_scene_pipeline as scenes
    from queue_manager import QueueManager
    import illustration_flow
    from audit_illustration_wardrobe_e2e import _load_llm_runtime

    directory = ROOT / ".work/c_slot_aliases" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (directory / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    records = [json.loads(line) for line in (ROOT / "logs/lighbd_history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    selection = next(row for row in records if row.get("execution_id") == "94ea27580ad7497bb739dc3da158099e" and row.get("attempt") == 1 and row.get("llm_slot") == "llm3")
    call1 = next(row for row in records if row.get("execution_id") == "fe02139a69014e37a17cb67f4e15786d")
    source_input = next(message["content"] for message in call1["input"] if message["role"] == "user")
    slotted = source_input.split("# CURRENT WITH SERVER SLOT BOUNDARIES\n", 1)[1].split("\n\n# SEGMENT TO SLOT ALIGNMENT", 1)[0].strip()
    catalog, segments = pipeline._segment_current_context(pipeline.remove_slot_markers(slotted), slotted_context=slotted)
    mapping, _catalog, reason = pipeline.build_segment_slot_map(slotted, segments)
    if reason or len(mapping) != len(set(mapping.values())):
        raise RuntimeError(f"One-to-one candidate construction failed: reason={reason}, mapping={mapping}")
    messages = copy.deepcopy(selection["input"])
    span = selection["input_variables"][1][-1]
    content = messages[1]["content"]
    messages[1]["content"] = content[:span["start"]] + catalog + content[span["end"]:]
    save("recorded_input.json", {"source_execution_id": selection["execution_id"], "source_call1_id": call1["execution_id"], "target_slotted": slotted})
    save("select_input.json", {"messages": messages, "mapping": mapping})

    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    keys = json.loads((ROOT / "key/llm_keys.json").read_text(encoding="utf-8"))
    llm_service.LOG_DIR = str(directory / "llm_logs")
    llm_service.HISTORY_PATH = str(directory / "llm_logs/llm_history.jsonl")
    llm_service.HISTORY_BACKUP_DIR = str(directory / "llm_logs/backups")
    llm_service.HISTORY_BACKUP_PATH = str(directory / "llm_logs/backups/llm_history.jsonl.bak")
    lighbd_service.LOG_DIR = str(directory / "lb_details")
    lighbd_service.LIGHBD_HISTORY_PATH = str(directory / "lb_details/lighbd_history.jsonl")
    _load_llm_runtime(config, keys)
    manager = QueueManager()
    manager.get_config = lambda: config
    events = []
    async def notify(event_type, data):
        events.append({"type": event_type, "data": copy.deepcopy(data)})
    manager.notify_frontend = notify
    parent = None
    async def stream(event):
        if event.get("queue_subtask"):
            await manager.update_subtask(parent, copy.deepcopy(event["queue_subtask"]),
                                         {key: value for key, value in event.items() if key != "queue_subtask"})
        events.append({"type": "llm_stream", "data": copy.deepcopy(event)})
    def parse(response):
        return scenes.parse_plan(response, segments=segments, slot_map=mapping,
                                 candidates=list(mapping.values()), names=["Doyun"],
                                 toggles={"output_count_min": 12, "output_count_max": 16, "character_limit": 2},
                                 stage="CALL2-SELECT")
    def validate(response):
        try:
            parse(response)
            return True, ""
        except ValueError as error:
            return False, str(error)
    trace = []
    async def execute(item):
        response = await pipeline._call_pipeline_llm("CALL2-SELECT", pipeline._normalize_messages(messages),
                                                    stream, result_validator=validate, json_mode=True,
                                                    history_ids_sink=trace)
        selected = parse(response)
        return {"selected": selected, "response": response, "slots": [mapping[row["anchor_segment"]] for row in selected]}
    manager.process_illustration_context = execute
    try:
        parent = await manager.add_item("illustration_llm_build", "C-Slot 1:1 verification · risu_7421b9f", {}, priority=0)
        result = await asyncio.wait_for(parent.completion_future, timeout=300)
        save("result.json", {**result, "trace_ids": trace, "queue_parent": parent.to_dict(),
                             "flow": illustration_flow.snapshot(), "routing": config["llm_routing"]["illustration_call2_select"]})
        print(f"PASS: selected={len(result['selected'])}, unique_slots={len(set(result['slots']))}, trace={trace}, artifacts={directory}", flush=True)
    finally:
        save("queue_events.json", events)
        pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        traceback.print_exc()
        raise
