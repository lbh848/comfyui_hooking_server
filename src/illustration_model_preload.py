"""Load illustration models through the normal Comfy execution queue, without sampling."""
from __future__ import annotations

import asyncio
import copy
import json
import logging
from pathlib import Path
import uuid

import aiohttp

from illustration_json import SCHEMA, workflow_json_defaults


async def run_model_load(operation, *, label: str) -> None:
    logger = logging.getLogger(__name__)
    logger.info("[특수 전략] %s 시작", label)
    try:
        await operation
        logger.info("[특수 전략] %s 완료", label)
    except Exception:
        # Loading ahead is optional. A failed warmup must not discard finished images.
        logger.exception("[특수 전략] %s 실패", label)


async def load_local_llm_after_generation(toggles: dict) -> None:
    if not toggles.get("local_llm_load_after_generation"):
        return
    from llm_sp_workflow import analyze
    from llm_settings import LLM_SP, LLM_SP_SLOTS
    from modes import llm_service

    config = llm_service.get_config()
    # CHARACTER-RESOLVE is the next illustration's first LLM call. Match its
    # local slot, especially its context size, so manager.start can reuse it.
    slot_id = llm_service.routing_primary_slot("illustration_character_resolve")
    if slot_id not in LLM_SP_SLOTS:
        slot_id = LLM_SP["id"]
    context_size = config.get(f"llm_context_size{slot_id[3:]}", LLM_SP["context_size"])
    logging.getLogger(__name__).info("[특수 전략] 공방-LLM 로드 설정: slot=%s context_size=%s", slot_id, context_size)
    await run_model_load(analyze([], preload_only=True, slot_id=slot_id,
        context_size=context_size,
        reasoning_budget=config.get("llm_reasoning_budget_tokens_sp") or 2048), label="공방-LLM 로드")


def image_model_load_graph(workflow: dict) -> dict:
    specs = json.loads((Path(__file__).parent / "comfy_installer/illustration_v2_specs.json").read_text(encoding="utf-8"))
    model_types = {name for name, spec in specs.items() if spec.get("model") and "MODEL" in spec["output"]}
    model_types.update({"UNETLoader", "CheckpointLoaderSimple", "CheckpointLoaderKJ"})
    payload = json.dumps({"schema": SCHEMA, "fields": workflow_json_defaults(workflow), "assets": {}}, ensure_ascii=False)
    graph = {}
    for identity, node in workflow.items():
        if node.get("class_type") not in model_types:
            continue
        loader = copy.deepcopy(node)
        inputs = loader.get("inputs", {})
        if "prompt_json" in inputs:
            inputs["prompt_json"] = payload
        if any(isinstance(value, list) for value in inputs.values()):
            raise ValueError("그림 모델 로더의 연결된 입력은 미리 로드할 수 없습니다.")
        graph[str(identity)] = loader
    if not graph or len(graph) > 2:
        raise ValueError("미리 로드할 그림 모델 로더가 1~2개인 워크플로우가 필요합니다.")
    inputs = {"model" if index == 0 else "secondary_model": [identity, 0]
              for index, identity in enumerate(graph)}
    graph["soya_model_preload"] = {"class_type": "SoyaModelPreload_mdsoya", "inputs": inputs}
    return graph


async def load_image_model(workflow: dict, *, base_url: str) -> None:
    graph = image_model_load_graph(workflow)
    prompt_id = str(uuid.uuid4())
    completed = False
    submitted = False
    timeout = aiohttp.ClientTimeout(total=30, connect=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def request(method, path, **kwargs):
            async with session.request(method, base_url + path, **kwargs) as response:
                if response.status != 200:
                    raise RuntimeError(f"그림 모델 로드 요청 실패: {path} (HTTP {response.status}). Comfy 노드 설치를 확인하세요.")
                return await response.json()
        try:
            async with asyncio.timeout(600):
                submitted = True
                result = await request("POST", "/prompt", json={"prompt": graph, "prompt_id": prompt_id})
                if result.get("prompt_id") != prompt_id:
                    raise RuntimeError("그림 모델 로드 작업 ID가 일치하지 않습니다.")
                while True:
                    history = await request("GET", f"/history/{prompt_id}")
                    record = history.get(prompt_id, {})
                    status = record.get("status", {})
                    if status.get("status_str") == "error":
                        completed = True
                        details = next((message[1].get("exception_message", "") for message in status.get("messages", [])
                                        if message[0] == "execution_error"), "Comfy 실행 로그를 확인하세요.")
                        raise RuntimeError(f"그림 모델 로드 실패: {details}")
                    if status.get("completed"):
                        completed = True
                        if not record.get("outputs", {}).get("soya_model_preload", {}).get("text"):
                            raise RuntimeError("그림 모델 로드 완료 결과가 없습니다.")
                        return
                    await asyncio.sleep(0.5)
        finally:
            if submitted and not completed:
                try:
                    async with session.post(base_url + f"/api/jobs/{prompt_id}/cancel", timeout=aiohttp.ClientTimeout(total=5)) as response:
                        response.raise_for_status()
                except (aiohttp.ClientError, TimeoutError):
                    logging.warning("그림 모델 로드 작업 취소 실패: %s", prompt_id)
