"""Resolve selected training workflows and apply their execution settings."""
import copy
import json
from pathlib import Path
from app_paths import SOURCE_DIR, resolve_install_path

TRAINING_OPTIMIZERS = ("default", "AdamW", "AdamW8bit", "Muon")
EPOCH_TRAINING_NODES = {"anima": "md_soya_AnimaTrainOptionsV2", "moss": "md_soya_MossTrainOptionsV2"}


def training_optimizer_type(value="default") -> str:
    if not isinstance(value, str) or value not in TRAINING_OPTIMIZERS:
        raise ValueError("지원하는 학습 옵티마이저를 선택하세요.")
    return value


def training_workflow_source_path(training_config: dict, project_root: Path = SOURCE_DIR) -> str:
    selected = training_config.get("workflow_source_path", "")
    if not isinstance(selected, str):
        raise ValueError("학습 워크플로우 경로를 확인하세요.")
    workflow_path = selected.strip()
    if not workflow_path:
        raise ValueError("학습 워크플로우를 선택하세요.")
    workflow_path = resolve_install_path(workflow_path, project_root)
    if not workflow_path.is_file():
        raise ValueError(f"학습 워크플로우 파일을 찾을 수 없습니다: {workflow_path}")
    return str(workflow_path)


def training_workflow_accumulation_steps(training_config: dict) -> int:
    """Read the shared batch-1 accumulation setting used to size epoch training."""
    from illustration_json import workflow_json_settings_nodes

    workflow_path = training_workflow_source_path(training_config)
    workflow = json.loads(Path(workflow_path).read_text(encoding="utf-8-sig"))
    nodes = workflow_json_settings_nodes(workflow, widget_names=("gradient_accumulation_steps",))
    options_type = EPOCH_TRAINING_NODES.get(training_config.get("profile", "anima"))
    accumulation_steps = []
    for node in nodes.values():
        if not isinstance(node, dict) or node.get("class_type") != options_type or node.get("mode", 0) != 0:
            continue
        inputs = node.get("inputs", {})
        gradient_accumulation_steps = inputs.get("gradient_accumulation_steps", 1)
        if type(gradient_accumulation_steps) is not int or gradient_accumulation_steps < 1:
            raise ValueError("학습 워크플로우의 gradient_accumulation_steps 값을 확인하세요.")
        accumulation_steps.append(gradient_accumulation_steps)
    if not accumulation_steps:
        raise ValueError("Epochs 학습에는 선택한 프로필의 V2 학습 워크플로우가 필요합니다.")
    if len(set(accumulation_steps)) != 1:
        raise ValueError("학습 워크플로우의 Gradient Accumulation 설정을 통일하세요.")
    return accumulation_steps[0]


def build_training_workflow(api_workflow: dict, positive_text: str, negative_text: str, total_steps: int, optimizer_type: str = "default", *, training_config: dict | None = None) -> dict:
    """Bind captions and an explicit step limit without duplicating the dataset."""
    if type(total_steps) is not int or total_steps < 1:
        raise ValueError("전체 STEP은 1 이상이어야 합니다.")
    optimizer_type = training_optimizer_type(optimizer_type)
    from lora_training_steps import training_step_settings
    steps = training_step_settings(training_config or {})
    workflow = copy.deepcopy(api_workflow)
    step_options = 0
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        title = node.get("_meta", {}).get("title", "")
        if title == "긍정프롬프트":
            node["inputs"]["value"] = positive_text
        elif title == "부정프롬프트":
            node["inputs"]["value"] = negative_text
        if node.get("class_type") == "md_soya_ReferenceTrainOptions" or node.get("class_type") in EPOCH_TRAINING_NODES.values():
            # Replace the legacy N_IMG * STEP_PER_IMAGE link with the exact count.
            node.setdefault("inputs", {})["steps_override"] = total_steps
            if node.get("class_type") in EPOCH_TRAINING_NODES.values():
                node["inputs"]["max_train_epochs"] = steps["max_train_epochs"] if steps["step_mode"] == "epochs" else 0
            if optimizer_type != "default":
                node["inputs"]["optimizer_type"] = optimizer_type
            step_options += 1
    if not step_options:
        raise ValueError("선택한 학습 워크플로우에 전체 STEP을 전달할 학습 옵션 노드가 없습니다.")
    return workflow
