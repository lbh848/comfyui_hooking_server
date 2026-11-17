"""Resolve selected training workflows and apply their execution settings."""
import copy
import json
from pathlib import Path
from app_paths import SOURCE_DIR, resolve_install_path

TRAINING_OPTIMIZERS = ("default", "AdamW", "AdamW8bit", "Muon")


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


def training_workflow_epoch_settings(training_config: dict) -> dict:
    """Read the selected workflow's literal epoch settings for preview and submission."""
    if not training_config.get("workflow_source_path", ""):
        return {}
    from illustration_json import workflow_json_settings_nodes

    workflow_path = training_workflow_source_path(training_config)
    workflow = json.loads(Path(workflow_path).read_text(encoding="utf-8-sig"))
    nodes = workflow_json_settings_nodes(workflow, widget_names=("max_train_epochs", "gradient_accumulation_steps"))
    epoch_settings = []
    for node in nodes.values():
        if not isinstance(node, dict) or node.get("class_type") != "md_soya_AnimaTrainOptionsV2" or node.get("mode", 0) != 0:
            continue
        inputs = node.get("inputs", {})
        settings = {"max_train_epochs": inputs.get("max_train_epochs", 0),
                    "gradient_accumulation_steps": inputs.get("gradient_accumulation_steps", 1)}
        for key, minimum in (("max_train_epochs", 0), ("gradient_accumulation_steps", 1)):
            if type(settings[key]) is not int or settings[key] < minimum:
                raise ValueError(f"학습 워크플로우의 {key} 값을 확인하세요.")
        epoch_settings.append(settings)
    if any(settings != epoch_settings[0] for settings in epoch_settings):
        raise ValueError("학습 워크플로우의 Epochs와 Gradient Accumulation 설정을 통일하세요.")
    return epoch_settings[0] if epoch_settings else {}


def build_training_workflow(api_workflow: dict, positive_text: str, negative_text: str, total_steps: int, optimizer_type: str = "default") -> dict:
    """Bind captions and an explicit step limit without duplicating the dataset."""
    if type(total_steps) is not int or total_steps < 1:
        raise ValueError("전체 STEP은 1 이상이어야 합니다.")
    optimizer_type = training_optimizer_type(optimizer_type)
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
        if node.get("class_type") in ("md_soya_ReferenceTrainOptions", "md_soya_AnimaTrainOptionsV2"):
            # Replace the legacy N_IMG * STEP_PER_IMAGE link with the exact count.
            node.setdefault("inputs", {})["steps_override"] = total_steps
            if optimizer_type != "default":
                node["inputs"]["optimizer_type"] = optimizer_type
            step_options += 1
    if not step_options:
        raise ValueError("선택한 학습 워크플로우에 전체 STEP을 전달할 학습 옵션 노드가 없습니다.")
    return workflow
