"""Require the workflow explicitly selected in the training settings."""
from pathlib import Path


def training_workflow_source_path(training_config: dict) -> str:
    selected = training_config.get("workflow_source_path", "")
    if not isinstance(selected, str):
        raise ValueError("학습 워크플로우 경로를 확인하세요.")
    workflow_path = selected.strip()
    if not workflow_path:
        raise ValueError("학습 워크플로우를 선택하세요.")
    if not Path(workflow_path).is_file():
        raise ValueError(f"학습 워크플로우 파일을 찾을 수 없습니다: {workflow_path}")
    return str(workflow_path)
