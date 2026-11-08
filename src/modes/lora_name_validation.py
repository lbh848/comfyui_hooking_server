from lora_paths import validate_lora_storage_name


def validate_lora_project_name(name: str) -> str:
    """Return an error message when a project name cannot be used as-is as a directory."""
    return validate_lora_storage_name(name, "LoRA 프로젝트명")
