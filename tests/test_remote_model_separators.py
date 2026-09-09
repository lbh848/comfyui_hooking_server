"""원격 실행 편집 경로의 두 가지 차단 요인 회귀 테스트.

T1(Qwen Edit 실검증) 중 드러났다. 둘 다 **플랫폼 무관**이며, 둘 다 원격 실행을
완전히 막고 있었다.

1. **로컬 모델 요구** — `qwen_edit_mode` 가 실행 대상과 무관하게 로컬 파일을
   요구했다. cloud_direct 로 26.48 GiB 를 볼륨에 올려도:

       FileNotFoundError: Qwen Rapid AIO v19 체크포인트 다운로드가 완료되지 않았습니다

   모델은 원격에 멀쩡히 있는데 "다운로드가 안 끝났다"고 말한다. 같은 판정을
   asset_tool_mode.py:335 는 이미 하고 있었다 — 이 모드만 빠져 있었다.

2. **Windows 경로 구분자** — 배포 워크플로우가 Windows 에서 작성돼 체크포인트
   이름이 ``v19\\Qwen-...safetensors`` 다. **워커는 언제나 Linux** 라 ComfyUI 는
   ``v19/Qwen-...`` 로 나열하고 제출을 거부한다:

       ckpt_name: 'v19\\Qwen-...' not in ['v19/Qwen-...']   (HTTP 400)

   즉 **Windows 사용자가 Modal 로 돌려도 똑같이 막힌다.** 업로드 쪽
   workflow_assets 는 이미 같은 정규화를 하고 있었고 제출 쪽만 빠져 있었다.
"""

from pathlib import Path

from modal_backend.service import normalize_remote_model_separators

ROOT = Path(__file__).resolve().parents[1]


def test_backslash_model_reference_is_normalized():
    """이게 안 되면 Windows 에서 만든 워크플로우는 원격에서 영원히 400 이다."""
    workflow = {
        "3": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "v19\\Qwen-Rapid-AIO-NSFW-v19.safetensors"},
        }
    }
    changed = normalize_remote_model_separators(workflow)
    assert changed == 1
    assert workflow["3"]["inputs"]["ckpt_name"] == (
        "v19/Qwen-Rapid-AIO-NSFW-v19.safetensors"
    )


def test_prompt_text_with_backslash_is_left_alone():
    """프롬프트의 역슬래시까지 바꾸면 생성 결과가 달라진다."""
    workflow = {
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "a \\ backslash"}},
    }
    assert normalize_remote_model_separators(workflow) == 0
    assert workflow["4"]["inputs"]["text"] == "a \\ backslash"


def test_every_model_suffix_is_covered():
    """체크포인트만 고치고 LoRA·VAE 를 놓치면 다른 워크플로우가 같은 벽에 부딪힌다."""
    workflow = {
        str(index): {"inputs": {"name": f"sub\\model{suffix}"}}
        for index, suffix in enumerate(
            (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".onnx", ".gguf")
        )
    }
    assert normalize_remote_model_separators(workflow) == 7
    for node in workflow.values():
        assert "\\" not in node["inputs"]["name"]


def test_nested_structures_are_walked():
    """widgets_values 처럼 리스트 안에 들어 있어도 잡아야 한다."""
    workflow = {
        "9": {"inputs": {"list": ["a\\b.safetensors", {"deep": "c\\d.ckpt"}]}},
    }
    assert normalize_remote_model_separators(workflow) == 2
    assert workflow["9"]["inputs"]["list"][0] == "a/b.safetensors"
    assert workflow["9"]["inputs"]["list"][1]["deep"] == "c/d.ckpt"


def test_already_normalized_workflow_is_untouched():
    """정규화가 멱등이어야 재제출이 안전하다."""
    workflow = {"3": {"inputs": {"ckpt_name": "v19/model.safetensors"}}}
    assert normalize_remote_model_separators(workflow) == 0
    assert workflow["3"]["inputs"]["ckpt_name"] == "v19/model.safetensors"


def test_run_workflow_normalizes_before_submitting():
    """정규화가 제출 경로에 실제로 배선돼 있어야 한다."""
    source = (ROOT / "modal_backend" / "service.py").read_text(encoding="utf-8")
    run_workflow_at = source.index("async def run_workflow")
    body = source[run_workflow_at : run_workflow_at + 4000]
    assert "normalize_remote_model_separators(workflow)" in body
