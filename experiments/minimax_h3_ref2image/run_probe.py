from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


NODE_TYPE = "SoyaMiniMaxH3ReferenceToImage_mdsoya"
DEFAULT_REFERENCE = "comfy-installer-e2e-face.png"
MAX_REFERENCES = 9
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = Path(__file__).with_name("workflow_api.json")
REQUIRED_MODELS = (
    PROJECT_ROOT
    / "comfy"
    / "models"
    / "diffusion_models"
    / "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    PROJECT_ROOT
    / "comfy"
    / "models"
    / "text_encoders"
    / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    PROJECT_ROOT
    / "comfy"
    / "models"
    / "vae"
    / "minimax_h3_t1_image_vae_step1597.safetensors",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the MiniMax H3 T=1 REF2I probe")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument(
        "--reference",
        dest="references",
        action="append",
        default=None,
        help="Comfy input-relative image name; repeat this option for 2-9 references",
    )
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ref-image-size", choices=("match", "max"), default="match")
    parser.add_argument("--timeout", type=float, default=1800.0)
    return parser.parse_args()


def request_json(url, method="GET", payload=None, timeout=30.0):
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        print(
            "[H3_REF2IMAGE_PROBE] HTTP 실패: "
            f"method={method}, url={url}, status={exc.code}, body={response_body}",
            file=sys.stderr,
            flush=True,
        )
        raise
    except Exception as exc:
        print(
            "[H3_REF2IMAGE_PROBE] 요청 실패: "
            f"method={method}, url={url}, type={type(exc).__name__}, error={exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        raise


def validate_inputs(args):
    missing_models = [str(path) for path in REQUIRED_MODELS if not path.is_file()]
    if missing_models:
        print(
            "[H3_REF2IMAGE_PROBE] 필수 모델 누락: " + ", ".join(missing_models),
            file=sys.stderr,
            flush=True,
        )
        raise FileNotFoundError("MiniMax H3 REF2I 필수 모델이 없습니다")

    references = args.references or [DEFAULT_REFERENCE]
    if len(references) > MAX_REFERENCES:
        print(
            "[H3_REF2IMAGE_PROBE] 레퍼런스 개수 오류: "
            f"count={len(references)}, max={MAX_REFERENCES}",
            file=sys.stderr,
            flush=True,
        )
        raise ValueError(f"MiniMax H3 레퍼런스는 최대 {MAX_REFERENCES}장입니다")
    for index, reference in enumerate(references, start=1):
        reference_path = PROJECT_ROOT / "comfy" / "input" / reference
        if not reference_path.is_file():
            print(
                "[H3_REF2IMAGE_PROBE] 입력 이미지 누락: "
                f"index={index}, path={reference_path}",
                file=sys.stderr,
                flush=True,
            )
            raise FileNotFoundError(
                f"Comfy input 레퍼런스 이미지 {index}가 없습니다"
            )

    for name, value in (("width", args.width), ("height", args.height)):
        if value < 32 or value % 32 != 0:
            print(
                f"[H3_REF2IMAGE_PROBE] 해상도 오류: {name}={value}, expected=32의 양의 배수",
                file=sys.stderr,
                flush=True,
            )
            raise ValueError(f"{name}는 32의 양의 배수여야 합니다")
    if args.steps < 1:
        print(
            f"[H3_REF2IMAGE_PROBE] 스텝 수 오류: steps={args.steps}",
            file=sys.stderr,
            flush=True,
        )
        raise ValueError("steps는 1 이상이어야 합니다")


def build_workflow(args):
    try:
        workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(
            "[H3_REF2IMAGE_PROBE] 워크플로 로드 실패: "
            f"path={WORKFLOW_PATH}, type={type(exc).__name__}, error={exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        raise
    workflow = copy.deepcopy(workflow)
    references = args.references or [DEFAULT_REFERENCE]
    workflow["4"]["inputs"]["image"] = references[0]
    for index, reference in enumerate(references[1:], start=2):
        load_node_id = str(12 + index)
        workflow[load_node_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": reference},
        }
        workflow["5"]["inputs"][f"ref_image_{index}"] = [load_node_id, 0]
    workflow["5"]["inputs"]["width"] = args.width
    workflow["5"]["inputs"]["height"] = args.height
    workflow["5"]["inputs"]["ref_image_size"] = args.ref_image_size
    if args.prompt is not None:
        workflow["5"]["inputs"]["prompt"] = args.prompt
    workflow["8"]["inputs"]["noise_seed"] = args.seed
    workflow["10"]["inputs"]["steps"] = args.steps
    return workflow


def sample_vram(base_url, observations):
    try:
        stats = request_json(f"{base_url}/system_stats")
        devices = stats.get("devices", [])
        if not devices:
            print(
                "[H3_REF2IMAGE_PROBE] VRAM 관찰 생략: system_stats devices가 비어 있습니다",
                file=sys.stderr,
                flush=True,
            )
            return
        device = devices[0]
        total = int(device.get("vram_total", 0))
        free = int(device.get("vram_free", 0))
        if total <= 0 or free < 0:
            print(
                "[H3_REF2IMAGE_PROBE] VRAM 관찰값 오류: "
                f"vram_total={total}, vram_free={free}",
                file=sys.stderr,
                flush=True,
            )
            return
        observations.append((total, free))
    except Exception as exc:
        print(
            "[H3_REF2IMAGE_PROBE] VRAM 관찰 실패: "
            f"type={type(exc).__name__}, error={exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()


def wait_for_history(base_url, prompt_id, timeout, vram_observations):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sample_vram(base_url, vram_observations)
        history = request_json(f"{base_url}/history/{prompt_id}")
        record = history.get(prompt_id)
        if record is not None:
            return record
        time.sleep(2.0)
    print(
        f"[H3_REF2IMAGE_PROBE] 완료 대기 시간 초과: prompt_id={prompt_id}, timeout={timeout}",
        file=sys.stderr,
        flush=True,
    )
    raise TimeoutError("MiniMax H3 REF2I probe가 제한 시간 안에 끝나지 않았습니다")


def main():
    args = parse_args()
    try:
        validate_inputs(args)
        base_url = args.comfy_url.rstrip("/")
        object_info = request_json(f"{base_url}/object_info/{NODE_TYPE}")
        if NODE_TYPE not in object_info:
            print(
                "[H3_REF2IMAGE_PROBE] 실험 노드가 Comfy에 로드되지 않았습니다. "
                f"node={NODE_TYPE}; ComfyUI를 재시작하세요.",
                file=sys.stderr,
                flush=True,
            )
            raise RuntimeError("MiniMax H3 REF2I 실험 노드가 로드되지 않았습니다")

        workflow = build_workflow(args)
        client_id = str(uuid.uuid4())
        vram_observations = []
        sample_vram(base_url, vram_observations)
        started_at = time.monotonic()
        queued = request_json(
            f"{base_url}/prompt",
            method="POST",
            payload={"prompt": workflow, "client_id": client_id},
        )
        prompt_id = queued.get("prompt_id")
        if not prompt_id:
            print(
                f"[H3_REF2IMAGE_PROBE] prompt_id 누락: response={queued!r}",
                file=sys.stderr,
                flush=True,
            )
            raise RuntimeError("Comfy가 prompt_id를 반환하지 않았습니다")

        print(
            "[H3_REF2IMAGE_PROBE] 큐 등록 완료: "
            f"prompt_id={prompt_id}, size={args.width}x{args.height}, "
            f"steps={args.steps}, seed={args.seed}, references="
            f"{len(args.references or [DEFAULT_REFERENCE])}, "
            f"ref_image_size={args.ref_image_size}",
            flush=True,
        )
        record = wait_for_history(
            base_url,
            prompt_id,
            args.timeout,
            vram_observations,
        )
        elapsed = time.monotonic() - started_at
        status = record.get("status", {})
        if status.get("status_str") != "success":
            print(
                "[H3_REF2IMAGE_PROBE] 생성 실패: "
                f"prompt_id={prompt_id}, status={status!r}",
                file=sys.stderr,
                flush=True,
            )
            raise RuntimeError("MiniMax H3 REF2I Comfy 실행이 실패했습니다")

        output_images = []
        for output in record.get("outputs", {}).values():
            output_images.extend(output.get("images", []))
        if not output_images:
            print(
                f"[H3_REF2IMAGE_PROBE] 출력 이미지 누락: prompt_id={prompt_id}",
                file=sys.stderr,
                flush=True,
            )
            raise RuntimeError("MiniMax H3 REF2I 출력 이미지가 없습니다")

        vram_summary = "unavailable"
        if vram_observations:
            total_vram = max(total for total, _free in vram_observations)
            minimum_free = min(free for _total, free in vram_observations)
            observed_used = max(0, total_vram - minimum_free)
            vram_summary = (
                f"observed_gpu_used_peak={observed_used / (1024 ** 3):.2f}GiB, "
                f"minimum_free={minimum_free / (1024 ** 3):.2f}GiB, "
                f"samples={len(vram_observations)}"
            )
        print(
            "[H3_REF2IMAGE_PROBE] 생성 완료: "
            f"prompt_id={prompt_id}, elapsed={elapsed:.2f}s, {vram_summary}, "
            f"outputs={output_images}",
            flush=True,
        )
        return 0
    except Exception as exc:
        print(
            "[H3_REF2IMAGE_PROBE] 최종 실패: "
            f"type={type(exc).__name__}, error={exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
