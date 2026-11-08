# Gemma 4 이미지 분석 유틸리티

Base64 이미지와 지시 프롬프트를 담은 JSON을 받아 텍스트를 반환합니다. 요청마다 이전 대화를
넘기지 않으며, 응답이 끝나도 모델을 유지합니다. 다음 GPU 이미지 모델을 로드하기
직전에 llama.cpp를 종료하고, 이후 분석 요청에서 다시 로드합니다.

- 모델: [zaakirio/gemma-4-12b-it-uncensored-GGUF](https://huggingface.co/zaakirio/gemma-4-12b-it-uncensored-GGUF), Q4_K_M
- 비전: 같은 저장소의 `mmproj-gemma-4-12B-it-bf16.gguf`
- 생성 가속: 같은 저장소의 `mtp-gemma-4-12b-it-uncensored.gguf`를 사용하는 MTP, 최대 2토큰 초안.
- 컨텍스트: **16,384**, 팩의 최대 응답: **8,196** 토큰, 온도 **1.0**, 추론 모드 활성화(최대 **2,048** 토큰).
  샘플링은 `top_p=0.95`, `top_k=64`, `min_p=0`, 반복 페널티 **1.0**을 사용합니다.
- 8GB 이상 NVIDIA GPU 대상: `gpu_layers` 자동, 메모리 맞춤 활성화, 여유 VRAM 목표 **1,536MiB**.
  컨텍스트는 고정하고, 필요한 경우 레이어를 CPU로 오프로드합니다.
- CPU 오프로드에는 시스템 RAM이 추가로 필요합니다. 다른 GPU 작업의 메모리 점유량에
  따라 실제 배치와 속도는 달라집니다.

2026-11-03 RTX 4060 8GB에서 같은 768×768 이미지, Q4_K_M, 16K 컨텍스트,
추론 활성화, 512토큰 생성으로 3회씩 측정한 중앙값은 기존 **14.25 tok/s**,
MTP 2토큰 **19.79 tok/s**, MTP 3토큰 **18.77 tok/s**였습니다.
2토큰 설정의 생성 속도는 약 **39%** 높았고 요청 시간은 **37.64 → 27.59초**였습니다.
화면 표시 등으로 VRAM 일부를 사용 중인 환경의 측정이며, 전체 답변 시간이나
다른 GPU의 성능을 보장하는 수치는 아닙니다. 이미지 인식 품질 비교는 수행하지 않았습니다.

## 설치

V6 워크플로우 팩의 `유틸리티 → Gemma4 이미지 분석`에 포함됩니다.
`워크플로우 관리`에서 설치하면 GGUF 파일 3개(본문 모델, 비전 프로젝터, MTP 보조 모델)와 고정된 `comfyui-llamacpp`,
SOYA 노드 및 Windows CUDA 실행 파일을 준비합니다. `LLM_SP`는 팩에서 설치한
워크플로우 경로를 사용합니다.

프로젝트의 `src`에서 실행합니다. 모델 약 8.02GB와 Windows CUDA 런타임을 내려받고,
고정된 버전과 SHA-256을 확인합니다. 기존 설치 파일은 재사용합니다.

```powershell
uv run python -m scripts.install_gemma4_vision_utility --comfy-root ../data/comfy
```

기존 `comfyui-soya-custom-nodes`에 `Gemma 4 Image Analyze (Soya)`를 등록합니다.
GGUF 실행에는 고정 버전의 `comfyui-llamacpp`를 사용하며, 응답 표시는 기존
`comfyui-custom-scripts`의 `Show Text 🐍`를 사용합니다.
다른 운영체제의 NVIDIA CUDA 환경에서는 해당 플랫폼의 Gemma 4 Unified 지원
`llama-server` 실행 파일을 `--llama-binary`로 지정합니다. 기본 `gpu_device`는 **0**입니다.
커스텀 노드 설치 후 ComfyUI를 다시 시작합니다.

설치되는 워크플로우: `배포_공방_Gemma4_이미지분석_V1_0.json`.
수정된 기존 워크플로우에는 덮어쓰지 않으며 `--workflow-output`으로 새 경로를 지정할 수 있습니다.

## 사용

`LLM API 관리 → LLM 설정`에서 두 연결이 같은 워크플로우와 모델을 공유합니다.

- `공방-로컬 LLM (추론)`: 추론 토큰 한도를 1,024 / 2,048 / 4,096 중 선택합니다. 기본값은 2,048입니다.
- `공방-로컬 LLM (비추론)`: 추론을 사용하지 않습니다.

두 연결의 컨텍스트 기본값은 32K(32,768토큰)이며, 8K부터 256K까지 8K 단위로 설정할 수 있습니다.

`Comfy 실행 관리`에서 Comfy #1을 실행한 뒤 `테스트`로 호출할 수 있습니다.
이미지는 선택 사항입니다. 텍스트만 보낼 때는 JSON의 `image`를 생략하거나 `null`로 둡니다.
이미지를 추가할 때는 최대 1장을 Base64로 전달합니다.
LLM_SP의 응답은 생성되는 동안 `테스트`와 `LLM 로그`에 실시간으로 표시됩니다.
`LLM 로그` 상세의 `추론 내용`에는 모델이 반환한 추론 텍스트를 저장합니다.
출력 토큰 수와 속도는 llama.cpp가 보고한 추론 및 최종 답변 전체 사용량을 기준으로 합니다.
중지하면 해당 Gemma 생성과 Comfy 작업만 취소하고, 모델은 계속 유지합니다.

1. `Gemma 4 Image Analyze (Soya)`의 `prompt_json`에 요청 JSON을 넣고 실행합니다.
2. `분석 응답`에서 결과를 확인합니다. STRING 출력은 다른 노드에 연결할 수 있습니다.

워크플로우에는 실행 가능한 예제 이미지가 Base64로 들어 있습니다. 실제 요청은 다음
형식으로 전송합니다. `sha256`은 Base64 문자열이 아닌 원본 이미지 바이트의 SHA-256입니다.
노드는 기존 SOYA 이미지 디코더를 재사용하여 메모리에서 검증·디코딩합니다.
이미지를 ComfyUI 입력 폴더에 저장하거나 `/upload/image`로 전송할 필요가 없습니다.

```json
{
  "schema": "soya.gemma4.vision.v1",
  "prompt": "Analyze this image and describe the composition. Answer in Korean.",
  "image": {
    "encoding": "base64",
    "media_type": "image/png",
    "sha256": "<SHA-256 of the original image bytes>",
    "data": "<Base64 image bytes, without a data-URL prefix>"
  }
}
```

응답 후 모델을 유지하고, 각 요청은 이전 메시지나 프롬프트 캐시를 사용하지 않습니다.
분석 모델을 새로 로드할 때 ComfyUI의 GPU 모델을 먼저 내립니다.
`context_size`, `max_tokens`, `temperature`, `seed`, `gpu_layers`, `gpu_device`,
`binary_path`, `port`를 노드에서 조절할 수 있습니다.

API 사용 시 설치된 워크플로우를 기존 `/workflow/convert`에 보내 API 형식으로
변환합니다. 노드 `4`의 `inputs.prompt_json`에 위 JSON을 직렬화한 문자열을 넣습니다.
`/prompt`로 실행한 뒤 `/history/{prompt_id}`의 `outputs["5"].text`에서 응답을 읽습니다.
노드 `4`도 같은 텍스트를 `outputs["4"].text`로 반환하므로 표시 노드 없이 호출할 수 있습니다.

## 메모리 전환 범위

같은 ComfyUI 인스턴스에서 이미지 모델의 초기 장치 선택 또는 `load_models_gpu` 직전에
전환합니다. HIGH_VRAM에서도 체크포인트가 GPU에 먼저 생성되기 전에 해제합니다.
다른 ComfyUI 인스턴스나 외부 프로그램의 GPU 작업은 이 전환 대상이 아닙니다.
외부 작업으로 전환할 때는 ComfyUI의 모델 해제를 먼저 실행합니다.

추론과 프로세스 종료는 [comfyui-llamacpp](https://github.com/Setmaster/comfyui-llamacpp)의
기존 관리 기능을 사용합니다. SOYA 패키지의 `soya_gemma4_vision.py`가 입력과 실행을,
`soya_vram_handoff.py`가 이미지 모델 로드 시 해제를 담당합니다.
ComfyUI 코어 파일은 수정하지 않습니다. 커스텀 노드를 별도로 업데이트했다면
이 설치 모듈도 함께 확인해야 합니다. 노드와 런타임의 고정 버전은 `dependencies.json`,
모델 다운로드 정보와 SHA-256은 상위 `install_manifest.json`에 기록되어 있습니다.

별도 설치 스크립트도 같은 노드·런타임 설치 기능을 재사용합니다.
