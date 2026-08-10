# syntax=docker/dockerfile:1.7

# linux/amd64 digests are pinned so every publisher builds the same inputs.
ARG UV_IMAGE="ghcr.io/astral-sh/uv@sha256:3835041081b0148268695cf4177602e3e1ff0216ecf1f882bac9dea744fd8e99"
ARG CUDA_IMAGE="nvidia/cuda@sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7"

FROM ${UV_IMAGE} AS uv
FROM ${CUDA_IMAGE}

ARG PYTHON_VERSION="3.12"
ARG TORCH_VERSION="2.11.0"
ARG TORCHVISION_VERSION="0.26.0"
ARG TORCHAUDIO_VERSION="2.11.0"
ARG PYTORCH_CUDA_INDEX_URL="https://download.pytorch.org/whl/cu128"
ARG SAGEATTENTION_REF="eb615cf6cf4d221338033340ee2de1c37fbdba4a"
ARG SAGEATTENTION_ARCH_LIST="8.0;8.6;8.9;12.0"
ARG COMFY_REF="64b8457f55cd7fb54ca7a956d9c73b505e903e0c"
ARG VCS_REF="unknown"

LABEL org.opencontainers.image.title="SOYA ComfyUI CUDA runtime" \
      org.opencontainers.image.description="ComfyUI runtime with PyTorch cu128 and SageAttention kernels for Modal GPUs" \
      org.opencontainers.image.source="https://github.com/lbh848/comfyui_hooking_server" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.licenses="GPL-3.0-or-later"

COPY --from=uv /uv /uvx /usr/local/bin/

ENV DEBIAN_FRONTEND="noninteractive" \
    CUDA_HOME="/usr/local/cuda" \
    CC="/usr/bin/gcc" \
    CXX="/usr/bin/g++" \
    CUDAHOSTCXX="/usr/bin/g++" \
    PYTHONUNBUFFERED="1" \
    UV_LINK_MODE="copy" \
    UV_PYTHON_DOWNLOADS="automatic" \
    VIRTUAL_ENV="/opt/soya-runtime/.venv" \
    PATH="/opt/soya-runtime/.venv/bin:/usr/local/bin:/usr/local/cuda/bin:${PATH}"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/soya-runtime

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python "${PYTHON_VERSION}" --seed .venv

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "${VIRTUAL_ENV}/bin/python" \
        --index-url "${PYTORCH_CUDA_INDEX_URL}" \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}"

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python "${VIRTUAL_ENV}/bin/python" \
        ninja \
        packaging \
        requests \
        wheel

COPY verify_sageattention_image.py /opt/soya-runtime/verify_sageattention_image.py

# GPU hardware is not required for this step. SageAttention reads the explicit
# architecture list and nvcc emits every requested cubin in the Linux builder.
RUN --mount=type=cache,target=/root/.cache/uv \
    git init /tmp/SageAttention \
    && git -C /tmp/SageAttention remote add origin https://github.com/thu-ml/SageAttention.git \
    && git -C /tmp/SageAttention fetch --depth 1 origin "${SAGEATTENTION_REF}" \
    && git -C /tmp/SageAttention checkout --detach FETCH_HEAD \
    && cd /tmp/SageAttention \
    && TORCH_CUDA_ARCH_LIST="${SAGEATTENTION_ARCH_LIST}" \
       MAX_JOBS="2" \
       EXT_PARALLEL="1" \
       NVCC_APPEND_FLAGS="--threads 2" \
       uv pip install --python "${VIRTUAL_ENV}/bin/python" --no-build-isolation . \
    && cd /opt/soya-runtime \
    && python verify_sageattention_image.py --static \
    && rm -rf /tmp/SageAttention

RUN git init /root/ComfyUI \
    && git -C /root/ComfyUI remote add origin https://github.com/comfyanonymous/ComfyUI.git \
    && git -C /root/ComfyUI fetch --depth 1 origin "${COMFY_REF}" \
    && git -C /root/ComfyUI checkout --detach FETCH_HEAD \
    && uv pip install --python "${VIRTUAL_ENV}/bin/python" \
        --requirements /root/ComfyUI/requirements.txt \
    && rm -rf /root/ComfyUI/models

WORKDIR /root/ComfyUI
EXPOSE 8188

# Modal requires an empty compatible entrypoint and starts ComfyUI itself.
ENTRYPOINT []
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
