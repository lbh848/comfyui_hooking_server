from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGETS = (
    Path(__file__).with_name("workflow_canvas.json"),
    PROJECT_ROOT
    / "comfy"
    / "user"
    / "default"
    / "workflows"
    / "SOYA_USER"
    / "실험_이미지_H3_REF2I_T1_v1.json",
)


def socket(name, socket_type, link=None, *, widget=False, optional=False):
    value = {
        "localized_name": name,
        "name": name,
        "type": socket_type,
        "link": link,
    }
    if widget:
        value["widget"] = {"name": name}
    if optional:
        value["shape"] = 7
    return value


def output(name, socket_type, links=None):
    return {
        "localized_name": name,
        "name": name,
        "type": socket_type,
        "links": links,
    }


def node(
    node_id,
    node_type,
    pos,
    size,
    order,
    inputs,
    outputs,
    widgets_values,
    *,
    properties=None,
    title=None,
):
    value = {
        "id": node_id,
        "type": node_type,
        "pos": list(pos),
        "size": list(size),
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": inputs,
        "outputs": outputs,
        "properties": properties or {"Node name for S&R": node_type},
        "widgets_values": widgets_values,
    }
    if title is not None:
        value["title"] = title
    return value


def build_workflow():
    nodes = [
        node(
            1,
            "UNETLoader",
            (-1180, -540),
            (520, 90),
            0,
            [
                socket("unet_name", "COMBO", widget=True),
                socket("weight_dtype", "COMBO", widget=True),
            ],
            [output("MODEL", "MODEL", [1])],
            ["minimax_h3_ref2va_pruned_int8_convrot.safetensors", "default"],
            properties={
                "Node name for S&R": "UNETLoader",
                "models": [
                    {
                        "name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
                        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
                        "directory": "diffusion_models",
                    }
                ],
            },
        ),
        node(
            2,
            "CLIPLoader",
            (-1180, -390),
            (520, 120),
            1,
            [
                socket("clip_name", "COMBO", widget=True),
                socket("type", "COMBO", widget=True),
                socket("device", "COMBO", widget=True, optional=True),
            ],
            [output("CLIP", "CLIP", [4])],
            [
                "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                "minimax",
                "default",
            ],
            properties={
                "Node name for S&R": "CLIPLoader",
                "models": [
                    {
                        "name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                        "directory": "text_encoders",
                    }
                ],
            },
        ),
        node(
            3,
            "VAELoader",
            (-1180, -210),
            (520, 70),
            2,
            [socket("vae_name", "COMBO", widget=True)],
            [output("VAE", "VAE", [5, 14])],
            ["minimax_h3_t1_image_vae_step1597.safetensors"],
            properties={
                "Node name for S&R": "VAELoader",
                "models": [
                    {
                        "name": "minimax_h3_t1_image_vae_step1597.safetensors",
                        "url": "https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE/resolve/main/minimax_h3_t1_image_vae_step1597.safetensors",
                        "directory": "vae",
                    }
                ],
            },
        ),
        node(
            4,
            "LoadImage",
            (-1180, -80),
            (520, 430),
            3,
            [
                socket("image", "COMBO", widget=True),
                socket("upload", "IMAGEUPLOAD", widget=True),
            ],
            [
                output("IMAGE", "IMAGE", [6]),
                output("MASK", "MASK", None),
            ],
            ["comfy-installer-e2e-face.png", "image"],
        ),
        node(
            5,
            "SoyaMiniMaxH3ReferenceToImage_mdsoya",
            (-540, -250),
            (490, 600),
            5,
            [
                socket("clip", "CLIP", 4),
                socket("vae", "VAE", 5),
                socket("ref_image_1", "IMAGE", 6),
                socket("ref_image_2", "IMAGE", optional=True),
                socket("ref_image_3", "IMAGE", optional=True),
                socket("ref_image_4", "IMAGE", optional=True),
                socket("ref_image_5", "IMAGE", optional=True),
                socket("ref_image_6", "IMAGE", optional=True),
                socket("ref_image_7", "IMAGE", optional=True),
                socket("ref_image_8", "IMAGE", optional=True),
                socket("ref_image_9", "IMAGE", optional=True),
                socket("prompt", "STRING", widget=True),
                socket("width", "INT", widget=True),
                socket("height", "INT", widget=True),
                socket("ref_image_size", "COMBO", widget=True),
            ],
            [
                output("positive", "CONDITIONING", [7]),
                output("latent", "LATENT", [8]),
                output("diagnostics", "STRING", None),
            ],
            [
                "Create a polished still image using <Picture 1> as the character identity reference. Preserve the face, hair, accessories, outfit colors, and defining features while changing the composition to a waist-up portrait with natural lighting and a clean background.",
                512,
                512,
                "match",
            ],
            title="MiniMax H3 REF2I T=1 — 1~9 References",
        ),
        node(
            6,
            "MiniMaxH3SigmaShift",
            (-540, -540),
            (490, 120),
            4,
            [
                socket("model", "MODEL", 1),
                socket("shift_video", "FLOAT", widget=True),
                socket("shift_audio", "FLOAT", widget=True),
            ],
            [output("MODEL", "MODEL", [2, 3])],
            [12.0, 3.0],
            title="H3 Sampling Shift — video 12 / audio 3",
        ),
        node(
            7,
            "BasicGuider",
            (80, -540),
            (320, 70),
            6,
            [socket("model", "MODEL", 2), socket("conditioning", "CONDITIONING", 7)],
            [output("GUIDER", "GUIDER", [9])],
            [],
        ),
        node(
            8,
            "BasicScheduler",
            (80, -420),
            (320, 130),
            7,
            [
                socket("model", "MODEL", 3),
                socket("scheduler", "COMBO", widget=True),
                socket("steps", "INT", widget=True),
                socket("denoise", "FLOAT", widget=True),
            ],
            [output("SIGMAS", "SIGMAS", [10])],
            ["simple", 12, 1.0],
        ),
        node(
            9,
            "RandomNoise",
            (80, -240),
            (320, 90),
            8,
            [socket("noise_seed", "INT", widget=True)],
            [output("NOISE", "NOISE", [11])],
            [42, "fixed"],
        ),
        node(
            10,
            "KSamplerSelect",
            (80, -100),
            (320, 70),
            9,
            [socket("sampler_name", "COMBO", widget=True)],
            [output("SAMPLER", "SAMPLER", [12])],
            ["res_multistep"],
        ),
        node(
            11,
            "SamplerCustomAdvanced",
            (470, -430),
            (280, 180),
            10,
            [
                socket("noise", "NOISE", 11),
                socket("guider", "GUIDER", 9),
                socket("sampler", "SAMPLER", 12),
                socket("sigmas", "SIGMAS", 10),
                socket("latent_image", "LATENT", 8),
            ],
            [
                output("output", "LATENT", [13]),
                output("denoised_output", "LATENT", None),
            ],
            [],
        ),
        node(
            12,
            "VAEDecode",
            (820, -430),
            (260, 80),
            11,
            [socket("samples", "LATENT", 13), socket("vae", "VAE", 14)],
            [output("IMAGE", "IMAGE", [15])],
            [],
            title="Decode T=1 with H3 Image VAE",
        ),
        node(
            13,
            "SaveImage",
            (1140, -470),
            (430, 420),
            12,
            [
                socket("images", "IMAGE", 15),
                socket("filename_prefix", "STRING", widget=True),
            ],
            [],
            ["h3_ref2image_canvas"],
        ),
        node(
            14,
            "MarkdownNote",
            (-1180, -870),
            (1210, 250),
            13,
            [],
            [],
            (
                "# MiniMax H3 REF2I T=1 — local experiment\n\n"
                "Start with **512×512 / 12 steps / ref_image_size=match / one reference**. "
                "Connect additional Load Image nodes to ref_image_2 through ref_image_9 without gaps, "
                "and address them as `<Picture 2>` ... `<Picture 9>` in the prompt.\n\n"
                "The custom node must report a video latent shaped `[1,24,1,H/16,W/16]` and an "
                "empty audio latent `[1,32,2,0]`. Decode only with "
                "`minimax_h3_t1_image_vae_step1597.safetensors`. Do not use the video/audio VAE, "
                "CreateVideo, or SaveVideo here.\n\n"
                "Observed on RTX 4080 16GB: 768×768 with one `match` reference reached about "
                "15.75 GiB system-wide GPU use. Multi-reference role separation is experimental; "
                "a visually dominant pose/scene reference can override character identity."
            ),
            title="Read before running",
        ),
    ]

    links = [
        [1, 1, 0, 6, 0, "MODEL"],
        [2, 6, 0, 7, 0, "MODEL"],
        [3, 6, 0, 8, 0, "MODEL"],
        [4, 2, 0, 5, 0, "CLIP"],
        [5, 3, 0, 5, 1, "VAE"],
        [6, 4, 0, 5, 2, "IMAGE"],
        [7, 5, 0, 7, 1, "CONDITIONING"],
        [8, 5, 1, 11, 4, "LATENT"],
        [9, 7, 0, 11, 1, "GUIDER"],
        [10, 8, 0, 11, 3, "SIGMAS"],
        [11, 9, 0, 11, 0, "NOISE"],
        [12, 10, 0, 11, 2, "SAMPLER"],
        [13, 11, 0, 12, 0, "LATENT"],
        [14, 3, 0, 12, 1, "VAE"],
        [15, 12, 0, 13, 0, "IMAGE"],
    ]

    return {
        "id": "dfe67589-107e-4bad-86cd-7f4471ff62db",
        "revision": 0,
        "last_node_id": 14,
        "last_link_id": 15,
        "nodes": nodes,
        "links": links,
        "groups": [
            {
                "id": 1,
                "title": "Models and reference",
                "bounding": [-1210, -590, 580, 980],
                "color": "#3f789e",
                "flags": {},
            },
            {
                "id": 2,
                "title": "T=1 conditioning",
                "bounding": [-570, -590, 550, 980],
                "color": "#7a5c9e",
                "flags": {},
            },
            {
                "id": 3,
                "title": "Sampling",
                "bounding": [50, -590, 730, 620],
                "color": "#3f789e",
                "flags": {},
            },
            {
                "id": 4,
                "title": "Image output",
                "bounding": [790, -520, 810, 540],
                "color": "#4f8b62",
                "flags": {},
            },
        ],
        "config": {},
        "extra": {
            "ds": {"scale": 0.75, "offset": [1220, 760]},
            "frontendVersion": "1.48.7",
        },
        "version": 0.4,
    }


def main():
    try:
        existing = [str(path) for path in TARGETS if path.exists()]
        if existing:
            print(
                "[H3_REF2IMAGE_CANVAS] 기존 파일이 있어 덮어쓰지 않습니다: "
                + ", ".join(existing),
                file=sys.stderr,
                flush=True,
            )
            raise FileExistsError("캔버스 워크플로 대상 파일이 이미 존재합니다")

        serialized = json.dumps(build_workflow(), ensure_ascii=False, indent=2) + "\n"
        for path in TARGETS:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(serialized, encoding="utf-8")
            print(
                f"[H3_REF2IMAGE_CANVAS] 생성 완료: path={path}, bytes={path.stat().st_size}",
                flush=True,
            )
        return 0
    except Exception as exc:
        print(
            "[H3_REF2IMAGE_CANVAS] 생성 실패: "
            f"type={type(exc).__name__}, error={exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
