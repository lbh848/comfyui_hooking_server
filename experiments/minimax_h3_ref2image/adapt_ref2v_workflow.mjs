import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";


const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(SCRIPT_DIR, "..", "..");
const SOURCE = path.join(
  PROJECT_ROOT,
  "comfy",
  "user",
  "default",
  "workflows",
  "SOYA_USER",
  "배포_영상_H3_REF2V_v1.json",
);
const TARGETS = [
  path.join(SCRIPT_DIR, "workflow_canvas.json"),
  path.join(
    PROJECT_ROOT,
    "comfy",
    "user",
    "default",
    "workflows",
    "SOYA_USER",
    "실험_이미지_H3_REF2I_T1_v1.json",
  ),
];

const IMAGE_VAE_NAME = "minimax_h3_t1_image_vae_step1597.safetensors";
const IMAGE_VAE_URL =
  "https://huggingface.co/Mamad8/MiniMax-H3-Image-VAE/resolve/main/" +
  IMAGE_VAE_NAME;
const DEFAULT_STILL_PROMPT = [
  "subject_definitions:",
  "<Subject 1> is the primary subject sourced from <Picture 1>, preserving the visible identity, facial features, hairstyle, clothing, colors, accessories, and body proportions.",
  "",
  "summary:",
  "[reference generation] Create one finished still image containing exactly one instance of <Subject 1> in a single coherent composition.",
  "",
  "retention_analysis:",
  "<Subject 1> (appears in [Shot 1]): fully_preserved - Preserve the referenced identity, facial features, hairstyle, clothing, colors, accessories, and body proportions.",
  "",
  "detailed_description:",
  "A polished still image with stable anatomy, clean detail, coherent lighting, and a clearly readable composition.",
  "[Shot 1] Show exactly one <Subject 1> as the sole primary character. Keep the complete subject visually consistent with <Picture 1>. Do not duplicate the subject or present alternate poses in the same image.",
  "",
  "overall_soundscape:",
  "N/A",
  "",
  "non_diegetic_music:",
  "N/A",
].join("\n");
const REMOVED_NODE_IDS = new Set([120, 121, 130, 131, 150, 154]);
const FORBIDDEN_TYPES = new Set([
  "CreateVideo",
  "SaveVideo",
  "VAEDecodeAudio",
  "MiniMaxH3ReferenceToVideo",
]);


function clone(value) {
  return JSON.parse(JSON.stringify(value));
}


function requireCondition(condition, message) {
  if (!condition) {
    console.error(`[H3_REF2I_ADAPT] 검증 실패: ${message}`);
    throw new Error(message);
  }
}


function socket(name, type, { widget = false, optional = false } = {}) {
  const value = {
    localized_name: name,
    name,
    type,
    link: null,
  };
  if (widget) {
    value.widget = { name };
  }
  if (optional) {
    value.shape = 7;
  }
  return value;
}


function output(name, type) {
  return {
    localized_name: name,
    name,
    type,
    links: null,
  };
}


function nodeById(workflow, nodeId) {
  const node = workflow.nodes.find((item) => item.id === nodeId);
  requireCondition(Boolean(node), `필수 노드가 없습니다: id=${nodeId}`);
  return node;
}


function assertSourceShape(source) {
  const expectedTypes = new Map([
    [92, "SaveVideo"],
    [119, "VAELoader"],
    [120, "VAELoader"],
    [121, "VAEDecodeAudio"],
    [122, "VAEDecode"],
    [125, "SamplerCustomAdvanced"],
    [127, "UNETLoader"],
    [128, "CLIPLoader"],
    [130, "CreateVideo"],
    [131, "ComfyMathExpression"],
    [136, "MiniMaxH3ReferenceToVideo"],
    [144, "MiniMaxH3TeaCache"],
    [145, "PrimitiveStringMultiline"],
    [146, "RegexExtract"],
    [147, "RegexExtract"],
    [148, "RegexExtract"],
    [149, "RegexExtract"],
    [150, "RegexExtract"],
    [151, "RegexExtract"],
    [159, "LoadImagesFromPath_mdsoya"],
    [160, "SoyaOptionalImageByName_mdsoya"],
    [161, "SoyaOptionalImageByName_mdsoya"],
    [162, "SoyaOptionalImageByName_mdsoya"],
  ]);
  requireCondition(source.nodes.length === 37, `원본 노드 수가 바뀌었습니다: ${source.nodes.length}`);
  for (const [nodeId, expectedType] of expectedTypes) {
    const actualType = nodeById(source, nodeId).type;
    requireCondition(
      actualType === expectedType,
      `원본 노드 타입이 바뀌었습니다: id=${nodeId}, expected=${expectedType}, actual=${actualType}`,
    );
  }
}


function replaceWithImageOutput(node) {
  node.type = "SaveImage";
  node.size = [430, 420];
  node.inputs = [
    socket("images", "IMAGE"),
    socket("filename_prefix", "STRING", { widget: true }),
  ];
  node.outputs = [];
  node.properties = { "Node name for S&R": "SaveImage" };
  node.widgets_values = ["h3_ref2image"];
  node.title = "Save H3 T=1 image";
}


function replaceWithImageVae(node) {
  node.widgets_values = [IMAGE_VAE_NAME];
  node.properties = {
    "Node name for S&R": "VAELoader",
    models: [
      {
        name: IMAGE_VAE_NAME,
        url: IMAGE_VAE_URL,
        directory: "vae",
      },
    ],
  };
  node.title = "MiniMax H3 T=1 Image VAE";
}


function replaceWithRef2Image(node) {
  node.type = "SoyaMiniMaxH3ReferenceToImage_mdsoya";
  node.size = [480, 620];
  node.inputs = [
    socket("clip", "CLIP"),
    socket("vae", "VAE"),
    socket("ref_image_1", "IMAGE"),
    ...Array.from({ length: 8 }, (_, index) =>
      socket(`ref_image_${index + 2}`, "IMAGE", { optional: true }),
    ),
    socket("prompt", "STRING", { widget: true }),
    socket("width", "INT", { widget: true }),
    socket("height", "INT", { widget: true }),
    socket("ref_image_size", "COMBO", { widget: true }),
  ];
  node.outputs = [
    output("positive", "CONDITIONING"),
    output("latent", "LATENT"),
    output("diagnostics", "STRING"),
  ];
  node.properties = {
    "Node name for S&R": "SoyaMiniMaxH3ReferenceToImage_mdsoya",
  };
  node.widgets_values = [
    DEFAULT_STILL_PROMPT,
    960,
    544,
    "match",
  ];
  node.title = "MiniMax H3 REF2I T=1 — deployment transport";
}


function setNote(node, text, title) {
  node.widgets_values = [text];
  if (title) {
    node.title = title;
  }
}


function rebuildSocketLinks(workflow) {
  const nodes = new Map(workflow.nodes.map((node) => [node.id, node]));
  for (const node of workflow.nodes) {
    for (const input of node.inputs ?? []) {
      input.link = null;
    }
    for (const item of node.outputs ?? []) {
      item.links = null;
    }
  }

  for (const link of workflow.links) {
    const [linkId, originId, originSlot, targetId, targetSlot, socketType] = link;
    const origin = nodes.get(originId);
    const target = nodes.get(targetId);
    requireCondition(Boolean(origin), `링크 출발 노드가 없습니다: link=${linkId}, node=${originId}`);
    requireCondition(Boolean(target), `링크 도착 노드가 없습니다: link=${linkId}, node=${targetId}`);
    requireCondition(
      originSlot < (origin.outputs?.length ?? 0),
      `링크 출발 슬롯이 없습니다: link=${linkId}, slot=${originSlot}`,
    );
    requireCondition(
      targetSlot < (target.inputs?.length ?? 0),
      `링크 도착 슬롯이 없습니다: link=${linkId}, slot=${targetSlot}`,
    );
    requireCondition(
      origin.outputs[originSlot].type === socketType,
      `출발 소켓 타입 불일치: link=${linkId}, expected=${socketType}, actual=${origin.outputs[originSlot].type}`,
    );
    requireCondition(
      target.inputs[targetSlot].type === socketType,
      `도착 소켓 타입 불일치: link=${linkId}, expected=${socketType}, actual=${target.inputs[targetSlot].type}`,
    );
    if (origin.outputs[originSlot].links === null) {
      origin.outputs[originSlot].links = [];
    }
    origin.outputs[originSlot].links.push(linkId);
    target.inputs[targetSlot].link = linkId;
  }
}


function buildDerivedWorkflow(source) {
  assertSourceShape(source);
  const workflow = clone(source);
  workflow.id = "50882944-0fda-46d2-b77b-0a71156c7f2b";
  workflow.revision = 0;
  workflow.nodes = workflow.nodes.filter((node) => !REMOVED_NODE_IDS.has(node.id));

  replaceWithImageOutput(nodeById(workflow, 92));
  replaceWithImageVae(nodeById(workflow, 119));
  nodeById(workflow, 122).title = "Decode T=1 with H3 Image VAE";
  replaceWithRef2Image(nodeById(workflow, 136));

  setNote(
    nodeById(workflow, 116),
    "## MiniMax H3 REF2I T=1 — REF2V-derived experimental workflow\n\n" +
      "This canvas is derived from `배포_영상_H3_REF2V_v1.json`. The application transport, " +
      "PATH/PROMPT/W/H/DURATION/SEED transport, reference-image selection, Ref2VA model, " +
      "TeaCache, and 20-step sampler are retained. The duration value remains in the transport " +
      "for server compatibility but is ignored by T=1. Only its frame-count calculation and the " +
      "audio/video-output path are removed.\n\n" +
      "The application can continue to place `[1]` through `[3]` images in `soya_video` and " +
      "replace the multiline transport payload. The T=1 node supports up to nine references, " +
      "but this deployment-derived canvas intentionally preserves the existing three-slot transport.",
    "REF2V-derived T=1 experiment",
  );
  setNote(
    nodeById(workflow, 117),
    "## Model Links\n\n" +
      "**Image VAE (T=1 only)**\n\n" +
      `- [${IMAGE_VAE_NAME}](${IMAGE_VAE_URL})\n\n` +
      "**Diffusion model**\n\n" +
      "- [minimax_h3_ref2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors)\n\n" +
      "**Text encoder**\n\n" +
      "- [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors)\n\n" +
      "Use the image VAE only for this single-frame workflow. It does not replace the H3 video VAE.",
    "T=1 model files",
  );
  setNote(
    nodeById(workflow, 140),
    "| Profile | Aspect | Output | Steps | Cache |\n" +
      "|---|---|---|---|---|\n" +
      "| Base Ref2VA + T=1 Image VAE | 16:9 | **960 x 544** | **20** | **0.10** |\n\n" +
      "Start with one reference. Multi-reference role separation remains experimental.",
    "Experimental profile",
  );

  nodeById(workflow, 145).widgets_values = [
    "[PATH]\n" +
      "soya_video\n" +
      "[PROMPT]\n" +
      DEFAULT_STILL_PROMPT +
      "\n" +
      "[W]\n" +
      "960\n" +
      "[H]\n" +
      "544\n" +
      "[DURATION]\n" +
      "5\n" +
      "[SEED]\n" +
      "42\n" +
      "[END]",
  ];

  const decodeGroup = workflow.groups.find((group) => group.id === 4);
  requireCondition(Boolean(decodeGroup), "Decoding 그룹(id=4)이 없습니다");
  decodeGroup.title = "Decoding and save image";

  workflow.links = workflow.links.filter(
    (link) =>
      !REMOVED_NODE_IDS.has(link[1]) &&
      (!REMOVED_NODE_IDS.has(link[3]) || link[0] === 258),
  );
  const linkChanges = new Map([
    [258, { targetId: 92, targetSlot: 0 }],
    [297, { targetId: 136, targetSlot: 11 }],
    [300, { targetId: 136, targetSlot: 12 }],
    [303, { targetId: 136, targetSlot: 13 }],
    [311, { targetId: 136, targetSlot: 2 }],
    [314, { targetId: 136, targetSlot: 3 }],
    [317, { targetId: 136, targetSlot: 4 }],
  ]);
  for (const link of workflow.links) {
    const change = linkChanges.get(link[0]);
    if (change) {
      link[3] = change.targetId;
      link[4] = change.targetSlot;
    }
  }
  for (const linkId of linkChanges.keys()) {
    requireCondition(
      workflow.links.some((link) => link[0] === linkId),
      `치환할 원본 링크가 없습니다: link=${linkId}`,
    );
  }

  rebuildSocketLinks(workflow);
  workflow.last_node_id = Math.max(...workflow.nodes.map((node) => node.id));
  workflow.last_link_id = Math.max(...workflow.links.map((link) => link[0]));
  validateDerivedWorkflow(workflow);
  return workflow;
}


function validateDerivedWorkflow(workflow) {
  const types = new Set(workflow.nodes.map((node) => node.type));
  for (const type of FORBIDDEN_TYPES) {
    requireCondition(!types.has(type), `영상 전용 노드가 남았습니다: ${type}`);
  }
  for (const nodeId of [123, 124, 125, 126, 127, 128, 129, 144, 145, 146, 147, 148, 149, 151, 152, 153, 155, 156, 157, 158, 159, 160, 161, 162]) {
    nodeById(workflow, nodeId);
  }
  requireCondition(nodeById(workflow, 92).type === "SaveImage", "SaveImage 치환 실패");
  requireCondition(
    nodeById(workflow, 136).type === "SoyaMiniMaxH3ReferenceToImage_mdsoya",
    "REF2I 치환 실패",
  );
  requireCondition(
    nodeById(workflow, 119).widgets_values[0] === IMAGE_VAE_NAME,
    "Image VAE 치환 실패",
  );
  requireCondition(workflow.nodes.length === 31, `파생 노드 수가 예상과 다릅니다: ${workflow.nodes.length}`);
  requireCondition(workflow.links.length === 39, `파생 링크 수가 예상과 다릅니다: ${workflow.links.length}`);
}


function main() {
  try {
    const replace = process.argv.includes("--replace");
    requireCondition(fs.existsSync(SOURCE), `원본 워크플로가 없습니다: ${SOURCE}`);
    if (!replace) {
      const existing = TARGETS.filter((target) => fs.existsSync(target));
      requireCondition(
        existing.length === 0,
        "대상 파일이 이미 있습니다. 개발자 백업 후 --replace를 명시하십시오: " + existing.join(", "),
      );
    }

    const source = JSON.parse(fs.readFileSync(SOURCE, "utf8"));
    const workflow = buildDerivedWorkflow(source);
    const serialized = `${JSON.stringify(workflow, null, 2)}\n`;
    for (const target of TARGETS) {
      fs.mkdirSync(path.dirname(target), { recursive: true });
      fs.writeFileSync(target, serialized, "utf8");
      console.log(
        `[H3_REF2I_ADAPT] 생성 완료: source=${SOURCE}, target=${target}, ` +
          `nodes=${workflow.nodes.length}, links=${workflow.links.length}, bytes=${Buffer.byteLength(serialized, "utf8")}`,
      );
    }
    return 0;
  } catch (error) {
    console.error(
      `[H3_REF2I_ADAPT] 생성 실패: type=${error?.name ?? "Error"}, error=${error?.message ?? error}`,
    );
    console.error(error?.stack ?? error);
    return 1;
  }
}


process.exitCode = main();
