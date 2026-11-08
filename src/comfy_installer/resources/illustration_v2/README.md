# Illustration JSON V2

## ANIMA HF, FD and ED references

Both distribution illustration V4 workflows read `SNAPREF.secondary_reference`.
Each reference keeps its general `asset` and may include a `face_asset` for the
same character/card. `face` sends representative + face to 1st/HF/HD/FD/ED;
`representative` sends representative + second representative to every stage.
A missing secondary image falls back to the primary. `stage` sends only the
representative to 1st/HD/HF and only the face to FD/ED (primary if absent).
`none` sends only the representative everywhere. These modes apply to ANIMA.
`face_only` sends only the face to 1st/HF/HD/FD/ED, falling back to the primary
representative when the character has no face image.

ANIMA HF, including Spectrum, reapplies `LORA_DATA` to the JSON RGB masks with
the same character order and regional prompts as the first pass. This remains
active when references are disabled. Solo HF keeps the already applied global
character LoRA. FD/ED references follow the matched character and the existing
face/eye crop mask. HD crops RGB ownership into each detected hand before
resizing it to the processing resolution, keeping each character's references
in their own region. ILXL stages retain their current LoRA/detailer behavior.
The mixed workflow receives an additional CLIP connection for ANIMA HF during
the existing managed-workflow/submission upgrade.

## Spatial face context

Current V2 illustration workflows use the new `SoyaSpatialFaceContextV2_mdsoya`
node. The original Face Match and FD/ED classes are unchanged. The first sampler's
normalized `face_positions` output and JSON `REGION_MASK` RGB channel ownership
identify faces; no CLIP identity model, InsightFace model, or embedding cache is
loaded by this node. Each group's current image still goes through the existing
YOLO detection call, confidence filter and maximum-face limit. Detections are
assigned to named sampler positions by a global 1:1 nearest-center match.
Tracking boxes and RGB masks are used only for identity, never as crop bounds or
as pixel cutouts. Crops use the detected boxes, `FACE_CROP_TOP/BOTTOM` (including
the legacy default of 3.0), side expansion and integer rounding from Face Match.
Paths lacking tracking use RGB ownership/nearest-mask distance with a global 1:1
assignment, selects the largest face in solo mode, and allows an identity to recur
across manga panels. It adds no overlap/confidence rejection rule. Existing FD/ED
Voronoi exclusions, rounded face masks, and eye masks retain their behavior.

`comfy_installer.spatial_face_workflow` migrates both UI and API graphs, including
the geometry/detector Set/Get routes. Managed installation upgrades old pack files;
the server also upgrades old V2 API graphs at submission. Legacy node definitions
and non-V2 workflows keep their original behavior. New NPCs stage their actual
representative images without running the IPA utility on these workflows.
Spatial submissions omit `FACE_EMBEDDINGS`; the cache contract below applies only
to older graphs. SnapRef still uses the actual character images. If a selected
secondary FACE image does not exist yet, only the primary representative is used.

Both illustration V4 files use the `soya.illustration.v2` transport. The workflow
pack stays at v6. V2 classes are registered inside the existing
`comfyui-soya-custom-nodes` package and repository; there is no second package.

The ANIMA and ANIMA+ILXL asset V4 workflows now share this transport and these
nodes. Workshop submits a typed `prompt_json` document; `asset_json` normalizes
older tagged asset requests and resolves random seeds separately for each job.
Pose, IPAdapter and ControlNet fields and graph branches are absent. SnapRef,
character/style/face LoRAs, hires and face/hand/eye stages remain available.
The first sampler applies character LoRAs once; its upstream model bypasses the
legacy character batch loader, which remains available to hires and hand stages.
Face stages use spatial matching without identity embeddings. JSON and model
connections use the same per-group Set/Get routing as illustration workflows.
Editable asset sources are in the sibling `asset_v2` resource directory; the
distribution pack includes them with their reduced fixed model dependencies.

## ANIMA inpainting

`배포_공방_인페인트_ANIMA_V2_0.json` uses the same transport. The
`SoyaInpaintJSONV2_mdsoya` node decodes `SOURCE_ASSET` and `MASK_ASSET` from
`assets` and outputs the edit prompt, mask and sampler values. The server embeds
the queued source/mask PNG bytes as base64 inside `긍정프롬프트`, so no
`comfy/input/qwen_edit` copy is written and the workflow has no
`LoadImagesFromPath_mdsoya` node.

To migrate a legacy asset UI workflow with a compatible running ComfyUI:

```powershell
python -m scripts.migrate_asset_json_v2 source.json --output migrated.json
```

Reinstall the shared V2 node resources and restart ComfyUI after migration.

The `긍정프롬프트` string node contains a single JSON document. Every execution
node receives that exact string through `prompt_json`. There is no prompt-parser
node and no output-slot contract for prompt fields. Each V2 node looks up its
own named keys; the binding definitions are in `illustration_v2_specs.json`.

V2 model loaders, model/LoRA patches and text encoders compare their resolved
arguments after extracting their JSON fields. Each node instance keeps one
result for those arguments, so unrelated JSON changes do not reload weights,
rebuild the same patches or re-encode unchanged text. A changed model, loading
option, LoRA setting, CLIP reference or text replaces that node's entry. The
cache holds resolved arguments and results, never the full JSON or Base64 assets.
ComfyUI still owns device placement and can offload models. Clearing ComfyUI's
execution cache also clears these node instances. No ComfyUI core files change.
The V2 first sampler runs on every queued request, including identical JSON;
its downstream image stages run again while the JSON seed remains unchanged.

Load Diffusion Model V2 and CheckpointLoaderKJ V2 expose `model_key` instead
of a local model dropdown. The defaults select `fields.ANIMA_DIFFUSION_MODEL`
and `fields.ILXL_CHECKPOINT` respectively. Each value is an installed model
filename as listed by its original loader, including any model subfolder.
The `fields.` prefix is optional in `model_key`. Missing names and models
unavailable to that loader produce an explicit error before loading weights.
The selected workflow supplies defaults when a submitted prompt omits these
keys; explicit request values override them. Changing the model name replaces
the loader's cached result, while an unchanged name and options reuse it.

CLIP Text Encode V2 exposes an editable `json_key` text field. Its initial
value is the node's binding, such as `ANIMA_CONTENT`, `SDXL_CONTENT`,
`NEGATIVE_ANIMA` or `NEGATIVE_SDXL`. Users can enter any key under `fields`,
with or without the `fields.` prefix. Multiple keys separated by commas
are read once and joined in the entered order, skipping empty values.
`ANIMA_QUALITY, ANIMA_CONTENT` supplies the combined ANIMA prompt without
storing another copy in `ANIMA_ALL`. `ANIMA_CONTENT` already includes the
trigger and artist text produced by the editor, so do not add them again.
The combined text is passed to the text encoder after releasing the parsed
JSON. An empty key name or missing field produces an explicit error.
The key is selected by the user, independently of the connected CLIP model.
The JSON transport removes `ANIMA_ALL` and renames `SDXL` to `SDXL_CONTENT`,
preserving the SDXL text. Tagged editor drafts are normalized at submission.

The first sampler, Hiresfix and all face, hand and eye detailers expose `sampling_key`.
It selects a settings object under `fields`: `ANIMA_FIRST_SAMPLER`,
`ANIMA_FACE_DETAILER`, `ANIMA_HAND_DETAILER`, `ANIMA_EYE_DETAILER`,
`ILXL_FACE_DETAILER`, `ILXL_HAND_DETAILER`, `ILXL_EYE_DETAILER`,
`ANIMA_HIRESFIX` or `ILXL_HIRESFIX`.
The first sampler reads `steps`, `cfg`, `sampler_name` and `scheduler`.
Hiresfix (including Spectrum) and every detailer read those four settings plus
`denoise`. For example:

```json
"ANIMA_FACE_DETAILER": {
  "steps": 16,
  "cfg": 5.0,
  "sampler_name": "er_sde",
  "scheduler": "simple",
  "denoise": 0.22
}
```

The JSON values replace those local node widgets. Each workflow keeps its
existing values when migrated. Names, numeric ranges and sampler choices
are validated against the underlying node before execution. Generated
requests inherit these blocks from the workflow's JSON source; values
explicitly supplied in the request take precedence, including partial blocks.

The editor routes this string through one KJNodes `SetNode` named
`SOYA_PROMPT_JSON` and a `GetNode` in each processing group. These virtual nodes
keep prompt connections local to each group. Both the browser and the workflow
converter resolve them back to the original string node for execution.

MODEL, CLIP and VAE connections between processing groups also use named
Set/Get routes. ANIMA and ILXL routes are separate, as are LoRA stages and
the first sampler's model output. Each getter resolves to its original
source output; the local model-processing chains retain their direct links.

Each V2 node extracts its fields and decodes the assets it needs in a
short-lived preparation call. That call returns before the node's main
operation starts, releasing the full parsed payload and unused Base64 strings.
Only the selected inputs and decoded tensors remain in the operation. The
shared source string is still owned by ComfyUI for the other consumers.

```json
{
  "schema": "soya.illustration.v2",
  "fields": {
    "ANIMA_QUALITY": "masterpiece, best quality",
    "ANIMA_CONTENT": "1girl, standing in a garden",
    "NEGATIVE_ANIMA": "low quality, blurry",
    "SDXL_CONTENT": "masterpiece, best quality, 1girl, standing in a garden",
    "NEGATIVE_SDXL": "low quality, blurry",
    "IMG_W": 768,
    "IMG_H": 1024,
    "SEED": 1234,
    "CHAR_LIST": [],
    "FACE_EMBEDDINGS": [],
    "MULTI_CHAR": {"enable": false},
    "SNAPREF": {"enabled": false, "references": []}
  },
  "assets": {}
}
```

This excerpt illustrates the shape. Use the complete example in either
workflow for all quality, LoRA and detailer settings. Booleans, numbers, lists
and objects retain their JSON types. Negative prompts are part of the same
document. LoRA settings selected by the user remain supported; fixed workflow
LoRA loaders and ILXL IPAdapter application are removed. Samplers that depended
on turbo/DMD2 LoRAs use ordinary `er_sde` sampling defaults instead.

An asset is stored once under its SHA-256 ID:

```json
{
  "encoding": "base64",
  "name": "reference.png",
  "media_type": "image/png",
  "sha256": "<sha256-of-decoded-bytes>",
  "data": "<base64-encoded-file-bytes>"
}
```

`REGION_MASK` refers to an RGB mask asset ID. `FACE_EMBEDDINGS` contains
`{"CHAR": "name", "asset": "id"}` entries for CLIP identity caches.
`SNAPREF.references` has the same entry shape for representative images.
SnapRef v2a-pre2 accepts up to two references with the same `CHAR`; each is
encoded separately and their K/V tokens are combined for that character's region.
Drawing settings choose the second image from the selected card's face image or
second representative image. `strength_scale: "linear"` uses `strength` in 0–2
(0 disables references; 1 is the trained default). `fidelity` in 0–1 controls
the negative CFG branch's reference weight; it has no effect when CFG is 1.
Images are decoded from `BytesIO`; tensor caches use `torch.load` with
`weights_only=True`. Assets are never restored into ComfyUI INPUT. Model and
user-selected LoRA weights remain model dependencies, not input assets.

`illustration_json.build_illustration_json` is the server boundary. It accepts
the existing editor draft or a complete V2 JSON document, captures the current
request's mask and input files, and emits the transmitted JSON. Existing draft
editing, backups and older workflows retain their original format.

The installer updates shared SnapRef controls and mixed-precision layer routing,
then generates isolated V2 sampler/face-matching core modules with in-memory I/O.
Older JSON drafts without `strength_scale` retain their logarithmic strength
interpretation. The package initializer registers
the additional V2 IDs. Installation,
updates, Modal image builds and Vast setup install the same V2 resources.
Restart ComfyUI after installing the nodes; existing remote workers need a
new deployment with these resources.

To migrate a source workflow with a running compatible ComfyUI:

```powershell
python -m scripts.migrate_illustration_json_v2 source.json --output migrated.json
```

The migration updates the bundled node binding definitions. Install the V2
nodes again after migration. UI node IDs and positions are retained for the
remaining execution graph; obsolete parser, routing and removed branches
are eliminated.
