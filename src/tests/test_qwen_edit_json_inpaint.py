"""The V2 inpainting workflow carries source and mask inside its JSON prompt."""
import base64
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from comfy_allocation import CURRENT_COMFY_EXECUTION_TARGET
from modes.qwen_edit_mode import QwenEditMode


WORKFLOW_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "comfy_workflow_library"
    / "SOYA_DISTRIBUTION"
    / "v6"
    / "배포_공방_인페인트_ANIMA_V2_0.json"
)


def png(color, size=(16, 16)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


class QwenEditJsonInpaintTests(unittest.TestCase):
    def test_payload_embeds_png_bytes_as_assets(self):
        source = b"source-png-bytes"
        mask = b"mask-png-bytes"
        params = {
            "job_id": "job123",
            "edit_prompt": "Change the jacket",
            "negative_prompt": "blurry",
            "seed": 7,
            "steps": 30,
            "cfg": 4.0,
            "denoise": 0.8,
            "mask_grow": 8,
            "mask_blur": 4.0,
            "width": 1024,
            "height": 1024,
        }

        document = json.loads(
            QwenEditMode._build_inpaint_json_payload(
                params,
                {"source": source, "mask": mask},
            )
        )

        self.assertEqual(document["schema"], "soya.illustration.v2")
        fields = document["fields"]
        self.assertEqual(fields["EDIT_PROMPT"], "Change the jacket")
        self.assertEqual(fields["FILENAME_PREFIX"], "anima_inpainting/job123/output")
        for field, data, name in (
            ("SOURCE_ASSET", source, "source.png"),
            ("MASK_ASSET", mask, "mask.png"),
        ):
            key = fields[field]
            asset = document["assets"][key]
            self.assertEqual(key, hashlib.sha256(data).hexdigest())
            self.assertEqual(asset["name"], name)
            self.assertEqual(base64.b64decode(asset["data"]), data)

    def test_workflow_links_prompt_json_to_inpaint_node(self):
        workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in workflow["nodes"]}

        prompt_nodes = [
            node for node in nodes.values()
            if node["type"] == "PrimitiveStringMultiline"
            and node.get("title") == "긍정프롬프트"
        ]
        inpaint_nodes = [
            node for node in nodes.values()
            if node["type"] == "SoyaInpaintJSONV2_mdsoya"
        ]
        self.assertEqual(len(prompt_nodes), 1)
        self.assertEqual(len(inpaint_nodes), 1)
        prompt = json.loads(prompt_nodes[0]["widgets_values"][0])
        self.assertEqual(prompt["schema"], "soya.illustration.v2")
        self.assertIn("SOURCE_ASSET", prompt["fields"])
        self.assertIn("MASK_ASSET", prompt["fields"])
        self.assertFalse(any(
            node["type"] == "LoadImagesFromPath_mdsoya"
            for node in nodes.values()
        ))

        links = {link[0]: link for link in workflow["links"]}
        inpaint = inpaint_nodes[0]
        link = links[inpaint["inputs"][0]["link"]]
        self.assertEqual(link[1:3], [prompt_nodes[0]["id"], 0])

        targets = {}
        for output_index, output in enumerate(inpaint["outputs"]):
            for link_id in output["links"]:
                _, _, _, target_id, target_slot, _ = links[link_id]
                targets[(target_id, target_slot)] = output_index
        self.assertEqual(
            targets,
            {
                (13, 1): 0,
                (14, 1): 1,
                (11, 0): 2,
                (15, 1): 2,
                (10, 0): 3,
                (16, 4): 4,
                (16, 5): 5,
                (16, 6): 6,
                (16, 9): 7,
                (10, 1): 8,
                (10, 5): 9,
            },
        )

    def test_workflow_detection_uses_inpaint_node(self):
        api = {"1": {"class_type": "SoyaInpaintJSONV2_mdsoya", "inputs": {}}}
        self.assertEqual(QwenEditMode._json_inpaint_node_ids(api), ["1"])
        self.assertEqual(
            QwenEditMode._json_inpaint_node_ids(
                {"1": {"class_type": "SoyaQwenEditPromptParser_mdsoya", "inputs": {}}}
            ),
            [],
        )


class QwenEditJsonInpaintExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_injects_json_without_staging(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            comfy_input = root / "comfy" / "input"
            comfy_input.mkdir(parents=True)
            source_path = root / "source.png"
            source_path.write_bytes(png((10, 20, 30)))
            mode = QwenEditMode()
            mode.get_config = lambda: {
                "comfy_input_dir": str(comfy_input),
                "anima_inpainting_workflow_source_path": "workflow.json",
            }
            job_id = "job-json"
            mode._pending_inputs[job_id] = {
                "source": png((40, 50, 60)),
                "mask": png((255, 255, 255)),
            }

            async def load_workflow(config, edit_tool):
                return {
                    "1": {
                        "class_type": "PrimitiveStringMultiline",
                        "_meta": {"title": "긍정프롬프트"},
                        "inputs": {"value": ""},
                    },
                    "2": {
                        "class_type": "SoyaInpaintJSONV2_mdsoya",
                        "inputs": {"prompt_json": ["1", 0]},
                    },
                }, "workflow.json"

            captured = {}

            async def submit(workflow, progress_callback=None, input_paths=None):
                captured["workflow"] = workflow
                captured["input_paths"] = input_paths
                return png((90, 90, 90)), {}

            mode._load_workflow = load_workflow
            mode.submit_workflow_func = submit
            params = {
                "job_id": job_id,
                "source_path": str(source_path),
                "source_filename": "source.png",
                "source_prompt": {},
                "image_path": "qwen_edit",
                "mask_path": "qwen_edit",
                "edit_prompt": "edit",
                "negative_prompt": "",
                "seed": 1,
                "steps": 30,
                "cfg": 4.0,
                "denoise": 0.8,
                "mask_grow": 8,
                "mask_blur": 4.0,
                "width": 16,
                "height": 16,
                "edit_tool": "anima_inpainting",
            }
            token = CURRENT_COMFY_EXECUTION_TARGET.set("modal")
            try:
                await mode.execute(params)
            finally:
                CURRENT_COMFY_EXECUTION_TARGET.reset(token)

            payload = json.loads(
                captured["workflow"]["1"]["inputs"]["value"]
            )
            self.assertEqual(payload["schema"], "soya.illustration.v2")
            self.assertEqual(len(payload["assets"]), 2)
            self.assertIsNone(captured["input_paths"])
            self.assertFalse((comfy_input / "qwen_edit").exists())


if __name__ == "__main__":
    unittest.main()
