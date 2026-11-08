"""Background survives both legacy building and the browser-to-worker JSON contract."""
import importlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from asset_json import prepare_asset_prompt

AssetMode = importlib.import_module("modes.asset_mode").AssetMode
ROOT = Path(__file__).resolve().parents[1]
TAGS = {"background_presets": {"garden": ["flower garden", "sunlight"]},
        "expressions": {"portrait": ["smile", "upper body"]}}


class AssetBackgroundTests(unittest.TestCase):
    def test_server_builder_includes_background_in_each_active_model(self):
        mode = AssetMode()
        with patch.object(mode, "get_tags", return_value=TAGS):
            for profile in ("ilxl", "anima_only", "anima_ilxl"):
                with self.subTest(profile=profile):
                    positive, negative = mode.build_prompts(
                        "", "", "portrait", asset_workflow_type=profile, background_preset="garden")
                    self.assertEqual(positive.count("flower garden"), 2 if profile == "anima_ilxl" else 1)
                    self.assertIn("smile", positive)
                    cleared, _ = mode.build_prompts("", "", "portrait", asset_workflow_type=profile)
                    self.assertNotIn("flower garden", cleared)

    def test_browser_json_survives_worker_prompt_preparation(self):
        script = """
            import './frontend/static/asset_prompt.js';
            const tags = JSON.parse(process.argv[1]);
            const docs = [];
            for (const type of ['anima_only', 'anima_ilxl']) {
                const result = await ComfyAssetPrompt.buildAssetPromptData({ asset_workflow_type: type,
                    expression: 'portrait', background_preset: 'garden', raw_background_tags: ['night sky'] }, tags);
                docs.push(result.document);
            }
            process.stdout.write(JSON.stringify(docs));
        """
        result = subprocess.run(["node", "--input-type=module", "-e", script, json.dumps(TAGS)],
                                cwd=ROOT, text=True, capture_output=True, check=True)
        for document in json.loads(result.stdout):
            prepared = json.loads(prepare_asset_prompt(json.dumps(document), ""))
            for field in ("ANIMA_CONTENT", "SDXL_CONTENT"):
                if field not in document["fields"]:
                    continue
                self.assertEqual(prepared["fields"][field], document["fields"][field])
                self.assertIn("night sky", prepared["fields"][field])
                self.assertNotIn("flower garden", prepared["fields"][field])


if __name__ == "__main__":
    unittest.main()
