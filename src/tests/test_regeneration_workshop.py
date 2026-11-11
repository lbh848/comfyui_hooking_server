"""Regeneration queue items stay in the workshop that submitted them."""
from types import SimpleNamespace
import unittest

import illustration_flow
import queue_manager


def regenerate_item(workshop):
    return SimpleNamespace(
        id=f"regen-{workshop or 'default'}",
        type="regenerate",
        label="재생성",
        params={"workshop": workshop} if workshop else {},
        status="pending",
        error="",
        completed_at=None,
    )


class RegenerationWorkshopTests(unittest.IsolatedAsyncioTestCase):
    async def test_emotion_regeneration_creates_emotion_flow(self):
        item = regenerate_item("emotion")
        illustration_flow.queue_added(item)
        run = item._illustration_flow[0]
        self.assertEqual(run.get("workshop"), "emotion")
        self.assertIn(run["id"], illustration_flow._histories["emotion"])
        self.assertNotIn(run["id"], illustration_flow._histories["illustration"])

    async def test_untagged_regeneration_defaults_to_illustration(self):
        item = regenerate_item("")
        illustration_flow.queue_added(item)
        run = item._illustration_flow[0]
        self.assertEqual(run.get("workshop"), "illustration")
        self.assertIn(run["id"], illustration_flow._histories["illustration"])


class RegenerationBackupWorkshopTests(unittest.IsolatedAsyncioTestCase):
    async def test_emotion_regeneration_backup_keeps_workshop_and_store_info(self):
        manager = queue_manager.QueueManager()
        captured = {}

        async def generate(positive, negative, **kwargs):
            return b"\x89PNG\r\n\x1a\npayload", None

        async def save_backup(image_bytes, prompt_id, positive, negative, **kwargs):
            captured.update(kwargs)
            return "saved", image_bytes

        manager.generate_image_with_prompt = generate
        manager.save_backup = save_backup
        manager.get_current_illustration_workflow_snapshot = None
        source_info = {
            "emotion_asset": {"saved_path": "/store/emotion/card_1/outfit/joy/1.png"},
            "face_positions": [],
        }
        item = queue_manager.QueueItem(
            id="regen-emotion", type="regenerate", label="재생성", priority=0,
            params={
                "backup_name": "source", "positive": "p", "negative": "n",
                "bot_name": "test", "provider": "comfy", "workshop": "emotion",
                "source_conversion_info": source_info,
            },
        )

        await manager._handle_regenerate(item)

        self.assertEqual(captured.get("workshop"), "emotion")
        self.assertEqual(captured.get("conversion_info_snapshot"), source_info)


if __name__ == "__main__":
    unittest.main()
