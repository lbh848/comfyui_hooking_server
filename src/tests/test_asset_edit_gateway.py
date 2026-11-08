"""The asset workspace forwards EDIT masks only through its guarded POST route."""
import unittest

from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from settings_assets import allowed_asset_path, register_assets


class AssetEditGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_edit_mask_and_identity_reach_runtime(self):
        received = {}

        class Runtime:
            async def request(self, request, *, asset):
                path = "/api/" + request.match_info["path"]
                if not asset or not allowed_asset_path(path, request.method):
                    raise web.HTTPNotFound()
                reader = await request.multipart()
                async for part in reader:
                    received[part.name] = bytes(await part.read())
                return web.json_response({"success": True, "item_id": "test-edit"})

        app = web.Application()
        register_assets(app, Runtime(), "2")
        client = TestClient(TestServer(app))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        path = "/api/asset/asset_mode/qwen_edit/enqueue"
        self.assertEqual((await client.post(path)).status, 403)
        self.assertEqual((await client.post(path, headers={"X-Comfy-Settings": "1"})).status, 409)
        self.assertEqual(received, {})

        form = FormData()
        mask = b"\x89PNG\r\n\x1a\n-mask-bytes"
        form.add_field("mask", mask, filename="mask.png", content_type="image/png")
        values = {"character": "캐릭터", "outfit": "기본", "expression": "미소", "filename": "source.webp",
                  "edit_prompt": "Change the jacket", "negative_prompt": "blurry"}
        for key, value in values.items():
            form.add_field(key, value)
        response = await client.post(path, data=form, headers={"X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2"})
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["item_id"], "test-edit")
        self.assertEqual(received["mask"], mask)
        for key, value in values.items():
            self.assertEqual(received[key].decode("utf-8"), value)

    def test_edit_route_does_not_allow_other_methods_or_subpaths(self):
        path = "/api/asset_mode/qwen_edit/enqueue"
        self.assertTrue(allowed_asset_path(path, "POST"))
        for method in ("GET", "HEAD", "DELETE", "PUT"):
            self.assertFalse(allowed_asset_path(path, method))
        self.assertFalse(allowed_asset_path(path + "/extra", "POST"))


if __name__ == "__main__":
    unittest.main()
