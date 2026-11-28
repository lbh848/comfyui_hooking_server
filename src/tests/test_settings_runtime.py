"""Runtime settings preserve other config and only launch saved instance profiles."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from aiohttp.test_utils import TestClient, TestServer

from comfy_runtime import ComfyRuntimeError, ComfyRuntimeValidationError
from settings_runtime import SettingsRuntime
from settings_server import create_app
from settings_store import SettingsConflict
from configuration import read_config, write_config


class RuntimeSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "src"
        self.source.mkdir()
        self.runtime = SettingsRuntime(self.source)
        port_probe = patch.object(self.runtime.manager, "_port_is_in_use", return_value=False)
        self.port_probe = port_probe.start()
        self.addCleanup(port_probe.stop)

    def configuration(self, instance=1):
        return self.runtime.status()["instances"][instance - 1]["configuration"]

    def save(self, instance, configuration):
        body = copy.deepcopy(configuration)
        revision = body.pop("revision")
        return self.runtime.save(instance, body, revision)

    def test_profiles_persist_independently_and_preserve_unrelated_config(self):
        self.runtime.path.parent.mkdir()
        write_config(self.runtime.path, {"other": {"keep": True}, "comfyui_port_2": 9002})
        first = self.configuration()
        first["profile"]["auto_start"] = True
        first["profile"]["extra_args"] = '--output-directory "my outputs"'
        self.save(1, first)
        second = self.configuration(2)
        second["profile"]["cuda_device"] = 1
        self.save(2, second)
        restored = SettingsRuntime(self.source).status()["instances"]
        self.assertTrue(restored[0]["configuration"]["profile"]["auto_start"])
        self.assertEqual(restored[1]["configuration"]["port"], 9002)
        self.assertEqual(restored[1]["configuration"]["profile"]["cuda_device"], 1)
        self.assertFalse(restored[2]["configuration"]["profile"]["auto_start"])
        self.assertEqual(read_config(self.runtime.path)["other"], {"keep": True})

    def test_stale_save_and_start_are_rejected(self):
        old = self.configuration()
        changed = copy.deepcopy(old)
        changed["port"] = 9001
        self.save(1, changed)
        with self.assertRaises(SettingsConflict):
            self.save(1, old)
        with patch.object(self.runtime.manager, "start") as start:
            with self.assertRaises(SettingsConflict):
                self.runtime.start(1, old["revision"])
            start.assert_not_called()

    def test_invalid_ports_and_managed_arguments_do_not_write(self):
        for port in (8212, 0, 65536, True, 8211.5):
            with self.subTest(port=port):
                configuration = self.configuration()
                configuration["port"] = port
                with self.assertRaises(ComfyRuntimeValidationError):
                    self.save(1, configuration)
                self.assertFalse(self.runtime.path.exists())
        configuration = self.configuration()
        configuration["profile"]["extra_args"] = "--port 9000"
        with self.assertRaises(ComfyRuntimeValidationError):
            self.save(1, configuration)
        self.assertFalse(self.runtime.path.exists())

    def test_invalid_existing_config_is_not_overwritten(self):
        configuration = self.configuration()
        self.runtime.path.parent.mkdir()
        self.runtime.path.write_text("broken")
        with self.assertRaises(ValueError):
            self.save(1, configuration)
        self.assertEqual(self.runtime.path.read_text(), "broken")

    def test_busy_external_port_is_reported_without_claiming_the_process(self):
        self.port_probe.side_effect = lambda port: port == 8211
        instance = self.runtime.status()["instances"][0]
        self.assertTrue(instance["port_in_use"])
        self.assertFalse(instance["running"])
        self.assertIsNone(instance["pid"])

    def test_autostart_uses_saved_profiles_and_continues_after_failure(self):
        for instance in (1, 3):
            configuration = self.configuration(instance)
            configuration["profile"]["auto_start"] = True
            self.save(instance, configuration)
        with patch.object(self.runtime.manager, "start", side_effect=[ComfyRuntimeError("busy port"), {}]) as start:
            self.runtime.autostart()
        self.assertEqual([call.kwargs["instance_id"] for call in start.call_args_list], [1, 3])
        self.assertEqual([call.kwargs["port"] for call in start.call_args_list], [8211, 8213])
        self.assertEqual(self.runtime.status()["instances"][0]["startup_error"], "busy port")


class RuntimeApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.source = Path(self.temporary.name) / "src"
        self.source.mkdir()
        dist = self.source / "dist"
        dist.mkdir()
        (dist / "assets").mkdir()
        (dist / "index.html").write_text("fixture")
        self.runtime = SettingsRuntime(self.source)
        self.installer = Mock()
        self.installer.status.return_value = {"state": "idle"}
        self.client = TestClient(TestServer(create_app(dist, self.source, runtime=self.runtime, installer=self.installer)))
        await self.client.start_server()
        self.headers = {"Host": "127.0.0.1:8201", "X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2"}

    async def asyncTearDown(self):
        await self.client.close()
        self.temporary.cleanup()

    async def test_guards_and_instance_scope(self):
        path = "/api/setting/comfy-runtime"
        for headers in ({"Host": "127.0.0.1:8201"}, {**self.headers, "Origin": "http://outside.example"}):
            self.assertEqual((await self.client.get(path, headers=headers)).status, 403)
        response = await self.client.get(path, headers=self.headers)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual([i["instance_id"] for i in (await response.json())["instances"]], [1, 2, 3])
        self.assertEqual((await self.client.post(path + "/4/stop", headers=self.headers, json={})).status, 400)
        self.assertEqual((await self.client.post(path + "/1/start", headers=self.headers, json={})).status, 428)

    async def test_start_uses_saved_profile_and_stop_targets_same_instance(self):
        configuration = self.runtime.status()["instances"][1]["configuration"]
        with patch.object(self.runtime.manager, "start", return_value={"ok": True}) as start, \
                patch.object(self.runtime.manager, "stop", return_value={"ok": True}) as stop:
            response = await self.client.post("/api/setting/comfy-runtime/2/start", headers={**self.headers, "If-Match": configuration["revision"]}, json={"port": 1})
            self.assertEqual(response.status, 200)
            self.assertEqual(start.call_args.kwargs["instance_id"], 2)
            self.assertEqual(start.call_args.kwargs["port"], 8212)
            response = await self.client.post("/api/setting/comfy-runtime/2/stop", headers=self.headers, json={})
            self.assertEqual(response.status, 200)
            stop.assert_called_once_with(instance_id=2)

    async def test_install_and_runtime_do_not_overlap(self):
        configuration = self.runtime.status()["instances"][0]["configuration"]
        self.installer.status.return_value = {"state": "running"}
        with patch.object(self.runtime.manager, "start") as start:
            response = await self.client.post("/api/setting/comfy-runtime/1/start", headers={**self.headers, "If-Match": configuration["revision"]}, json={})
            self.assertEqual(response.status, 409)
            start.assert_not_called()
        with patch.object(self.runtime.manager, "is_running", return_value=True):
            response = await self.client.post("/api/setting/install", headers=self.headers, json={"install_mode": "standard"})
            self.assertEqual(response.status, 409)
            self.installer.start_basic_install.assert_not_called()
