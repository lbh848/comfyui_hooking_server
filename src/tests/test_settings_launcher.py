"""Exercise the built Windows launcher with isolated settings and a tiny server."""
from pathlib import Path
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.request import Request, urlopen

import psutil

SOURCE = Path(__file__).resolve().parents[1]
EXECUTABLE = SOURCE.parent / "Comfy공방.exe"
SERVER = '''import json, os, threading, time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
print("한글 실행 로그: 공방 준비 중 🌸", flush=True)
while Path("wait-for-qa").exists():
    time.sleep(.1)
if Path("fail-for-qa").exists():
    raise UnicodeEncodeError("cp949", "🌸", 0, 1, "인코딩 오류 테스트")
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        data = json.dumps({"name":"Comfy공방", "mode":"settings", "pid":os.getpid()}, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_POST(self):
        self.send_response(200)
        self.end_headers()
        threading.Thread(target=self.server.shutdown, daemon=True).start()
    def log_message(self, *args): pass
server = ThreadingHTTPServer(("127.0.0.1", 8201), Handler)
print("서버 준비 완료", flush=True)
server.serve_forever()
server.server_close()
print("서버 정상 종료", flush=True)
'''


def make_fixture(directory, *, failure=False, waiting=False, executable=EXECUTABLE, port=8201):
    root = Path(directory) / "한글 경로 공방"
    source = root / "src"
    python = source / ".venv" / "Scripts"
    python.mkdir(parents=True)
    shutil.copy2(executable, root / EXECUTABLE.name)
    script = (SOURCE / "run_settings.ps1").read_text(encoding="ascii")
    (source / "run_settings.ps1").write_text(script.replace("8201", str(port)), encoding="ascii")
    shutil.copy2(SOURCE / ".venv" / "Scripts" / "python.exe", python / "python.exe")
    shutil.copy2(SOURCE / ".venv" / "pyvenv.cfg", python.parent / "pyvenv.cfg")
    ui = source / "settings_ui"
    for part in ("src", "public", "dist"):
        (ui / part).mkdir(parents=True)
    (ui / "dist" / "index.html").write_text("fixture", encoding="utf-8")
    (source / "settings_server.py").write_text(SERVER.replace("8201", str(port)), encoding="utf-8")
    if failure:
        (source / "fail-for-qa").touch()
    if waiting:
        (source / "wait-for-qa").touch()
    return root


def startup_log(root):
    return "\n".join(path.read_text(encoding="utf-8") for path in (root / "data" / "logs").glob("*.log"))


def wait_for(predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.1)
    raise AssertionError("Timed out waiting for launcher")


def running_server(port=8201):
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/setting/status", timeout=.5) as response:
            return json.load(response)
    except OSError:
        return None


class LauncherSourceEncodingTests(unittest.TestCase):
    def test_entrypoints_are_ascii_without_bom(self):
        paths = [SOURCE / "run_en.bat", SOURCE / "run_settings.ps1"]
        paths.extend((SOURCE / "packaging" / "windows").glob("*.cs"))
        paths.extend((SOURCE / "packaging" / "windows").glob("*.ps1"))
        paths.extend((SOURCE / "packaging" / "windows").glob("*.manifest"))
        for path in paths:
            with self.subTest(path=path.name):
                self.assertTrue(path.read_bytes().isascii(), "Keep startup sources ASCII-only, without BOM")


@unittest.skipUnless(sys.platform == "win32", "Windows launcher")
class SettingsLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not EXECUTABLE.is_file():
            raise unittest.SkipTest("Build the Windows launcher first")
        cls.port = 8201
        cls.executable = EXECUTABLE
        with socket.socket() as probe:
            port_busy = probe.connect_ex(("127.0.0.1", 8201)) == 0
        if port_busy:
            # Exercise the same sources on a free port without stopping the user's app.
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                cls.port = probe.getsockname()[1]
            temporary = tempfile.TemporaryDirectory(prefix="comfy-launcher-build-")
            cls.addClassCleanup(temporary.cleanup)
            root = Path(temporary.name)
            packaging = root / "src" / "packaging" / "windows"
            shutil.copytree(SOURCE / "packaging" / "windows", packaging)
            launcher = packaging / "SettingsLauncher.cs"
            launcher.write_text(launcher.read_text(encoding="ascii").replace("8201", str(cls.port)), encoding="ascii")
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
                 str(packaging / "build_launcher.ps1")],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=30,
            )
            if result.returncode:
                raise RuntimeError(result.stdout.decode("utf-8", errors="replace"))
            cls.executable = root / EXECUTABLE.name

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="comfy-launcher-")
        self.addCleanup(self.temporary.cleanup)

    def launch(self, root):
        # The launcher must override a legacy console environment before Python starts.
        env = {**os.environ, "PYTHONUTF8": "0", "PYTHONIOENCODING": "cp949:strict"}
        process = subprocess.Popen([str(root / EXECUTABLE.name), "--no-browser"], env=env,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        def cleanup():
            try:
                owner = psutil.Process(process.pid)
                children = owner.children(recursive=True)
                for child in children:
                    try: child.kill()
                    except psutil.NoSuchProcess: pass
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=10)
                psutil.wait_procs(children, timeout=10)
            except psutil.NoSuchProcess:
                process.wait(timeout=10)
        self.addCleanup(cleanup)
        return process

    def make_fixture(self, **kwargs):
        return make_fixture(self.temporary.name, executable=self.executable, port=self.port, **kwargs)

    def running_server(self):
        return running_server(self.port)

    def test_executable_uses_gui_subsystem(self):
        data = EXECUTABLE.read_bytes()
        header = struct.unpack_from("<I", data, 0x3c)[0]
        self.assertEqual(struct.unpack_from("<H", data, header + 24 + 68)[0], 2)

    def test_success_duplicate_open_and_clean_exit(self):
        root = self.make_fixture()
        owner = self.launch(root)
        status = wait_for(self.running_server)
        wait_for(lambda: "서버 준비 완료" in startup_log(root))
        self.assertIsNone(owner.poll(), "Closing the splash must keep the server running")
        self.assertIn("한글 실행 로그: 공방 준비 중 🌸", startup_log(root))
        self.assertNotIn("\ufffd", startup_log(root))
        stages = [line for line in startup_log(root).splitlines() if line.startswith("COMFY_STARTUP|")]
        self.assertTrue(stages)
        self.assertTrue(all(line.isascii() for line in stages))
        self.assertIn("COMFY_STARTUP|checking|Checking the startup environment.", stages)
        logs_before = set((root / "data" / "logs").glob("*.log"))
        duplicate = self.launch(root)
        self.assertEqual(duplicate.wait(timeout=15), 0)
        self.assertEqual(set((root / "data" / "logs").glob("*.log")), logs_before,
                         "An existing server must bypass the splash and startup process entirely")
        self.assertEqual(self.running_server()["pid"], status["pid"])
        # Also check the ASCII-only script's server-name comparison and fast path.
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
             str(root / "src" / "run_settings.ps1"), "-LauncherMode"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(b"COMFY_STARTUP|ready|Connecting to the running workspace.", result.stdout)
        with urlopen(Request(f"http://127.0.0.1:{self.port}/test-stop", data=b"{}"), timeout=2):
            pass
        self.assertEqual(owner.wait(timeout=15), 0)
        self.assertIsNone(self.running_server())

    def test_encoding_failure_keeps_error_window_and_log(self):
        root = self.make_fixture(failure=True)
        owner = self.launch(root)
        wait_for(lambda: "UnicodeEncodeError" in startup_log(root))
        wait_for(lambda: "COMFY_STARTUP|error|" in startup_log(root))
        self.assertIsNone(owner.poll(), "An initial failure must remain visible until closed")
        text = startup_log(root)
        self.assertIn("인코딩 오류 테스트", text)
        self.assertIn("한글 실행 로그: 공방 준비 중 🌸", text)
        self.assertNotIn("\ufffd", text)
        self.assertIn("COMFY_STARTUP|error|Comfy Workshop exited with code 1.", text)
        self.assertIsNone(self.running_server())

    def test_script_parse_failure_is_readable_before_script_initializes_encoding(self):
        root = self.make_fixture()
        (root / "src" / "run_settings.ps1").write_text('Write-Output "한글 구문 오류 테스트', encoding="utf-8-sig")
        owner = self.launch(root)
        wait_for(lambda: "ParserError" in startup_log(root))
        self.assertIn("한글 구문 오류 테스트", startup_log(root))
        self.assertNotIn("\ufffd", startup_log(root))
        self.assertIsNone(owner.poll())


if __name__ == "__main__":
    unittest.main()
