"""Run installed node math/contracts with Comfy's actual isolated dependencies."""
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("test_file", ["test_soya_first_sampler.py", "test_spatial_lora.py"])
def test_installed_sampler_contracts_in_comfy_environment(test_file):
    node_tests = ROOT / "comfy/custom_nodes/comfyui-soya-custom-nodes/tests"
    script = """
import importlib.util, sys, unittest
spec = importlib.util.spec_from_file_location('_installed_sampler_contract', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
sys.exit(0 if result.wasSuccessful() else 1)
"""
    completed = subprocess.run([
        "uv", "run", "--no-project", "--python", str(ROOT / "comfy/.venv/Scripts/python.exe"),
        "python", "-c", script, str(node_tests / test_file),
    ], cwd=ROOT / "comfy", text=True, encoding="utf-8", errors="replace",
       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    print(completed.stdout)
    assert completed.returncode == 0, completed.stdout
