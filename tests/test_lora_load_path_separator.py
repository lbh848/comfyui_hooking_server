import json
import subprocess
from pathlib import Path


FRONTEND = (
    Path(__file__).resolve().parents[1] / "frontend" / "index.html"
).read_text(encoding="utf-8")


def _function(name: str, next_name: str) -> str:
    return FRONTEND.split(f"function {name}", 1)[1].split(
        f"function {next_name}", 1
    )[0]


def test_lora_path_helpers_round_trip_windows_and_posix_paths() -> None:
    script = f"""
function combineLoraLoadPath{_function('combineLoraLoadPath', 'stripManagedLoraPath')}
function stripManagedLoraPath{_function('stripManagedLoraPath', 'updateClampStatus')}
const result = {{
  windows: combineLoraLoadPath('C:\\\\ComfyUI\\\\models\\\\loras', 'SOYA_CHAR_LORA', 'SOYA_BOT_LORA'),
  posix: combineLoraLoadPath('/opt/comfy/models/loras/', 'SOYA_CHAR_LORA', 'SOYA_BOT_LORA'),
  strippedWindows: stripManagedLoraPath('C:\\\\ComfyUI\\\\models\\\\loras\\\\SOYA_CHAR_LORA\\\\SOYA_BOT_LORA'),
  strippedPosix: stripManagedLoraPath('/opt/comfy/models/loras/SOYA_CHAR_LORA/SOYA_BOT_LORA'),
}};
process.stdout.write(JSON.stringify(result));
"""
    completed = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "windows": "C:\\ComfyUI\\models\\loras\\SOYA_CHAR_LORA\\SOYA_BOT_LORA",
        "posix": "/opt/comfy/models/loras/SOYA_CHAR_LORA/SOYA_BOT_LORA",
        "strippedWindows": "C:\\ComfyUI\\models\\loras",
        "strippedPosix": "/opt/comfy/models/loras",
    }
