# PR B — Windows 외 플랫폼(macOS) 지원 추가 (17 Commits)

> 이 문서는 리뷰 편의를 위해 작성되었으며, 머지(Merge) 전에 삭제하셔도 무방합니다.
> 하단의 "For the reviewing agent"는 Claude Code 등 AI 에이전트가 코드를 검증하고
> 질의응답을 수행할 수 있도록 영어로 작성되었습니다.

## 주요 내용 및 목적

현재 Windows 외 환경에서는 `onnxruntime-directml` (Windows 전용) 패키지가 조건 없이
의존성에 포함되어 있어 `uv sync` 단계부터 실패합니다. 이를 우회하더라도
설치기(Installer)가 OS 이름에서 실행을 차단하거나 폰트, cloudflared, Real-ESRGAN
등을 모두 Windows용 Asset으로 다운로드하는 문제가 있습니다.

이러한 제한을 해소하는 17건의 수정을 포함하며, **기존 Windows 환경의 동작은 전혀
변경하지 않는 것**을 최우선 설계 목표로 삼았습니다.

## Windows 환경 안전성 검증

macOS 환경에서 작업하여 Windows에서의 실제 실행 테스트는 불가했으나, 코드를 전수
리뷰(Code Review)하여 안전성을 검증했습니다. 전체 PR 중 플랫폼에 민감한 구문이
포함된 추가/삭제 라인(추가 66줄, 삭제 23줄)을 개별적으로 확인했습니다.

- `onnxruntime-directml`은 `sys_platform == 'win32'` 마커를 유지하여 Windows에서만
  설치되도록 보존했습니다.
- Real-ESRGAN과 cloudflared는 Windows에서 기존과 완벽하게 동일한 URL, 파일명,
  SHA256 해시값을 사용해 다운로드됩니다.
- 폰트는 Windows 후보군 뒤에 macOS 후보군을 추가하여 기존 폰트 인식 우선순위에
  영향을 주지 않도록 처리했습니다.
- ffmpeg은 `not IS_WINDOWS` 조건일 때만 시스템 PATH를 확인하며, Windows의 기존
  다운로드 및 검증 로직은 통째로 유지됩니다.

단, 한 가지 변경 사항이 존재합니다. 본 PR이 아닌 **PR C**에 포함된 LoRA 경로 결합
로직입니다. 자세한 내용은 PR C 문서에 기재해 두었습니다.

## 검증 환경

macOS 26.6.2 (arm64) / Python 3.12.12 / uv 0.12.5 환경에서 클린 설치부터 구동
완료까지 정상 동작함을 확인했습니다. Linux 환경은 별도 검증을 거치지 않아 설치
허용 OS 목록에 포함하지 않았습니다.

## 포함된 테스트 (신규 10건, 수정 4건, +719줄)

대부분 플랫폼별 분기 로직을 하드코딩된 표 형태로 락(Lock)을 거는 테스트로,
Windows Runner 없이도 Regression(회귀 버그)을 방지할 수 있습니다.

| 파일명 | 검증 목적 |
|---|---|
| `test_video_tools_platform.py` | Real-ESRGAN 및 ffmpeg의 OS별 Asset 파일명 및 SHA256 해시 검증 |
| `test_cloudflared_asset.py` | 7가지 OS/아키텍처 조합별 Asset 파일명 매칭 검증 |
| `test_system_font_candidates.py` | 플랫폼별 폰트 후보군이 상호 충돌하지 않도록 검증 |
| `test_installer_platform_gate.py` | 설치기가 허용하는 OS 목록 검증 |
| `test_installer_preinstall_wheels.py` | 사전 설치 Wheel의 타겟 플랫폼 선언 무결성 검증 |
| `test_installer_runtime_probe.py` | Probe 종료 시 발생할 수 있는 Race Condition(경합) 방지 |
| `test_installer_e2e_python_path.py` | venv 인터프리터의 심볼릭 링크(Symbolic Link) 처리 검증 |
| `test_onnx_coreml_provider.py` | CoreML이 자동 선택에 개입하지 못하도록 제한 |
| `test_no_browser.py` | `NO_BROWSER` 플래그의 기본값이 "브라우저 열기"로 유지되는지 검증 |
| `test_installer_tab_platform_label.py` | UI 텍스트 출력 검증 |

---

## For the reviewing agent

### Scope

17 commits, 29 files, +1163/−119. Base: `01e2dc1`.
Design constraint: **every change is a condition, not a platform fork.** No code
path is added that only macOS can reach; Windows keeps its exact prior behaviour.

### Why this ordering matters

The chain fails at the first link. Without `fix(deps)`, `uv sync` fails on any
non-Windows host, so nothing downstream is reachable. Review in commit order.

| # | Commit | Windows impact |
|---|---|---|
| 1 | `feat(ui)` move data-patch import out of the Windows-only installer tab | UI placement only |
| 2 | `fix(deps)` onnxruntime split | `onnxruntime-directml; sys_platform == 'win32'` — **unchanged on Windows** |
| 3 | `feat(installer)` Darwin gate | Allow-list *widened* from `{Windows}` to `{Windows, Darwin}` |
| 4 | `feat(installer)` skip foreign prebuilt wheels | Absent `platforms` key ⇒ install (backward compatible). The one wheel declares `["Windows"]` |
| 5 | `refactor(installer)` extract probe script | Pure move. Exists so #6 is a small diff |
| 6 | `fix(installer)` probe exit race | `os._exit(0)` after explicit flushes — see below |
| 7 | `fix(installer)` E2E venv interpreter | `python.parent.resolve() / python.name`; on Windows `Scripts\python.exe` is a real file, not a symlink, so identical result |
| 8 | `feat(fonts)` macOS font candidates | Appended **after** Windows entries in all three candidate lists |
| 9 | `feat(video-tools)` Real-ESRGAN/FFmpeg off Windows | Same archive name **and sha256** under the `win32` key; `_resolve_tool` returns the project-local path unconditionally when `IS_WINDOWS` |
| 10 | `fix(server)` cloudflared platform/arch | `("cloudflared-windows-amd64.exe", "cloudflared.exe")` — byte-identical |
| 11 | `feat(server)` `NO_BROWSER` | `""`/`0`/`false` ⇒ opens; a parametrised test locks the default |
| 12 | `feat(onnx)` CoreML provider | Additive; CoreML deliberately **excluded** from `auto_device_key` |
| 13 | `feat` `run_en.sh` | New file. Windows keeps `run_en.bat` untouched |
| 14 | `fix(test)` managed Python layout | Branches on `os.name == "nt"` back to the original paths |
| 15 | `fix(test)` CPU threads | Pins `logical_cpu_count` to 16 — **strengthens** it (previously passed only on ≥12-core hosts) |
| 16 | `fix(test)` bubble golden font | Skips only when the resolved font is not `malgun.ttf`; on Windows it is the first candidate, so it still runs |
| 17 | `fix(ui)` installer tab label | Text only |

### Commits worth a closer read

**#6 probe exit.** `onnxruntime` starts a telemetry upload thread at import. If
that thread is mid-upload during interpreter teardown, it segfaults. The observed
failure was `code=-11` (SIGSEGV) *after* the probe had already printed its JSON
correctly, with a macOS crash report naming
`TransmissionPolicyManager::uploadAsync`.

The fix flushes stdout and stderr, then `os._exit(0)`, skipping `atexit` and GC.
Deliberate and safe **here specifically** — a one-shot subprocess with no state
beyond the streams — but exactly the kind of thing worth objecting to, so it is
called out rather than buried.

**#12 CoreML.** Intentionally absent from auto-selection: unsupported-op fallback
and FP16 numerics can change output silently, so it is offered in the UI but never
auto-chosen. Windows priority (CUDA > DirectML > CPU) is untouched.

**#16 bubble golden.** The risk is a test that silently stops running on Windows.
It does not: `malgun.ttf` is the first Windows candidate, so the skip never
triggers there. A font-independent structural test was added alongside, so
non-Windows hosts still assert something.

### Verifying the Windows-safety claim yourself

```bash
# added platform-sensitive lines, production code only
git diff 01e2dc1..HEAD -- . ':!tests' | grep -E '^\+' | \
  grep -E 'os\.name|sys\.platform|platform\.system|platform\.machine|os\.sep|\.exe|Scripts/|killpg|chmod|shutil\.which'

# the more important direction: Windows paths that DISAPPEARED
git diff 01e2dc1..HEAD -- . ':!tests' | grep -E '^-' | \
  grep -E 'os\.name|sys\.platform|platform\.system|\.exe|Windows'
```

The second query returns hits in commits 3, 9, 10, 17. Each is a rewrite with an
equivalent, not a deletion:

| Removed | Replaced by |
|---|---|
| `if platform.system() != "Windows"` gate | allow-list still containing `Windows` |
| four hardcoded `*.exe` paths | `f"...{EXE_SUFFIX}"`, where `EXE_SUFFIX == ".exe"` on Windows |
| three `if os.name != "nt"` refusals | platform-package table; Windows download flow untouched |
| Windows branch of `_ensure_cloudflared` | `_cloudflared_asset("Windows", …)` returning identical values |

### Tests added or changed

10 new files, 4 modified, +719 lines. The design intent is that **platform choices
are locked in tables, not inferred**, so a Windows regression is caught by reading
a test on any OS:

- `test_video_tools_platform.py` — asset name *and sha256* per OS.
- `test_cloudflared_asset.py` — all 7 OS/arch combinations, including the 32-bit
  ARM Linux asset that the original code ignored.
- `test_system_font_candidates.py` — the candidate list exists in three modules;
  this locks them together so fixing one and forgetting the others fails loudly.
- `test_installer_platform_gate.py`, `test_installer_preinstall_wheels.py`,
  `test_installer_runtime_probe.py`, `test_installer_e2e_python_path.py`
- `test_onnx_coreml_provider.py` — asserts CoreML is *not* in auto-selection.
- `test_no_browser.py` — parametrised over `""`, `"0"`, `"false"`, `"False"`,
  `"FALSE"`, `" 0 "`; the point is that the default does not change.

Two review findings came out of writing these and are folded in: `NO_BROWSER=FALSE`
was read as "off" (case-sensitive comparison), and cloudflared ignored
`cloudflared-linux-arm`.

### Environment

macOS 26.6.2 (arm64), Python 3.12.12, uv 0.12.5. Verified by running the installer
from a clean state through to a working ComfyUI, then the pipelines.

### What was not verified

- **No Windows run.** The central limitation. Everything above is a read. If you
  can run the suite on Windows once, that closes the only real gap.
- Linux is deliberately **not** claimed — the gate allows `{Windows, Darwin}` only.

### Questions the maintainer may reasonably ask

- *"Why not just branch on platform everywhere?"* — The same conditions apply to
  Windows machines without NVIDIA and to low-VRAM GPUs. A platform fork helps
  macOS only; a condition helps all three, and leaves one code path to maintain.
- *"Does the uv.lock diff have to be that large?"* — It is generated. Regenerating
  on your machine is fine and expected.
- *"Can I take the test fixes but not the platform work?"* — Yes; #14–#16 are
  independent and improve determinism on Windows too.
- *"What about Linux?"* — Not claimed, not gated in. Adding `"Linux"` to
  `SUPPORTED_INSTALL_PLATFORMS` is one line, but it is untested and should not be
  part of this PR.
