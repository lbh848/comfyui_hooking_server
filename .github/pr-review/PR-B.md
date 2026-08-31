# PR B — Windows 밖에서도 돌게 합니다 (17커밋)

> 이 문서는 **리뷰를 돕기 위한 것**이며, 병합 전에 지우셔도 됩니다.
> 아래 "For the reviewing agent"는 Claude Code 같은 에이전트가 읽고 검증과 질문
> 응대에 쓰도록 작성한 것이라 영어로 되어 있습니다.

## 무엇을, 왜

현재는 Windows 밖에서 **`uv sync` 조차 실패합니다**(`onnxruntime-directml` 이
Windows 전용인데 무조건 의존성에 있습니다). 그 앞을 통과해도 설치기가 OS 이름으로
막고, 폰트·cloudflared·Real-ESRGAN 이 전부 Windows 자산을 받습니다.

17건이 그 사슬을 풉니다. **Windows 동작은 한 줄도 바꾸지 않는 것이 설계
목표였습니다.**

## Windows 가 깨지지 않는다는 근거

맥에서 작업했고 **Windows 를 돌려볼 수 없어서**, 검증을 실행이 아니라 **전수
독해**로 했습니다. 시리즈 전체에서 플랫폼에 민감한 구문이 든 **추가된 줄 66개**와
**삭제된 줄 23개**를 뽑아 하나씩 확인했습니다.

핵심만 옮기면 다음과 같습니다.

- `onnxruntime-directml` 은 `sys_platform == 'win32'` 마커로 **그대로 남습니다**
- Real-ESRGAN 은 Windows 에서 **같은 아카이브 이름과 같은 sha256** 을 받습니다
- cloudflared 는 Windows 에서 **같은 URL·같은 파일명**을 받습니다
- 폰트는 macOS 후보를 Windows 후보 **뒤에** 붙였으므로 잡히는 폰트가 바뀌지 않습니다
- ffmpeg 은 `not IS_WINDOWS` 일 때만 시스템 PATH 로 빠지며, Windows 의
  다운로드·검증 흐름은 통째로 그대로입니다

**실제로 달라지는 것이 딱 하나 있습니다.** 이 PR 이 아니라 PR C 의 LoRA 경로
결합입니다. 해당 PR 문서에 표로 적어 두었습니다.

## 검증 환경

macOS 26.6.2 (arm64) · Python 3.12.12 · uv 0.12.5 에서 무설치 상태부터 설치기를
완주시켜 확인했습니다. Linux 는 검증하지 않아 설치 게이트에 넣지 않았습니다.

## 함께 들어가는 테스트

신규 10개, 수정 4개(+719줄)입니다. 대부분 **플랫폼 선택을 표로 잠그는** 테스트라
Windows 러너 없이도 회귀를 잡습니다.

| 파일 | 무엇을 잠그는가 |
|---|---|
| `test_video_tools_platform.py` | Real-ESRGAN·ffmpeg 의 OS별 자산과 sha256 |
| `test_cloudflared_asset.py` | 7가지 OS·아키텍처 조합의 자산 이름 |
| `test_system_font_candidates.py` | 폰트 후보 세 벌이 서로 어긋나지 않게 |
| `test_installer_platform_gate.py` | 설치 게이트가 허용하는 OS |
| `test_installer_preinstall_wheels.py` | 사전 휠의 대상 플랫폼 선언 |
| `test_installer_runtime_probe.py` | 프로브 종료 경합 회귀 |
| `test_installer_e2e_python_path.py` | venv 인터프리터 심볼릭 링크 처리 |
| `test_onnx_coreml_provider.py` | CoreML 이 자동 선택에 끼어들지 않게 |
| `test_no_browser.py` | `NO_BROWSER` 기본값이 "열기"로 유지되게 |
| `test_installer_tab_platform_label.py` | UI 문구 |

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
