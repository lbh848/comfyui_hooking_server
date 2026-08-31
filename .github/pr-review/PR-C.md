# PR C — Cloud Direct 모델 다운로드 지원 (35 Commits)

> 이 문서는 리뷰 편의를 위해 작성되었으며, 머지(Merge) 전에 삭제하셔도 무방합니다.
> 하단의 "For the reviewing agent"는 Claude Code 등 AI 에이전트가 코드를 검증하고
> 질의응답을 수행할 수 있도록 영어로 작성되었습니다.

## 주요 내용 및 목적

**문제**: 기존에는 로컬 디스크만이 유일한 모델 저장소였습니다. 따라서 Modal
환경에서만 이미지를 생성하더라도 매니페스트 전체(약 145 GiB)를 로컬에 다운로드한 후
다시 원격 Volume으로 업로드해야 했습니다. 이는 로컬 GPU가 없거나 VRAM이 부족한
사용자에게는 사실상 사용이 불가능한 구조적 제약이었습니다.

**해결**: `modal_model_source=cloud_direct` 설정 시, Modal Worker가 원격 저장소에서
Volume으로 모델을 직접 다운로드합니다. 이 경우 로컬 설치기는 로컬에서 직접 실행되는
작업용 모델만 선택적으로 내려받습니다.

이 기능은 특정 플랫폼(OS)에 종속되지 않으며 **작업 분배 방식**에 따라 동작합니다.
`model_scope.py` 내에 플랫폼 분기 코드가 없음을 테스트로 보장합니다. 따라서
macOS뿐만 아니라 NVIDIA GPU가 없는 Windows 기기나 VRAM이 부족한 환경에서도 동일하게
스토리지 및 네트워크 절감 효과를 얻을 수 있습니다.

(실측 결과: 로컬 다운로드 용량이 6.42 GiB(6개) → 0.00 GiB(0개)로 감소, 원격 직접
다운로드는 138.69 GiB(45개) 수행됨)

기본값은 기존과 동일한 `local_first`이므로, 별도로 설정을 변경하지 않으면 기존 동작
방식이 그대로 유지됩니다.

## 원격 구동 실측 결과

Mock(단순 모의 테스트)가 아닌 실제 Modal Worker 환경에서 전체 프로세스를 구동하여
확인한 결과입니다.

| 항목 | 상세 내용 |
|---|---|
| Modal 환경 | `modal 1.5.3`, environment `main`, `scaledown_window` 15초 |
| GPU 자원 | 대부분 L4 사용. (영상 처리 워크플로우에 한해 RTX-PRO-6000(sm_120, NVFP4 네이티브) 환경에서 확인) |
| VRAM 모드 | `auto` 설정. (Upstream 기본값인 `highvram` 사용 시 Warm container 내에서 워크플로우 전환 중 OOM이 발생하여, 2회 실측 후 `auto`로 변경하여 해결함) |
| 로컬 환경 | macOS 26.6.2 (arm64) / Python 3.12.12 / uv 0.12.5 |
| Workflow 검증 | 총 23개 중 20개 정상 실행 확인. (나머지 3개는 단순 변형 파생본이라 생략, `outfit` 워크플로우는 미배포 대상이므로 검증에서 제외함) |

원격에서 정상 구동을 확인한 작업군: 삽화, 에셋(2종), qwen_edit, LoRA 학습(캐릭터/
스타일), 영상 처리, 태그 분석, 유틸리티/디버그, 얼굴 추출.

## Windows 환경 변경 사항 안내

LoRA 로드 경로 결합 시 구분자(Separator)를 역슬래시(`\`)로 하드코딩하던 것을,
사용자가 입력한 경로의 구분자를 그대로 따르도록 개선했습니다.

Windows 환경에서 `E:\models\loras`로 입력하면 기존처럼 `\`가 사용됩니다. 단, 사용자가
`E:/models/loras` 형태로 슬래시를 입력하면 결과값도 `/`로 출력됩니다. Windows
시스템은 두 구분자를 모두 정상 처리하므로 오류가 발생하지는 않으나, 변경 사항이
존재하므로 기록해 둡니다.

## 커밋 분할(35개) 사유

기능을 한 번에 작성한 것이 아니라, 원격 실행 환경으로 마이그레이션하면서 발생하는
오류(텍스트 수신 포트 불일치, 참조 이미지 폴더 누락, Worker 이미지 내 모듈 누락,
Redirect 과정의 토큰 유출 등)를 실측을 통해 찾아내며 단계적으로 수정한 기록입니다.
리뷰 편의를 위해 "Modal 전송 계층"과 "Cloud Direct 취득" 기능 등 두 개의 PR로
분리하는 것이 낫다고 판단되시면 말씀해 주시기 바랍니다.

## 포함된 테스트 (신규 23건, 수정 3건, +3011줄)

실제 원격 접속 없이 Mock/Stub을 활용해 검증되는 Contract Test(계약 테스트)들입니다.

| 분류 | 파일명 |
|---|---|
| 취득 경로 | `test_installer_model_source.py`, `test_modal_model_source.py`, `test_modal_volume_target_path.py`, `test_modal_user_copy_matching.py` |
| 설치 모드 | `test_cloud_only_install_mode.py`, `test_local_model_preflight.py` |
| Inventory | `test_modal_model_inventory.py`, `test_modal_model_cleanup.py` |
| 원격 배분 | `test_comfy_allocation_preflight.py`, `test_execution_target_outside_queue.py`, `test_remote_tag_analysis_wiring.py`, `test_remote_edit_preconditions.py` |
| 전송 결함 회귀 | `test_remote_model_separators.py`, `test_modal_model_sync_redirect_auth.py`, `test_modal_peak_vram.py` |
| LoRA 경로 | `test_lora_load_path_separator.py`, `test_modal_lora_root_guard.py`, `test_lora_training_staging_cleanup.py`, `test_lora_training_timeout_ceiling.py`, `test_modal_delete_outbox_backup.py`, `test_modal_lora_delete_idempotent.py` |
| 진단 | `test_comfy_connection_diagnostics.py`, `test_face_extract_error_reporting.py` |

---

## For the reviewing agent

### Scope

35 commits, 44 files, +5819/−144. Base: `01e2dc1`.
Largest of the three PRs. It is a chronological record of removing one blocker at
a time, each verified by an actual remote run.

### The central design claim, and how to falsify it

> The condition is **allocation**, not platform.

```bash
grep -rn 'platform' comfy_installer/model_scope.py
# → no hits. If this ever returns a platform branch, the design has been violated.
```

`tests/test_installer_model_source.py` locks the behaviour: models are kept when a
task that uses them is allocated to a non-remote target, regardless of OS.

The practical consequence is that this is **not** a macOS feature. A Windows user
with no NVIDIA card, or one with 8 GB of VRAM, hits the identical wall and gets
the identical benefit.

### How the pieces fit

1. **Acquisition path** — `sync_models_from_source` on the worker downloads from
   the source repository straight into the Modal volume. Authenticated models use
   a Modal Secret. (commits 4–8)
2. **Installer scoping** — `model_scope.py` maps *tasks → workflow bindings →
   manifest model_ids* and keeps only what locally-executed tasks need. (commit 9)
3. **Diagnostics** — reports which locally-run tasks lack models and recommends a
   cloud-only configuration when that is what the machine is. (commit 10)
4. **Inventory / cleanup** — see what is on the volume and delete it, because
   **volume storage bills even when idle**. (commits 11–12)
5. **Transport fixes** — the long tail: path separators, text-sender port,
   reference-dir staging, artifact recovery, worker image contents, redirect auth.
6. **Cloud-only install mode** — one choice that applies the whole configuration.

### Commits worth a closer read

**`fix(modal)` installed user copy did not match the pack source.** This is the
one that made `cloud_direct` silently do *nothing*. Read it first — it shows the
failure mode this area is prone to: everything reports success, no model is
fetched.

**`fix(modal)` worker image missing `remote_comfy_vram`.** Deploy succeeds; the
container then crash-loops with `ModuleNotFoundError`. **The deploy log looks
clean** — it only surfaces when a function is called. If you review one commit for
its lesson rather than its diff, make it this one.

Note the constraint it encodes: `add_local_*` must come **last** in an image
chain, and Modal rejects a build step placed after it. That is why sources are
attached per concrete image via `with_local_python_sources()` rather than on the
shared `runtime_image` — the web app appends `.env()` on top of that base.

**`fix(modal)` redirect auth leak.** civitai hands large files to S3 via a
presigned URL. `urllib` carries `Authorization` across the redirect; S3 reads it
as a signature and rejects with `Missing x-amz-content-sha256` (HTTP 400). The
tell was that five small LoRAs succeeded and only one 6.46 GiB checkpoint failed —
same token, same code path. The fix strips the header only when the host changes.
Measured: 6,938,042,346 B in 96.7 s, sha256 matching the manifest.

**`fix(modal)` Windows path separators.** Distributed workflows are authored on
Windows, so nested model names contain `v19\Model.safetensors`. The worker is
always Linux and lists `v19/Model.safetensors`, so submission is refused with
HTTP 400. **This is not a macOS problem — a Windows user running on Modal hits it
identically.** Only strings ending in a model extension are rewritten, and only on
the remote submission path (`ModalService.run_workflow`); local execution never
sees a rewritten dict.

### Windows safety

Reviewed by reading every platform-sensitive added and removed line. The
remote-side normalizations (`\` → `/`) all live in code that runs against the
Linux worker: `ModalService.run_workflow`, `workflow_assets`, `modal_app`
(container-side), and `client_cli` display formatting.

**One genuine behaviour change**, in `fix(lora)` — `combineLoraLoadPath` in
`frontend/index.html`:

```js
const separator =
    normalizedBase.includes('\\') && !normalizedBase.includes('/') ? '\\' : '/';
```

| base on Windows | before | after |
|---|---|---|
| `E:\models\loras` | `\` | `\` unchanged |
| `E:/models/loras` | `\` | `/` |
| `E:\models/loras` | `\` | `/` |
| `loras` (relative) | `\` | `/` |

Every result is still a valid Windows path. The reason for inferring rather than
branching: the browser cannot reliably know the server's OS, and the path the user
typed is the better signal. If you prefer a server-provided flag, that is a
reasonable request and a small change.

### Test environment — what was actually exercised

| | |
|---|---|
| Modal client | `modal 1.5.3`, environment `main`, `scaledown_window` 15 s |
| GPU | **L4** for most work; RTX-PRO-6000 (sm_120) only for video, where NVFP4 was confirmed to run natively |
| VRAM mode | `auto`. Upstream's `highvram` default passes `--highvram` to ComfyUI, which pins models in VRAM instead of offloading to CPU. On a warm container, switching workflow families (ILXL↔anima, asset→LoRA training) then OOMs — a 22 GiB L4 already holding 21.5 GiB. Reproduced twice; `auto` resolved both. **The value is baked into the image, so changing it requires a redeploy.** |
| Local host | macOS 26.6.2 (arm64), Python 3.12.12, uv 0.12.5 |
| Model scope after | 0 local models / 0.00 GiB; 45 remote / 138.69 GiB |
| Workflow coverage | 20 of 23 executed end to end |

Verified remotely, end to end: illustration, asset generation (two workflow
families), qwen_edit, LoRA training (character and style), video generation, tag
analysis, utility/debug, face extract.

A caution worth passing on: in video runs, `generation_time` in the `*_info.json`
sidecar is the **post-process** elapsed time, not the render time. Measured on 120
frames: 588 s remote render vs 689 s local Real-ESRGAN + AVIF encode. Using that
field to compare GPU cost measures the wrong machine.

### Tests added or changed

23 new files, 3 modified, +3011 lines. **None require Modal or a GPU** — they are
contract tests over stubs, so they run in any CI.

Grouped by what they lock:

- **Acquisition** — `test_installer_model_source.py` (the design-claim test),
  `test_modal_model_source.py`, `test_modal_volume_target_path.py`,
  `test_modal_user_copy_matching.py`
- **Install modes** — `test_cloud_only_install_mode.py`,
  `test_local_model_preflight.py`
- **Inventory/cleanup** — `test_modal_model_inventory.py`,
  `test_modal_model_cleanup.py`
- **Remote allocation** — `test_comfy_allocation_preflight.py`,
  `test_execution_target_outside_queue.py`, `test_remote_tag_analysis_wiring.py`,
  `test_remote_edit_preconditions.py`
- **Transport regressions** — `test_remote_model_separators.py`,
  `test_modal_model_sync_redirect_auth.py`, `test_modal_peak_vram.py`
- **LoRA paths** — `test_lora_load_path_separator.py`,
  `test_modal_lora_root_guard.py`, `test_lora_training_staging_cleanup.py`,
  `test_lora_training_timeout_ceiling.py`, `test_modal_delete_outbox_backup.py`,
  `test_modal_lora_delete_idempotent.py`
- **Diagnostics** — `test_comfy_connection_diagnostics.py`,
  `test_face_extract_error_reporting.py`

`test_modal_lora_delete_idempotent.py` is worth a look for its reasoning: it
asserts that "target missing" is decided by **re-querying the Volume**, not by
matching exception text, because `Volume.remove_file` raises `InvalidError` (not
`NotFoundError`) for a missing path and the same exception type also covers
read-only volumes and write conflicts. Matching on the message would swallow real
failures.

### What was NOT verified

- **No Windows run.** As with PR B, the Windows review is a read, not an
  execution.
- **`outfit`** remains local-only. Its workflow is not distributed
  (`excluded_filenames`), so its model requirements cannot even be determined. It
  has no manifest bindings, so it demands no local models.
- Three workflows have never been executed: `H3_REF2V`, `H3_FLF2V` (standard,
  non-fast), and `배포_삽화_v1_1`. All are variants of something already proven.
- The `/prompt` hook's prompt assembly was never checked against a real RisuAI
  session — synthetic requests cannot exercise it.

### Questions the maintainer may reasonably ask

- *"Is this macOS-specific?"* — No, and that is the main design claim. Check
  `model_scope.py` for a platform branch; there is none, and a test enforces it.
- *"35 commits is a lot."* — Agreed. Natural split: "Modal transport fixes" vs
  "cloud_direct acquisition". Say the word and it can be resubmitted as two.
- *"What happens to existing Windows users who do nothing?"* — Default stays
  `local_first`, the current behaviour. Nothing changes unless
  `modal_model_source` is set to `cloud_direct` or cloud-only install is chosen.
- *"Does storage cost anything?"* — Yes, which is why inventory and cleanup are
  here. Modal volumes bill on stored bytes, not just compute.
- *"Do the tests need a Modal account?"* — No. Every test here runs offline.
