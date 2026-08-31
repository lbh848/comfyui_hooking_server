# PR A — 기타 버그 수정 및 테스트 환경 개선 (10 Commits)

> 이 문서는 리뷰 편의를 위해 작성되었으며, 머지(Merge) 전에 삭제하셔도 무방합니다.
> 하단의 "For the reviewing agent"는 Claude Code 등 AI 에이전트가 코드를 검증하고
> 질의응답을 수행할 수 있도록 영어로 작성되었습니다.

## 주요 내용 및 목적

플랫폼 및 클라우드 환경과 무관한 10건의 수정 사항이며, 크게 세 가지로 분류됩니다.

**설정 초기화 버그 수정**: 빈 폼(Form) 상태로 저장할 경우 기존 설정값이 덮어씌워지는
문제를 해결했습니다. 향후 동일한 문제가 발생했을 때 원인을 쉽게 추적할 수 있도록,
저장 시 diff 로그를 남기는 기능을 추가했습니다.

**잔여 파일 정리**: qwen_edit 실행 후 staging 폴더가 디스크에 남는 문제를
해결했습니다.

**테스트 환경 의존성 제거 (가장 중요)**: 새로 체크아웃(Checkout)한 환경에서 pytest가
Collection 단계에서 크래시(Crash)나는 문제를 수정했습니다. 저장소에 없는 파일을
참조하는 3개의 테스트로 인해 전체 테스트 스위트가 중단되어, 현재 저장소에 CI를
연동할 수 없는 상태입니다. 마지막 커밋에서 이를 해결했습니다.

이 문제를 해결해야 다른 PR들의 CI 파이프라인이 정상적으로 통과(Green)할 수 있으므로,
이 PR을 가장 먼저 머지해 주시길 권장합니다.

## 검증 환경

macOS 26.6.2 (arm64) / Python 3.12.12 / uv 0.12.5 환경에서 확인했습니다.
본 PR은 Modal이나 GPU 자원을 사용하지 않으므로 원격 실행 없이 로컬에서 완벽하게
재현 및 검증이 가능합니다.

## 포함된 테스트 (신규 3건, 수정 12건, +418줄)

| 파일명 | 검증 목적 |
|---|---|
| `test_frontend_settings_form_guard.py` | 빈 폼이 기존 설정을 덮어쓰지 못하도록 방지 |
| `test_config_save_diagnostics.py` | 저장 diff 및 출처 로그 데이터 무결성 검증 |
| `test_qwen_edit_staging_cleanup.py` | 실행 후 staging 폴더가 남지 않도록 검증 |
| `test_bot_one_click_workflow.py` (수정) | FACE 백업의 롤백 전용 정책 및 복구 동작 검증 |
| `test_video_engine_runtime.py` (수정) | 종료 시그널이 실제 OS로 전송되지 않도록 샌드박싱 처리 |
| Collection 실패 3종 (수정) | 대상 파일이 없을 경우 실패(Fail) 대신 스킵(Skip) 처리 |

## Revert 안내

각 커밋은 독립적으로 구성되어 있습니다. 불필요한 항목은 상호 의존성 없이
`git revert` 명령어로 개별 취소할 수 있습니다.

---

## For the reviewing agent

You are helping review this PR. This section gives you facts to verify claims and
answer the maintainer's questions. Everything here is reproducible.

### Scope

10 commits, 20 files, +606/−33. Base: `01e2dc1`.
No commit here touches platform detection or Modal/cloud code paths.

### The one blocking claim, and how to check it

```bash
git checkout 01e2dc1
python -m pytest -q
# → ERROR collecting tests/test_minimax_h3_ref2image_experiment.py
#   ERROR collecting tests/test_soya_minimax_h3_ref2image.py
#   ERROR collecting tests/test_soya_optional_image_by_name.py
#   !!!! Interrupted: 3 errors during collection !!!!
#   3 errors in 1.43s     ← zero tests ran
```

These modules read paths absent from a fresh clone: experiment artifacts that
moved to local-only storage, and a custom node under `comfy/` (untracked, created
by the installer). A module-scope `FileNotFoundError` aborts collection for the
**whole suite**, not just those files.

The fix converts them to `pytest.skip(..., allow_module_level=True)`. Skipping is
correct rather than lenient: when the artifact is absent there is nothing to
assert, and counting its absence as a failure makes the baseline depend on local
machine state.

After the fix the suite runs. On a checkout without local data expect a
substantial number of failures for unrelated reasons (missing workflow packs,
absent models) — **compare failure sets, never counts.**

### Commit-by-commit

| # | Commit | What to look at |
|---|---|---|
| 1 | `fix(test)` devil gradient | Test sampled background *outside* the panel, so it asserted on nothing. |
| 2 | `feat(config)` save diff + origin log | Adds `_origin` to save payloads, logs a before/after diff. Diagnostic only. This is the tool that would have caught #3 in minutes. |
| 3 | `fix(settings)` unfilled form guard | **Real data loss.** An unfilled settings form could be submitted and overwrite stored config. |
| 4 | `fix(test)` SSE flake | Response body read after server shutdown — intermittent. |
| 5 | `fix(qwen_edit)` staging cleanup | Staging folder survived the run and accumulated. |
| 6 | `fix(queue)` `preset_import_classify` order | See below. |
| 7 | `fix(video-engine)` signal seam | See below. |
| 8 | `test(data-patch)` FACE rollback | Test asserted a permanent backup the code deliberately stopped creating. Test was stale, code was right. |
| 9 | `test` skip untracked artifacts | The collection fix above. |
| 10 | `test(installer)` ml-dtypes comment | Comment only. Weakest patch here — drop it freely if unwanted. |

### Two commits that deserve a closer read

**#6 `fix(queue)`** — when a new LLM queue type is added without a `missing_after`
rule, normalization appends it to the **end** for existing installs while fresh
installs get declaration order. Same version, two different priority orders,
silently. `lora_prompt_review` had a rule; `preset_import_classify` did not.

The test change (`11,12` → `12,13`) is an absolute-rank assertion following
declaration order. The *relative* assertion immediately above it is the invariant
that matters, and it passes unchanged.

**#7 `fix(video-engine)`** — the POSIX stop path called
`os.killpg(process.pid, ...)`, signalling the **real OS**. The test's fake process
has a hardcoded pid (44321). In practice no such pid existed and the test failed
with `ProcessLookupError`; had it existed, an unrelated process group would have
been killed.

Production behaviour is unchanged — `start()` uses `start_new_session=True`, so
the child's pgid equals its pid and `killpg` is correct. The call is only moved
into `_signal_process_group()` so tests can substitute it, matching the existing
`_port_is_in_use` / `_create_windows_job` seams. Windows guards are intact:
`killpg` sits inside `if os.name != "nt"`, and Windows still takes
`_terminate_windows_job` / `CTRL_BREAK_EVENT`.

### Tests added or changed

3 new files, 12 modified, +418 lines. All are pure unit/contract tests — no
network, no GPU, no Modal.

- `test_frontend_settings_form_guard.py` — the guard for #3.
- `test_config_save_diagnostics.py` — locks the diff/origin invariants for #2.
- `test_qwen_edit_staging_cleanup.py` — asserts the staging directory is gone.
- `test_bot_one_click_workflow.py` — rewritten to the new FACE contract, plus a
  new rollback test. **Note the failure point:** injecting failure at `os.replace`
  leaves the original untouched, so the test passes even with the restore code
  deleted. The injection was moved to the post-swap prompt-JSON removal, and
  mutation-verified.
- `test_video_engine_runtime.py` — substitutes `_signal_process_group`.

### Environment

macOS 26.6.2 (arm64), Python 3.12.12, uv 0.12.5. This PR requires none of that
specifically; it is pure-Python and platform-neutral.

### What was not verified

- **No Windows execution.** Nobody on this side has a Windows machine. The review
  was a read of every added and removed platform-sensitive line, not a run.
- Commit #3 was verified by its regression test, not by a manual UI session.

### Questions the maintainer may reasonably ask

- *"Why is a comment worth a commit (#10)?"* — It is not, strongly. Drop it.
- *"Can I take only some of these?"* — Yes. Every commit is independent.
- *"Does this depend on the other PRs?"* — No. It is the base the others are
  easier to evaluate on, but there is no code dependency in either direction.
