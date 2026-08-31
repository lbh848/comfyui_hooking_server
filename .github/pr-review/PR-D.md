# PR D — macOS App Bundle 및 GitHub Actions Build (1 Commit)

> 이 문서는 리뷰 편의를 위해 작성되었으며, 머지(Merge) 전에 삭제하셔도 무방합니다.
> 하단의 "For the reviewing agent"는 Claude Code 등 AI 에이전트가 코드를 검증하고
> 질의응답을 수행할 수 있도록 영어로 작성되었습니다.

## 주요 내용 및 목적

**로컬 macOS 환경 없이도 macOS 배포판을 자동 생성할 수 있도록 구축했습니다.** PR B를
통해 macOS 지원 코드가 병합되더라도 빌드할 Mac 기기가 없으면 배포가 불가능합니다.
이를 해결하기 위해 GitHub Actions Runner가 빌드를 대신 수행하도록 구성했습니다.
(공개 저장소이므로 macOS Runner 사용 시간은 무료입니다.)

**코드를 바이너리로 패키징하지 않습니다 (PyInstaller 미사용).** 본 앱은 구동 시 앱
내부 폴더에 ComfyUI(약 12 GiB), 모델, `config.json`, `logs` 등을 직접 생성하는
구조입니다. macOS의 `/Applications` 디렉토리는 읽기 전용(Read-only) 정책이 적용되어,
해당 위치에서 구동 시 초기 설치 단계부터 권한 오류가 발생합니다. 따라서 App Bundle
내부에는 소스 코드만 포함하고, 최초 실행 시
`~/Library/Application Support/SoyaComfy` (사용자 디렉토리)로 복사한 뒤 기존의
`run_en.sh` 스크립트를 띄우는 방식을 채택했습니다. 이를 통해 실행 경로를 단일화하여
macOS 전용 분기 로직이 생기는 것을 방지했습니다.

**Ad-hoc(임시) 서명을 적용했습니다.** Apple Developer 계정 등록이 필요한
공증(Notarization) 프로세스 대신, 무료로 가능한 임시 서명만 수행합니다. Apple
Silicon Mac은 서명되지 않은 실행 파일을 OS 단에서 원천 거부하므로 이 과정은
필수적입니다. 사용자는 앱 최초 실행 전 터미널에서 다음 명령어를 단 한 번만 실행해야
합니다.

```bash
xattr -dr com.apple.quarantine /Applications/SoyaComfy.app
```

## 검증 내역

포크된 저장소에서 Workflow를 직접 실행하여 `Build arm64 = success` 처리됨을
확인했습니다. 생성된 산출물은 zip 파일 기준 5 MiB (압축 해제 시 17 MB)입니다.

빌드 전 로컬 환경(macOS 26.6.2, arm64)에서 Launcher 스크립트의 세 가지 분기를 모두
직접 테스트했습니다.

| 테스트 시나리오 | 검증 결과 |
|---|---|
| 최초 실행 | 259개 파일이 정상 복사되며, `run_en.sh` 스크립트의 실행 권한이 유지됨을 확인 |
| 동일 버전 재실행 | 파일 복사 로직을 건너뛰며, 기존 사용자의 `config.json`이 보존됨을 확인 |
| 앱 버전 업데이트 | `key/`, `logs/`, `comfy/models/`, `workflow_backup/`, `config.json` 등 주요 사용자 데이터를 보존하고 코드 폴더 내 잔여 파일만 정상 정리함을 확인 |

추가 안전장치로, CI 단계에서 Payload 내부에 `config.json`, `key/`, `comfy/`,
`logs/`, `models/`, `.git/` 등 사용자 데이터나 개발 산출물이 섞여 들어갔는지 매번
검증하며, 하나라도 발견될 시 빌드를 강제 실패(Fail) 처리하도록 스크립트를
작성했습니다.

Intel Mac(`macos-13`) 빌드는 대상에서 제외했습니다. 해당 Runner의 큐(Queue) 대기
시간이 과도하게 길고, Intel Mac 환경에서는 로컬 Diffusion 구동이 불가하여 어차피
원격 클라우드 연동 경로를 써야 하므로 arm64 아키텍처와 사용 조건이 완전히 동일하기
때문입니다.

## 의존성 안내

`run_en.sh` 스크립트가 포함된 **PR B가 본 PR보다 먼저 병합되어야만** 정상 동작합니다.

---

## For the reviewing agent

### Scope

1 commit, 3 new files. Adds `packaging/macos/` and `.github/workflows/`.
**Touches no existing file.** Reverting is deleting three files.

```
packaging/macos/build_app.sh     bundle builder (macOS only)
packaging/macos/launcher.sh      Contents/MacOS entry point
.github/workflows/macos-package.yml
```

### Why not PyInstaller

The application writes into its own directory tree: the installer clones ComfyUI
to `<project_root>/comfy` (~12 GiB), models land beside it, and `config.json`,
`key/`, `logs/`, `workflow_backup/` are all siblings. A frozen bundle in
`/Applications` is read-only, so this fails at first install.

Making it work would require relocating the project root — but the root is
derived independently in **~45 production modules, 228 lines** of the form:

```python
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
```

Under PyInstaller `__file__` points inside the bundle. Fixing that is a refactor
of the whole codebase and should be its own decision, not a side effect of
packaging. The copy-on-first-run approach sidesteps it entirely.

Freezing also would not remove the runtime dependencies that actually matter:
`uv` is still needed to build ComfyUI's **separate** venv
(`comfy_installer/dependency_installer.py`), and `git` is still needed to clone
ComfyUI and custom nodes (`comfy_installer/node_installer.py`).

### How the bundle is built

Payload comes from `git archive HEAD` — **tracked files only**. This is
deliberate: an exclude list is something you forget to update, whereas
`git archive` cannot emit an untracked file at all. A handful of directories
useless inside an app (`tests`, `tools`, `upstream_patches`, `.github`,
`packaging`, `updater_dev`) are then removed.

The CI job asserts the payload contains none of `config.json`, `key`, `comfy`,
`logs`, `models`, `.git`, and fails the build if any appears.

### Launcher behaviour — three paths, all tested

| Situation | Behaviour |
|---|---|
| First run | `ditto` payload → `~/Library/Application Support/SoyaComfy`, write version file |
| Same version | No copy. User data untouched |
| Version changed | `rsync` **only** the paths in `CODE_PATHS`; user data untouched |

`CODE_PATHS` is an allow-list, not a deny-list, so a new user-data directory
cannot be destroyed by an upgrade simply because nobody remembered to exclude it.

Measured locally on arm64 before submission:

```
first run   259 files copied, run_en.sh executable
second run  no-op, user config.json preserved
upgrade     key/, logs/, comfy/models/, workflow_backup/, config.json  → all survived
            stale file in modes/ → removed        (rsync --delete, code paths only)
```

`SOYA_DATA_ROOT` overrides the destination, which is how the above was tested
without touching a real install.

### Things that look like details but are not

- **`ditto -c -k`, not `zip`.** A plain `zip` loses the executable bit, so the
  downloaded bundle will not launch.
- **`ditto`, not `cp -R`,** for the payload copy — `cp` drops macOS metadata.
- **Ad-hoc `codesign` is mandatory**, not cosmetic. Apple Silicon refuses
  unsigned executables outright; this is separate from Gatekeeper/quarantine.
- **Terminal is opened deliberately.** First run installs uv, syncs Python
  dependencies and fetches video tools — minutes of output. A silent double-click
  looks like a hang.

### CI result

The workflow ran on this branch in the fork before submission:
**`Build arm64 = success`.** The Intel job was removed — `macos-13` runners queue
for a long time, and an Intel Mac cannot run local diffusion anyway, so it would
be using the remote path that arm64 also uses.

Tagging `v*` attaches the zip to a GitHub release with the quarantine
instructions in the notes. On non-tag pushes it only uploads an artifact.

### What was NOT verified

- **The bundle has never been launched end to end.** The copy, upgrade, signing,
  and archive steps are all verified; what is *not* verified is a full
  double-click → install ComfyUI → generate an image run from inside the bundle.
  That needs a clean macOS machine.
- **No Intel build.** Deliberate, as above.
- **Upgrade across a real version bump** was simulated by editing the version
  file, not by installing two successive releases.

### Questions the maintainer may reasonably ask

- *"Do I have to maintain this?"* — It is three self-contained files that touch
  nothing else. If it rots, deleting them restores the status quo exactly.
- *"Why is there no icon?"* — None was invented. `AppIcon.icns` can be dropped in
  and referenced from `Info.plist` whenever you want one.
- *"Can this also build a Windows package?"* — Not as written, and it would be a
  separate piece of work. Windows already has `run_en.bat` and a working install
  story, so the gap this closes does not exist there.
- *"Does the bundle include models?"* — No. 17 MB, code only. Models and ComfyUI
  are fetched at first run exactly as they are today.
