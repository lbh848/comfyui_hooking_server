# comfyui_hooking_server — V6 정리 기반

설치 폴더는 아래 구조를 사용합니다. 개발용 `.work`는 루트에 두고 Git에서 제외합니다.

```text
설치 폴더/
├─ src/          소스, 기본 리소스, 실행기, 개발·배포 도구
├─ config/       config.json, key/ 인증 정보, 업데이터·원격 모델 설정
├─ data/         모델, 생성물, 워크플로우, 프리셋, 프롬프트, 로그, 백업
├─ 삽화.lnk
├─ 에셋.lnk
└─ 영상.lnk
```

Windows 바로가기는 `src/create_shortcuts.ps1`을 실행해 생성합니다.
설치 폴더를 옮겼을 때도 다시 실행하면 됩니다. 바로가기는 Git에서 제외합니다.
수동 실행은 설치 루트에서 `src\run_en.bat illustration` 또는 `src\run_en.bat asset`입니다.

현재 세 바로가기는 **기존 공용 엔진과 화면의 실행 진입점**입니다. 에셋 진입점은 기존
`에셋 생성` 탭을 선택하고, 삽화·영상 진입점은 기존 기본 화면을 엽니다.
전용 화면 분리, React 도입, 기능 제거, Lazy 설치, 워크플로우 내장화는 다음 작업입니다.
이번 정리에서는 기존 생성 동작과 시작 시 다운로드 절차를 유지합니다.

경로 정책은 [app_paths.py](app_paths.py)에서 관리합니다. 기존 엔진의 `BASE_DIR`와
서비스의 `project_root`는 `src`를 가리키며, 쓰기 대상은 `data_root()` 또는
`config_root()`로 구합니다. 작업 디렉터리에 의존하는 경로를 새로 추가하지 않습니다.
기본 프롬프트·자동완성 파일은 `src`에 배포하고 첫 실행 시 `data`의 누락된 파일만 채웁니다.
사용자 수정 파일은 덮어쓰지 않습니다. V5 설치본의 데이터 자동 이관은 포함하지 않았으므로
기존 설치본 위에 덮어쓰는 배포에는 별도 이관 절차가 필요합니다.

프런트엔드는 빌드 도구 없이 실행됩니다.

- `frontend/index.html`: 기존 화면 마크업
- `frontend/static/app.css`: 기존 인라인 스타일
- `frontend/static/app.js`: 기존 인라인 JavaScript
- `frontend/static/entry.js`: 실행 진입점 선택
- `frontend/illustration_*.js`: 기존 삽화 보조 모듈

개발 명령은 `src`에서 실행합니다(`cd src`). `uv sync`, `uv run python launch.py`,
`uv run python -m unittest discover -s tests -v`를 사용할 수 있습니다.
프로젝트 전체 의존성 설치 없이 정리 회귀 테스트만 실행하려면:

```powershell
uv run --no-project --with aiohttp --with argon2-cffi --with cryptography python -m unittest discover -s tests -v
```

## macOS 14 이상 (Apple Silicon)

설치 루트에서 Windows는 src\run_en.bat, macOS는 bash src/run_en.sh로 실행합니다.
macOS는 Git과 FFmpeg이 필요합니다(xcode-select --install, brew install ffmpeg).
앱 번들은 src/packaging/macos/build_app.sh 또는 macOS package 워크플로우로 만들 수 있습니다.
빌드 환경에도 uv가 필요합니다. 번들에는 소스 코드가 들어가며, 첫 실행에 Python과 의존성을 준비합니다.
ComfyUI와 모델은 앱 안의 설치 탭에서 별도로 설치합니다.

작업 폴더는 ~/Library/Application Support/SoyaComfy입니다.
업데이트는 배포 파일 목록과 이전 내용을 비교합니다. 직접 수정한 프롬프트·팩 명세·추가 파일은 보존하고,
충돌하는 새 배포 파일과 교체 전 파일은 작업 폴더의 data/backups/app_updates/에 보관합니다.
첫 실행이 중단돼도 다시 실행하면 이어서 파일을 준비합니다.
앱 실행 안내는 [macOS 시작 가이드](packaging/macos/FIRST_RUN.txt)를 참고하세요.

## License

This project is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Third-party components and data retain their respective licenses. The bundled
Korean autocomplete CSV is published by
[mengod-hf/danbooru-tags-kr](https://huggingface.co/datasets/mengod-hf/danbooru-tags-kr).
Its upstream license notice specifies CC BY-SA 4.0 for Korean localization
contributions and includes MIT notices and other third-party attributions.
See [the bundled CSV's source and license notices](auto_complete/README.md).
