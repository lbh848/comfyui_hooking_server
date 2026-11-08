# comfyui_hooking_server — V6 정리 기반

설치 폴더는 아래 구조를 사용합니다. 개발용 `.work`는 `src` 안에 두고 Git에서 제외합니다.

```text
설치 폴더/
├─ src/          소스, 기본 리소스, 실행기, 개발·배포 도구
├─ config/       settings.json 앱·LLM 공통 설정, key/ 인증 정보, 업데이터·원격 모델 설정
├─ data/         모델, 생성물, 워크플로우, 프리셋, 프롬프트, 로그, 백업
├─ 삽화.lnk
├─ 에셋.lnk
└─ 영상.lnk
```

Windows에서 새 Comfy공방 화면은 설치 폴더의 **`Comfy공방.exe`를 더블클릭**해 실행합니다.
이미 실행 중이면 시작 창 없이 브라우저만 바로 엽니다.
작은 시작 창에서 준비 단계와 실행 로그를 확인할 수 있습니다. 별도의 CMD 창은 뜨지 않습니다.
서버가 준비되면 브라우저가 열리고 시작 창과 로그 영역은 자동으로 닫힙니다.
시작에 실패하면 오류와 로그가 남아 있으며 X로 닫을 수 있습니다.
시작 로그는 `data/logs/launcher-날짜-프로세스번호.log`에도 저장됩니다.
실행 스크립트와 Windows 실행기 소스는 BOM 없는 ASCII로 유지하며, 준비 단계와 스크립트 오류 로그는 영문을 사용합니다.
시작 창의 한글 안내는 실행기 안에서 표시하고, 서버 출력은 UTF-8로 받아 한글 경로나 오류 로그도 읽을 수 있게 합니다.
좌측 상단의 **Comfy공방**을 누르면 설정/삽화/에셋 화면을 전환합니다.
에셋 작업 공간(`/asset/`)에서 **에셋 생성**과 **에셋 일괄 생성**을 사용할 수 있습니다.
에셋 일괄 생성에서 슬롯별 설정·복제·순서 변경·일괄 설정, 체인 프리셋 저장·불러오기를 사용할 수 있습니다.
에셋의 배경은 **그림체 설정**에서 선택하고 그림체 프리셋의 기본값·적용값과 함께 저장합니다. 에셋 생성 화면의 **그림체 태그 → 배경**에서도 조절할 수 있으며, 일반·일괄·마스크 기반 생성에 모두 반영됩니다. 캐릭터를 바꿔도 선택한 배경은 유지됩니다.
이미지 생성 옆의 **마스크 기반 생성**에서 빈 캔버스에 사각형을 그려 1인 위치 제어 또는 최대 3인 생성을 할 수 있습니다. 영역마다 캐릭터와 복장·표정·자연어를 지정합니다. 외모·LoRA·참조 이미지는 캐릭터 설정의 값을 사용하며 이 창에는 편집 항목을 표시하지 않습니다. 배경은 기존 배경 드롭다운 선택값을 사용합니다. 박스 좌표로 만든 기본 구도는 **추가 동작·상황**에 미리 채워지며, 편집한 내용이 공통 구도로 전달됩니다. 직접 편집한 내용은 박스를 바꿔도 유지되고 **박스 배치로 다시 채우기**로 갱신할 수 있습니다. 현재 그림체와 그림 모델로 새 이미지를 생성하며, 결과는 첫 번째 영역 캐릭터의 복장·표정 폴더에 저장합니다.
생성 전에 **워크플로우 설정**에서 사용할 에셋 워크플로우를 지정하세요.
ANIMA Ref 이미지와 LoRA를 지원하며, 에셋 작업 공간에서는 IPA와 포즈를 사용하지 않습니다.
생성 진행 상황과 취소는 **작업 내역 관리 → 작업 큐**에서 확인합니다.
종료는 우측 상단의 **프로그램 종료** 버튼을 누르세요. 콘솔에서 직접 `src/run_settings.ps1`을 실행한 경우에는 `Ctrl+C`도 사용할 수 있습니다.
프로그램 종료는 관리 중인 Comfy와 작업 서버도 함께 정리하며, 브라우저 탭은 직접 닫으면 됩니다.

기존 화면의 Windows 바로가기는 `src/create_shortcuts.ps1`을 실행해 생성합니다.
설치 폴더를 옮겼을 때도 다시 실행하면 됩니다. 바로가기는 Git에서 제외합니다.
수동 실행은 설치 루트에서 `src\run_en.bat illustration` 또는 `src\run_en.bat asset`입니다.

현재 세 바로가기는 **기존 공용 엔진과 화면의 실행 진입점**입니다. 에셋 진입점은 기존
`에셋 생성` 탭을 선택하고, 삽화·영상 진입점은 기존 기본 화면을 엽니다.
React 설정 화면과 설치 실행 방법은 [settings_ui/README.md](settings_ui/README.md)에 있습니다.
설정 화면은 로컬 전용 `http://127.0.0.1:8201/setting`으로 열립니다.
외부 삽화 요청은 `8202` 포트에서 받습니다. **삽화 → 외부 연결**에서 랜덤 주소(Cloudflare)와 고정 주소(Tailscale)를 발급할 수 있으며, 설정 화면과 관리 API는 `8201` 로컬 전용으로 유지됩니다.
초기 설치는 암호화된 동봉 팩의 기본 워크플로우 5개를 ComfyUI에 읽기 전용으로 설치합니다.
Anima 2.9B (`Anima-2.9B-preview-v1.safetensors`)와 기존 Anima용 텍스트 인코더·VAE,
설치 매니페스트의 `models/loras/`에 등록된 LoRA도 워크플로우 참조 여부와 관계없이 초기 설치에 포함합니다.
모델 다운로드 정보와 기본 제공 LoRA 목록은 `comfy_installer/resources/install_manifest.json` 한 곳에서 관리합니다.
`workflows.soyawfp`에는 워크플로우와 고정 의존성의 모델 ID만 담으며, 설치 명세를 동봉하거나 팩을 불러올 때 덮어쓰지 않습니다.
따라서 기본 모델·LoRA를 추가할 때 팩을 다시 만들 필요가 없습니다. 프로그램 프리셋에서 선택하는
디퓨전 모델·체크포인트·LoRA 기본값은 팩의 고정 의존성에서 제외합니다.
CPU 환경이나 클라우드 직접 다운로드 설정에서도 이 기본 모델과 LoRA는 로컬에 설치하며, Civitai 다운로드에는 등록한 API 키를 사용합니다.
동봉 팩은 총 16개이며, 삽화는 ANIMA+ILXL과 ANIMA 각각 하나씩 제공합니다. ANIMA 고속·고속 8GB 파일은 ANIMA 기본형으로 통합했습니다.
에셋 ANIMA V4 기반의 `배포_공방_ANIMA_V4_투명배경`은 Anima-RGBA VAE로 생성한 알파 채널을 후처리 뒤 다시 합쳐 RGBA PNG로 저장합니다.
RGBA LoRA는 기존 그림 모델의 공통 LoRA 설정으로 적용하며, `transparent background` 문구는 배경 프리셋에서 지정합니다.
삽화 데이터 준비의 데이터 패치는 삽화유틸리티 V2_0으로 FACE 이미지만 생성하며, 임베딩 캐시를 생성하거나 완료 조건으로 요구하지 않습니다.
이번 정리에서는 기존 생성 동작과 시작 시 다운로드 절차를 유지합니다.

경로 정책은 [app_paths.py](app_paths.py)에서 관리합니다. 기존 엔진의 `BASE_DIR`와
서비스의 `project_root`는 `src`를 가리키며, 쓰기 대상은 `data_root()` 또는
`config_root()`로 구합니다. 작업 디렉터리에 의존하는 경로를 새로 추가하지 않습니다.
기본 프롬프트·자동완성 파일은 `src`에 배포하고 첫 실행 시 `data`의 누락된 파일만 채웁니다.
삽화용 봇 설정에는 `src/defaults/illustration/bot.json`의 **공방-테스트봇** 하나를 기본으로 포함합니다.
새 설치에서는 이 봇이 활성화됩니다. 삽화용 봇 설정의 **배포 풀기 / 배포 잠그기**와
**봇 구분**으로 배포 봇과 사용자 봇을 관리합니다. 배포 봇은 기본적으로 읽기 전용이며,
잠금을 해제한 뒤 저장한 캐릭터·이미지·지침·설정은 배포 파일에도 반영됩니다.
배포 파일은 업데이트 시 백업 없이 교체됩니다. 보관할 설정은 **봇 복제**로 사용자 봇에 저장하세요.
사용자 봇은 업데이트 후에도 유지됩니다.
설정과 프리셋의 자동 백업은 내용이 달라질 때 저장 전 원본을 보관하며, 파일별로 중복 없이 최근 100개를 유지합니다.
프로그램 시작 시에도 배포·사용자 프리셋, 숨긴 프리셋과 체인 프리셋을 한 번 확인해 백업합니다.
같은 내용으로 다시 저장하거나 프로그램을 다시 시작해도 중복 사본을 만들지 않습니다.
설정은 `data/backups/config/`, 프리셋은 `data/backups/presets/`에 보관합니다.
기존 봇 이력은 `data/asset_data/backups/`, 태그 프리셋 이력은 `data/asset_data/backup/`을 계속 사용합니다.
이 백업은 지정된 JSON 파일만 다루며 ComfyUI, 가상환경, 모델 파일을 복사하지 않습니다.

### 배포·사용자 채널과 Git 업데이트

`src/` 전체는 Git 제외 대상이 아닙니다. `src/presets/*.json`, `src/presets/chain_presets/*.json`, `src/defaults/illustration/bot.json`,
`src/prompts/bot_system_prompt/presets.json`은 배포 채널이며 소스와 함께 커밋해야 배포에 포함됩니다.
배포 풀기로 수정한 파일도 업데이트 시 새 버전으로 교체됩니다. 배포에서 삭제된 프리셋 파일도 제거됩니다.
직접 유지할 내용은 사용자 채널로 복제하거나 이동하세요. 편집·시작 백업은 업데이트가 덮어쓰지 않는 `data/`에 남습니다.

사용자 채널은 `config/*_presets.json`, `data/asset_data/`, `data/chain_presets/` 등에 저장합니다.
체인 프리셋도 배포·사용자 그룹으로 나뉩니다. 배포 체인은 기본 읽기 전용이며, 에셋 일괄 생성에서 복제로 사용자 체인을 만들거나 배포 풀기 후 그룹을 이동할 수 있습니다.
`config/`와 `data/`는 `.gitkeep`만 추적하며, 사용자 데이터와 백업은 Git에 포함하지 않습니다.
일반 Git 업데이트와 앱 번들 업데이트 모두 이 두 경로의 사용자 파일을 교체하지 않습니다.
이름이 같은 배포 프리셋이 추가되어도 사용자 프리셋을 삭제하거나 배포 프리셋으로 바꾸지 않습니다.

사용자 수정 파일은 덮어쓰지 않습니다. V5 설치본의 데이터 자동 이관은 포함하지 않았으므로
기존 설치본 위에 덮어쓰는 배포에는 별도 이관 절차가 필요합니다.

기존 Actor 프런트엔드는 빌드 도구 없이 실행됩니다. 새 React 설정 화면은 별도로 빌드합니다.

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
배포 프리셋(src/presets/presets.json)은 예외로, 직접 수정했어도 확인·백업 없이 새 파일 전체로 교체합니다.
유지할 프리셋은 프리셋 관리에서 비배포로 저장하세요.
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
