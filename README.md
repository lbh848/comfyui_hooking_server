# comfyui_hooking_server

## macOS 14 이상 (Apple Silicon)

Windows는 run_en.bat, macOS는 bash run_en.sh로 실행합니다.
macOS는 Git과 FFmpeg이 필요합니다(xcode-select --install, brew install ffmpeg).
앱 번들은 packaging/macos/build_app.sh 또는 macOS package 워크플로우로 만들 수 있습니다.
빌드 환경에도 uv가 필요합니다. 번들에는 소스 코드가 들어가며, 첫 실행에 Python과 의존성을 준비합니다.
ComfyUI와 모델은 앱 안의 설치 탭에서 별도로 설치합니다.

작업 폴더는 ~/Library/Application Support/SoyaComfy입니다.
업데이트는 배포 파일 목록과 이전 내용을 비교합니다. 직접 수정한 프롬프트·팩 명세·추가 파일은 보존하고,
충돌하는 새 배포 파일과 교체 전 파일은 작업 폴더의 backups/app_updates/에 보관합니다.
첫 실행이 중단돼도 다시 실행하면 이어서 파일을 준비합니다.
앱 실행 안내는 [macOS 시작 가이드](packaging/macos/FIRST_RUN.txt)를 참고하세요.

## License

This project is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
