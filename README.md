# comfyui_hooking_server

## macOS (Apple Silicon)

릴리스에서 `SoyaComfy-*-arm64.zip` 을 내려받아 쓰실 수 있습니다.
zip 안의 `먼저 읽어주세요.txt` 에 자세한 안내가 있습니다.

먼저 Git 과 FFmpeg 이 필요합니다.

```bash
xcode-select --install
brew install ffmpeg
```

응용 프로그램 폴더로 옮기신 뒤, 임시 서명 앱이라 격리 속성을 한 번 지워야 합니다.

```bash
xattr -dr com.apple.quarantine /Applications/SoyaComfy.app
```

저장소에서 직접 실행하실 때는 `./run_en.sh` 를 쓰시면 됩니다
(Windows 는 `run_en.bat`).

## License

This project is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
