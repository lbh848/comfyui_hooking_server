"""
view_prompt.py
JSON 프롬프트를 보기 좋게 출력하는 뷰어
\n → 실제 줄바꿈 변환
"""
import json
import sys
import os


def view_prompt(json_str: str):
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"\n[JSON 파싱 에러] {e}")
        return

    print("\n" + "=" * 80)

    # model 정보
    if "model" in data:
        print(f"  Model: {data['model']}")
    if "temperature" in data:
        print(f"  Temperature: {data['temperature']}")
    if "stream" in data:
        print(f"  Stream: {data['stream']}")
    print("=" * 80)

    messages = data.get("messages", [])
    if not messages:
        print("\n  (messages 없음)")
        return

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")

        print(f"\n{'─' * 80}")
        print(f"  [{role}]  (message {i + 1}/{len(messages)})")
        print(f"{'─' * 80}")

        # \n → 실제 줄바꿈
        formatted = content.replace("\\n", "\n")
        print(formatted)

    print(f"\n{'=' * 80}")
    print(f"  총 {len(messages)}개 메시지")
    print(f"{'=' * 80}\n")


def main():
    print("=" * 80)
    print("  PROMPT VIEWER  |  \\n → 줄바꿈 변환")
    print("  JSON을 붙여넣고 Enter 두 번 치면 출력됩니다")
    print("  종료: q 입력 또는 Ctrl+C")
    print("=" * 80)

    while True:
        print("\n▼ JSON 입력 (빈 줄 Enter = 실행):")
        lines = []
        empty_count = 0

        while True:
            try:
                line = input()
            except EOFError:
                line = ""
            except KeyboardInterrupt:
                print("\n\n종료합니다.")
                return

            if line.strip() == "q":
                print("종료합니다.")
                return

            if line.strip() == "":
                empty_count += 1
                if empty_count >= 1 and lines:
                    break
                # 빈 줄은 무시하지만 누적 (JSON 구조 유지를 위해 마지막 빈 줄만 제외)
                continue
            else:
                empty_count = 0
                lines.append(line)

        json_str = "\n".join(lines)
        view_prompt(json_str)

        # 연속 사용 안내
        print("다른 프롬프트를 보려면 JSON을 붙여넣으세요. (q = 종료)")


if __name__ == "__main__":
    # 파일 경로를 인자로 받은 경우
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        if not os.path.exists(filepath):
            print(f"파일 없음: {filepath}")
            sys.exit(1)
        with open(filepath, "r", encoding="utf-8") as f:
            json_str = f.read()
        view_prompt(json_str)
    else:
        main()
