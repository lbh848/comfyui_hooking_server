"""
view_prompt_ui.py
JSON 프롬프트 뷰어 - tkinter GUI
\n → 실제 줄바꿈 변환
"""
import json
import tkinter as tk
from tkinter import scrolledtext, font


def format_prompt(json_str: str) -> str:
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return f"[JSON 파싱 에러] {e}"

    lines = []
    sep = "─" * 80
    thick = "═" * 80

    # 메타 정보
    meta_parts = []
    if "model" in data:
        meta_parts.append(f"Model: {data['model']}")
    if "temperature" in data:
        meta_parts.append(f"Temperature: {data['temperature']}")
    if "stream" in data:
        meta_parts.append(f"Stream: {data['stream']}")
    if meta_parts:
        lines.append(thick)
        lines.append("  " + "  |  ".join(meta_parts))
        lines.append(thick)

    messages = data.get("messages", [])
    if not messages:
        return "(messages 없음)"

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        formatted = content.replace("\\n", "\n")

        lines.append(f"\n{sep}")
        lines.append(f"  [{role}]  (message {i + 1}/{len(messages)})")
        lines.append(sep)
        lines.append(formatted)

    lines.append(f"\n{thick}")
    lines.append(f"  총 {len(messages)}개 메시지")
    lines.append(thick)

    return "\n".join(lines)


def on_convert():
    raw = input_text.get("1.0", tk.END).strip()
    if not raw:
        return
    result = format_prompt(raw)
    output_text.config(state=tk.NORMAL)
    output_text.delete("1.0", tk.END)
    output_text.insert("1.0", result)
    output_text.config(state=tk.DISABLED)


def on_clear():
    input_text.delete("1.0", tk.END)
    output_text.config(state=tk.NORMAL)
    output_text.delete("1.0", tk.END)
    output_text.config(state=tk.DISABLED)


def on_paste():
    input_text.delete("1.0", tk.END)
    try:
        clipped = root.clipboard_get()
        input_text.insert("1.0", clipped)
    except tk.TclError:
        pass


# ── UI 구성 ──
root = tk.Tk()
root.title("Prompt Viewer  |  \\n → 줄바꿈 변환")
root.geometry("1200x800")
root.configure(bg="#1e1e2e")

# 폰트
mono = font.Font(family="Consolas", size=10)
header_font = font.Font(family="맑은 고딕", size=11, weight="bold")

# 색상 (dark theme)
BG = "#1e1e2e"
FG = "#cdd6f4"
INPUT_BG = "#313244"
OUTPUT_BG = "#1e1e1e"
ACCENT = "#89b4fa"
BTN_BG = "#45475a"
BTN_FG = "#cdd6f4"
BORDER = "#585b70"

root.option_add("*TCombobox*Listbox*Background", BG)

# ── 상단 버튼 프레임 ──
btn_frame = tk.Frame(root, bg=BG)
btn_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

tk.Label(btn_frame, text="입력 (JSON 붙여넣기)", font=header_font,
         bg=BG, fg=ACCENT).pack(side=tk.LEFT)

btn_style = dict(font=header_font, bg=BTN_BG, fg=BTN_FG,
                 activebackground=ACCENT, activeforeground="#1e1e2e",
                 relief=tk.FLAT, padx=15, pady=4, cursor="hand2")

tk.Button(btn_frame, text="📋 붙여넣기", command=on_paste, **btn_style).pack(side=tk.RIGHT, padx=2)
tk.Button(btn_frame, text="🗑 지우기", command=on_clear, **btn_style).pack(side=tk.RIGHT, padx=2)
tk.Button(btn_frame, text="▶ 변환", command=on_convert, **btn_style).pack(side=tk.RIGHT, padx=2)

# ── 입력 영역 ──
input_text = scrolledtext.ScrolledText(
    root, font=mono, wrap=tk.WORD,
    bg=INPUT_BG, fg=FG, insertbackground=FG,
    selectbackground=ACCENT, selectforeground="#1e1e2e",
    height=12, relief=tk.FLAT, borderwidth=2
)
input_text.pack(fill=tk.BOTH, expand=False, padx=10, pady=(0, 5))

# ── 출력 라벨 ──
tk.Label(root, text="결과", font=header_font,
         bg=BG, fg=ACCENT).pack(anchor=tk.W, padx=10)

# ── 출력 영역 ──
output_text = scrolledtext.ScrolledText(
    root, font=mono, wrap=tk.WORD,
    bg=OUTPUT_BG, fg="#a6e3a1",
    insertbackground=FG,
    selectbackground=ACCENT, selectforeground="#1e1e2e",
    relief=tk.FLAT, borderwidth=2
)
output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
output_text.config(state=tk.DISABLED)

# 단축키
root.bind("<Control-Return>", lambda e: on_convert())
root.bind("<Control-v>", lambda e: on_paste())

root.mainloop()
