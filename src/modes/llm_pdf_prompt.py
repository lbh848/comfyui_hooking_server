"""Gemini/Vertex용 대화 transcript PDF 변환.

대화의 자연어 흐름과 role 순서를 하나의 transcript로 보존하고, 기존 이미지
파트는 PDF 뒤에 원래 순서대로 첨부한다. 기본 PDF는 3pt 텍스트와
``ActualText``로 긴 문맥을 담는다. 이미지 모드는 9pt 글자를 손실 없는
그림으로 그려 PDF에 담고 네이티브 텍스트 레이어를 포함하지 않는다.
"""

from __future__ import annotations

from app_paths import data_root

import base64
import copy
import json
import math
import re
import zlib
from pathlib import Path


PDF_MIME_TYPE = "application/pdf"
PDF_FILENAME = "llm-conversation.pdf"
PDF_MEDIA_RESOLUTION = "MEDIA_RESOLUTION_LOW"
PDF_FONT_SIZE = 3.0
PDF_IMAGE_FONT_SIZE = 9.0

# Image-only PDF transport defaults. These mirror the settings exposed by the
# image prompt panel; the native-text PDF path keeps its own fixed layout.
PDF_IMAGE_DEFAULT_OPTIONS = {
    "media_resolution": "medium",
    "page_width": 1191,
    "page_height": 1684,
    "margin": 12,
    "font_size": 16,
    "line_spacing": 1.2,
    "font_family": (
        '"Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", '
        '"Noto Sans CJK KR", sans-serif'
    ),
    "line_break": "character",
    "color_mode": "grayscale",
    "emoji": "keep",
    "newline": "actual",
    "preserve_system_prompt": True,
}
_PDF_IMAGE_MEDIA_RESOLUTIONS = frozenset({"low", "medium", "high"})
_PDF_IMAGE_LINE_BREAKS = frozenset({"character", "word"})
_PDF_IMAGE_COLOR_MODES = frozenset({"grayscale", "auto"})
_PDF_IMAGE_EMOJI_MODES = frozenset({"keep", "strip"})
_PDF_IMAGE_NEWLINE_MODES = frozenset({"actual", "collapse"})
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF☀-➿️‍]"
)

_PAGE_WIDTH = 595.28
_PAGE_HEIGHT = 841.89
_PAGE_MARGIN = 10.0
_CHARACTER_WIDTH_RATIO = 0.5

PDF_TRANSCRIPT_BOOTSTRAP = """The attached PDF contains the complete role-labelled conversation transcript.
Extract and read its embedded native text, preserve the role and message order, and continue the task from that context.
Interpret U+2028 as one newline and U+2029 as two newlines.
Any image markers in the transcript refer, in numerical order, to the image parts attached after the PDF."""

PDF_BASE64_TRANSCRIPT_BOOTSTRAP = """The attached PDF's embedded native text is one UTF-8 Base64 block.
Extract that block exactly, Base64-decode it as UTF-8, then treat the decoded value as the complete role-labelled conversation transcript and continue the task from that context.
Interpret U+2028 as one newline and U+2029 as two newlines after decoding.
Any image markers in the decoded transcript refer, in numerical order, to the image parts attached after the PDF."""

IMAGE_TRANSCRIPT_BOOTSTRAP = """The attached PDF contains pictures of the complete role-labelled conversation transcript, with no embedded text layer.
Read the pictured text in page order, preserve the roles and message order, and continue the task from that context. Visual line wrapping does not add new messages.
Any numbered image markers in the transcript refer to the original image parts attached after the PDF, not to its pictured text pages."""

IMAGE_BASE64_TRANSCRIPT_BOOTSTRAP = """The visible text across the attached PDF pages is one UTF-8 Base64 block, with no embedded text layer.
Read it in page order, ignore visual line wrapping, Base64-decode the complete block as UTF-8, then continue the task using the decoded role-labelled conversation transcript.
Any numbered image markers in the decoded transcript refer to the original image parts attached after the PDF, not to its pictured text pages."""


def _pdf_number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _utf16_hex(value: str) -> str:
    return value.encode("utf-16-be").hex().upper()


def _newline_marker(count: int) -> str:
    if count <= 0:
        return ""
    if count == 1:
        return "\u2028"
    return "\u2029" + ("\u2028" * (count - 2))


def normalize_image_options(options: dict | None = None) -> dict:
    """Return image-PDF render options with validated defaults."""
    merged = dict(PDF_IMAGE_DEFAULT_OPTIONS)
    if isinstance(options, dict):
        for key in merged:
            if options.get(key) is not None:
                merged[key] = options[key]

    def integer(key: str, minimum: int, maximum: int) -> int:
        try:
            value = int(merged[key])
        except (TypeError, ValueError):
            value = PDF_IMAGE_DEFAULT_OPTIONS[key]
        return max(minimum, min(maximum, value))

    def number(key: str, minimum: float, maximum: float) -> float:
        try:
            value = float(merged[key])
        except (TypeError, ValueError):
            value = PDF_IMAGE_DEFAULT_OPTIONS[key]
        return max(minimum, min(maximum, value))

    merged["media_resolution"] = (
        str(merged["media_resolution"]).strip().lower()
        if str(merged["media_resolution"]).strip().lower() in _PDF_IMAGE_MEDIA_RESOLUTIONS
        else PDF_IMAGE_DEFAULT_OPTIONS["media_resolution"]
    )
    merged["page_width"] = integer("page_width", 256, 4096)
    merged["page_height"] = integer("page_height", 256, 4096)
    merged["margin"] = integer("margin", 0, 256)
    merged["font_size"] = integer("font_size", 6, 128)
    merged["line_spacing"] = number("line_spacing", 0.8, 3.0)
    merged["font_family"] = str(merged["font_family"] or PDF_IMAGE_DEFAULT_OPTIONS["font_family"])
    merged["line_break"] = (
        str(merged["line_break"]).strip().lower()
        if str(merged["line_break"]).strip().lower() in _PDF_IMAGE_LINE_BREAKS
        else PDF_IMAGE_DEFAULT_OPTIONS["line_break"]
    )
    merged["color_mode"] = (
        str(merged["color_mode"]).strip().lower()
        if str(merged["color_mode"]).strip().lower() in _PDF_IMAGE_COLOR_MODES
        else PDF_IMAGE_DEFAULT_OPTIONS["color_mode"]
    )
    merged["emoji"] = (
        str(merged["emoji"]).strip().lower()
        if str(merged["emoji"]).strip().lower() in _PDF_IMAGE_EMOJI_MODES
        else PDF_IMAGE_DEFAULT_OPTIONS["emoji"]
    )
    merged["newline"] = (
        str(merged["newline"]).strip().lower()
        if str(merged["newline"]).strip().lower() in _PDF_IMAGE_NEWLINE_MODES
        else PDF_IMAGE_DEFAULT_OPTIONS["newline"]
    )
    merged["preserve_system_prompt"] = bool(merged["preserve_system_prompt"])
    return merged


_FONT_NAME_FILES = {
    "malgun gothic": ("malgun.ttf", "malgunsl.ttf"),
    "apple sd gothic neo": ("AppleSDGothicNeo.ttc",),
    "noto sans kr": ("NotoSansKR-VF.ttf", "NotoSansKR-Regular.otf", "NotoSansKR-Regular.ttf"),
    "noto sans cjk kr": ("NotoSansCJKkr-Regular.otf", "NotoSansKR-VF.ttf"),
    "noto sans": ("NotoSansKR-VF.ttf", "NotoSans-Regular.ttf"),
    "segoe ui": ("segoeui.ttf",),
}


def _font_search_dirs() -> list[Path]:
    dirs = [data_root() / "fonts", Path("C:/Windows/Fonts"), Path("/usr/share/fonts"), Path("/System/Library/Fonts")]
    return [directory for directory in dirs if directory.is_dir()]


def _resolve_image_font_path(font_family: str) -> Path | None:
    names = [
        name.strip().strip('"').strip("'").lower()
        for name in str(font_family or "").split(",")
        if name.strip()
    ]
    search_dirs = _font_search_dirs()
    for name in names:
        candidates = list(_FONT_NAME_FILES.get(name, ()))
        normalized = re.sub(r"[^a-z0-9]+", "", name)
        for directory in search_dirs:
            for path in directory.iterdir():
                if not path.is_file() or path.suffix.lower() not in (".ttf", ".otf", ".ttc"):
                    continue
                stem = re.sub(r"[^a-z0-9]+", "", path.stem.lower())
                if stem == normalized or stem.startswith(normalized):
                    return path
        for directory in search_dirs:
            for filename in candidates:
                path = directory / filename
                if path.is_file():
                    return path
    fallback = data_root() / "fonts" / "NotoSansKR-VF.ttf"
    return fallback if fallback.is_file() else None


def _load_image_font(font_family: str, font_size: int):
    from PIL import ImageFont

    path = _resolve_image_font_path(font_family)
    if path is not None:
        try:
            return ImageFont.truetype(str(path), size=font_size)
        except OSError:
            pass
    return ImageFont.load_default(size=font_size)


def _strip_emoji(text: str) -> str:
    return _EMOJI_RE.sub("", text)


def _char_chunks(text: str, font, usable_width: float, width_scale: float = 1.0) -> list[str]:
    chunks: list[str] = []
    current = ""
    for character in text:
        if current and font.getlength(current + character) * width_scale > usable_width:
            chunks.append(current)
            current = character
        else:
            current += character
    chunks.append(current)
    return chunks


def _wrap_image_line(text: str, font, usable_width: float, line_break: str,
                     width_scale: float = 1.0) -> list[str]:
    if not text:
        return [""]
    if line_break != "word":
        return _char_chunks(text, font, usable_width, width_scale)
    chunks: list[str] = []
    current = ""
    for token in re.findall(r"\S+|\s+", text):
        candidate = current + token
        if not current or font.getlength(candidate.rstrip()) * width_scale <= usable_width:
            current = candidate
            continue
        if current.strip():
            chunks.append(current.rstrip())
        current = token.lstrip()
        if font.getlength(current) * width_scale > usable_width:
            broken = _char_chunks(current, font, usable_width, width_scale)
            chunks.extend(broken[:-1])
            current = broken[-1]
    if current or not chunks:
        chunks.append(current.rstrip())
    return chunks


def _paginate_image_text(
    text: str, *,
    width: int, height: int, margin: int, font, font_size: int,
    line_spacing: float, line_break: str, emoji: str, newline: str,
    character_width_ratio: float = _CHARACTER_WIDTH_RATIO,
) -> tuple[list[list[str]], int]:
    usable_width = max(1, width - (2 * margin))
    width_scale = max(0.1, character_width_ratio / _CHARACTER_WIDTH_RATIO)
    usable_height = max(1, height - (2 * margin))
    line_height = max(1, round(font_size * line_spacing))
    lines_per_page = max(1, usable_height // line_height)
    source = str(text)
    if emoji == "strip":
        source = _strip_emoji(source)
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    if newline == "collapse":
        source = re.sub(r"[ \t]*\n[ \t]*", " ", source)
    rendered_lines: list[str] = []
    for logical_line in source.split("\n"):
        if not logical_line:
            rendered_lines.append("")
            continue
        rendered_lines.extend(
            _wrap_image_line(logical_line, font, usable_width, line_break, width_scale)
        )
    if not rendered_lines:
        rendered_lines.append("")
    return [
        rendered_lines[offset:offset + lines_per_page]
        for offset in range(0, len(rendered_lines), lines_per_page)
    ], line_height


def _paginate_text(
    text: str, *, character_width_ratio: float = _CHARACTER_WIDTH_RATIO,
    font=None,
) -> list[list[dict]]:
    usable_width = _PAGE_WIDTH - (2 * _PAGE_MARGIN)
    usable_height = _PAGE_HEIGHT - (2 * _PAGE_MARGIN)
    characters_per_line = max(
        1,
        math.floor(usable_width / (PDF_FONT_SIZE * character_width_ratio)),
    )
    lines_per_page = max(1, math.floor(usable_height / PDF_FONT_SIZE))
    if font is not None:
        # Image pages use actual glyph advances, including full-width CJK.
        # Reserve the font's full vertical metrics so adjacent rows never overlap.
        lines_per_page = max(1, math.floor(usable_height * 2 / sum(font.getmetrics())))
        glyph_widths = {
            character: font.getlength(character) * character_width_ratio / _CHARACTER_WIDTH_RATIO
            for character in set(text)
        }

    rendered_lines: list[dict] = []
    logical_lines = str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    last_text_line: dict | None = None
    last_text_index = -1

    for logical_index, line in enumerate(logical_lines):
        if not line:
            rendered_lines.append({"glyphs": "", "actual_text": ""})
            continue

        if last_text_line is not None:
            last_text_line["actual_text"] += _newline_marker(
                logical_index - last_text_index
            )

        if font is None:
            chunks = [line[offset:offset + characters_per_line]
                      for offset in range(0, len(line), characters_per_line)]
        else:
            chunks = []
            start, width = 0, 0.0
            for offset, character in enumerate(line):
                advance = glyph_widths[character]
                if width + advance > usable_width * 2 and offset > start:
                    chunks.append(line[start:offset])
                    start, width = offset, 0.0
                width += advance
            chunks.append(line[start:])
        for chunk_index, chunk in enumerate(chunks):
            actual_text = chunk
            if last_text_line is None and chunk_index == 0:
                actual_text = _newline_marker(logical_index) + actual_text
            rendered_lines.append(
                {"glyphs": chunk, "actual_text": actual_text}
            )

        last_text_line = rendered_lines[-1]
        last_text_index = logical_index

    if last_text_line is not None:
        last_text_line["actual_text"] += _newline_marker(
            len(logical_lines) - last_text_index - 1
        )

    if not rendered_lines:
        rendered_lines.append({"glyphs": "", "actual_text": ""})
    return [
        rendered_lines[offset:offset + lines_per_page]
        for offset in range(0, len(rendered_lines), lines_per_page)
    ]


def _deflated_stream(data: bytes) -> bytes:
    compressed = zlib.compress(data)
    return (
        f"<< /Length {len(compressed)} /Filter /FlateDecode >>\nstream\n".encode(
            "ascii"
        )
        + compressed
        + b"\nendstream"
    )


def _build_to_unicode_cmap(character_ids: dict[str, int]) -> bytes:
    mappings = [
        f"<{character_id:04X}><{_utf16_hex(character)}>"
        for character, character_id in character_ids.items()
    ]
    blocks = []
    for offset in range(0, len(mappings), 100):
        block = mappings[offset:offset + 100]
        blocks.extend(
            [
                f"{len(block)} beginbfchar",
                *block,
                "endbfchar",
            ]
        )
    cmap = "\n".join(
        [
            "/CIDInit /ProcSet findresource begin",
            "12 dict begin",
            "begincmap",
            "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
            "/CMapName /LighbdUnicode-UCS def",
            "/CMapType 2 def",
            "1 begincodespacerange",
            "<0000><FFFF>",
            "endcodespacerange",
            *blocks,
            "endcmap",
            "CMapName currentdict /CMap defineresource pop",
            "end",
            "end",
        ]
    )
    return _deflated_stream(cmap.encode("ascii"))


def _build_page_stream(lines: list[dict], character_ids: dict[str, int]) -> bytes:
    commands = [
        "BT",
        f"/F0 {_pdf_number(PDF_FONT_SIZE)} Tf",
        f"{_pdf_number(PDF_FONT_SIZE)} TL",
        (
            "1 0 0 1 "
            f"{_pdf_number(_PAGE_MARGIN)} "
            f"{_pdf_number(_PAGE_HEIGHT - _PAGE_MARGIN - PDF_FONT_SIZE)} Tm"
        ),
    ]
    for index, line in enumerate(lines):
        glyphs = line["glyphs"]
        if glyphs:
            glyph_hex = "".join(
                f"{character_ids[character]:04X}" for character in glyphs
            )
            actual_text_hex = _utf16_hex(line["actual_text"])
            commands.extend(
                [
                    f"/Span << /ActualText <FEFF{actual_text_hex}> >> BDC",
                    f"<{glyph_hex}> Tj",
                    "EMC",
                ]
            )
        if index < len(lines) - 1:
            commands.append("T*")
    commands.append("ET")
    return _deflated_stream("\n".join(commands).encode("ascii"))


def _assemble_pdf(objects: list[bytes]) -> bytes:
    header = b"%PDF-1.7\n%\xff\xff\xff\xff\n"
    chunks = [header]
    offsets = [0]
    byte_offset = len(header)
    for object_number, payload in enumerate(objects, start=1):
        offsets.append(byte_offset)
        chunk = (
            f"{object_number} 0 obj\n".encode("ascii")
            + payload
            + b"\nendobj\n"
        )
        chunks.append(chunk)
        byte_offset += len(chunk)

    xref_offset = byte_offset
    xref_lines = [
        f"xref\n0 {len(objects) + 1}",
        "0000000000 65535 f ",
        *(f"{offset:010d} 00000 n " for offset in offsets[1:]),
        "trailer",
        f"<< /Size {len(objects) + 1} /Root 1 0 R >>",
        "startxref",
        str(xref_offset),
        "%%EOF",
    ]
    chunks.append(("\n".join(xref_lines) + "\n").encode("ascii"))
    return b"".join(chunks)


def build_text_pdf(
    text: str, *, character_width_ratio: float = _CHARACTER_WIDTH_RATIO,
    prompt_format: str = "pdf", image_options: dict | None = None,
) -> tuple[bytes, int]:
    """Build a native-text PDF or a text-layer-free pictured PDF."""
    if prompt_format == "image":
        from PIL import Image, ImageDraw

        options = normalize_image_options(image_options)
        page_width = options["page_width"]
        page_height = options["page_height"]
        margin = options["margin"]
        font = _load_image_font(options["font_family"], options["font_size"])
        pages, line_height = _paginate_image_text(
            str(text).expandtabs(4),
            width=page_width,
            height=page_height,
            margin=margin,
            font=font,
            font_size=options["font_size"],
            line_spacing=options["line_spacing"],
            line_break=options["line_break"],
            emoji=options["emoji"],
            newline=options["newline"],
            character_width_ratio=character_width_ratio,
        )
        # The pictured canvas is drawn at 2x density; the PDF page uses points.
        page_width_pt = page_width / 2
        page_height_pt = page_height / 2
        color_mode = "L" if options["color_mode"] == "grayscale" else "RGB"
        background = 255 if color_mode == "L" else (255, 255, 255)
        text_color = 0 if color_mode == "L" else (0, 0, 0)
        color_space = "/DeviceGray" if color_mode == "L" else "/DeviceRGB"
        page_references = " ".join(f"{3 + index * 3} 0 R" for index in range(len(pages)))
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            (f"<< /Type /Pages /Kids [{page_references}] /Count {len(pages)} "
             f"/MediaBox [0 0 {_pdf_number(page_width_pt)} {_pdf_number(page_height_pt)}] >>").encode("ascii"),
        ]
        for page_index, lines in enumerate(pages):
            with Image.new(color_mode, (page_width, page_height), background) as image:
                draw = ImageDraw.Draw(image)
                for line_index, line in enumerate(lines):
                    draw.text(
                        (margin, margin + line_index * line_height),
                        line, font=font, fill=text_color, anchor="lt",
                    )
                # Lossless pixels only: no hidden text, font, ActualText or OCR layer.
                pixels = zlib.compress(image.tobytes())
                width, height = image.size
            content_number = 4 + page_index * 3
            image_number = content_number + 1
            objects.extend([
                (f"<< /Type /Page /Parent 2 0 R /Contents {content_number} 0 R "
                 f"/Resources << /XObject << /PicturedText {image_number} 0 R >> >> >>").encode("ascii"),
                _deflated_stream(
                    (f"q {_pdf_number(page_width_pt)} 0 0 {_pdf_number(page_height_pt)} 0 0 cm "
                     "/PicturedText Do Q").encode("ascii")
                ),
                (f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
                 f"/ColorSpace {color_space} /BitsPerComponent 8 /Filter /FlateDecode "
                 f"/Length {len(pixels)} >>\nstream\n").encode("ascii") + pixels + b"\nendstream",
            ])
        return _assemble_pdf(objects), len(pages)

    pages = _paginate_text(str(text), character_width_ratio=character_width_ratio)
    character_ids: dict[str, int] = {}
    for page in pages:
        for line in page:
            for character in line["glyphs"]:
                if character in character_ids:
                    continue
                if len(character_ids) >= 65535:
                    raise ValueError("PDF에 서로 다른 문자를 65,535개보다 많이 넣을 수 없습니다")
                character_ids[character] = len(character_ids) + 1

    page_object_number = lambda page_index: 7 + (page_index * 2)
    page_references = " ".join(
        f"{page_object_number(index)} 0 R" for index in range(len(pages))
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            f"<< /Type /Pages /Kids [{page_references}] /Count {len(pages)} "
            f"/MediaBox [0 0 {_pdf_number(_PAGE_WIDTH)} {_pdf_number(_PAGE_HEIGHT)}] "
            "/Resources << /Font << /F0 3 0 R >> >> >>"
        ).encode("ascii"),
        (
            b"<< /Type /Font /Subtype /Type0 /BaseFont /LighbdUnicode "
            b"/Encoding /Identity-H /DescendantFonts [4 0 R] /ToUnicode 6 0 R >>"
        ),
        (
            b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /LighbdUnicode "
            b"/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
            b"/FontDescriptor 5 0 R /DW 500 /CIDToGIDMap /Identity >>"
        ),
        (
            b"<< /Type /FontDescriptor /FontName /LighbdUnicode /Flags 4 "
            b"/FontBBox [0 -200 1000 800] /ItalicAngle 0 /Ascent 800 "
            b"/Descent -200 /CapHeight 700 /StemV 80 /MissingWidth 500 >>"
        ),
        _build_to_unicode_cmap(character_ids),
    ]
    for page_index, page_lines in enumerate(pages):
        content_object_number = page_object_number(page_index) + 1
        objects.extend(
            [
                (
                    f"<< /Type /Page /Parent 2 0 R "
                    f"/Contents {content_object_number} 0 R >>"
                ).encode("ascii"),
                _build_page_stream(page_lines, character_ids),
            ]
        )
    return _assemble_pdf(objects), len(pages)


def _plain_content(content, media_parts: list[dict]) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)

    text_parts = []
    for part in content:
        if isinstance(part, str):
            text_parts.append(part)
            continue
        if not isinstance(part, dict):
            text_parts.append(str(part))
            continue
        part_type = part.get("type")
        if part_type == "text":
            text_parts.append(str(part.get("text", "")))
        elif part_type == "image_url":
            media_parts.append(copy.deepcopy(part))
            text_parts.append(
                f"[Image {len(media_parts)} is attached separately after the transcript PDF.]"
            )
        elif isinstance(part.get("text"), str):
            text_parts.append(part["text"])
        else:
            text_parts.append(
                "[NON-TEXT CONTENT] " + json.dumps(part, ensure_ascii=False)
            )
    return "\n".join(text_parts)


def build_conversation_transcript(messages: list) -> tuple[str, list[dict]]:
    """메시지를 role-labelled 자연어 transcript와 별도 이미지 파트로 나눈다."""
    transcript_sections = [
        "ComfyUI Hooking Server Conversation Transcript",
        "Continue the task using the complete role-labelled transcript below.",
    ]
    media_parts: list[dict] = []
    for index, raw_message in enumerate(messages or [], start=1):
        if not isinstance(raw_message, dict):
            transcript_sections.extend(
                [f"--- UNKNOWN {index} ---", str(raw_message)]
            )
            continue
        role = str(raw_message.get("role", "unknown")).upper()
        name = str(raw_message.get("name", "") or "").strip()
        heading = f"--- {role} {index}" + (f" ({name})" if name else "") + " ---"
        section = [heading, _plain_content(raw_message.get("content", ""), media_parts)]

        tool_calls = raw_message.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                section.append(
                    "[TOOL CALL]\n" + json.dumps(tool_call, ensure_ascii=False)
                )
        if raw_message.get("tool_call_id") is not None:
            section.append(f"[TOOL CALL ID] {raw_message['tool_call_id']}")
        transcript_sections.append("\n".join(section))
    return "\n\n".join(transcript_sections), media_parts


def prepare_pdf_prompt_messages(
    messages: list,
    *,
    encode_transcript: bool,
    character_width_ratio: float = _CHARACTER_WIDTH_RATIO,
    prompt_format: str = "pdf",
    image_options: dict | None = None,
    preserve_system_prompt: bool | None = None,
) -> tuple[list, dict]:
    """대화를 PDF 한 파트와 원본 이미지 파트들로 변환한다."""
    options = normalize_image_options(image_options) if prompt_format == "image" else None
    if preserve_system_prompt is None:
        preserve_system_prompt = bool(options and options["preserve_system_prompt"])
    preserved_system = ""
    transcript_messages = list(messages or [])
    if (
        preserve_system_prompt
        and transcript_messages
        and isinstance(transcript_messages[0], dict)
        and transcript_messages[0].get("role") == "system"
    ):
        preserved_media: list[dict] = []
        preserved_system = _plain_content(
            transcript_messages[0].get("content", ""), preserved_media
        ).strip()
        transcript_messages = transcript_messages[1:]

    transcript, media_parts = build_conversation_transcript(transcript_messages)
    if preserved_system:
        media_parts = preserved_media + media_parts
    pdf_text = (
        base64.b64encode(transcript.encode("utf-8")).decode("ascii")
        if encode_transcript
        else transcript
    )
    layout_options = {}
    if character_width_ratio != _CHARACTER_WIDTH_RATIO:
        layout_options["character_width_ratio"] = character_width_ratio
    if prompt_format == "image":
        layout_options["prompt_format"] = "image"
        layout_options["image_options"] = options
    pdf_bytes, page_count = build_text_pdf(pdf_text, **layout_options)
    pdf_data_url = (
        f"data:{PDF_MIME_TYPE};base64,"
        + base64.b64encode(pdf_bytes).decode("ascii")
    )
    bootstrap = (
        PDF_BASE64_TRANSCRIPT_BOOTSTRAP
        if encode_transcript
        else PDF_TRANSCRIPT_BOOTSTRAP
    )
    if prompt_format == "image":
        bootstrap = IMAGE_BASE64_TRANSCRIPT_BOOTSTRAP if encode_transcript else IMAGE_TRANSCRIPT_BOOTSTRAP
    transformed = [{"role": "system", "content": bootstrap}]
    if preserved_system:
        bootstrap += (
            "\n\nThe original system message is preserved below in plain text, "
            "not inside the attached PDF, and must be read as plain text."
        )
        transformed[0]["content"] = bootstrap
        preserved_message = {
            "role": "system",
            "content": (
                "### PRESERVED ORIGINAL SYSTEM PROMPT (PLAIN TEXT — "
                "DO NOT BASE64-DECODE)\n" + preserved_system
            ),
        }
        if encode_transcript:
            # The protocol is prepended later; mark this message so the encoder
            # leaves the instruction text readable instead of hiding it in a
            # Base64 block the model has to decode from a pictured PDF.
            preserved_message["_base64_exempt"] = True
        transformed.append(preserved_message)
    transformed.append({
        "role": "user",
        "content": [
            {
                "type": "file",
                "file": {
                    "filename": PDF_FILENAME,
                    "file_data": pdf_data_url,
                },
            },
            *media_parts,
        ],
    })
    return transformed, {
        "page_count": page_count,
        "pdf_bytes": len(pdf_bytes),
        "transcript_chars": len(transcript),
        "encoded_chars": len(pdf_text),
        "image_count": len(media_parts),
        "base64_transcript": encode_transcript,
        "character_width_ratio": character_width_ratio,
        "prompt_format": prompt_format,
        "font_size": options["font_size"] if options else PDF_FONT_SIZE,
        "preserved_system_chars": len(preserved_system),
        "image_options": options or {},
    }

