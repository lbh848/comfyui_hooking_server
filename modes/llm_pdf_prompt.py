"""Gemini/Vertex용 대화 transcript PDF 변환.

대화의 자연어 흐름과 role 순서를 하나의 transcript로 보존하고, 기존 이미지
파트는 PDF 뒤에 원래 순서대로 첨부한다. 기본 PDF는 1pt 텍스트와
``ActualText``로 긴 문맥을 담는다. 이미지 모드는 9pt 글자를 손실 없는
그림으로 그려 PDF에 담고 네이티브 텍스트 레이어를 포함하지 않는다.
"""

from __future__ import annotations

import base64
import copy
import json
import math
import zlib
from pathlib import Path


PDF_MIME_TYPE = "application/pdf"
PDF_FILENAME = "llm-conversation.pdf"
PDF_MEDIA_RESOLUTION = "MEDIA_RESOLUTION_LOW"
PDF_FONT_SIZE = 1.0
PDF_IMAGE_FONT_SIZE = 9.0

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
    prompt_format: str = "pdf",
) -> tuple[bytes, int]:
    """1pt 네이티브 텍스트 또는 텍스트 레이어 없는 9pt 그림 PDF를 만든다."""
    if prompt_format == "image":
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.truetype(
            str(Path(__file__).resolve().parents[1] / "fonts" / "NotoSansKR-VF.ttf"),
            size=round(PDF_IMAGE_FONT_SIZE * 2),
        )
        pages = _paginate_text(
            str(text).expandtabs(4), font=font, character_width_ratio=character_width_ratio,
        )
        page_references = " ".join(f"{3 + index * 3} 0 R" for index in range(len(pages)))
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            (f"<< /Type /Pages /Kids [{page_references}] /Count {len(pages)} "
             f"/MediaBox [0 0 {_pdf_number(_PAGE_WIDTH)} {_pdf_number(_PAGE_HEIGHT)}] >>").encode("ascii"),
        ]
        for page_index, lines in enumerate(pages):
            with Image.new("L", (math.ceil(_PAGE_WIDTH * 2), math.ceil(_PAGE_HEIGHT * 2)), 255) as image:
                draw = ImageDraw.Draw(image)
                for line_index, line in enumerate(lines):
                    draw.text(
                        (_PAGE_MARGIN * 2, _PAGE_MARGIN * 2 + line_index * sum(font.getmetrics())),
                        line["glyphs"], font=font, fill=0, anchor="lt",
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
                    (f"q {_pdf_number(_PAGE_WIDTH)} 0 0 {_pdf_number(_PAGE_HEIGHT)} 0 0 cm "
                     "/PicturedText Do Q").encode("ascii")
                ),
                (f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
                 f"/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode "
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
) -> tuple[list, dict]:
    """대화를 PDF 한 파트와 원본 이미지 파트들로 변환한다."""
    transcript, media_parts = build_conversation_transcript(messages)
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
    transformed = [
        {"role": "system", "content": bootstrap},
        {
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
        },
    ]
    return transformed, {
        "page_count": page_count,
        "pdf_bytes": len(pdf_bytes),
        "transcript_chars": len(transcript),
        "encoded_chars": len(pdf_text),
        "image_count": len(media_parts),
        "base64_transcript": encode_transcript,
        "character_width_ratio": character_width_ratio,
        "prompt_format": prompt_format,
    }

