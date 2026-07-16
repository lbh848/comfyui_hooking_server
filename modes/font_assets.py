"""font_assets - 말풍선 모드 폰트 관리(자동 다운로드/업로드/목록/로드).

- fonts/ 폴더(프로젝트 루트, git 배포 제외)에 폰트 파일 보관.
- 기본 폰트: Noto Sans KR Medium(Korean subset 변수폰트, 최초 1회 자동 다운로드).
- 사용자 업로드 폰트(.ttf/.otf/.ttc)도 fonts/ 에 저장해 드롭박스에서 선택.
- 변수폰트는 variation(wght=500)으로 Medium 을 적용해 로드한다.

모든 실패 경로 print + traceback (CLAUDE.md 에러 로깅).
"""

from __future__ import annotations

import os
import shutil
import traceback
from pathlib import Path
from typing import Optional

from PIL import ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FONT_DIR = PROJECT_ROOT / "fonts"

# 기본 제공 폰트. 최초 사용 시 자동 다운로드.
# Noto Sans KR Korean subset 변수폰트(SIL Open Font License 1.1).
# 변수폰트라 wght 축 variation 으로 Medium(500)을 지정해 로드한다.
BUILTIN_FONTS = {
    "noto-sans-kr-medium": {
        "name": "Noto Sans KR Medium",
        "filename": "NotoSansKR-VF.ttf",
        "url": "https://github.com/googlefonts/noto-cjk/raw/main/Sans/Variable/TTF/Subset/NotoSansKR-VF.ttf",
        "license_filename": "LICENSE-OFL.txt",
        "license_url": "https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Sans/LICENSE",
        "variation": [("wght", 500.0)],
    },
}

# 시스템 기본 폰트(font_path/번들 모두 사용 불가일 때 fallback 후보)
SYSTEM_FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/seguiemj.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

FONT_EXTENSIONS = (".ttf", ".otf", ".ttc")

SYSTEM_FONT_ID = "system"


def _ensure_font_dir() -> None:
    try:
        FONT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[FONT_ASSETS] 폰트 폴더 생성 실패({FONT_DIR}): {e}")
        traceback.print_exc()


def ensure_font(font_id: str) -> Optional[str]:
    """번들 폰트가 fonts/ 에 없으면 다운로드한다. 로컬 경로 반환(실패 시 None)."""
    meta = BUILTIN_FONTS.get(font_id)
    if not meta:
        return None
    _ensure_font_dir()
    target = FONT_DIR / meta["filename"]
    if target.is_file() and target.stat().st_size > 0:
        return str(target)
    tmp_part = FONT_DIR / f".{meta['filename']}.part"
    try:
        import urllib.request

        print(f"[FONT_ASSETS] 폰트 다운로드 시작: {meta['name']} <- {meta['url']}")
        urllib.request.urlretrieve(meta["url"], tmp_part)
        if not tmp_part.is_file() or tmp_part.stat().st_size == 0:
            raise RuntimeError("다운로드된 파일이 비어있음")
        os.replace(tmp_part, target)
        # 라이센스 전문 저장(OFL 고지 의무). 실패해도 폰트 사용엔 영향 없음.
        try:
            urllib.request.urlretrieve(
                meta["license_url"], FONT_DIR / meta["license_filename"]
            )
        except Exception as le:
            print(f"[FONT_ASSETS] 라이센스 파일 다운로드 실패(무시 가능): {le}")
        print(
            f"[FONT_ASSETS] 폰트 설치 완료: {target} ({target.stat().st_size} bytes)"
        )
        return str(target)
    except Exception as e:
        print(f"[FONT_ASSETS] 폰트 다운로드 실패({meta['name']}): {e}")
        traceback.print_exc()
        if tmp_part.exists():
            try:
                tmp_part.unlink()
            except Exception:
                pass
        return None


def list_fonts() -> list:
    """드롭박스용 폰트 목록. 항목: {id, name, source, installed}."""
    result = [
        {
            "id": SYSTEM_FONT_ID,
            "name": "기본 (시스템 폰트)",
            "source": "system",
            "installed": True,
        }
    ]
    for fid, meta in BUILTIN_FONTS.items():
        path = FONT_DIR / meta["filename"]
        installed = path.is_file() and path.stat().st_size > 0
        result.append(
            {
                "id": fid,
                "name": meta["name"],
                "source": "builtin",
                "installed": bool(installed),
            }
        )
    _ensure_font_dir()
    builtin_filenames = {m["filename"] for m in BUILTIN_FONTS.values()}
    try:
        for entry in sorted(FONT_DIR.iterdir(), key=lambda p: p.name.lower()):
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in FONT_EXTENSIONS:
                continue
            if entry.name in builtin_filenames:
                continue
            result.append(
                {
                    "id": entry.name,
                    "name": entry.stem,
                    "source": "upload",
                    "installed": True,
                }
            )
    except Exception as e:
        print(f"[FONT_ASSETS] 업로드 폰트 목록 조회 실패: {e}")
        traceback.print_exc()
    return result


def resolve_font(
    font_id: Optional[str], legacy_path: Optional[str] = None
) -> tuple:
    """font_id → (경로, variation). 시스템 폰트는 (None, None).

    번들 미설치 시 자동 다운로드 시도. 알 수 없는 id 면 legacy_path/시스템 폴백.
    variation 은 [("wght", 500.0)] 형태(변수폰트) 또는 None.
    """
    font_id = (font_id or "").strip()
    if not font_id or font_id == SYSTEM_FONT_ID:
        if legacy_path and os.path.isfile(legacy_path):
            return legacy_path, None
        return None, None
    if font_id in BUILTIN_FONTS:
        path = ensure_font(font_id)
        if path:
            return path, BUILTIN_FONTS[font_id].get("variation")
        print(f"[FONT_ASSETS] 번들 폰트 사용 불가 → 시스템 폰트 fallback: {font_id}")
        return None, None
    # 업로드 폰트(파일명 == id)
    candidate = FONT_DIR / font_id
    if candidate.is_file():
        return str(candidate), None
    # legacy: font_id 자체가 경로일 수도 있음
    if os.path.isfile(font_id):
        return font_id, None
    print(
        f"[FONT_ASSETS] 알 수 없는 폰트 id → legacy_path/시스템 폰트 fallback: {font_id!r}"
    )
    if legacy_path and os.path.isfile(legacy_path):
        return legacy_path, None
    return None, None


def load_font(
    size: int,
    font_id: Optional[str] = None,
    legacy_path: Optional[str] = None,
):
    """font_id(또는 legacy_path)로 폰트 로드. 변수폰트는 variation 적용.

    시스템 폰트는 SYSTEM_FONT_CANDIDATES 순회. 최후 PIL 기본 폰트.
    """
    fs = max(10, int(size or 28))
    path, variation = resolve_font(font_id, legacy_path)
    if path:
        try:
            if variation:
                try:
                    return ImageFont.truetype(path, fs, variation=variation)
                except (TypeError, Exception):
                    # variation 미지원 Pillow / 비변수폰트 → 일반 로드
                    return ImageFont.truetype(path, fs)
            return ImageFont.truetype(path, fs)
        except Exception as e:
            print(f"[FONT_ASSETS] 폰트 로드 실패({path}): {e} → 시스템 폰트 fallback")
    for cand in SYSTEM_FONT_CANDIDATES:
        if os.path.isfile(cand):
            try:
                return ImageFont.truetype(cand, fs)
            except Exception:
                continue
    print("[FONT_ASSETS] ⚠ 사용 가능한 TTF 폰트 없음 → PIL 비트맵 폰트(font_size 무시됨)")
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def save_uploaded_font(filename: str, file_bytes: bytes) -> str:
    """업로드 폰트를 fonts/ 에 저장. 안전한 파일명 정규화. 저장된 경로 반환."""
    _ensure_font_dir()
    safe = os.path.basename(filename or "")
    if not safe:
        raise ValueError("잘못된 파일명")
    ext = os.path.splitext(safe)[1].lower()
    if ext not in FONT_EXTENSIONS:
        raise ValueError(f"지원하지 않는 폰트 형식: {ext}")
    target = FONT_DIR / safe
    # 덮어쓰기 전 백업(기존 파일 있으면 .bak 보존)
    if target.exists():
        backup = FONT_DIR / f"{safe}.bak"
        try:
            shutil.copy2(target, backup)
        except Exception as e:
            print(f"[FONT_ASSETS] 기존 폰트 백업 실패(무시): {e}")
    target.write_bytes(file_bytes)
    print(f"[FONT_ASSETS] 업로드 폰트 저장: {target} ({len(file_bytes)} bytes)")
    return str(target)


def delete_font(font_id: str) -> bool:
    """업로드 폰트 삭제. 번들/시스템 폰트는 삭제 불가."""
    font_id = (font_id or "").strip()
    if not font_id or font_id == SYSTEM_FONT_ID or font_id in BUILTIN_FONTS:
        return False
    target = FONT_DIR / os.path.basename(font_id)
    if target.is_file() and target.suffix.lower() in FONT_EXTENSIONS:
        target.unlink()
        print(f"[FONT_ASSETS] 폰트 삭제: {target}")
        return True
    return False