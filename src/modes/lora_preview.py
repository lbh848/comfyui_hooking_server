"""Choose a folder preview without decoding images or reading their prompts."""
import json
import os


def training_preview_filename(directory: str) -> str:
    try:
        with os.scandir(directory) as entries:
            filenames = sorted(entry.name for entry in entries if entry.is_file()
                               and not entry.name.startswith("_")
                               and os.path.splitext(entry.name)[1].lower() in {".png", ".jpg", ".jpeg", ".webp"})
    except FileNotFoundError:
        return ""
    try:
        with open(os.path.join(directory, "_representative.json"), encoding="utf-8") as source:
            representative = json.load(source).get("filename", "")
        if representative in filenames:
            return representative
    except (OSError, ValueError, AttributeError):
        pass
    return filenames[0] if filenames else ""
