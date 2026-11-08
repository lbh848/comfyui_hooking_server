"""Search the complete filename index without opening images or caption files."""
import unicodedata


def matches_image_name(filename: str, search: str) -> bool:
    def normalize(value: str) -> str:
        return unicodedata.normalize("NFKC", value).lower().replace("_", " ")
    name = normalize(filename)
    return all(term in name for term in normalize(search).split())
