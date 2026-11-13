"""Manga page resolutions offered to the DETAIL-MANGA prompt.

The list is shared by the prompt builder and the settings UI so the bot setting
can limit which resolutions the model may choose from. Values are base sizes;
the bot's manga scale is applied afterwards.
"""

MANGA_PAGE_SIZES = (
    (1152, 1728),
    (1408, 1408),
    (1056, 1888),
    (1536, 1152),
    (1728, 1152),
)

MANGA_PAGE_SIZE_TOKENS = tuple(f"{width}x{height}" for width, height in MANGA_PAGE_SIZES)
DEFAULT_MANGA_PAGE_SIZES = ",".join(MANGA_PAGE_SIZE_TOKENS)
_SIZES_BY_TOKEN = {
    token: (width, height)
    for token, (width, height) in zip(MANGA_PAGE_SIZE_TOKENS, MANGA_PAGE_SIZES)
}


def parse_manga_page_sizes(value) -> list[tuple[int, int]]:
    """Return the selected resolutions in canonical order.

    ``value`` is the saved comma-separated token string, but a list of tokens or
    ``[width, height]`` pairs is accepted as well. Unknown tokens are dropped.
    An empty result means no usable selection, so callers use every default.
    """
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        return []
    wanted = set()
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            try:
                size = (int(item[0]), int(item[1]))
            except (TypeError, ValueError):
                continue
        else:
            token = str(item).strip().lower().replace("×", "x").replace(" ", "")
            size = _SIZES_BY_TOKEN.get(token)
        if size in _SIZES_BY_TOKEN.values():
            wanted.add(size)
    return [size for size in MANGA_PAGE_SIZES if size in wanted]
