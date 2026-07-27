"""Embedded Danbooru tag retrieval.

The retrieval implementation is derived from ``joykst96/danbooru-tag-rag``
and is distributed under the bundled MIT license.
"""

from .installer import (
    HF_ARCHIVE_PATH,
    HF_MANIFEST_PATH,
    HF_REPO_ID,
    HF_REVISION,
    DanbooruRagInstallError,
    DanbooruRagIndexInstaller,
)
from .service import (
    DanbooruRagError,
    DanbooruRagIndexNotInstalledError,
    DanbooruRagService,
    get_danbooru_rag_service,
)

__all__ = [
    "HF_ARCHIVE_PATH",
    "HF_MANIFEST_PATH",
    "HF_REPO_ID",
    "HF_REVISION",
    "DanbooruRagError",
    "DanbooruRagInstallError",
    "DanbooruRagIndexInstaller",
    "DanbooruRagIndexNotInstalledError",
    "DanbooruRagService",
    "get_danbooru_rag_service",
]
