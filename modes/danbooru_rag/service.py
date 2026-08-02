"""In-process Danbooru tag vector search.

This module is adapted from the MIT-licensed ``joykst96/danbooru-tag-rag``
project.  Only the vector retrieval path used by Character Maker is embedded;
the original FastAPI/uvicorn sidecar and LLM pipeline are intentionally absent.
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
import threading
import traceback
from typing import Any


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "auto_complete" / "danbooru_rag_data"
DEFAULT_INDEX_PATH = DEFAULT_DATA_ROOT / "lancedb_b"
DEFAULT_MODEL_CACHE = PROJECT_ROOT / "models" / "danbooru_rag"
TABLE_NAME = "danbooru_tags_b"
EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-large"
EMBEDDING_MODEL_LOCAL_NAME = "multilingual-e5-large"
EMBEDDING_DIMENSION = 1024
QUERY_PREFIX = "query: "


class DanbooruRagError(RuntimeError):
    """Base exception for embedded Danbooru RAG failures."""


class DanbooruRagIndexNotInstalledError(DanbooruRagError):
    """Raised when a search is attempted without the downloaded index."""


def _distance_to_similarity(distance: float) -> float:
    similarity = 1.0 - float(distance) / 2.0
    return max(0.0, min(1.0, similarity))


class DanbooruRagService:
    """Thread-safe lazy owner of the embedding model and LanceDB table."""

    def __init__(
        self,
        *,
        index_path: str | os.PathLike[str] | None = None,
        model_cache: str | os.PathLike[str] | None = None,
    ) -> None:
        configured_index = os.environ.get("DANBOORU_RAG_INDEX_PATH")
        configured_data_root = os.environ.get("DANBOORU_RAG_DATA_ROOT")
        configured_cache = os.environ.get("DANBOORU_RAG_MODEL_CACHE")
        self.index_path = Path(
            index_path
            or configured_index
            or (
                Path(configured_data_root) / "lancedb_b"
                if configured_data_root
                else DEFAULT_INDEX_PATH
            )
        ).resolve()
        self.model_cache = Path(
            model_cache or configured_cache or DEFAULT_MODEL_CACHE
        ).resolve()
        self.embedding_device = (
            str(os.environ.get("DANBOORU_RAG_DEVICE") or "").strip() or None
        )
        self._model: Any = None
        self._database: Any = None
        self._table: Any = None
        self._lock = threading.RLock()
        self._last_error = ""
        self._missing_reported = False
        # warmup 진행 중 표시. status()는 이벤트 루프 스레드에서 락 없이 읽으므로
        # warmup이 잡고 있는 RLock를 기다리지 않는다(서버 먹통 방지).
        self._loading = False

    def index_available(self) -> bool:
        table_dir = self.index_path / f"{TABLE_NAME}.lance"
        available = self.index_path.is_dir() and table_dir.is_dir()
        if available:
            self._missing_reported = False
        elif not self._missing_reported:
            print(
                "[DANBOORU_RAG] 설치된 인덱스 없음: "
                f"path={str(self.index_path)!r}"
            )
            self._missing_reported = True
        return available

    def _require_index(self) -> None:
        if self.index_available():
            return
        raise DanbooruRagIndexNotInstalledError(
            "Danbooru RAG 인덱스가 설치되지 않았습니다. "
            "설정에서 Hugging Face 인덱스를 먼저 설치하세요."
        )

    def _get_table(self) -> Any:
        if self._table is not None:
            return self._table
        self._require_index()
        try:
            import lancedb

            logger.info("LanceDB 연결: %s", self.index_path)
            database = lancedb.connect(str(self.index_path))
            listed = database.list_tables()
            table_names = list(getattr(listed, "tables", []) or [])
            if TABLE_NAME not in table_names:
                print(
                    "[DANBOORU_RAG] LanceDB 테이블 없음: "
                    f"path={str(self.index_path)!r}, tables={table_names!r}"
                )
                raise DanbooruRagIndexNotInstalledError(
                    f"설치된 인덱스에 {TABLE_NAME!r} 테이블이 없습니다."
                )
            self._database = database
            self._table = database.open_table(TABLE_NAME)
            logger.info(
                "LanceDB 준비 완료: table=%s, rows=%s",
                TABLE_NAME,
                self._table.count_rows(),
            )
            return self._table
        except DanbooruRagError:
            raise
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            print(
                "[DANBOORU_RAG] LanceDB 로드 실패: "
                f"path={str(self.index_path)!r}, error={self._last_error}"
            )
            traceback.print_exc()
            raise DanbooruRagError(
                f"Danbooru RAG 인덱스를 열지 못했습니다: {exc}"
            ) from exc

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            self.model_cache.mkdir(parents=True, exist_ok=True)
            local_model = self.model_cache / EMBEDDING_MODEL_LOCAL_NAME
            model_source = (
                str(local_model) if local_model.is_dir() else EMBEDDING_MODEL_ID
            )
            logger.info(
                "임베딩 모델 로드: source=%s, device=%s",
                model_source,
                self.embedding_device or "auto",
            )
            self._model = SentenceTransformer(
                model_source,
                device=self.embedding_device,
                cache_folder=str(self.model_cache),
            )
            logger.info("임베딩 모델 로드 완료")
            return self._model
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            print(
                "[DANBOORU_RAG] 임베딩 모델 로드 실패: "
                f"model={EMBEDDING_MODEL_ID!r}, cache={str(self.model_cache)!r}, "
                f"error={self._last_error}"
            )
            traceback.print_exc()
            raise DanbooruRagError(
                f"Danbooru RAG 임베딩 모델을 불러오지 못했습니다: {exc}"
            ) from exc

    def warmup(self) -> dict[str, Any]:
        """Load the table and embedding model without starting another server."""
        with self._lock:
            self._loading = True
            try:
                table = self._get_table()
                self._get_model()
                self._last_error = ""
                return {
                    "success": True,
                    "loaded": True,
                    "row_count": int(table.count_rows()),
                    "variant": "b",
                    "mode": "embedded",
                }
            except Exception as exc:
                if not self._last_error:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                print(
                    "[DANBOORU_RAG] 내장 서비스 준비 실패: "
                    f"error={self._last_error}"
                )
                if not isinstance(exc, DanbooruRagError):
                    traceback.print_exc()
                raise
            finally:
                self._loading = False

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        threshold: float = 0.0,
        categories: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        clean_query = str(query or "").strip()
        if not clean_query:
            print("[DANBOORU_RAG] 빈 검색어 거부")
            raise DanbooruRagError("Danbooru RAG 검색어가 비어 있습니다.")
        safe_top_k = max(1, min(20, int(top_k)))
        safe_threshold = max(-1.0, min(1.0, float(threshold)))
        include_categories = (
            {int(value) for value in categories} if categories else None
        )

        with self._lock:
            try:
                table = self._get_table()
                model = self._get_model()
                embedding = model.encode(
                    f"{QUERY_PREFIX}{clean_query}",
                    normalize_embeddings=True,
                )
                vector = embedding.tolist()
                if len(vector) != EMBEDDING_DIMENSION:
                    print(
                        "[DANBOORU_RAG] 임베딩 차원 불일치: "
                        f"query={clean_query!r}, expected={EMBEDDING_DIMENSION}, "
                        f"actual={len(vector)}"
                    )
                    raise DanbooruRagError(
                        "Danbooru RAG 임베딩 차원이 인덱스와 맞지 않습니다."
                    )

                fetch_k = safe_top_k if not include_categories else safe_top_k * 4
                raw_results = table.search(vector).limit(fetch_k).to_list()
                results: list[dict[str, Any]] = []
                for row in raw_results:
                    category = int(row["category"])
                    if (
                        include_categories is not None
                        and category not in include_categories
                    ):
                        continue
                    score = round(
                        _distance_to_similarity(row.get("_distance", 0.0)),
                        4,
                    )
                    if score < safe_threshold:
                        continue
                    aliases_value = row.get("aliases")
                    aliases = (
                        list(aliases_value) if aliases_value is not None else []
                    )
                    results.append(
                        {
                            "tag": str(row["tag"]),
                            "score": score,
                            "category": category,
                            "frequency": int(row.get("frequency", 0)),
                            "major": str(row.get("major") or ""),
                            "minor": str(row.get("minor") or ""),
                            "definition": str(row.get("definition") or ""),
                            "aliases": aliases,
                        }
                    )
                    if len(results) >= safe_top_k:
                        break
                self._last_error = ""
                return results
            except DanbooruRagError:
                raise
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                print(
                    "[DANBOORU_RAG] 검색 실패: "
                    f"query={clean_query!r}, top_k={safe_top_k}, "
                    f"categories={sorted(include_categories) if include_categories else None}, "
                    f"error={self._last_error}"
                )
                traceback.print_exc()
                raise DanbooruRagError(
                    f"Danbooru RAG 검색에 실패했습니다: {exc}"
                ) from exc

    def lexical_search(
        self,
        term: str,
        *,
        top_k: int = 20,
        categories: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Find tag-name fragments supplied by the LLM retrieval planner."""
        clean_term = str(term or "").strip().casefold().replace(" ", "_")
        clean_term = clean_term.replace("%", "").replace("\\", "")[:200]
        if not clean_term or not any(char.isalnum() for char in clean_term):
            print(
                "[DANBOORU_RAG] 유효하지 않은 문자열 검색어 거부: "
                f"term={term!r}, normalized={clean_term!r}"
            )
            raise DanbooruRagError("Danbooru 태그 문자열 검색어가 올바르지 않습니다.")
        safe_top_k = max(1, min(50, int(top_k)))
        include_categories = (
            {int(value) for value in categories} if categories else None
        )
        escaped_term = clean_term.replace("'", "''")
        predicates = [f"tag LIKE '%{escaped_term}%'"]
        if include_categories:
            category_values = ", ".join(
                str(value) for value in sorted(include_categories)
            )
            predicates.append(f"category IN ({category_values})")
        where_clause = " AND ".join(f"({value})" for value in predicates)

        with self._lock:
            try:
                table = self._get_table()
                raw_results = (
                    table.search()
                    .where(where_clause)
                    .limit(min(200, safe_top_k * 5))
                    .to_list()
                )
                results: list[dict[str, Any]] = []
                for row in raw_results:
                    tag = str(row.get("tag") or "")
                    if not tag:
                        print(
                            "[DANBOORU_RAG] 문자열 검색 행에 태그 없음: "
                            f"term={clean_term!r}, row={row!r}"
                        )
                        continue
                    folded_tag = tag.casefold()
                    if folded_tag == clean_term:
                        score = 1.0
                    elif folded_tag.startswith(clean_term) or folded_tag.endswith(clean_term):
                        score = 0.98
                    else:
                        score = 0.95
                    aliases_value = row.get("aliases")
                    results.append(
                        {
                            "tag": tag,
                            "score": score,
                            "category": int(row.get("category", 0)),
                            "frequency": int(row.get("frequency", 0)),
                            "major": str(row.get("major") or ""),
                            "minor": str(row.get("minor") or ""),
                            "definition": str(row.get("definition") or ""),
                            "aliases": (
                                list(aliases_value)
                                if aliases_value is not None
                                else []
                            ),
                        }
                    )
                results.sort(
                    key=lambda item: (
                        -float(item["score"]),
                        -int(item["frequency"]),
                        str(item["tag"]),
                    )
                )
                if not results:
                    print(
                        "[DANBOORU_RAG] 문자열 검색 결과 없음: "
                        f"term={clean_term!r}, categories="
                        f"{sorted(include_categories) if include_categories else None}"
                    )
                self._last_error = ""
                return results[:safe_top_k]
            except DanbooruRagError:
                raise
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                print(
                    "[DANBOORU_RAG] 문자열 검색 실패: "
                    f"term={clean_term!r}, categories="
                    f"{sorted(include_categories) if include_categories else None}, "
                    f"filter={where_clause!r}, error={self._last_error}"
                )
                traceback.print_exc()
                raise DanbooruRagError(
                    f"Danbooru 태그 문자열 검색에 실패했습니다: {exc}"
                ) from exc

    def unload(self) -> dict[str, Any]:
        """Release in-process handles and model memory."""
        with self._lock:
            was_loaded = self._model is not None or self._table is not None
            self._table = None
            self._database = None
            self._model = None
            self._last_error = ""
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                print(
                    "[DANBOORU_RAG] 모델 메모리 추가 정리 스킵: "
                    f"error={type(exc).__name__}: {exc}"
                )
            logger.info("내장 RAG 언로드 완료: was_loaded=%s", was_loaded)
            return {
                "success": True,
                "loaded": False,
                "was_loaded": bool(was_loaded),
                "mode": "embedded",
            }

    def status(self) -> dict[str, Any]:
        # 주의: self._lock를 잡지 않는다. warmup이 모델 로드 동안 RLock를
        # 장시간 쥐고 있으므로, 여기서 락을 잡으려 하면 이벤트 루프(메인)
        # 스레드가 블록되어 서버 전체가 먹통이 된다. 대신 최선의 읽기로
        # 스냅샷을 반환한다. _model/_table은 lazy 1회 세팅 후 unload 전까지
        # 안정적이므로 락 없이 읽어도 안전하다.
        installed = self.index_available()
        loaded = self._model is not None and self._table is not None
        row_count = 0
        if self._table is not None:
            try:
                row_count = int(self._table.count_rows())
            except Exception as exc:
                print(
                    "[DANBOORU_RAG] 상태용 행 수 조회 실패: "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
        return {
            "mode": "embedded",
            "variant": "b",
            "installed": bool(installed),
            "loaded": bool(loaded),
            "ready": bool(installed and loaded),
            "loading": bool(self._loading),
            "row_count": row_count,
            "index_path": str(self.index_path),
            "model_cache": str(self.model_cache),
            "error": self._last_error,
        }


_service = DanbooruRagService()


def get_danbooru_rag_service() -> DanbooruRagService:
    return _service
