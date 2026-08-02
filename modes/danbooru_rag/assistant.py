"""LLM-assisted knowledge search over the embedded Danbooru tag index."""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from typing import Any, Awaitable, Callable

from modes import llm_prompt_edit, llm_service

from .service import DanbooruRagError, get_danbooru_rag_service


PLAN_TASK_KEY = "character_maker_draft"
ANSWER_TASK_KEY = "character_maker_feedback"
MAX_QUESTION_LENGTH = 1000
MAX_SEARCH_QUERIES = 5
RAG_RESULTS_PER_QUERY = 20
MAX_CANDIDATES = 60
RAG_COLD_START_TIMEOUT_SECONDS = 300.0
RAG_SEARCH_TIMEOUT_SECONDS = 60.0

CATEGORY_LABELS = {
    0: "일반 태그",
    1: "작가",
    3: "작품",
    4: "캐릭터",
    5: "메타",
}
ALLOWED_CATEGORIES = frozenset(CATEGORY_LABELS)


def format_display_tag(tag: str, category: int) -> str:
    """Convert an index tag to the prompt-ready format used by this project."""
    spaced = str(tag or "").replace("_", " ")
    escaped = spaced.replace("(", r"\(").replace(")", r"\)")
    if int(category) == 3:
        return f"(\\{escaped})\\"
    return escaped


class DanbooruKnowledgeError(RuntimeError):
    """Base error for the LLM-assisted Danbooru knowledge pipeline."""


class DanbooruKnowledgeQueryError(DanbooruKnowledgeError):
    """Raised when the user request itself is invalid."""


def _parse_json_object(raw: Any, *, stage: str) -> tuple[dict[str, Any] | None, str]:
    try:
        parsed = llm_prompt_edit.parse_llm_json(raw)
    except Exception as exc:
        reason = f"{stage} 응답 JSON 해석 실패: {type(exc).__name__}: {exc}"
        print(f"[DANBOORU_KNOWLEDGE] {reason}; raw={str(raw)[:1000]!r}")
        traceback.print_exc()
        return None, reason
    if not isinstance(parsed, dict):
        reason = f"{stage} 응답이 JSON 객체가 아닙니다."
        print(
            f"[DANBOORU_KNOWLEDGE] {reason}; "
            f"type={type(parsed).__name__}, raw={str(raw)[:1000]!r}"
        )
        return None, reason
    return parsed, ""


def parse_search_plan(raw: Any) -> tuple[dict[str, Any] | None, str]:
    parsed, reason = _parse_json_object(raw, stage="질문 분석")
    if parsed is None:
        return None, reason

    interpretation = parsed.get("interpretation")
    if not isinstance(interpretation, str) or not interpretation.strip():
        reason = "질문 분석 응답의 interpretation이 비어 있습니다."
        print(f"[DANBOORU_KNOWLEDGE] {reason}; parsed={parsed!r}")
        return None, reason

    raw_queries = parsed.get("search_queries")
    if not isinstance(raw_queries, list):
        reason = "질문 분석 응답의 search_queries가 배열이 아닙니다."
        print(f"[DANBOORU_KNOWLEDGE] {reason}; parsed={parsed!r}")
        return None, reason
    search_queries: list[str] = []
    seen_queries: set[str] = set()
    for value in raw_queries:
        if not isinstance(value, str):
            reason = "질문 분석 응답의 검색 단위에 문자열이 아닌 값이 있습니다."
            print(f"[DANBOORU_KNOWLEDGE] {reason}; value={value!r}")
            return None, reason
        query = value.strip()[:300]
        folded = query.casefold()
        if query and folded not in seen_queries:
            seen_queries.add(folded)
            search_queries.append(query)
    if not search_queries:
        reason = "질문 분석 응답에 사용할 검색 단위가 없습니다."
        print(f"[DANBOORU_KNOWLEDGE] {reason}; parsed={parsed!r}")
        return None, reason
    if len(search_queries) > MAX_SEARCH_QUERIES:
        reason = f"질문 분석 검색 단위는 최대 {MAX_SEARCH_QUERIES}개여야 합니다."
        print(
            f"[DANBOORU_KNOWLEDGE] {reason}; "
            f"count={len(search_queries)}, queries={search_queries!r}"
        )
        return None, reason

    raw_lexical_terms = parsed.get("lexical_terms")
    if not isinstance(raw_lexical_terms, list):
        reason = "질문 분석 응답의 lexical_terms가 배열이 아닙니다."
        print(f"[DANBOORU_KNOWLEDGE] {reason}; parsed={parsed!r}")
        return None, reason
    lexical_terms: list[str] = []
    seen_terms: set[str] = set()
    for value in raw_lexical_terms:
        if not isinstance(value, str):
            reason = "질문 분석 응답의 문자열 검색 단위에 문자열이 아닌 값이 있습니다."
            print(f"[DANBOORU_KNOWLEDGE] {reason}; value={value!r}")
            return None, reason
        term = value.strip()[:200]
        if term and not any(char.isalnum() for char in term):
            reason = "질문 분석 응답의 문자열 검색 단위에 문자나 숫자가 없습니다."
            print(f"[DANBOORU_KNOWLEDGE] {reason}; term={term!r}")
            return None, reason
        folded = term.casefold()
        if term and folded not in seen_terms:
            seen_terms.add(folded)
            lexical_terms.append(term)
    if not lexical_terms:
        reason = "질문 분석 응답에 태그명 문자열 검색 단위가 없습니다."
        print(f"[DANBOORU_KNOWLEDGE] {reason}; parsed={parsed!r}")
        return None, reason
    if len(lexical_terms) > MAX_SEARCH_QUERIES:
        reason = f"문자열 검색 단위는 최대 {MAX_SEARCH_QUERIES}개여야 합니다."
        print(
            f"[DANBOORU_KNOWLEDGE] {reason}; "
            f"count={len(lexical_terms)}, terms={lexical_terms!r}"
        )
        return None, reason

    raw_categories = parsed.get("categories")
    if not isinstance(raw_categories, list):
        reason = "질문 분석 응답의 categories가 배열이 아닙니다."
        print(f"[DANBOORU_KNOWLEDGE] {reason}; parsed={parsed!r}")
        return None, reason
    categories: list[int] = []
    for value in raw_categories:
        if isinstance(value, bool) or not isinstance(value, int):
            reason = "질문 분석 응답의 카테고리는 정수여야 합니다."
            print(f"[DANBOORU_KNOWLEDGE] {reason}; value={value!r}")
            return None, reason
        if value not in ALLOWED_CATEGORIES:
            reason = f"지원하지 않는 Danbooru 카테고리입니다: {value!r}"
            print(f"[DANBOORU_KNOWLEDGE] {reason}; allowed={sorted(ALLOWED_CATEGORIES)}")
            return None, reason
        if value not in categories:
            categories.append(value)
    if not categories:
        reason = "질문 분석 응답에 검색할 Danbooru 카테고리가 없습니다."
        print(f"[DANBOORU_KNOWLEDGE] {reason}; parsed={parsed!r}")
        return None, reason

    return {
        "interpretation": interpretation.strip()[:1000],
        "search_queries": search_queries,
        "lexical_terms": lexical_terms,
        "categories": categories,
    }, ""


def validate_search_plan(raw: Any) -> tuple[bool, str]:
    parsed, reason = parse_search_plan(raw)
    return parsed is not None, reason


def parse_grounded_answer(
    raw: Any,
    *,
    allowed_tags: set[str],
) -> tuple[dict[str, Any] | None, str]:
    parsed, reason = _parse_json_object(raw, stage="근거 선별")
    if parsed is None:
        return None, reason

    answer = parsed.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        reason = "근거 선별 응답의 answer가 비어 있습니다."
        print(f"[DANBOORU_KNOWLEDGE] {reason}; parsed={parsed!r}")
        return None, reason

    status = parsed.get("status")
    if status not in {"answered", "ambiguous", "not_found"}:
        reason = "근거 선별 응답의 status가 answered/ambiguous/not_found 중 하나가 아닙니다."
        print(f"[DANBOORU_KNOWLEDGE] {reason}; status={status!r}")
        return None, reason

    confidence = parsed.get("confidence")
    if confidence not in {"high", "medium", "low"}:
        reason = "근거 선별 응답의 confidence가 high/medium/low 중 하나가 아닙니다."
        print(f"[DANBOORU_KNOWLEDGE] {reason}; confidence={confidence!r}")
        return None, reason

    raw_evidence = parsed.get("evidence_tags")
    if not isinstance(raw_evidence, list):
        reason = "근거 선별 응답의 evidence_tags가 배열이 아닙니다."
        print(f"[DANBOORU_KNOWLEDGE] {reason}; parsed={parsed!r}")
        return None, reason

    canonical_by_folded = {tag.casefold(): tag for tag in allowed_tags}
    evidence_tags: list[str] = []
    for value in raw_evidence:
        if not isinstance(value, str) or not value.strip():
            reason = "근거 선별 응답의 evidence_tags에 유효하지 않은 값이 있습니다."
            print(f"[DANBOORU_KNOWLEDGE] {reason}; value={value!r}")
            return None, reason
        canonical = canonical_by_folded.get(value.strip().casefold())
        if canonical is None:
            reason = f"RAG 후보에 없는 태그를 근거로 선택했습니다: {value!r}"
            print(
                f"[DANBOORU_KNOWLEDGE] {reason}; "
                f"allowed_count={len(allowed_tags)}"
            )
            return None, reason
        if canonical not in evidence_tags:
            evidence_tags.append(canonical)
    if status != "not_found" and not evidence_tags:
        reason = "답을 찾았다고 했지만 RAG 근거 태그가 하나도 없습니다."
        print(f"[DANBOORU_KNOWLEDGE] {reason}; parsed={parsed!r}")
        return None, reason

    return {
        "answer": answer.strip()[:5000],
        "status": status,
        "confidence": confidence,
        "evidence_tags": evidence_tags[:20],
    }, ""


def _plan_messages(question: str) -> list[dict[str, str]]:
    category_guide = ", ".join(
        f"{category}={label}" for category, label in CATEGORY_LABELS.items()
    )
    return [
        {
            "role": "system",
            "content": (
                "You plan evidence retrieval from a Danbooru tag knowledge index. "
                "Understand the user's natural-language question using ordinary reasoning; "
                "never use hard-coded keyword matching. Do not answer from memory. "
                "The index rows contain tag, category, frequency, major, minor, definition, "
                "and aliases. For character-to-work questions, character rows often carry "
                "the work title in minor and relationship details in definition. "
                "Preserve names and proper nouns, and create up to five short Korean or English "
                "semantic retrieval queries that can find evidence even when a name is ambiguous. "
                "Also produce up to five lexical_terms containing likely Danbooru tag-name or "
                "proper-name fragments; use spaces rather than underscores. These terms drive "
                "literal tag-name lookup in parallel with vector retrieval. "
                f"Allowed category IDs are {category_guide}. "
                "Return exactly one JSON object with interpretation, search_queries, lexical_terms, "
                "categories. search_queries and lexical_terms are non-empty string arrays and "
                "categories is a non-empty integer array."
            ),
        },
        {
            "role": "user",
            "content": json.dumps({"question": question}, ensure_ascii=False),
        },
    ]


def _answer_messages(
    question: str,
    plan: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Answer a Danbooru knowledge question in Korean using only the supplied RAG "
                "candidate rows. Never add facts from memory and never invent a tag. Interpret "
                "major, minor, definition, aliases, category, and frequency as evidence. "
                "When multiple entities share the requested name, do not silently choose one: "
                "list the plausible matches and their works, then explain the ambiguity briefly. "
                "In the Korean answer, render tag names with spaces instead of underscores "
                "(for example, blonde hair rather than blonde_hair), and copy each candidate's "
                "display_tag exactly when mentioning a prompt-ready tag. "
                "Use exact candidate tag strings in evidence_tags. If the candidates do not "
                "support an answer, say so and return status=not_found with an empty evidence_tags. "
                "Return exactly one JSON object with answer, status, confidence, evidence_tags. "
                "status must be answered, ambiguous, or not_found; confidence must be high, "
                "medium, or low."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": question,
                    "interpretation": plan["interpretation"],
                    "search_queries": plan["search_queries"],
                    "lexical_terms": plan["lexical_terms"],
                    "candidate_pool": candidates,
                },
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]


class DanbooruKnowledgeAssistant:
    """Two-pass LLM planner and grounded answerer around Danbooru RAG."""

    def __init__(
        self,
        *,
        rag_service: Any | None = None,
        llm_caller: Callable[..., Awaitable[str]] | None = None,
    ) -> None:
        self.rag_service = rag_service or get_danbooru_rag_service()
        self.llm_caller = llm_caller or llm_service.callLLMTask

    async def _call_llm(
        self,
        *,
        task_key: str,
        call_name: str,
        messages: list[dict[str, str]],
        validator: Callable[[Any], tuple[bool, str]],
    ) -> str:
        context = llm_service.create_llm_execution_context(
            task_key,
            call_name=call_name,
            json_mode=True,
        )
        try:
            raw = await self.llm_caller(
                task_key,
                messages,
                json_mode=True,
                result_validator=validator,
                execution_context=context,
            )
        except Exception as exc:
            print(
                f"[DANBOORU_KNOWLEDGE] LLM 호출 예외: stage={call_name!r}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise DanbooruKnowledgeError(f"{call_name} 중 LLM 호출에 실패했습니다: {exc}") from exc
        if not isinstance(raw, str) or not raw.strip():
            print(
                f"[DANBOORU_KNOWLEDGE] LLM 빈 응답: stage={call_name!r}, "
                f"type={type(raw).__name__}"
            )
            raise DanbooruKnowledgeError(f"{call_name} 중 LLM 응답이 비어 있습니다.")
        if raw.strip().startswith("[LLM 실패]"):
            print(
                f"[DANBOORU_KNOWLEDGE] LLM 최종 실패: "
                f"stage={call_name!r}, result={raw[:1000]!r}"
            )
            raise DanbooruKnowledgeError(raw)
        return raw

    async def _ensure_rag_ready(self) -> dict[str, Any]:
        try:
            status = await asyncio.to_thread(self.rag_service.status)
            if not isinstance(status, dict):
                print(
                    "[DANBOORU_KNOWLEDGE] RAG 상태 형식 오류: "
                    f"type={type(status).__name__}, value={status!r}"
                )
                raise DanbooruKnowledgeError("Danbooru RAG 상태 응답이 올바르지 않습니다.")
            if status.get("loaded"):
                return status
            if not status.get("installed"):
                print(
                    "[DANBOORU_KNOWLEDGE] RAG 인덱스 미설치: "
                    f"index_path={status.get('index_path')!r}"
                )
                raise DanbooruKnowledgeError(
                    "Danbooru RAG 인덱스가 설치되지 않았습니다. 설정에서 먼저 설치하세요."
                )
            return await asyncio.wait_for(
                asyncio.to_thread(self.rag_service.warmup),
                timeout=RAG_COLD_START_TIMEOUT_SECONDS,
            )
        except DanbooruKnowledgeError:
            raise
        except asyncio.TimeoutError as exc:
            print(
                "[DANBOORU_KNOWLEDGE] RAG 최초 준비 시간 초과: "
                f"timeout={RAG_COLD_START_TIMEOUT_SECONDS}"
            )
            traceback.print_exc()
            raise DanbooruKnowledgeError(
                "Danbooru RAG 최초 준비 시간이 초과되었습니다."
            ) from exc
        except DanbooruRagError as exc:
            print(
                "[DANBOORU_KNOWLEDGE] RAG 준비 실패: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise DanbooruKnowledgeError(str(exc)) from exc
        except Exception as exc:
            print(
                "[DANBOORU_KNOWLEDGE] RAG 준비 예외: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise DanbooruKnowledgeError(f"Danbooru RAG 준비에 실패했습니다: {exc}") from exc

    async def _search(
        self,
        query: str,
        *,
        categories: set[int],
    ) -> list[dict[str, Any]]:
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    self.rag_service.search,
                    query,
                    top_k=RAG_RESULTS_PER_QUERY,
                    threshold=0.0,
                    categories=categories,
                ),
                timeout=RAG_SEARCH_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            print(
                f"[DANBOORU_KNOWLEDGE] RAG 검색 시간 초과: query={query!r}, "
                f"categories={sorted(categories)}, timeout={RAG_SEARCH_TIMEOUT_SECONDS}"
            )
            traceback.print_exc()
            raise DanbooruKnowledgeError(
                f"Danbooru RAG 검색이 {RAG_SEARCH_TIMEOUT_SECONDS:g}초를 초과했습니다."
            ) from exc
        except DanbooruRagError as exc:
            print(
                f"[DANBOORU_KNOWLEDGE] RAG 검색 실패: query={query!r}, "
                f"categories={sorted(categories)}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise DanbooruKnowledgeError(str(exc)) from exc
        except Exception as exc:
            print(
                f"[DANBOORU_KNOWLEDGE] RAG 검색 예외: query={query!r}, "
                f"categories={sorted(categories)}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise DanbooruKnowledgeError(f"Danbooru RAG 검색에 실패했습니다: {exc}") from exc
        if not isinstance(results, list):
            print(
                f"[DANBOORU_KNOWLEDGE] RAG 결과 형식 오류: query={query!r}, "
                f"type={type(results).__name__}, value={results!r}"
            )
            raise DanbooruKnowledgeError("Danbooru RAG 검색 결과 형식이 올바르지 않습니다.")
        if not results:
            print(
                f"[DANBOORU_KNOWLEDGE] 검색 단위 결과 없음: query={query!r}, "
                f"categories={sorted(categories)}"
            )
        return [item for item in results if isinstance(item, dict) and item.get("tag")]

    async def _lexical_search(
        self,
        term: str,
        *,
        categories: set[int],
    ) -> list[dict[str, Any]]:
        lexical_search = getattr(self.rag_service, "lexical_search", None)
        if not callable(lexical_search):
            print(
                "[DANBOORU_KNOWLEDGE] RAG 문자열 검색 함수를 찾지 못했습니다: "
                f"service={type(self.rag_service).__name__}, term={term!r}"
            )
            raise DanbooruKnowledgeError(
                "현재 Danbooru RAG 서비스가 태그명 문자열 검색을 지원하지 않습니다."
            )
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    lexical_search,
                    term,
                    top_k=RAG_RESULTS_PER_QUERY,
                    categories=categories,
                ),
                timeout=RAG_SEARCH_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            print(
                f"[DANBOORU_KNOWLEDGE] 태그명 문자열 검색 시간 초과: term={term!r}, "
                f"categories={sorted(categories)}, timeout={RAG_SEARCH_TIMEOUT_SECONDS}"
            )
            traceback.print_exc()
            raise DanbooruKnowledgeError(
                f"Danbooru 태그명 검색이 {RAG_SEARCH_TIMEOUT_SECONDS:g}초를 초과했습니다."
            ) from exc
        except DanbooruRagError as exc:
            print(
                f"[DANBOORU_KNOWLEDGE] 태그명 문자열 검색 실패: term={term!r}, "
                f"categories={sorted(categories)}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise DanbooruKnowledgeError(str(exc)) from exc
        except Exception as exc:
            print(
                f"[DANBOORU_KNOWLEDGE] 태그명 문자열 검색 예외: term={term!r}, "
                f"categories={sorted(categories)}, error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise DanbooruKnowledgeError(
                f"Danbooru 태그명 문자열 검색에 실패했습니다: {exc}"
            ) from exc
        if not isinstance(results, list):
            print(
                f"[DANBOORU_KNOWLEDGE] 태그명 문자열 결과 형식 오류: term={term!r}, "
                f"type={type(results).__name__}, value={results!r}"
            )
            raise DanbooruKnowledgeError(
                "Danbooru 태그명 문자열 검색 결과 형식이 올바르지 않습니다."
            )
        if not results:
            print(
                f"[DANBOORU_KNOWLEDGE] 문자열 검색 단위 결과 없음: term={term!r}, "
                f"categories={sorted(categories)}"
            )
        return [item for item in results if isinstance(item, dict) and item.get("tag")]

    async def answer(self, question: str) -> dict[str, Any]:
        if not isinstance(question, str):
            print(
                "[DANBOORU_KNOWLEDGE] 질문 타입 오류: "
                f"type={type(question).__name__}, value={question!r}"
            )
            raise DanbooruKnowledgeQueryError("질문은 문자열이어야 합니다.")
        clean_question = question.strip()
        if not clean_question:
            print("[DANBOORU_KNOWLEDGE] 빈 자연어 질문 거부")
            raise DanbooruKnowledgeQueryError("찾고 싶은 내용을 자연어로 입력하세요.")
        if len(clean_question) > MAX_QUESTION_LENGTH:
            print(
                "[DANBOORU_KNOWLEDGE] 질문 길이 초과: "
                f"length={len(clean_question)}, max={MAX_QUESTION_LENGTH}"
            )
            raise DanbooruKnowledgeQueryError(
                f"질문은 {MAX_QUESTION_LENGTH}자 이하여야 합니다."
            )

        started = time.perf_counter()
        raw_plan = await self._call_llm(
            task_key=PLAN_TASK_KEY,
            call_name="단부르 지식 검색 · 질문 분석",
            messages=_plan_messages(clean_question),
            validator=validate_search_plan,
        )
        plan, reason = parse_search_plan(raw_plan)
        if plan is None:
            print(
                f"[DANBOORU_KNOWLEDGE] 검증 통과 후 질문 분석 파싱 실패: "
                f"question={clean_question!r}, reason={reason}, raw={raw_plan[:1000]!r}"
            )
            raise DanbooruKnowledgeError(reason)

        await self._ensure_rag_ready()
        categories = set(plan["categories"])
        vector_batches = await asyncio.gather(
            *(
                self._search(query, categories=categories)
                for query in plan["search_queries"]
            )
        )
        lexical_batches = await asyncio.gather(
            *(
                self._lexical_search(term, categories=categories)
                for term in plan["lexical_terms"]
            )
        )
        candidates_by_tag: dict[str, dict[str, Any]] = {}

        def merge_candidate(
            item: dict[str, Any],
            *,
            matched_value: str,
            match_type: str,
        ) -> None:
            tag = str(item.get("tag") or "").strip()
            if not tag:
                print(
                    f"[DANBOORU_KNOWLEDGE] 태그 없는 RAG 행 스킵: "
                    f"match_type={match_type}, value={matched_value!r}, row={item!r}"
                )
                return
            folded = tag.casefold()
            existing = candidates_by_tag.get(folded)
            if existing is None:
                aliases = item.get("aliases")
                candidate = {
                    "tag": tag,
                    "display_tag": format_display_tag(
                        tag,
                        int(item.get("category", 0)),
                    ),
                    "category": int(item.get("category", 0)),
                    "category_label": CATEGORY_LABELS.get(
                        int(item.get("category", 0)), "알 수 없음"
                    ),
                    "frequency": int(item.get("frequency", 0)),
                    "major": str(item.get("major") or ""),
                    "minor": str(item.get("minor") or ""),
                    "definition": str(item.get("definition") or "")[:1000],
                    "aliases": (
                        [str(value) for value in aliases][:30]
                        if isinstance(aliases, list)
                        else []
                    ),
                    "score": float(item.get("score", 0.0)),
                    "matched_queries": [matched_value],
                    "match_types": [match_type],
                }
                candidates_by_tag[folded] = candidate
                return
            existing["score"] = max(
                float(existing.get("score", 0.0)),
                float(item.get("score", 0.0)),
            )
            if matched_value not in existing["matched_queries"]:
                existing["matched_queries"].append(matched_value)
            if match_type not in existing["match_types"]:
                existing["match_types"].append(match_type)

        for search_query, results in zip(plan["search_queries"], vector_batches):
            for item in results:
                merge_candidate(
                    item,
                    matched_value=search_query,
                    match_type="vector",
                )
        for lexical_term, results in zip(plan["lexical_terms"], lexical_batches):
            for item in results:
                merge_candidate(
                    item,
                    matched_value=lexical_term,
                    match_type="lexical",
                )

        candidates = sorted(
            candidates_by_tag.values(),
            key=lambda item: (
                -float(item.get("score", 0.0)),
                -int(item.get("frequency", 0)),
                str(item.get("tag") or ""),
            ),
        )[:MAX_CANDIDATES]
        if not candidates:
            print(
                f"[DANBOORU_KNOWLEDGE] 전체 RAG 후보 없음: "
                f"question={clean_question!r}, plan={plan!r}"
            )
            raise DanbooruKnowledgeError(
                "질문과 관련된 Danbooru RAG 후보를 찾지 못했습니다."
            )

        allowed_tags = {item["tag"] for item in candidates}
        answer_validator = lambda raw: (
            lambda parsed_reason: (parsed_reason[0] is not None, parsed_reason[1])
        )(parse_grounded_answer(raw, allowed_tags=allowed_tags))
        raw_answer = await self._call_llm(
            task_key=ANSWER_TASK_KEY,
            call_name="단부르 지식 검색 · 근거 선별",
            messages=_answer_messages(clean_question, plan, candidates),
            validator=answer_validator,
        )
        grounded, reason = parse_grounded_answer(
            raw_answer,
            allowed_tags=allowed_tags,
        )
        if grounded is None:
            print(
                f"[DANBOORU_KNOWLEDGE] 검증 통과 후 근거 응답 파싱 실패: "
                f"question={clean_question!r}, reason={reason}, raw={raw_answer[:1000]!r}"
            )
            raise DanbooruKnowledgeError(reason)

        evidence_set = set(grounded["evidence_tags"])
        evidence = [item for item in candidates if item["tag"] in evidence_set]
        evidence.sort(key=lambda item: grounded["evidence_tags"].index(item["tag"]))
        display_answer = grounded["answer"].replace("_", " ")
        for item in sorted(evidence, key=lambda value: len(value["tag"]), reverse=True):
            base_display = item["tag"].replace("_", " ")
            prompt_display = item["display_tag"]
            if prompt_display not in display_answer:
                display_answer = display_answer.replace(base_display, prompt_display)
        return {
            "success": True,
            "question": clean_question,
            "interpretation": plan["interpretation"],
            "search_queries": plan["search_queries"],
            "lexical_terms": plan["lexical_terms"],
            "categories": [
                {"id": value, "label": CATEGORY_LABELS[value]}
                for value in plan["categories"]
            ],
            "answer": display_answer,
            "status": grounded["status"],
            "confidence": grounded["confidence"],
            "evidence": evidence,
            "candidate_count": len(candidates),
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }
