from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modes.danbooru_rag.assistant import (
    ANSWER_TASK_KEY,
    PLAN_TASK_KEY,
    DanbooruKnowledgeAssistant,
    DanbooruKnowledgeError,
    DanbooruKnowledgeQueryError,
    format_display_tag,
    parse_grounded_answer,
    parse_search_plan,
)
from modes.danbooru_rag.service import DanbooruRagService


@pytest.fixture(autouse=True)
def lighbd_records(monkeypatch):
    from modes import lighbd_service

    records: list[dict] = []
    monkeypatch.setattr(lighbd_service, "_log_lighbd_history", records.append)
    return records


class FakeRagService:
    def __init__(self, rows_by_query: dict[str, list[dict]]) -> None:
        self.rows_by_query = rows_by_query
        self.search_calls: list[dict] = []
        self.lexical_calls: list[dict] = []

    def status(self) -> dict:
        return {
            "installed": True,
            "loaded": True,
            "index_path": "test/index",
        }

    def search(self, query, *, top_k, threshold, categories):
        self.search_calls.append(
            {
                "query": query,
                "top_k": top_k,
                "threshold": threshold,
                "categories": set(categories),
            }
        )
        return list(self.rows_by_query.get(query, []))

    def lexical_search(self, term, *, top_k, categories):
        self.lexical_calls.append(
            {
                "term": term,
                "top_k": top_k,
                "categories": set(categories),
            }
        )
        return list(self.rows_by_query.get(term, []))


def _row(
    tag: str,
    *,
    category: int,
    minor: str,
    definition: str,
    score: float,
) -> dict:
    return {
        "tag": tag,
        "category": category,
        "frequency": 100,
        "major": "캐릭터" if category == 4 else "머리카락",
        "minor": minor,
        "definition": definition,
        "aliases": [],
        "score": score,
    }


@pytest.mark.asyncio
async def test_two_pass_search_answers_general_tag_question_from_rag_only(
    lighbd_records,
) -> None:
    rag = FakeRagService(
        {
            "금발 머리 Danbooru 일반 태그": [
                _row(
                    "blonde_hair",
                    category=0,
                    minor="머리 색",
                    definition="금색 또는 금발 머리카락.",
                    score=0.91,
                )
            ]
        }
    )
    calls: list[tuple[str, dict]] = []

    async def fake_llm(task_key, messages, **kwargs):
        payload = json.loads(messages[-1]["content"])
        calls.append((task_key, payload))
        kwargs["metadata_sink"].update(
            {"completion_tokens": 12, "prompt_tokens": 34, "tps": 5.5}
        )
        if len(calls) == 1:
            return json.dumps(
                {
                    "interpretation": "금발 머리를 표현하는 일반 태그 확인",
                    "search_queries": ["금발 머리 Danbooru 일반 태그"],
                    "lexical_terms": ["blonde hair"],
                    "categories": [0],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "answer": "금발 머리를 뜻하는 태그는 blonde_hair입니다.",
                "status": "answered",
                "confidence": "high",
                "evidence_tags": ["blonde hair"],
            },
            ensure_ascii=False,
        )

    assistant = DanbooruKnowledgeAssistant(rag_service=rag, llm_caller=fake_llm)
    result = await assistant.answer("금발 머리를 뜻하는 태그가 뭐지?")

    assert [task for task, _ in calls] == [
        "danbooru_tag_search",
        "danbooru_tag_search",
    ]
    assert PLAN_TASK_KEY == ANSWER_TASK_KEY == "danbooru_tag_search"
    assert rag.search_calls[0]["categories"] == {0}
    assert rag.lexical_calls[0]["term"] == "blonde hair"
    assert result["answer"].endswith("blonde hair입니다.")
    assert [item["tag"] for item in result["evidence"]] == ["blonde_hair"]
    assert result["evidence"][0]["display_tag"] == "blonde hair"
    assert calls[1][1]["candidate_pool"][0]["tag"] == "blonde_hair"
    assert [record["call_name"] for record in lighbd_records] == [
        "단부르 지식 검색 · 질문 분석",
        "단부르 지식 검색 · 근거 선별",
    ]
    assert all(record["task_key"] == "danbooru_tag_search" for record in lighbd_records)
    assert all(record["status"] == "ok" for record in lighbd_records)
    assert all(record["completion_tokens"] == 12 for record in lighbd_records)


@pytest.mark.asyncio
async def test_ambiguous_character_name_keeps_multiple_grounded_matches() -> None:
    query = "Alisa 캐릭터 풀네임과 작품"
    rag = FakeRagService(
        {
            query: [
                _row(
                    "alisa_(pgr)",
                    category=4,
                    minor="",
                    definition="Punishing: Gray Raven의 캐릭터.",
                    score=0.84,
                )
            ],
            "Alisa": [
                _row(
                    "alisa_ilinichina_amiella",
                    category=4,
                    minor="갓 이터",
                    definition="갓 이터 시리즈의 주요 캐릭터.",
                    score=0.94,
                ),
                _row(
                    "alisa_mikhailovna_kujou",
                    category=4,
                    minor="가끔씩 툭하고 러시아어로 부끄러워하는 옆자리의 아랴 양",
                    definition="작품의 여주인공이며 아랴라는 별명으로 불림.",
                    score=0.92,
                ),
            ]
        }
    )
    call_index = 0

    async def fake_llm(_task_key, _messages, **_kwargs):
        nonlocal call_index
        call_index += 1
        if call_index == 1:
            return json.dumps(
                {
                    "interpretation": "Alisa라는 이름을 쓰는 캐릭터의 정식 이름 확인",
                    "search_queries": [query],
                    "lexical_terms": ["Alisa"],
                    "categories": [4],
                }
            )
        return json.dumps(
            {
                "answer": "Alisa는 동명이인이 있어 두 후보를 함께 확인해야 합니다.",
                "status": "ambiguous",
                "confidence": "high",
                "evidence_tags": [
                    "alisa_ilinichina_amiella",
                    "alisa_mikhailovna_kujou",
                ],
            }
        )

    assistant = DanbooruKnowledgeAssistant(rag_service=rag, llm_caller=fake_llm)
    result = await assistant.answer("alisa라는 캐릭터의 풀네임이 어떻게 되지?")

    assert result["status"] == "ambiguous"
    assert [item["minor"] for item in result["evidence"]] == [
        "갓 이터",
        "가끔씩 툭하고 러시아어로 부끄러워하는 옆자리의 아랴 양",
    ]


def test_grounded_answer_rejects_a_tag_outside_candidate_pool() -> None:
    parsed, reason = parse_grounded_answer(
        json.dumps(
            {
                "answer": "근거 없는 답",
                "status": "answered",
                "confidence": "high",
                "evidence_tags": ["invented_tag"],
            }
        ),
        allowed_tags={"blonde_hair"},
    )

    assert parsed is None
    assert "RAG 후보에 없는 태그" in reason


def test_grounded_answer_converts_display_tag_to_raw_candidate_tag() -> None:
    parsed, reason = parse_grounded_answer(
        json.dumps(
            {
                "answer": "공식 캐릭터 태그는 suou yuki입니다.",
                "status": "answered",
                "confidence": "high",
                "evidence_tags": ["suou yuki"],
            }
        ),
        allowed_tags={"suou_yuki"},
        display_tag_aliases={"suou yuki": "suou_yuki"},
    )

    assert reason == ""
    assert parsed is not None
    assert parsed["evidence_tags"] == ["suou_yuki"]


@pytest.mark.asyncio
async def test_llm_retry_failure_keeps_original_response_in_details(
    lighbd_records,
) -> None:
    async def fake_llm(_task_key, _messages, **kwargs):
        kwargs["metadata_sink"].update(
            {"completion_tokens": 7, "prompt_tokens": 11}
        )
        kwargs["on_attempt_failure"](
            {
                "phase": "primary",
                "slot": "llm1",
                "attempt": 1,
                "total_attempts": 2,
                "reason": "JSON 검증 실패",
                "result": "원본 비정상 응답",
            }
        )
        return "[LLM 실패] danbooru_tag_search primary 재시도 소진: JSON 검증 실패"

    assistant = DanbooruKnowledgeAssistant(
        rag_service=FakeRagService({}),
        llm_caller=fake_llm,
    )

    with pytest.raises(DanbooruKnowledgeError):
        await assistant._call_llm(
            task_key="danbooru_tag_search",
            call_name="단부르 지식 검색 · 질문 분석",
            messages=[{"role": "user", "content": "질문"}],
            validator=lambda _raw: (False, "JSON 검증 실패"),
        )

    assert len(lighbd_records) == 2
    assert lighbd_records[0]["output"] == "원본 비정상 응답"
    assert lighbd_records[0]["error"].startswith("[재시도 primary llm1 1/2]")
    assert lighbd_records[1]["output"] == "원본 비정상 응답"
    assert lighbd_records[1]["error"].startswith("[LLM 실패]")


def test_prompt_display_format_preserves_project_character_and_copyright_syntax() -> None:
    assert format_display_tag("blonde_hair", 0) == "blonde hair"
    assert format_display_tag("shifty_(nikke)", 4) == r"shifty \(nikke\)"
    assert format_display_tag(
        "tokidoki_bosotto_roshia-go_de_dereru_tonari_no_alya-san",
        3,
    ) == "(\\tokidoki bosotto roshia-go de dereru tonari no alya-san)\\"


def test_search_plan_supports_all_danbooru_knowledge_categories() -> None:
    plan, reason = parse_search_plan(
        json.dumps(
            {
                "interpretation": "복합 태그 지식 질문",
                "search_queries": ["query one", "query two"],
                "lexical_terms": ["tag one"],
                "categories": [0, 1, 3, 4, 5],
            }
        )
    )

    assert reason == ""
    assert plan is not None
    assert plan["categories"] == [0, 1, 3, 4, 5]


def test_lexical_search_finds_full_names_and_filters_categories(tmp_path) -> None:
    class FakeQuery:
        def __init__(self, rows):
            self.rows = rows
            self.where_clause = ""
            self.limit_value = 0

        def where(self, value):
            self.where_clause = value
            return self

        def limit(self, value):
            self.limit_value = value
            return self

        def to_list(self):
            return self.rows[: self.limit_value]

    class FakeTable:
        def __init__(self):
            self.query = None

        def search(self):
            self.query = FakeQuery(
                [
                    {
                        "tag": "alisa_mikhailovna_kujou",
                        "category": 4,
                        "frequency": 1072,
                        "major": "캐릭터",
                        "minor": "작품명",
                        "definition": "여주인공",
                        "aliases": ["아랴"],
                    }
                ]
            )
            return self.query

    table = FakeTable()
    service = DanbooruRagService(index_path=tmp_path)
    service._table = table

    result = service.lexical_search("Alisa", top_k=10, categories={4})

    assert result[0]["tag"] == "alisa_mikhailovna_kujou"
    assert result[0]["score"] == 0.98
    assert "tag LIKE '%alisa%'" in table.query.where_clause
    assert "category IN (4)" in table.query.where_clause


@pytest.mark.asyncio
async def test_search_api_returns_assistant_result_and_rejects_bad_question(
    monkeypatch,
) -> None:
    import server

    class FakeAssistant:
        async def answer(self, question):
            if not question:
                raise DanbooruKnowledgeQueryError("질문이 비어 있습니다.")
            return {
                "success": True,
                "question": question,
                "answer": "blonde_hair",
                "evidence": [],
            }

    monkeypatch.setattr(server, "_danbooru_knowledge_assistant", FakeAssistant())
    app = web.Application()
    app.router.add_post(
        "/api/danbooru_rag/assist",
        server.handle_api_danbooru_knowledge_search,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        success = await client.post(
            "/api/danbooru_rag/assist",
            json={"question": "금발 머리 태그가 뭐지?"},
        )
        success_payload = await success.json()
        invalid = await client.post("/api/danbooru_rag/assist", json={})
        invalid_payload = await invalid.json()
    finally:
        await client.close()

    assert success.status == 200
    assert success_payload["answer"] == "blonde_hair"
    assert invalid.status == 400
    assert invalid_payload["success"] is False


def test_memo_ui_exposes_llm_assisted_danbooru_search_tab() -> None:
    frontend = (
        Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")

    assert 'id="memo-tab-search"' in frontend
    assert 'id="memo-panel-search"' in frontend
    assert 'id="memo-knowledge-query"' in frontend
    assert "searchDanbooruKnowledge(event)" in frontend
    assert "fetch('/api/danbooru_rag/assist'" in frontend
    assert "LLM 질문 분석 → Danbooru RAG → LLM 근거 선별" in frontend
    assert "copyMemoKnowledgeTag" in frontend
    assert ".replaceAll('_', ' ')" in frontend
    assert "Number(item.category) === 3" in frontend
    assert "key: 'danbooru_tag_search'" in frontend
    assert "label: '단부르 태그 검색'" in frontend


def test_server_registers_one_dedicated_danbooru_llm_route() -> None:
    import server

    route = server.DEFAULT_CONFIG["llm_routing"]["danbooru_tag_search"]

    assert route["primary"] == "llm1"
    assert route["json_mode"] is True
    assert route["max_retries"] == 1
    assert "danbooru_tag_search_plan" not in server.DEFAULT_CONFIG["llm_routing"]
    assert "danbooru_tag_search_answer" not in server.DEFAULT_CONFIG["llm_routing"]
