from pathlib import Path

import server


ROOT = Path(__file__).resolve().parents[1]


def test_video_instruction_translation_route_is_registered_in_backend_and_frontend() -> None:
    route = server.DEFAULT_CONFIG["llm_routing"]["video_instruction_translate"]
    assert route["primary"] == "llm1"

    frontend = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert (
        "{ key: 'video_instruction_translate', label: '영상 연출 영어 번역', "
        "kind: '텍스트', modality: 'text', group: 'video_pipeline' }"
    ) in frontend


def test_existing_video_compose_route_does_not_override_new_translation_route() -> None:
    merged = server._merge_llm_routing_config(
        {
            "llm_routing": {
                "video_prompt_i2v_compose": {"primary": "llm2"},
                "video_instruction_translate": {"primary": "llm7"},
            }
        }
    )

    assert merged["video_prompt_i2v_compose"]["primary"] == "llm2"
    assert merged["video_instruction_translate"]["primary"] == "llm7"
