from modes import autocomplete_service


def test_english_search_prioritizes_exact_tag_before_prefix_matches(monkeypatch):
    monkeypatch.setattr(autocomplete_service, "_loaded", True)
    monkeypatch.setattr(
        autocomplete_service,
        "_tags",
        [
            {
                "name": "blue hair",
                "description": "파란 머리카락",
                "plain_kws": [],
            },
            {
                "name": "blue",
                "description": "파란색",
                "plain_kws": [],
            },
        ],
    )

    assert autocomplete_service.search_tags("blue", limit=1) == [
        {"name": "blue", "description": "파란색"}
    ]
