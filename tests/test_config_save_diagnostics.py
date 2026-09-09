"""설정 저장 진단 로그의 계약.

설정이 조용히 되돌아가는 사고를 로그만으로 추적하려고 넣은 계측이다. 계측이
요청을 깨뜨리거나 저장 내용을 오염시키면 안 된다.
"""

import server


def test_diff_lists_only_changed_keys():
    old = {"a": 1, "b": "x", "c": [1, 2]}
    new = {"a": 1, "b": "y", "c": [1, 2]}
    assert server._config_diff_keys(old, new) == ["b"]


def test_dict_key_order_is_not_a_change():
    """폼 스냅샷은 키 순서가 매번 다르다. 순서로 diff 가 뜨면 로그가 쓸모없어진다."""
    old = {"a": {"x": 1, "y": 2}}
    new = {"a": {"y": 2, "x": 1}}
    assert server._config_diff_keys(old, new) == []


def test_added_and_removed_keys_are_marked():
    """추가(+)와 제거(-)를 값 변경과 구분해야 원인이 보인다."""
    assert server._config_diff_keys({"a": 1}, {"b": 1}) == ["+b", "-a"]


def test_unserializable_values_do_not_raise():
    """계측이 요청 처리를 깨뜨리면 안 된다."""
    sentinel = object()
    assert server._config_diff_keys({"a": sentinel}, {"a": sentinel}) == []
    assert server._config_diff_keys({"a": sentinel}, {"a": object()}) == ["a"]


def test_origin_hint_is_not_a_config_key():
    """_origin 은 진단용 메타 필드다. 저장 대상이 되면 모든 사용자 설정이 오염된다."""
    assert "_origin" not in server.DEFAULT_CONFIG
