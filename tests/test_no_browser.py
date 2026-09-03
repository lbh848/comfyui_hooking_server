"""NO_BROWSER 환경변수 판정.

기본 동작(브라우저 열기)이 바뀌지 않는 것이 이 테스트의 핵심이다.
"""

import pytest

import server


@pytest.mark.parametrize("value", ["", "0", "false", "False", "FALSE", " 0 "])
def test_browser_opens_by_default(monkeypatch, value):
    monkeypatch.setenv("NO_BROWSER", value)
    assert server._browser_autostart_disabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes"])
def test_browser_is_disabled_when_asked(monkeypatch, value):
    monkeypatch.setenv("NO_BROWSER", value)
    assert server._browser_autostart_disabled() is True


def test_browser_opens_when_unset(monkeypatch):
    monkeypatch.delenv("NO_BROWSER", raising=False)
    assert server._browser_autostart_disabled() is False
