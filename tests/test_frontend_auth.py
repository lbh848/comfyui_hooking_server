import json
import sys
from pathlib import Path

import pytest
from aiohttp import CookieJar, web
from aiohttp.test_utils import TestClient, TestServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frontend_auth import (
    SESSION_COOKIE_NAME,
    FrontendAuthController,
    FrontendAuthManager,
    LoginAttemptLimiter,
)


SERVER_SOURCE = ROOT / "server.py"


def _write_frontend_files(tmp_path: Path) -> tuple[Path, Path]:
    index_file = tmp_path / "index.html"
    login_file = tmp_path / "login.html"
    index_file.write_text("<html>protected dashboard</html>", encoding="utf-8")
    login_file.write_text("<html>frontend-auth-card</html>", encoding="utf-8")
    return index_file, login_file


async def _start_auth_app(tmp_path: Path):
    auth_file = tmp_path / "key" / "frontend_auth.json"
    index_file, login_file = _write_frontend_files(tmp_path)
    manager = FrontendAuthManager(auth_file)
    controller = FrontendAuthController(
        manager,
        index_file=index_file,
        login_file=login_file,
    )
    app = web.Application()
    controller.register_routes(app)

    async def plugin_prompt(_request):
        return web.json_response({"plugin": "available"})

    async def unprotected_api(_request):
        return web.json_response({"api": "available"})

    app.router.add_post("/prompt", plugin_prompt)
    app.router.add_get("/api/example", unprotected_api)

    client = TestClient(TestServer(app), cookie_jar=CookieJar(unsafe=True))
    await client.start_server()
    return auth_file, manager, client


def test_password_file_contains_hash_and_not_plaintext(tmp_path):
    auth_file = tmp_path / "key" / "frontend_auth.json"
    manager = FrontendAuthManager(auth_file)

    token = manager.setup_password("correct horse battery staple")

    record_text = auth_file.read_text(encoding="utf-8")
    record = json.loads(record_text)
    assert "correct horse battery staple" not in record_text
    assert record["algorithm"] == "pbkdf2_sha256"
    assert record["iterations"] >= 100_000
    assert record["salt"]
    assert record["password_hash"]
    assert record["session_secret"]
    assert manager.verify_password("correct horse battery staple") is True
    assert manager.verify_password("wrong password") is False
    assert manager.verify_session(token) is True


def test_deleting_auth_file_returns_to_setup_and_invalidates_session(tmp_path):
    auth_file = tmp_path / "key" / "frontend_auth.json"
    manager = FrontendAuthManager(auth_file)
    old_token = manager.setup_password("first password")

    auth_file.unlink()

    assert manager.state == "setup"
    assert manager.verify_session(old_token) is False

    new_token = manager.setup_password("second password")
    assert manager.state == "ready"
    assert manager.verify_session(old_token) is False
    assert manager.verify_session(new_token) is True
    assert manager.verify_password("first password") is False
    assert manager.verify_password("second password") is True


def test_corrupt_auth_file_is_error_instead_of_open_setup(tmp_path):
    auth_file = tmp_path / "key" / "frontend_auth.json"
    auth_file.parent.mkdir(parents=True)
    auth_file.write_text("{not valid json", encoding="utf-8")

    manager = FrontendAuthManager(auth_file)

    assert manager.state == "error"
    with pytest.raises(FileExistsError):
        manager.setup_password("replacement password")


def test_login_limiter_expires_old_failures():
    now = [100.0]
    limiter = LoginAttemptLimiter(
        max_failures=2,
        window_seconds=30,
        clock=lambda: now[0],
    )

    assert limiter.is_blocked("client") is False
    assert limiter.record_failure("client") == 1
    assert limiter.record_failure("client") == 2
    assert limiter.is_blocked("client") is True

    now[0] = 131.0
    assert limiter.is_blocked("client") is False


def test_frontend_index_body_is_cached_and_reloaded_after_file_change(tmp_path):
    auth_file = tmp_path / "key" / "frontend_auth.json"
    index_file, login_file = _write_frontend_files(tmp_path)
    controller = FrontendAuthController(
        FrontendAuthManager(auth_file),
        index_file=index_file,
        login_file=login_file,
    )

    first = controller._load_index_body()
    second = controller._load_index_body()

    assert second is first
    assert first == b"<html>protected dashboard</html>"

    index_file.write_text(
        "<html>updated protected dashboard</html>",
        encoding="utf-8",
    )
    updated = controller._load_index_body()

    assert updated == b"<html>updated protected dashboard</html>"
    assert updated is not first


@pytest.mark.asyncio
async def test_only_root_is_guarded_and_plugin_routes_remain_available(tmp_path):
    auth_file, manager, client = await _start_auth_app(tmp_path)
    try:
        root_response = await client.get("/")
        assert root_response.status == 200
        assert "frontend-auth-card" in await root_response.text()

        plugin_response = await client.post("/prompt", json={"prompt": {}})
        assert plugin_response.status == 200
        assert await plugin_response.json() == {"plugin": "available"}

        api_response = await client.get("/api/example")
        assert api_response.status == 200
        assert await api_response.json() == {"api": "available"}

        setup_response = await client.post(
            "/api/frontend-auth/setup",
            json={"password": "dashboard password"},
        )
        assert setup_response.status == 200
        assert (await setup_response.json())["success"] is True
        assert SESSION_COOKIE_NAME in client.session.cookie_jar.filter_cookies(
            client.make_url("/")
        )

        protected_response = await client.get("/")
        assert protected_response.status == 200
        assert "protected dashboard" in await protected_response.text()
        assert protected_response.headers["Accept-Ranges"] == "none"
        assert protected_response.headers["Content-Type"] == "text/html; charset=utf-8"

        auth_file.unlink()

        reset_response = await client.get("/")
        assert reset_response.status == 200
        assert "frontend-auth-card" in await reset_response.text()
        assert manager.state == "setup"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_login_rejects_wrong_password_and_accepts_correct_password(tmp_path):
    auth_file, manager, client = await _start_auth_app(tmp_path)
    manager.setup_password("correct password")
    try:
        wrong_response = await client.post(
            "/api/frontend-auth/login",
            json={"password": "wrong password"},
        )
        assert wrong_response.status == 401

        unauthenticated_root = await client.get("/")
        assert "frontend-auth-card" in await unauthenticated_root.text()

        login_response = await client.post(
            "/api/frontend-auth/login",
            json={"password": "correct password"},
        )
        assert login_response.status == 200

        authenticated_root = await client.get("/")
        assert "protected dashboard" in await authenticated_root.text()
    finally:
        await client.close()


def test_server_registers_root_auth_without_auth_middleware_on_plugin_routes():
    source = SERVER_SOURCE.read_text(encoding="utf-8")

    assert "frontend_auth_controller.register_routes(app)" in source
    assert 'app.router.add_post("/prompt", handle_prompt)' in source
    assert 'app.router.add_get("/ws", handle_ws)' in source
    assert "middlewares=[log_middleware, cors_middleware]" in source
