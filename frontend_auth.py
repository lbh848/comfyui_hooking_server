import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable

from aiohttp import web


AUTH_VERSION = 1
AUTH_ALGORITHM = "pbkdf2_sha256"
PBKDF2_ITERATIONS = 600_000
SESSION_COOKIE_NAME = "soya_frontend_session"
SESSION_LIFETIME_SECONDS = 30 * 24 * 60 * 60
MIN_PASSWORD_LENGTH = 4
MAX_PASSWORD_LENGTH = 256


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


class FrontendAuthManager:
    """Password hashing and signed browser sessions for the dashboard root."""

    def __init__(
        self,
        auth_file: str | Path,
        *,
        clock: Callable[[], float] = time.time,
    ):
        self.auth_file = Path(auth_file)
        self._clock = clock
        self._state = "unknown"
        self._record: dict | None = None
        self._fingerprint: tuple[int, int] | None = None
        self.refresh(force=True)

    @property
    def state(self) -> str:
        self.refresh()
        return self._state

    def refresh(self, *, force: bool = False) -> str:
        try:
            stat = self.auth_file.stat()
        except FileNotFoundError:
            if force or self._state != "setup":
                print(
                    f"[FRONTEND_AUTH] 인증 파일이 없습니다. "
                    f"새 비밀번호 설정이 필요합니다: {self.auth_file}"
                )
            self._state = "setup"
            self._record = None
            self._fingerprint = None
            return self._state
        except Exception as exc:
            print(
                f"[FRONTEND_AUTH] 인증 파일 상태 확인 실패: "
                f"{type(exc).__name__}: {exc} path={self.auth_file}"
            )
            traceback.print_exc()
            self._state = "error"
            self._record = None
            self._fingerprint = None
            return self._state

        fingerprint = (stat.st_mtime_ns, stat.st_size)
        if (
            not force
            and self._state in {"ready", "error"}
            and fingerprint == self._fingerprint
        ):
            return self._state

        try:
            with self.auth_file.open("r", encoding="utf-8") as file:
                record = json.load(file)
            self._validate_record(record)
        except Exception as exc:
            print(
                f"[FRONTEND_AUTH] 인증 파일 로드 실패: "
                f"{type(exc).__name__}: {exc} path={self.auth_file}"
            )
            traceback.print_exc()
            self._state = "error"
            self._record = None
            self._fingerprint = fingerprint
            return self._state

        if force or self._state != "ready" or fingerprint != self._fingerprint:
            print(f"[FRONTEND_AUTH] 인증 파일 로드 완료: {self.auth_file}")
        self._state = "ready"
        self._record = record
        self._fingerprint = fingerprint
        return self._state

    @staticmethod
    def _validate_record(record: object) -> None:
        if not isinstance(record, dict):
            raise ValueError("인증 파일 최상위 값은 객체여야 합니다.")
        if record.get("version") != AUTH_VERSION:
            raise ValueError(f"지원하지 않는 인증 파일 버전: {record.get('version')!r}")
        if record.get("algorithm") != AUTH_ALGORITHM:
            raise ValueError(f"지원하지 않는 해시 알고리즘: {record.get('algorithm')!r}")

        iterations = record.get("iterations")
        if not isinstance(iterations, int) or iterations < 100_000:
            raise ValueError(f"유효하지 않은 PBKDF2 반복 횟수: {iterations!r}")

        salt = _b64decode(str(record.get("salt", "")))
        password_hash = _b64decode(str(record.get("password_hash", "")))
        session_secret = _b64decode(str(record.get("session_secret", "")))
        if len(salt) < 16:
            raise ValueError("인증 파일 salt 길이가 너무 짧습니다.")
        if len(password_hash) != 32:
            raise ValueError("인증 파일 password_hash 길이가 올바르지 않습니다.")
        if len(session_secret) < 32:
            raise ValueError("인증 파일 session_secret 길이가 너무 짧습니다.")

    @staticmethod
    def validate_new_password(password: object) -> str:
        if not isinstance(password, str):
            raise ValueError("비밀번호는 문자열이어야 합니다.")
        if len(password) < MIN_PASSWORD_LENGTH:
            raise ValueError(f"비밀번호는 최소 {MIN_PASSWORD_LENGTH}자여야 합니다.")
        if len(password) > MAX_PASSWORD_LENGTH:
            raise ValueError(f"비밀번호는 최대 {MAX_PASSWORD_LENGTH}자까지 사용할 수 있습니다.")
        return password

    def setup_password(self, password: object) -> str:
        password = self.validate_new_password(password)
        if self.refresh(force=True) != "setup":
            print(
                f"[FRONTEND_AUTH] 새 비밀번호 설정 거부: "
                f"state={self._state} path={self.auth_file}"
            )
            raise FileExistsError("인증 파일이 이미 있거나 사용할 수 없는 상태입니다.")

        salt = secrets.token_bytes(24)
        password_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            PBKDF2_ITERATIONS,
            dklen=32,
        )
        record = {
            "version": AUTH_VERSION,
            "algorithm": AUTH_ALGORITHM,
            "iterations": PBKDF2_ITERATIONS,
            "salt": _b64encode(salt),
            "password_hash": _b64encode(password_hash),
            "session_secret": _b64encode(secrets.token_bytes(32)),
        }

        created = False
        try:
            self.auth_file.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            descriptor = os.open(self.auth_file, flags, 0o600)
            created = True
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
                json.dump(record, file, ensure_ascii=False, indent=2)
                file.write("\n")
            print(f"[FRONTEND_AUTH] 새 인증 파일 생성 완료: {self.auth_file}")
        except Exception as exc:
            print(
                f"[FRONTEND_AUTH] 새 인증 파일 생성 실패: "
                f"{type(exc).__name__}: {exc} path={self.auth_file}"
            )
            traceback.print_exc()
            if created:
                try:
                    self.auth_file.unlink(missing_ok=True)
                    print(f"[FRONTEND_AUTH] 불완전한 인증 파일 제거 완료: {self.auth_file}")
                except Exception as cleanup_exc:
                    print(
                        f"[FRONTEND_AUTH] 불완전한 인증 파일 제거 실패: "
                        f"{type(cleanup_exc).__name__}: {cleanup_exc} "
                        f"path={self.auth_file}"
                    )
                    traceback.print_exc()
            raise

        if self.refresh(force=True) != "ready":
            print(
                f"[FRONTEND_AUTH] 생성한 인증 파일 검증 실패: "
                f"state={self._state} path={self.auth_file}"
            )
            raise RuntimeError("생성한 인증 파일을 다시 읽을 수 없습니다.")
        return self.create_session()

    def verify_password(self, password: object) -> bool:
        if self.refresh() != "ready" or self._record is None:
            print(
                f"[FRONTEND_AUTH] 비밀번호 확인 건너뜀: "
                f"state={self._state} path={self.auth_file}"
            )
            return False
        if not isinstance(password, str):
            print(
                f"[FRONTEND_AUTH] 비밀번호 확인 실패: "
                f"입력 타입={type(password).__name__}"
            )
            return False

        try:
            salt = _b64decode(self._record["salt"])
            expected = _b64decode(self._record["password_hash"])
            actual = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt,
                self._record["iterations"],
                dklen=32,
            )
            return hmac.compare_digest(actual, expected)
        except Exception as exc:
            print(
                f"[FRONTEND_AUTH] 비밀번호 해시 확인 예외: "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return False

    def create_session(self) -> str:
        if self.refresh() != "ready" or self._record is None:
            print(
                f"[FRONTEND_AUTH] 세션 생성 실패: "
                f"state={self._state} path={self.auth_file}"
            )
            raise RuntimeError("인증 설정이 준비되지 않았습니다.")

        payload = {
            "version": AUTH_VERSION,
            "expires_at": int(self._clock()) + SESSION_LIFETIME_SECONDS,
            "nonce": _b64encode(secrets.token_bytes(18)),
        }
        payload_encoded = _b64encode(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        secret = _b64decode(self._record["session_secret"])
        signature = hmac.new(
            secret,
            payload_encoded.encode("ascii"),
            hashlib.sha256,
        ).digest()
        return f"{payload_encoded}.{_b64encode(signature)}"

    def verify_session(self, token: object) -> bool:
        if self.refresh() != "ready" or self._record is None:
            if token:
                print(
                    f"[FRONTEND_AUTH] 기존 세션 거부: "
                    f"state={self._state} path={self.auth_file}"
                )
            return False
        if not isinstance(token, str) or not token:
            return False

        try:
            payload_encoded, signature_encoded = token.split(".", 1)
            secret = _b64decode(self._record["session_secret"])
            expected = hmac.new(
                secret,
                payload_encoded.encode("ascii"),
                hashlib.sha256,
            ).digest()
            actual = _b64decode(signature_encoded)
            if not hmac.compare_digest(actual, expected):
                print("[FRONTEND_AUTH] 세션 확인 실패: 서명이 일치하지 않습니다.")
                return False

            payload = json.loads(_b64decode(payload_encoded).decode("utf-8"))
            if payload.get("version") != AUTH_VERSION:
                print(
                    f"[FRONTEND_AUTH] 세션 확인 실패: "
                    f"version={payload.get('version')!r}"
                )
                return False
            if int(payload.get("expires_at", 0)) <= int(self._clock()):
                print("[FRONTEND_AUTH] 세션 확인 실패: 세션이 만료되었습니다.")
                return False
            if not payload.get("nonce"):
                print("[FRONTEND_AUTH] 세션 확인 실패: nonce가 없습니다.")
                return False
            return True
        except Exception as exc:
            print(
                f"[FRONTEND_AUTH] 세션 확인 예외: "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return False


class LoginAttemptLimiter:
    def __init__(
        self,
        *,
        max_failures: int = 10,
        window_seconds: int = 5 * 60,
        clock: Callable[[], float] = time.time,
    ):
        self.max_failures = max_failures
        self.window_seconds = window_seconds
        self._clock = clock
        self._failures: dict[str, deque[float]] = defaultdict(deque)

    def _recent(self, client_id: str) -> deque[float]:
        attempts = self._failures[client_id]
        cutoff = self._clock() - self.window_seconds
        while attempts and attempts[0] <= cutoff:
            attempts.popleft()
        return attempts

    def is_blocked(self, client_id: str) -> bool:
        return len(self._recent(client_id)) >= self.max_failures

    def record_failure(self, client_id: str) -> int:
        attempts = self._recent(client_id)
        attempts.append(self._clock())
        return len(attempts)

    def clear(self, client_id: str) -> None:
        self._failures.pop(client_id, None)


class FrontendAuthController:
    """aiohttp handlers that guard only GET / and leave plugin routes untouched."""

    def __init__(
        self,
        manager: FrontendAuthManager,
        *,
        index_file: str | Path,
        login_file: str | Path,
    ):
        self.manager = manager
        self.index_file = Path(index_file)
        self.login_file = Path(login_file)
        self.login_limiter = LoginAttemptLimiter()

    def register_routes(self, app: web.Application) -> None:
        app.router.add_get("/", self.handle_frontend)
        app.router.add_get("/api/frontend-auth/status", self.handle_status)
        app.router.add_post("/api/frontend-auth/setup", self.handle_setup)
        app.router.add_post("/api/frontend-auth/login", self.handle_login)
        app.router.add_post("/api/frontend-auth/logout", self.handle_logout)

    @staticmethod
    def _client_id(request: web.Request) -> str:
        connecting_ip = request.headers.get("CF-Connecting-IP", "").strip()
        if connecting_ip:
            return connecting_ip[:128]
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",", 1)[0].strip()[:128]
        return (request.remote or "unknown")[:128]

    @staticmethod
    def _uses_https(request: web.Request) -> bool:
        if request.secure:
            return True
        forwarded_proto = request.headers.get("X-Forwarded-Proto", "")
        return forwarded_proto.split(",", 1)[0].strip().lower() == "https"

    def _set_session_cookie(self, request: web.Request, response: web.StreamResponse, token: str) -> None:
        response.set_cookie(
            SESSION_COOKIE_NAME,
            token,
            max_age=SESSION_LIFETIME_SECONDS,
            httponly=True,
            secure=self._uses_https(request),
            samesite="Strict",
            path="/",
        )

    @staticmethod
    def _no_store(response: web.StreamResponse) -> web.StreamResponse:
        response.headers["Cache-Control"] = "no-store"
        return response

    async def _read_password(self, request: web.Request) -> object:
        try:
            body = await request.json()
        except Exception as exc:
            print(
                f"[FRONTEND_AUTH] 요청 JSON 파싱 실패: "
                f"{type(exc).__name__}: {exc} "
                f"client={self._client_id(request)} path={request.path}"
            )
            traceback.print_exc()
            raise web.HTTPBadRequest(
                text=json.dumps(
                    {"success": False, "error": "요청 JSON 형식이 올바르지 않습니다."},
                    ensure_ascii=False,
                ),
                content_type="application/json",
            )
        if not isinstance(body, dict):
            print(
                f"[FRONTEND_AUTH] 요청 본문 거부: "
                f"입력 타입={type(body).__name__} "
                f"client={self._client_id(request)} path={request.path}"
            )
            raise web.HTTPBadRequest(
                text=json.dumps(
                    {"success": False, "error": "요청 본문은 객체여야 합니다."},
                    ensure_ascii=False,
                ),
                content_type="application/json",
            )
        return body.get("password")

    async def handle_frontend(self, request: web.Request) -> web.StreamResponse:
        token = request.cookies.get(SESSION_COOKIE_NAME)
        if self.manager.verify_session(token):
            if not self.index_file.is_file():
                print(f"[FRONTEND_AUTH] 프런트엔드 파일 없음: {self.index_file}")
                return web.Response(
                    text="Frontend not found. frontend/index.html 필요",
                    status=404,
                )
            return web.FileResponse(
                self.index_file,
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )

        print(
            f"[FRONTEND_AUTH] 프런트엔드 로그인 필요: "
            f"client={self._client_id(request)} state={self.manager.state}"
        )
        if not self.login_file.is_file():
            print(f"[FRONTEND_AUTH] 로그인 화면 파일 없음: {self.login_file}")
            return web.Response(
                text="Frontend login page not found. frontend/login.html 필요",
                status=404,
            )
        return web.FileResponse(
            self.login_file,
            headers={"Cache-Control": "no-store"},
        )

    async def handle_status(self, request: web.Request) -> web.Response:
        state = self.manager.state
        authenticated = self.manager.verify_session(
            request.cookies.get(SESSION_COOKIE_NAME)
        )
        return self._no_store(
            web.json_response(
                {
                    "success": state != "error",
                    "state": state,
                    "authenticated": authenticated,
                    "auth_file": "key/frontend_auth.json",
                },
                status=503 if state == "error" else 200,
            )
        )

    async def handle_setup(self, request: web.Request) -> web.Response:
        client_id = self._client_id(request)
        if self.manager.state != "setup":
            print(
                f"[FRONTEND_AUTH] 초기 비밀번호 설정 요청 거부: "
                f"client={client_id} state={self.manager.state}"
            )
            return self._no_store(
                web.json_response(
                    {
                        "success": False,
                        "error": "이미 비밀번호가 설정되어 있거나 인증 파일에 오류가 있습니다.",
                    },
                    status=409,
                )
            )

        password = await self._read_password(request)
        try:
            token = self.manager.setup_password(password)
        except ValueError as exc:
            print(
                f"[FRONTEND_AUTH] 초기 비밀번호 유효성 검사 실패: "
                f"client={client_id} reason={exc}"
            )
            return self._no_store(
                web.json_response(
                    {"success": False, "error": str(exc)},
                    status=400,
                )
            )
        except FileExistsError as exc:
            print(
                f"[FRONTEND_AUTH] 초기 비밀번호 설정 경합 실패: "
                f"client={client_id} reason={exc}"
            )
            return self._no_store(
                web.json_response(
                    {
                        "success": False,
                        "error": "다른 요청에서 비밀번호 설정이 먼저 완료되었습니다.",
                    },
                    status=409,
                )
            )
        except Exception as exc:
            print(
                f"[FRONTEND_AUTH] 초기 비밀번호 설정 예외: "
                f"{type(exc).__name__}: {exc} client={client_id}"
            )
            traceback.print_exc()
            return self._no_store(
                web.json_response(
                    {"success": False, "error": "비밀번호 설정 중 서버 오류가 발생했습니다."},
                    status=500,
                )
            )

        print(f"[FRONTEND_AUTH] 초기 비밀번호 설정 성공: client={client_id}")
        response = web.json_response({"success": True, "state": "ready"})
        self._set_session_cookie(request, response, token)
        return self._no_store(response)

    async def handle_login(self, request: web.Request) -> web.Response:
        client_id = self._client_id(request)
        state = self.manager.state
        if state == "setup":
            print(f"[FRONTEND_AUTH] 로그인 실패: 설정 필요 client={client_id}")
            return self._no_store(
                web.json_response(
                    {"success": False, "error": "먼저 새 비밀번호를 설정해주세요."},
                    status=409,
                )
            )
        if state == "error":
            print(f"[FRONTEND_AUTH] 로그인 실패: 인증 파일 오류 client={client_id}")
            return self._no_store(
                web.json_response(
                    {
                        "success": False,
                        "error": "인증 파일을 읽을 수 없습니다. 서버 로그를 확인해주세요.",
                    },
                    status=503,
                )
            )
        if self.login_limiter.is_blocked(client_id):
            print(f"[FRONTEND_AUTH] 로그인 시도 제한: client={client_id}")
            return self._no_store(
                web.json_response(
                    {
                        "success": False,
                        "error": "로그인 시도가 너무 많습니다. 5분 뒤 다시 시도해주세요.",
                    },
                    status=429,
                )
            )

        password = await self._read_password(request)
        if not self.manager.verify_password(password):
            failure_count = self.login_limiter.record_failure(client_id)
            print(
                f"[FRONTEND_AUTH] 비밀번호 불일치: "
                f"client={client_id} failures={failure_count}"
            )
            return self._no_store(
                web.json_response(
                    {"success": False, "error": "비밀번호가 올바르지 않습니다."},
                    status=401,
                )
            )

        self.login_limiter.clear(client_id)
        try:
            token = self.manager.create_session()
        except Exception as exc:
            print(
                f"[FRONTEND_AUTH] 로그인 세션 생성 예외: "
                f"{type(exc).__name__}: {exc} client={client_id}"
            )
            traceback.print_exc()
            return self._no_store(
                web.json_response(
                    {"success": False, "error": "로그인 세션 생성에 실패했습니다."},
                    status=500,
                )
            )

        print(f"[FRONTEND_AUTH] 로그인 성공: client={client_id}")
        response = web.json_response({"success": True, "state": "ready"})
        self._set_session_cookie(request, response, token)
        return self._no_store(response)

    async def handle_logout(self, request: web.Request) -> web.Response:
        print(f"[FRONTEND_AUTH] 로그아웃: client={self._client_id(request)}")
        response = web.json_response({"success": True})
        response.del_cookie(SESSION_COOKIE_NAME, path="/")
        return self._no_store(response)
