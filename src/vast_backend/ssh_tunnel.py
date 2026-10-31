"""Paramiko SSH 연결을 로컬 HTTP 포트 포워더로 노출한다."""
from __future__ import annotations

import select
import errno
import socketserver
import threading
import traceback
from typing import Any


DEFAULT_LOCAL_PORT = 18188
LOCAL_PORT_ATTEMPTS = 100


class _TunnelServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[socketserver.BaseRequestHandler],
        *,
        transport: Any,
        remote_host: str,
        remote_port: int,
    ) -> None:
        self.transport = transport
        self.remote_host = remote_host
        self.remote_port = remote_port
        super().__init__(server_address, handler_class)


class _ForwardHandler(socketserver.BaseRequestHandler):
    server: _TunnelServer

    def handle(self) -> None:
        channel = None
        try:
            channel = self.server.transport.open_channel(
                "direct-tcpip",
                (self.server.remote_host, self.server.remote_port),
                self.client_address,
            )
            if channel is None:
                print(
                    "[VAST][TUNNEL][ERROR] SSH direct-tcpip 채널 생성 결과가 "
                    f"비어 있습니다: client={self.client_address}, "
                    f"remote={self.server.remote_host}:{self.server.remote_port}"
                )
                return
            while True:
                readable, _, _ = select.select([self.request, channel], [], [], 1.0)
                if self.request in readable:
                    data = self.request.recv(65536)
                    if not data:
                        return
                    channel.sendall(data)
                if channel in readable:
                    data = channel.recv(65536)
                    if not data:
                        return
                    self.request.sendall(data)
        except (OSError, EOFError) as exc:
            print(
                "[VAST][TUNNEL] 포워딩 연결 종료: "
                f"client={self.client_address}, error={type(exc).__name__}: {exc}"
            )
        except Exception as exc:
            print(
                "[VAST][TUNNEL][ERROR] 포워딩 실패: "
                f"client={self.client_address}, "
                f"remote={self.server.remote_host}:{self.server.remote_port}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
        finally:
            if channel is not None:
                try:
                    channel.close()
                except Exception as exc:
                    print(
                        "[VAST][TUNNEL][ERROR] SSH 채널 닫기 실패: "
                        f"error={type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()


class ComfySshTunnel:
    """기존 Paramiko SSHClient를 소유하는 로컬 127.0.0.1 포워더."""

    def __init__(
        self,
        ssh_client: Any,
        *,
        remote_host: str = "127.0.0.1",
        remote_port: int = 8188,
        local_port: int = DEFAULT_LOCAL_PORT,
    ) -> None:
        self._ssh_client = ssh_client
        self._remote_host = remote_host
        self._remote_port = int(remote_port)
        self._preferred_local_port = int(local_port)
        self._server: _TunnelServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def local_port(self) -> int:
        if self._server is None:
            raise RuntimeError("Vast SSH 터널이 시작되지 않았습니다.")
        return int(self._server.server_address[1])

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.local_port}"

    def start(self) -> str:
        if self._server is not None:
            return self.url
        transport = self._ssh_client.get_transport()
        if transport is None or not transport.is_active():
            print("[VAST][TUNNEL][ERROR] 활성 SSH transport가 없습니다.")
            raise RuntimeError("활성 Vast SSH 연결이 없어 터널을 만들 수 없습니다.")
        transport.set_keepalive(30)
        try:
            server: _TunnelServer | None = None
            for offset in range(LOCAL_PORT_ATTEMPTS):
                candidate = self._preferred_local_port + offset
                if not 1 <= candidate <= 65535:
                    print(
                        "[VAST][TUNNEL][ERROR] 로컬 포트 후보 범위 초과: "
                        f"preferred={self._preferred_local_port}, candidate={candidate}"
                    )
                    raise ValueError(f"잘못된 Vast 터널 로컬 포트: {candidate}")
                try:
                    server = _TunnelServer(
                        ("127.0.0.1", candidate),
                        _ForwardHandler,
                        transport=transport,
                        remote_host=self._remote_host,
                        remote_port=self._remote_port,
                    )
                    break
                except OSError as exc:
                    address_in_use = (
                        exc.errno == errno.EADDRINUSE
                        or getattr(exc, "winerror", None) == 10048
                    )
                    if not address_in_use:
                        print(
                            "[VAST][TUNNEL][ERROR] 로컬 포트 바인딩 실패: "
                            f"port={candidate}, error={type(exc).__name__}: {exc}"
                        )
                        traceback.print_exc()
                        raise
                    print(
                        "[VAST][TUNNEL] 로컬 포트 사용 중, 다음 후보 시도: "
                        f"port={candidate}, error={exc}"
                    )
                    if offset == LOCAL_PORT_ATTEMPTS - 1:
                        print(
                            "[VAST][TUNNEL][ERROR] 사용 가능한 로컬 포트를 찾지 못함: "
                            f"range={self._preferred_local_port}-"
                            f"{self._preferred_local_port + LOCAL_PORT_ATTEMPTS - 1}"
                        )
                        traceback.print_exc()
                        raise RuntimeError(
                            "Vast ComfyUI SSH 터널에 사용할 로컬 포트가 없습니다."
                        ) from exc
            if server is None:
                print(
                    "[VAST][TUNNEL][ERROR] 로컬 터널 서버 생성 결과가 비어 있습니다: "
                    f"preferred={self._preferred_local_port}"
                )
                raise RuntimeError("Vast ComfyUI SSH 터널 서버를 만들지 못했습니다.")
            thread = threading.Thread(
                target=server.serve_forever,
                name="vast-comfy-ssh-tunnel",
                daemon=True,
            )
            thread.start()
            self._server = server
            self._thread = thread
            print(
                "[VAST][TUNNEL] 시작: "
                f"{self.url} -> {self._remote_host}:{self._remote_port}"
            )
            return self.url
        except Exception as exc:
            print(
                "[VAST][TUNNEL][ERROR] 터널 시작 실패: "
                f"remote={self._remote_host}:{self._remote_port}, "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            raise

    def close(self) -> None:
        server, thread = self._server, self._thread
        self._server = None
        self._thread = None
        if server is not None:
            try:
                server.shutdown()
                server.server_close()
            except Exception as exc:
                print(
                    "[VAST][TUNNEL][ERROR] 로컬 포워더 종료 실패: "
                    f"error={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
        try:
            self._ssh_client.close()
        except Exception as exc:
            print(
                "[VAST][TUNNEL][ERROR] SSH 연결 종료 실패: "
                f"error={type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2)
            if thread.is_alive():
                print("[VAST][TUNNEL][ERROR] 터널 스레드가 제한 시간 안에 종료되지 않았습니다.")
        print("[VAST][TUNNEL] 종료")
