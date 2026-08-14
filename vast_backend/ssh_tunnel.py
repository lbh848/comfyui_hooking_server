"""Paramiko SSH 연결을 로컬 HTTP 포트 포워더로 노출한다."""
from __future__ import annotations

import select
import socketserver
import threading
import traceback
from typing import Any


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
    ) -> None:
        self._ssh_client = ssh_client
        self._remote_host = remote_host
        self._remote_port = int(remote_port)
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
            server = _TunnelServer(
                ("127.0.0.1", 0),
                _ForwardHandler,
                transport=transport,
                remote_host=self._remote_host,
                remote_port=self._remote_port,
            )
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
