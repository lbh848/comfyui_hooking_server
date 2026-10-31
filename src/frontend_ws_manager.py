import asyncio
import time
import traceback
from typing import Any


FRONTEND_WS_REPLACED_CLOSE_CODE = 4001
FRONTEND_WS_REPLACED_REASON = b"replaced by newer frontend connection"


class FrontendWsConnectionManager:
    """Keep exactly one acknowledged frontend WebSocket active."""

    def __init__(self, heartbeat_interval: float, stale_timeout: float):
        self.heartbeat_interval = heartbeat_interval
        self.stale_timeout = stale_timeout
        self.connections: dict[str, dict[str, Any]] = {}
        self._register_lock = asyncio.Lock()

    async def register(self, client_id: str, ws: Any) -> bool:
        """Acknowledge *ws*, make it active, and explicitly retire its predecessor."""
        async with self._register_lock:
            try:
                await ws.send_json({
                    "type": "connection_ready",
                    "data": {
                        "client_id": client_id,
                        "heartbeat_interval_seconds": self.heartbeat_interval,
                        "stale_timeout_seconds": self.stale_timeout,
                    },
                })
            except Exception as exc:
                print(
                    f"[FE-WS] connection_ready 송신 실패 client={client_id[:8]} "
                    f"err={type(exc).__name__}: {exc}"
                )
                traceback.print_exc()
                try:
                    await ws.close()
                except Exception as close_exc:
                    print(
                        f"[FE-WS] 승인 실패 연결 종료 실패 client={client_id[:8]} "
                        f"err={type(close_exc).__name__}: {close_exc}"
                    )
                    traceback.print_exc()
                return False

            replaced = list(self.connections.items())
            self.connections.clear()
            self.connections[client_id] = {"ws": ws, "last_pong": time.time()}

            if replaced:
                replaced_ids = [old_id[:8] for old_id, _entry in replaced]
                print(
                    f"[FE-WS] 기존 연결 {len(replaced)}개를 새 연결로 교체: "
                    f"old={replaced_ids} new={client_id[:8]}"
                )

            for old_id, entry in replaced:
                old_ws = entry.get("ws")
                if old_ws is None:
                    print(f"[FE-WS] 기존 연결에 ws 누락 client={old_id[:8]} entry={entry}")
                    continue
                if old_ws.closed:
                    print(f"[FE-WS] 기존 연결이 이미 닫힘 client={old_id[:8]}")
                    continue
                try:
                    await old_ws.send_json({
                        "type": "connection_replaced",
                        "data": {
                            "reason": "newer_connection",
                            "active_client_id": client_id,
                        },
                    })
                except Exception as exc:
                    print(
                        f"[FE-WS] 교체 알림 송신 실패 client={old_id[:8]} "
                        f"err={type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()
                try:
                    await old_ws.close(
                        code=FRONTEND_WS_REPLACED_CLOSE_CODE,
                        message=FRONTEND_WS_REPLACED_REASON,
                    )
                except Exception as exc:
                    print(
                        f"[FE-WS] 기존 연결 종료 실패 client={old_id[:8]} "
                        f"err={type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()

            return True

    def unregister(self, client_id: str, ws: Any) -> bool:
        """Remove the connection only when it is still the active instance."""
        entry = self.connections.get(client_id)
        if entry is None or entry.get("ws") is not ws:
            return False
        self.connections.pop(client_id, None)
        return True
