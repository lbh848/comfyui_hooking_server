"""Vast 인스턴스 수동 E2E 실행 도구 — 실제 비용 발생, 성공 후 수동 파괴."""

import asyncio
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vast_backend.service import VastService  # noqa: E402


WORKFLOW = "comfy/user/default/workflows/SOYA_USER/배포_영상_H3_I2V_v1.json"


async def main() -> None:
    service = VastService(PROJECT_ROOT, lambda: {"vast_enabled": True})
    account = await service.account_status()
    print("잔액:", account.get("balance_usd"), "USD", flush=True)
    if not account.get("balance_usd"):
        print("잔액 없음 — 중단", flush=True)
        raise SystemExit(1)

    adopt_id = None
    if len(sys.argv) >= 3 and sys.argv[1] == "adopt":
        adopt_id = int(sys.argv[2])
        print(f"기존 인스턴스 재활용: {adopt_id}", flush=True)

    offers = await service.offers(
        gpu_names=["RTX 3090"],
        min_disk_gb=80,
        limit=400,
    )
    if not offers["offers"]:
        print("오퍼 없음 — 중단", flush=True)
        raise SystemExit(1)

    plan = service.wizard_plan(
        workflow_files=[{"path": WORKFLOW, "name": "영상 H3 I2V"}],
        lora_files=[],
    )
    print("PLAN:", plan["totals"], flush=True)
    disk = plan["totals"]["recommended_disk_gb"]
    print(f"디스크 {disk}GB로 생성 시작", flush=True)

    payload = service.prepare_install_payload()
    print(
        "install payload: manifest",
        len(payload["manifest_bytes"]),
        "B / script",
        len(payload["script_bytes"]),
        "B / 로컬 노드",
        [node["name"] for node in payload["local_nodes"]],
        flush=True,
    )

    for attempt in range(5):
        offers = await service.offers(
            gpu_names=["RTX 3090"],
            min_disk_gb=80,
            limit=400,
        )
        if not offers["offers"]:
            print("오퍼 없음 — 중단", flush=True)
            raise SystemExit(1)
        target = offers["offers"][0]
        ask_id = target["id"]
        print(
            f"[시도 {attempt + 1}] 오퍼: id={ask_id} "
            f"${target['dph_total']:.3f}/h "
            f"RAM={target['cpu_ram_gb']}GB disk={target['disk_gb']:.0f}GB",
            flush=True,
        )

        await service.start_launch(
            ask_id=ask_id,
            disk_gb=disk,
            model_plan=plan,
            lora_files=[],
            install_payload=payload,
            adopt_instance_id=adopt_id,
        )
        last = ""
        while service.launch["state"] not in {"ready", "error", "destroyed"}:
            await asyncio.sleep(10)
            signature = f"{service.launch['state']} | " + " ; ".join(
                f"{step['key']}:{step['state']}" for step in service.launch["steps"]
            )
            if signature != last:
                print(
                    f"[{service.launch['instance_id']}] {signature}",
                    flush=True,
                )
                last = signature
        if service.launch["state"] == "ready":
            break
        error = str(service.launch.get("error") or "")
        if "no_such_ask" in error:
            print("오퍼 레이스 패배 — 다음 오퍼로 재시도", flush=True)
            adopt_id = None
            continue
        break

    print(
        "최종:",
        service.launch["state"],
        "err=",
        service.launch.get("error"),
        flush=True,
    )
    print("comfy_url:", service.launch.get("comfy_base_url"), flush=True)
    await service.close()


if __name__ == "__main__":
    asyncio.run(main())
