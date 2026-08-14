"""Vast 인스턴스 풀플로우 테스트 — 생성→준비→ready 까지 (테스트 후 수동 파괴)."""
import asyncio
import sys

from vast_backend.service import VastService

WORKFLOW = "comfy/user/default/workflows/SOYA_USER/배포_영상_H3_I2V_v1.json"

s = VastService(".", lambda: {"vast_enabled": True})


async def main() -> None:
    acct = await s.account_status()
    print("잔액:", acct.get("balance_usd"), "USD", flush=True)
    if not acct.get("balance_usd"):
        print("잔액 없음 — 중단", flush=True)
        sys.exit(1)

    adopt_id = None
    if len(sys.argv) >= 3 and sys.argv[1] == "adopt":
        adopt_id = int(sys.argv[2])
        print(f"기존 인스턴스 재활용: {adopt_id}", flush=True)

    offers = await s.offers(gpu_names=["RTX 3090"], min_disk_gb=80, limit=400)
    if not offers["offers"]:
        print("오퍼 없음 — 중단", flush=True)
        sys.exit(1)

    plan = s.wizard_plan(
        workflow_files=[{"path": WORKFLOW, "name": "영상 H3 I2V"}],
        lora_files=[],
    )
    print("PLAN:", plan["totals"], flush=True)
    disk = plan["totals"]["recommended_disk_gb"]
    print(f"디스크 {disk}GB로 생성 시작", flush=True)

    payload = s.prepare_install_payload()
    print(
        "install payload: manifest",
        len(payload["manifest_bytes"]),
        "B / script",
        len(payload["script_bytes"]),
        "B / 로컬 노드",
        [n["name"] for n in payload["local_nodes"]],
        flush=True,
    )

    for attempt in range(5):
        offers = await s.offers(gpu_names=["RTX 3090"], min_disk_gb=80, limit=400)
        if not offers["offers"]:
            print("오퍼 없음 — 중단", flush=True)
            sys.exit(1)
        target = offers["offers"][0]
        ask_id = target["id"]
        print(
            f"[시도 {attempt + 1}] 오퍼: id={ask_id} ${target['dph_total']:.3f}/h "
            f"RAM={target['cpu_ram_gb']}GB disk={target['disk_gb']:.0f}GB",
            flush=True,
        )

        await s.start_launch(
            ask_id=ask_id,
            disk_gb=disk,
            model_plan=plan,
            lora_files=[],
            install_payload=payload,
            adopt_instance_id=adopt_id,
        )
        last = ""
        while s.launch["state"] not in {"ready", "error", "destroyed"}:
            await asyncio.sleep(10)
            steps = s.launch["steps"]
            sig = f"{s.launch['state']} | " + " ; ".join(
                f"{x['key']}:{x['state']}" for x in steps
            )
            if sig != last:
                print(f"[{s.launch['instance_id']}] {sig}", flush=True)
                last = sig
        if s.launch["state"] == "ready":
            break
        err = str(s.launch.get("error") or "")
        if "no_such_ask" in err:
            print("오퍼 레이스 패배 — 다음 오퍼로 재시도", flush=True)
            adopt_id = None  # 첫 시도만 기존 인스턴스 재활용
            continue
        break

    print("최종:", s.launch["state"], "err=", s.launch.get("error"), flush=True)
    print("comfy_url:", s.launch.get("comfy_base_url"), flush=True)
    await s.close()


asyncio.run(main())
