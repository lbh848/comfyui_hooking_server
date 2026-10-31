"""Shared launcher; application-specific frontends will use these entry points."""
from __future__ import annotations

import argparse
import os
import runpy
import socket
import webbrowser

from app_paths import SOURCE_DIR, initialize_layout


APPS = ("illustration", "asset", "video")


def app_url(app: str) -> str:
    if app not in APPS:
        raise ValueError(f"Unknown application: {app}")
    return f"http://127.0.0.1:8189/?app={app}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app", choices=APPS, nargs="?", default="illustration")
    parser.add_argument("--print-url", action="store_true", help="Print the entry URL without starting the engine")
    args = parser.parse_args()
    url = app_url(args.app)
    if args.print_url:
        print(url)
        return
    os.chdir(SOURCE_DIR)
    os.environ["SOYA_APP"] = args.app
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.3)
        running = probe.connect_ex(("127.0.0.1", 8189)) == 0
    if running:
        if os.environ.get("NO_BROWSER", "").lower() not in {"1", "true", "yes", "on"}:
            webbrowser.open(url)
        return
    initialize_layout()
    runpy.run_path(str(SOURCE_DIR / "server.py"), run_name="__main__")


if __name__ == "__main__":
    main()
