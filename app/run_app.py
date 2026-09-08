"""Click Run Python File in VS Code to launch the BTC simulator app."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
import socket
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_URL = "http://127.0.0.1:8000"
HEALTH_URL = f"{APP_URL}/api/health"


def open_browser_when_ready() -> None:
    """Open the UI only after Uvicorn is accepting requests."""
    for _ in range(100):
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=0.5) as response:
                if response.status == 200:
                    webbrowser.open(APP_URL)
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.1)

    print(f"The server did not become ready. Try opening {APP_URL} manually.")


def running_build() -> str | None:
    """The build id of whatever already answers on the port, if it answers at all."""
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=1.0) as response:
            return json.loads(response.read()).get("build_id")
    except (OSError, urllib.error.URLError, ValueError):
        return None


def listening_pids() -> list[str]:
    """PIDs holding port 8000, so the message can name the process to stop."""
    try:
        output = subprocess.run(["netstat", "-ano"], capture_output=True, text=True,
                                timeout=5, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    pids = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].upper() == "TCP" and parts[1].endswith(":8000") \
                and parts[3].upper() == "LISTENING" and parts[4] not in pids:
            pids.append(parts[4])
    return pids


def report_port_conflict() -> None:
    expected = current_build_id()
    running = running_build()
    print("Port 8000 is already in use by another simulator process.\n")
    if running is None:
        print("  It does not answer /api/health, so it may not be this application.")
    elif expected is None:
        print(f"  It answers as build {running!r}; this checkout's build could not be read")
        print("  (dependencies may be missing for this interpreter).")
    elif running == expected:
        print(f"  It is running the current build ({running}); just open {APP_URL}.")
        return
    else:
        print(f"  It is running an OLDER build: {running!r} (this checkout is {expected!r}).")
        print("  The browser reads index.html and app.js from disk, so you would see the")
        print("  new interface talking to the old API — which fails when you run a")
        print("  simulation. Stop that process and start this one again.")
    for pid in listening_pids():
        print(f"\n  Stop it with:  taskkill /PID {pid} /F")
    print(f"\n  Then run app/run_app.py again; it will serve {APP_URL}.")


def current_build_id() -> str | None:
    """The build id in this checkout. None if the dependencies are not installed."""
    try:
        from app.main import BUILD_ID
    except Exception:
        return None
    return BUILD_ID


def main() -> None:
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))

    # A server left running from an earlier session keeps the port and answers
    # with whatever code it loaded back then, while the browser reads index.html
    # and app.js fresh from disk. The result is a new page talking to an old API:
    # /api routes 404 or return a stale contract and the UI reports an error that
    # looks like a simulation failure. Diagnose that explicitly rather than
    # printing a generic "port in use".
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 8000)) == 0:
            report_port_conflict()
            return

    try:
        import uvicorn
    except ImportError:
        print("Uvicorn is not installed.")
        print(f'Run:  "{sys.executable}" -m pip install -r "{PROJECT_ROOT / "requirements.txt"}"')
        input("Press Enter to close...")
        return

    try:
        import yaml  # noqa: F401
    except ImportError:
        print(f"This Python interpreter is missing PyYAML: {sys.executable}")
        print(f'Run:  "{sys.executable}" -m pip install -r "{PROJECT_ROOT / "requirements.txt"}"')
        print("Or launch with the configured Python 3.12 interpreter if it is already installed:")
        print(f'      "C:\\Users\\kai\\AppData\\Local\\Programs\\Python\\Python312\\python.exe" "{Path(__file__)}"')
        return

    print("Liquidation Impact Simulator")
    print(f"Opening {APP_URL}")
    print("Stop the application with the red Stop button in VS Code or Ctrl+C.\n")

    threading.Thread(target=open_browser_when_ready, daemon=True).start()
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
