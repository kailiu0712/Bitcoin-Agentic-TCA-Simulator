"""Click Run Python File in VS Code to launch the BTC simulator app."""

from __future__ import annotations

import os
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
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


def main() -> None:
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))

    try:
        import uvicorn
    except ImportError:
        print("Uvicorn is not installed.")
        print(f'Run:  "{sys.executable}" -m pip install -r "{PROJECT_ROOT / "requirements.txt"}"')
        input("Press Enter to close...")
        return

    print("BTC Execution Simulator")
    print(f"Opening {APP_URL}")
    print("Stop the application with the red Stop button in VS Code or Ctrl+C.\n")

    threading.Thread(target=open_browser_when_ready, daemon=True).start()
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
