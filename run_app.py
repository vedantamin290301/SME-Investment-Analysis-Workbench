from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
APP_FILE = APP_DIR / "app.py"
PORT = 8501
URL = f"http://localhost:{PORT}"


def wait_until_ready(url: str, timeout_seconds: int = 30) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return response.status == 200
        except Exception:
            time.sleep(1)
    return False


def main() -> int:
    if not APP_FILE.exists():
        print(f"Could not find Streamlit app at {APP_FILE}")
        return 1

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_FILE),
        "--server.port",
        str(PORT),
        "--server.headless",
        "true",
    ]

    print("Starting SME Investment Analysis Workbench...")
    print(f"Opening {URL}")
    process = subprocess.Popen(command, cwd=APP_DIR)

    if wait_until_ready(URL):
        webbrowser.open(URL)
    else:
        print("The app is still starting. Open this URL manually if the browser does not appear:")
        print(URL)

    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
