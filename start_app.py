from __future__ import annotations

import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
BACKEND_PORT = 8000
FRONTEND_PORT = 5173
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}/api/health"
FRONTEND_URL = f"http://127.0.0.1:{FRONTEND_PORT}"

BACKEND_COMMAND = [
    r"D:\Anaconda\envs\jzy\python.exe",
    "-m",
    "uvicorn",
    "Back_end.api_server:app",
    "--host",
    "127.0.0.1",
    "--port",
    str(BACKEND_PORT),
]

FRONTEND_COMMAND = [
    r"D:\Node.js\node.exe",
    str(ROOT / "Front_end" / "node_modules" / "vite" / "bin" / "vite.js"),
    "--host",
    "0.0.0.0",
    "--port",
    str(FRONTEND_PORT),
]


def find_port_pids(ports: list[int]) -> list[int]:
    result = subprocess.run(
        ["netstat", "-ano"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
    )
    pids: set[int] = set()

    for line in result.stdout.splitlines():
        if "LISTENING" not in line:
            continue

        parts = [part for part in line.split() if part]
        if len(parts) < 5:
            continue

        local_address = parts[1]
        pid_text = parts[-1]

        if not pid_text.isdigit():
            continue

        for port in ports:
            if local_address.endswith(f":{port}"):
                pids.add(int(pid_text))

    return sorted(pids)


def kill_processes(pids: list[int]) -> None:
    for pid in pids:
        subprocess.run(
            ["taskkill", "/F", "/PID", str(pid)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )


def start_process(command: list[str], cwd: Path, log_name: str) -> subprocess.Popen:
    OUTPUT_DIR.mkdir(exist_ok=True)
    log_path = OUTPUT_DIR / log_name
    log_file = open(log_path, "w", encoding="utf-8")
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0

    return subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        startupinfo=startupinfo,
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NO_WINDOW
        ),
    )


def wait_until_ready(url: str, timeout_seconds: int) -> bool:
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2):
                return True
        except URLError:
            time.sleep(1)

    return False


def main() -> None:
    ports = [FRONTEND_PORT, BACKEND_PORT]
    old_pids = find_port_pids(ports)

    if old_pids:
        print(f"Cleaning old processes on ports {ports}: {old_pids}")
        kill_processes(old_pids)
        time.sleep(2)

    print("Starting backend...")
    start_process(BACKEND_COMMAND, ROOT, "backend.log")

    print("Starting frontend...")
    start_process(FRONTEND_COMMAND, ROOT / "Front_end", "frontend.log")

    backend_ready = wait_until_ready(BACKEND_URL, timeout_seconds=20)
    frontend_ready = wait_until_ready(FRONTEND_URL, timeout_seconds=20)

    print(f"Backend ready: {backend_ready} -> {BACKEND_URL}")
    print(f"Frontend ready: {frontend_ready} -> {FRONTEND_URL}")
    print(f"Backend log: {OUTPUT_DIR / 'backend.log'}")
    print(f"Frontend log: {OUTPUT_DIR / 'frontend.log'}")

    if not backend_ready or not frontend_ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
