#!/usr/bin/env python3
"""
Simulate.py

This launcher does the following:
 1. Runs `initiate.py` in the current (main) terminal.
 2. After 30 seconds, opens a new GNOME Terminal window to run `RML_yolo.py`.
 3. After 5 more seconds, opens another GNOME Terminal window to run `missioncorrect.py`.
 4. If you hit Ctrl+C in the main terminal, it will send SIGINT to all three child processes (including the two GNOME‐Terminal windows), so they all shut down together.
"""

import subprocess
import time
import signal
import sys
from pathlib import Path


BASE_DIR = Path(__file__).parent.resolve()
INITIATE_SCRIPT       = BASE_DIR / "Scripts/Initiate.py"
RML_YOLO_SCRIPT       = BASE_DIR / "Scripts/RML_yolo.py"
MISSIONCORRECT_SCRIPT = BASE_DIR / "Scripts/Missioncorect.py"

def launch_in_new_gnome_terminal(python_script: Path) -> subprocess.Popen:
  
    return subprocess.Popen([
        "gnome-terminal",
        "--",                      
        "bash", "-c",
        f"source ~/yolo_env/bin/activate; python3 \"{python_script}\"; exec bash"
    ])

# ── GLOBALS ──────────────────────────────────────────────────────────────────────
child_processes = []  # will hold Popen objects for initiate.py, gnome-terminal#1, gnome-terminal#2

def on_sigint(signum, frame):
    
    print("\n[!] Caught Ctrl+C in main terminal. Terminating all children…")
    for p in child_processes:
        try:
            # Try sending a SIGINT first
            p.send_signal(signal.SIGINT)
            time.sleep(0.1)
            # If it’s still alive, force‐kill it
            if p.poll() is None:
                p.kill()
        except Exception:
            pass

    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, on_sigint)

    if not INITIATE_SCRIPT.exists():
        print(f"[ERROR] Cannot find {INITIATE_SCRIPT}")
        sys.exit(1)

    print("[+] Launching initiate.py in this terminal…")
    p1 = subprocess.Popen(
        ["python3", str(INITIATE_SCRIPT)],
    )
    child_processes.append(p1)
    print(f"    → PID(initiate.py) = {p1.pid}")

    print("[…] Waiting 30 seconds before starting RML_yolo.py …")
    time.sleep(30)

    if not RML_YOLO_SCRIPT.exists():
        print(f"[ERROR] Cannot find {RML_YOLO_SCRIPT}")
        sys.exit(1)

    print("[+] Opening a new GNOME Terminal for RML_yolo.py …")
    p2 = launch_in_new_gnome_terminal(RML_YOLO_SCRIPT)
    child_processes.append(p2)
    print(f"    → PID(gnome-terminal for RML_yolo.py) = {p2.pid}")

    # 4) Wait 5 more seconds
    print("[…] Waiting 5 seconds before starting missioncorrect.py …")
    time.sleep(5)

    if not MISSIONCORRECT_SCRIPT.exists():
        print(f"[ERROR] Cannot find {MISSIONCORRECT_SCRIPT}")
        sys.exit(1)

    print("[+] Opening another GNOME Terminal for missioncorrect.py …")
    p3 = launch_in_new_gnome_terminal(MISSIONCORRECT_SCRIPT)
    child_processes.append(p3)
    print(f"    → PID(gnome-terminal for missioncorrect.py) = {p3.pid}")

    print("[…] Launcher is now waiting for all scripts to finish. Press Ctrl+C to kill them all.")
    while True:
        if all(p.poll() is not None for p in child_processes):
            print("[+] All child processes have exited. Exiting launcher.")
            break
        time.sleep(1)


if __name__ == "__main__":
    main()
