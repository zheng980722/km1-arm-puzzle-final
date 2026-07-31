"""Map two active-low Jetson header buttons to competition shell scripts."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import time


# JetPack 6.2 reports a "super" compatible string that Jetson.GPIO 2.1.7
# does not recognize yet. The pin table itself is the standard Orin NX table.
os.environ.setdefault("JETSON_MODEL_NAME", "JETSON_ORIN_NX")

import Jetson.GPIO as GPIO  # noqa: E402


S1_PIN = 7
S2_PIN = 15
S1_SCRIPT = Path("/home/wheeltec/111.sh")
S2_SCRIPT = Path("/home/wheeltec/222.sh")
DEBOUNCE_S = 0.06
RELEASE_TO_ARM_S = 0.30
POLL_S = 0.02


class DebouncedInput:
    def __init__(self, initial_value: int) -> None:
        self.stable = initial_value
        self.candidate = initial_value
        self.candidate_since = time.monotonic()

    def update(self, value: int, now: float) -> None:
        if value != self.candidate:
            self.candidate = value
            self.candidate_since = now
        elif value != self.stable and now - self.candidate_since >= DEBOUNCE_S:
            self.stable = value


def _stop_child(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=12.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def _run_script(script: Path, stop_requested: list[bool]) -> int:
    if not script.is_file() or not os.access(script, os.X_OK):
        print(f"[buttons] script is missing or not executable: {script}", flush=True)
        return 126

    print(f"[buttons] executing {script}", flush=True)
    process = subprocess.Popen([str(script)], start_new_session=True)
    while process.poll() is None:
        if stop_requested[0]:
            _stop_child(process)
            break
        time.sleep(0.1)
    return process.wait()


def main() -> None:
    stop_requested = [False]

    def request_stop(signum, _frame):
        stop_requested[0] = True
        print(f"[buttons] received signal {signum}; stopping", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(S1_PIN, GPIO.IN)
    GPIO.setup(S2_PIN, GPIO.IN)

    s1 = DebouncedInput(GPIO.input(S1_PIN))
    s2 = DebouncedInput(GPIO.input(S2_PIN))
    armed = False
    released_since: float | None = None

    print(
        "[buttons] ready: S1 physical pin 7 -> ~/111.sh (Task1); "
        "S2 physical pin 15 -> ~/222.sh (Task2); active-low",
        flush=True,
    )

    try:
        while not stop_requested[0]:
            now = time.monotonic()
            s1.update(GPIO.input(S1_PIN), now)
            s2.update(GPIO.input(S2_PIN), now)

            both_released = s1.stable == GPIO.HIGH and s2.stable == GPIO.HIGH
            if not armed:
                if both_released:
                    if released_since is None:
                        released_since = now
                    elif now - released_since >= RELEASE_TO_ARM_S:
                        armed = True
                        print("[buttons] armed", flush=True)
                else:
                    released_since = None
                time.sleep(POLL_S)
                continue

            if s1.stable == GPIO.LOW and s2.stable == GPIO.LOW:
                print("[buttons] both buttons pressed; ignoring until release", flush=True)
                armed = False
                released_since = None
            elif s1.stable == GPIO.LOW:
                armed = False
                released_since = None
                return_code = _run_script(S1_SCRIPT, stop_requested)
                print(f"[buttons] {S1_SCRIPT} exited with code {return_code}", flush=True)
            elif s2.stable == GPIO.LOW:
                armed = False
                released_since = None
                return_code = _run_script(S2_SCRIPT, stop_requested)
                print(f"[buttons] {S2_SCRIPT} exited with code {return_code}", flush=True)

            time.sleep(POLL_S)
    finally:
        GPIO.cleanup()
        print("[buttons] GPIO released", flush=True)


if __name__ == "__main__":
    main()
