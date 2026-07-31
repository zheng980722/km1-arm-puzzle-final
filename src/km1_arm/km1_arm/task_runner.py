"""Run one competition task and stop its ROS launch after completion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


WORKSPACE = Path("/home/wheeltec/WorkSpace/km1_arm_ws")
DEFAULT_OUTPUT_ROOT = WORKSPACE / "button_runs"
SERIAL_DEVICE = Path("/dev/ttyCH341USB0")
CAMERA_DEVICE = Path("/dev/video0")


def _stop_process_group(process: subprocess.Popen, label: str) -> None:
    if process.poll() is not None:
        return

    for sig, timeout_s in (
        (signal.SIGINT, 8.0),
        (signal.SIGTERM, 4.0),
        (signal.SIGKILL, 1.0),
    ):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout_s)
            return
        except subprocess.TimeoutExpired:
            print(f"[{label}] process did not stop after {sig.name}", flush=True)


def _new_timing_file(
    output_root: Path,
    files_before_start: set[Path],
) -> tuple[Path, dict] | None:
    candidates = sorted(
        output_root.glob("*/13_timing.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        if path in files_before_start:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if any(event.get("event") == "all_done" for event in payload.get("events", [])):
            return path, payload
    return None


def run_task(task: int, output_base: Path, timeout_s: float, enable_motion: bool) -> int:
    if task not in (1, 2):
        raise ValueError(f"unsupported task: {task}")

    missing = [str(path) for path in (SERIAL_DEVICE, CAMERA_DEVICE) if not path.exists()]
    if missing:
        print(f"[Task{task}] missing device(s): {', '.join(missing)}", file=sys.stderr)
        return 2

    output_root = output_base / f"task{task}"
    output_root.mkdir(parents=True, exist_ok=True)
    files_before_start = set(output_root.glob("*/13_timing.json"))

    command = [
        "ros2",
        "launch",
        "km1_arm",
        "competition_once.launch.py",
        f"competition_task:={task}",
        f"diagnostic_dir:={output_root}",
        f"enable_motion:={'true' if enable_motion else 'false'}",
    ]
    print(f"[Task{task}] starting: {' '.join(command)}", flush=True)
    process = subprocess.Popen(command, start_new_session=True)
    started_monotonic = time.monotonic()
    interrupted = False

    def request_stop(signum, _frame):
        nonlocal interrupted
        interrupted = True
        print(f"[Task{task}] received signal {signum}; stopping", flush=True)

    previous_handlers = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }

    try:
        while True:
            if interrupted:
                _stop_process_group(process, f"Task{task}")
                return 130

            completed = _new_timing_file(output_root, files_before_start)
            if completed is not None:
                timing_path, timing = completed
                elapsed_s = float(timing.get("total_elapsed_s", 0.0))
                within_limit = bool(timing.get("within_limit", False))
                print(
                    f"[Task{task}] completed in {elapsed_s:.3f}s; "
                    f"within_limit={within_limit}; run={timing_path.parent}",
                    flush=True,
                )
                # The timing file is written after the final comparison image.
                # Give ROS logs a brief moment to flush before releasing devices.
                time.sleep(1.0)
                _stop_process_group(process, f"Task{task}")
                return 0 if within_limit else 3

            return_code = process.poll()
            if return_code is not None:
                print(
                    f"[Task{task}] ROS launch exited early with code {return_code}",
                    file=sys.stderr,
                    flush=True,
                )
                return return_code or 4

            elapsed = time.monotonic() - started_monotonic
            if elapsed >= timeout_s:
                print(
                    f"[Task{task}] timed out after {elapsed:.1f}s; stopping ROS launch",
                    file=sys.stderr,
                    flush=True,
                )
                _stop_process_group(process, f"Task{task}")
                return 124

            time.sleep(0.2)
    finally:
        _stop_process_group(process, f"Task{task}")
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timeout", type=float, default=130.0)
    parser.add_argument(
        "--no-motion",
        action="store_true",
        help="run the visual pipeline without sending arm motion commands",
    )
    args = parser.parse_args()
    raise SystemExit(
        run_task(
            task=args.task,
            output_base=args.output_base,
            timeout_s=args.timeout,
            enable_motion=not args.no_motion,
        )
    )


if __name__ == "__main__":
    main()
