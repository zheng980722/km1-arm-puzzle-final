#!/usr/bin/env python3
"""Camera-assisted bidirectional calibration for the modified ID4 tool yaw.

Only servo ID4 is moved.  The other arm axes remain in their current pose and
the electromagnet is explicitly disabled before the sweep.  A red line on the
rotating tool is segmented in the overhead camera image and measured modulo
180 degrees.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


TOOL_SERVO_ID = 4
MAGNET_SERVO_ID = 5
TOOL_CENTER_PWM = 1500
TOOL_PWM_PER_DEG = 2000.0 / 270.0
MAGNET_OFF_PWM = 1500
MEASURED_ANGLES_DEG = (-70.0, -35.0, 0.0, 35.0, 70.0)


def angle_to_pwm(angle_deg: float) -> int:
    return int(round(TOOL_CENTER_PWM + TOOL_PWM_PER_DEG * angle_deg))


def line_angle_deg(rect) -> float:
    (_, _), (width, height), angle = rect
    if width < height:
        angle += 90.0
    return (float(angle) + 90.0) % 180.0 - 90.0


def wrapped_line_difference(a_deg: float, b_deg: float) -> float:
    """Return a-b for an undirected line, in [-90, 90) degrees."""

    return (float(a_deg) - float(b_deg) + 90.0) % 180.0 - 90.0


def detect_red_marker(frame, center_hint=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = (
        ((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 170))
        & (hsv[:, :, 1] >= 100)
        & (hsv[:, :, 2] >= 70)
    ).astype(np.uint8) * 255
    red = cv2.morphologyEx(
        red,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    )

    contours, _ = cv2.findContours(
        red,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    height, width = frame.shape[:2]
    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 20.0:
            continue
        rect = cv2.minAreaRect(contour)
        (cx, cy), (rect_w, rect_h), _ = rect
        long_side = max(float(rect_w), float(rect_h))
        short_side = max(1.0, min(float(rect_w), float(rect_h)))
        aspect = long_side / short_side
        if long_side < 18.0 or aspect < 2.0:
            continue

        if center_hint is None:
            if not (0.35 * width <= cx <= 0.65 * width):
                continue
            if cy < 0.78 * height:
                continue
            distance = math.hypot(cx - 0.5 * width, cy - 0.90 * height)
        else:
            distance = math.hypot(
                cx - float(center_hint[0]),
                cy - float(center_hint[1]),
            )
            if distance > 45.0:
                continue

        score = area * min(aspect, 12.0) / (1.0 + 0.08 * distance)
        candidates.append((score, contour, rect, area, aspect, distance))

    if not candidates:
        return None, red
    _, contour, rect, area, aspect, distance = max(
        candidates,
        key=lambda item: item[0],
    )
    center = tuple(float(value) for value in rect[0])
    result = {
        "center_px": [round(center[0], 3), round(center[1], 3)],
        "line_angle_deg_mod_180": round(line_angle_deg(rect), 4),
        "area_px2": round(area, 3),
        "aspect_ratio": round(aspect, 3),
        "distance_from_reference_px": round(distance, 3),
        "box_px": np.round(cv2.boxPoints(rect), 1).tolist(),
    }
    return result, red


def capture_stable_frame(capture, discard_count=8):
    frame = None
    for _ in range(discard_count):
        ok, candidate = capture.read()
        if ok:
            frame = candidate
        time.sleep(0.025)
    if frame is None:
        raise RuntimeError("camera frame capture failed")
    return frame


def save_annotated(path: Path, frame, marker, label: str) -> None:
    annotated = frame.copy()
    if marker is not None:
        box = np.asarray(marker["box_px"], dtype=np.int32)
        cv2.drawContours(annotated, [box], 0, (0, 255, 255), 3)
        cx, cy = marker["center_px"]
        cv2.circle(annotated, (round(cx), round(cy)), 5, (255, 0, 255), -1)
        details = (
            f"{label} measured={marker['line_angle_deg_mod_180']:.2f} deg"
        )
    else:
        details = f"{label} MARKER NOT FOUND"
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 60), (20, 20, 20), -1)
    cv2.putText(
        annotated,
        details,
        (18, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), annotated):
        raise RuntimeError(f"unable to write {path}")


class CalibrationPublisher(Node):
    def __init__(self):
        super().__init__("km1_yaw_calibration")
        self.publisher = self.create_publisher(String, "/km1/raw_command", 10)

    def wait_for_driver(self, timeout_s=5.0):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.publisher.get_subscription_count() > 0:
                return
        raise RuntimeError("/km1/raw_command has no subscriber")

    def send_servo(self, servo_id: int, pwm: int, time_ms: int):
        message = String()
        message.data = (
            f"#{int(servo_id):03d}P{int(pwm):04d}T{int(time_ms):04d}!"
        )
        self.publisher.publish(message)
        rclpy.spin_once(self, timeout_sec=0.15)


def analyse_measurements(measurements):
    summaries = {}
    for sweep_name in ("up", "down"):
        rows = [
            row
            for row in measurements
            if row["sweep"] == sweep_name and row.get("marker") is not None
        ]
        if len(rows) < 3:
            summaries[sweep_name] = {"valid": False, "samples": len(rows)}
            continue
        raw = np.asarray(
            [row["marker"]["line_angle_deg_mod_180"] for row in rows],
            dtype=float,
        )
        continuous = np.rad2deg(np.unwrap(np.deg2rad(2.0 * raw))) / 2.0
        commanded = np.asarray([row["command_deg"] for row in rows], dtype=float)
        pwm = np.asarray([row["pwm_us"] for row in rows], dtype=float)
        slope_command, intercept_command = np.polyfit(commanded, continuous, 1)
        slope_pwm, intercept_pwm = np.polyfit(pwm, continuous, 1)
        predicted = slope_command * commanded + intercept_command
        residual = continuous - predicted
        for row, measured in zip(rows, continuous):
            row["continuous_line_angle_deg"] = round(float(measured), 4)
        summaries[sweep_name] = {
            "valid": True,
            "samples": len(rows),
            "line_angle_per_command_deg": round(float(slope_command), 6),
            "line_angle_deg_per_pwm_us": round(float(slope_pwm), 8),
            "fit_intercept_deg": round(float(intercept_command), 5),
            "fit_rms_deg": round(
                float(np.sqrt(np.mean(np.square(residual)))),
                5,
            ),
            "expected_abs_deg_per_pwm_us": round(1.0 / TOOL_PWM_PER_DEG, 8),
            "pwm_fit_intercept_deg": round(float(intercept_pwm), 5),
        }

    hysteresis = []
    up = {
        row["command_deg"]: row
        for row in measurements
        if row["sweep"] == "up" and row.get("marker") is not None
    }
    down = {
        row["command_deg"]: row
        for row in measurements
        if row["sweep"] == "down" and row.get("marker") is not None
    }
    for command_deg in sorted(set(up) & set(down)):
        difference = wrapped_line_difference(
            down[command_deg]["marker"]["line_angle_deg_mod_180"],
            up[command_deg]["marker"]["line_angle_deg_mod_180"],
        )
        hysteresis.append({
            "command_deg": command_deg,
            "down_minus_up_line_angle_deg": round(difference, 4),
        })
    return summaries, hysteresis


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--move-time-ms", type=int, default=1000)
    parser.add_argument("--settle-s", type=float, default=0.45)
    parser.add_argument("--execute", action="store_true")
    return parser


def run(args):
    if not args.execute:
        raise RuntimeError("physical yaw sweep requires --execute")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    capture.set(cv2.CAP_PROP_FPS, 30)
    if not capture.isOpened():
        raise RuntimeError("unable to open camera")

    rclpy.init()
    node = CalibrationPublisher()
    measurements = []
    try:
        node.wait_for_driver()
        capture_stable_frame(capture, discard_count=25)
        node.send_servo(MAGNET_SERVO_ID, MAGNET_OFF_PWM, 100)
        node.send_servo(TOOL_SERVO_ID, TOOL_CENTER_PWM, args.move_time_ms)
        time.sleep(args.move_time_ms / 1000.0 + args.settle_s)
        baseline_frame = capture_stable_frame(capture)
        baseline, baseline_mask = detect_red_marker(baseline_frame)
        if baseline is None:
            raise RuntimeError("red marker was not found in the parked image")
        cv2.imwrite(str(output_dir / "baseline_red_mask.png"), baseline_mask)
        save_annotated(
            output_dir / "baseline.jpg",
            baseline_frame,
            baseline,
            "baseline pwm=1500",
        )
        center_hint = baseline["center_px"]

        sweeps = (
            ("up", -80.0, MEASURED_ANGLES_DEG),
            ("down", 80.0, tuple(reversed(MEASURED_ANGLES_DEG))),
        )
        for sweep_name, preload_deg, command_angles in sweeps:
            node.send_servo(
                TOOL_SERVO_ID,
                angle_to_pwm(preload_deg),
                args.move_time_ms,
            )
            time.sleep(args.move_time_ms / 1000.0 + args.settle_s)
            for command_deg in command_angles:
                pwm = angle_to_pwm(command_deg)
                node.send_servo(TOOL_SERVO_ID, pwm, args.move_time_ms)
                time.sleep(args.move_time_ms / 1000.0 + args.settle_s)
                frame = capture_stable_frame(capture)
                marker, mask = detect_red_marker(frame, center_hint=center_hint)
                label = f"{sweep_name} cmd={command_deg:+.0f} pwm={pwm}"
                stem = f"{sweep_name}_{command_deg:+04.0f}".replace("+", "p").replace("-", "m")
                save_annotated(output_dir / f"{stem}.jpg", frame, marker, label)
                cv2.imwrite(str(output_dir / f"{stem}_mask.png"), mask)
                measurements.append({
                    "sweep": sweep_name,
                    "command_deg": command_deg,
                    "pwm_us": pwm,
                    "marker": marker,
                })
    finally:
        try:
            node.send_servo(TOOL_SERVO_ID, TOOL_CENTER_PWM, args.move_time_ms)
            time.sleep(args.move_time_ms / 1000.0 + 0.2)
        except Exception:
            pass
        capture.release()
        node.destroy_node()
        rclpy.shutdown()

    summaries, hysteresis = analyse_measurements(measurements)
    result = {
        "baseline": baseline,
        "tool_center_pwm": TOOL_CENTER_PWM,
        "assumed_pwm_per_deg": TOOL_PWM_PER_DEG,
        "measurements": measurements,
        "fit": summaries,
        "hysteresis": hysteresis,
    }
    (output_dir / "yaw_calibration_measurements.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
