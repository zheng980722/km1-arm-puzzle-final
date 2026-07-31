#!/usr/bin/env python3
"""Guarded real-hardware test commands for the modified KM1 arm.

This program never opens the Arduino serial device.  It publishes commands to
the project's long-running ``km1_serial_driver`` node, which keeps the serial
port open and avoids the reset caused by one-shot serial connections.

All commands are preview-only unless ``--execute`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray

from .control_geometry import (
    MODIFIED_TOOL_LENGTH_MM,
    paper_to_robot,
    validate_paper_point,
)


DOCS_PATH = os.environ.get(
    "KM1_DOCS_PATH",
    "/home/wheeltec/WorkSpace/km1_arm_ws/docs",
)
if DOCS_PATH not in sys.path:
    sys.path.insert(0, DOCS_PATH)
from kinematics import Km1Kinematics  # noqa: E402


MAGNET_ON_PWM = 1100
MAGNET_OFF_PWM = 1500
TOOL_ROTATION_ID = 4
TOOL_CENTER_PWM = 1500
TOOL_PWM_PER_DEG = 2000.0 / 270.0

SAFE_TEST_Z_MIN_MM = 100.0
SAFE_TEST_Z_MAX_MM = 230.0
SAFE_PWM_MIN = 750
SAFE_PWM_MAX = 2250
PARK_PWMS = (1500, 2000, 2000, 850, 1500)


class ControlTestPublisher(Node):
    def __init__(self):
        super().__init__("km1_control_test")
        self.publisher = self.create_publisher(
            Int32MultiArray,
            "/km1/joint_command",
            10,
        )

    def wait_for_driver(self, timeout_s: float = 3.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.publisher.get_subscription_count() > 0:
                return
        raise RuntimeError(
            "No /km1/joint_command subscriber. Start km1_serial_driver first."
        )

    def publish(self, values: list[int]) -> None:
        message = Int32MultiArray()
        message.data = [int(value) for value in values]
        self.publisher.publish(message)
        rclpy.spin_once(self, timeout_sec=0.15)
        time.sleep(0.15)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="KM1 guarded control test through the project ROS driver"
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    move_paper = subparsers.add_parser(
        "move-paper",
        help="move to a point expressed in rectified A4 millimetres",
    )
    move_paper.add_argument("--paper-x", type=float, required=True)
    move_paper.add_argument("--paper-y", type=float, required=True)
    move_paper.add_argument("--z", type=float, required=True)
    move_paper.add_argument("--time-ms", type=int, default=3000)
    move_paper.add_argument(
        "--tool-length",
        type=float,
        default=MODIFIED_TOOL_LENGTH_MM,
    )
    move_paper.add_argument(
        "--edge-margin",
        type=float,
        default=20.0,
        help="minimum A4 edge clearance for the 40 mm diameter magnet",
    )
    move_paper.add_argument(
        "--allow-low-z",
        action="store_true",
        help="allow z below the uncalibrated 100 mm test floor",
    )
    move_paper.add_argument("--execute", action="store_true")

    move_robot = subparsers.add_parser(
        "move-robot",
        help="move to an explicit KM1 Cartesian coordinate",
    )
    move_robot.add_argument("--x", type=float, required=True)
    move_robot.add_argument("--y", type=float, required=True)
    move_robot.add_argument("--z", type=float, required=True)
    move_robot.add_argument("--time-ms", type=int, default=3000)
    move_robot.add_argument(
        "--tool-length",
        type=float,
        default=MODIFIED_TOOL_LENGTH_MM,
    )
    move_robot.add_argument("--allow-low-z", action="store_true")
    move_robot.add_argument("--execute", action="store_true")

    rotate = subparsers.add_parser(
        "rotate",
        help="rotate the modified gripper/electromagnet about servo ID4",
    )
    rotate.add_argument("--angle-deg", type=float, required=True)
    rotate.add_argument("--time-ms", type=int, default=1500)
    rotate.add_argument("--execute", action="store_true")

    magnet = subparsers.add_parser(
        "magnet",
        help="switch the electromagnet using the captured gripper PWM channel",
    )
    magnet.add_argument("state", choices=("on", "off"))
    magnet.add_argument("--execute", action="store_true")
    magnet.add_argument(
        "--confirm-on",
        action="store_true",
        help="second confirmation required for a real magnet-on command",
    )

    park = subparsers.add_parser(
        "park",
        help="move to the verified firmware park pose outside the A4 view",
    )
    park.add_argument("--time-ms", type=int, default=3000)
    park.add_argument("--execute", action="store_true")
    return parser


def ensure_test_z(z_mm: float, allow_low_z: bool) -> None:
    if z_mm > SAFE_TEST_Z_MAX_MM:
        raise ValueError(
            f"z={z_mm:.1f} mm exceeds the guarded test ceiling "
            f"{SAFE_TEST_Z_MAX_MM:.1f} mm"
        )
    if z_mm < SAFE_TEST_Z_MIN_MM and not allow_low_z:
        raise ValueError(
            f"z={z_mm:.1f} mm is below the uncalibrated guarded floor "
            f"{SAFE_TEST_Z_MIN_MM:.1f} mm; use --allow-low-z only after "
            "paper-surface Z has been measured"
        )


def solve_move(args, robot_x_mm: float, robot_y_mm: float) -> dict:
    ensure_test_z(args.z, args.allow_low_z)
    ik = Km1Kinematics(tool_length_mm=args.tool_length)
    pwms, alpha = ik.find_best_alpha(robot_x_mm, robot_y_mm, args.z)
    if pwms is None:
        raise RuntimeError(
            f"IK failed for robot ({robot_x_mm:.1f}, "
            f"{robot_y_mm:.1f}, {args.z:.1f}) mm"
        )
    if any(pwm < SAFE_PWM_MIN or pwm > SAFE_PWM_MAX for pwm in pwms):
        raise RuntimeError(
            f"Guard rejected PWM {pwms}; allowed test range is "
            f"[{SAFE_PWM_MIN}, {SAFE_PWM_MAX}] us"
        )
    values: list[int] = []
    for servo_id, pwm in enumerate(pwms):
        values.extend([servo_id, int(pwm), int(args.time_ms)])
    return {
        "robot_mm": [
            round(robot_x_mm, 3),
            round(robot_y_mm, 3),
            round(float(args.z), 3),
        ],
        "tool_length_mm": float(args.tool_length),
        "alpha_deg": alpha,
        "pwms": list(pwms),
        "joint_command": values,
    }


def execute_joint_command(values: list[int], magnet_off_first: bool) -> None:
    rclpy.init()
    node = ControlTestPublisher()
    try:
        node.wait_for_driver()
        if magnet_off_first:
            node.publish([5, MAGNET_OFF_PWM, 100])
        node.publish(values)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def run(args) -> dict:
    if args.action == "move-paper":
        validate_paper_point(args.paper_x, args.paper_y, args.edge_margin)
        robot_x, robot_y = paper_to_robot(args.paper_x, args.paper_y)
        result = solve_move(args, robot_x, robot_y)
        result["paper_mm"] = [
            round(float(args.paper_x), 3),
            round(float(args.paper_y), 3),
        ]
        if args.execute:
            execute_joint_command(result["joint_command"], magnet_off_first=True)
            result["executed"] = True
        else:
            result["executed"] = False
        return result

    if args.action == "move-robot":
        result = solve_move(args, args.x, args.y)
        if args.execute:
            execute_joint_command(result["joint_command"], magnet_off_first=True)
            result["executed"] = True
        else:
            result["executed"] = False
        return result

    if args.action == "rotate":
        if not -90.0 <= args.angle_deg <= 90.0:
            raise ValueError("Guarded ID4 rotation range is [-90, 90] degrees")
        pwm = int(round(TOOL_CENTER_PWM + TOOL_PWM_PER_DEG * args.angle_deg))
        result = {
            "angle_deg": float(args.angle_deg),
            "pwm": pwm,
            "joint_command": [TOOL_ROTATION_ID, pwm, int(args.time_ms)],
            "executed": False,
        }
        if args.execute:
            execute_joint_command(result["joint_command"], magnet_off_first=True)
            result["executed"] = True
        return result

    if args.action == "park":
        values: list[int] = []
        for servo_id, pwm in enumerate(PARK_PWMS):
            values.extend([servo_id, pwm, int(args.time_ms)])
        result = {
            "pose": "park",
            "pwms": list(PARK_PWMS),
            "joint_command": values,
            "executed": False,
        }
        if args.execute:
            execute_joint_command(values, magnet_off_first=True)
            result["executed"] = True
        return result

    pwm = MAGNET_ON_PWM if args.state == "on" else MAGNET_OFF_PWM
    if args.state == "on" and args.execute and not args.confirm_on:
        raise ValueError("Real magnet-on requires both --execute and --confirm-on")
    result = {
        "magnet": args.state,
        "pwm": pwm,
        "joint_command": [5, pwm, 100],
        "executed": False,
    }
    if args.execute:
        execute_joint_command(result["joint_command"], magnet_off_first=False)
        result["executed"] = True
    return result


def main():
    parser = build_parser()
    args, unknown = parser.parse_known_args()
    ros_unknown = [
        value for value in unknown if value not in ("--ros-args",)
    ]
    if ros_unknown:
        parser.error(f"unrecognized arguments: {' '.join(ros_unknown)}")
    try:
        result = run(args)
    except Exception as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
