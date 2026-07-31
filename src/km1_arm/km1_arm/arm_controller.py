#!/usr/bin/env python3
"""KM1 vertical-electromagnet pick/place controller.

Production data flow:

    vision_bridge -> /km1/control_plan -> this node -> serial_driver

The controller accepts only the complete schema-v2 vision envelope.  It
chooses reachable grasp points from the detected polygons, constrains the
electromagnet axis to vertical, performs all XY travel at a clearance height,
and releases above the raised paper without mechanical contact.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

sys.path.insert(0, "/home/wheeltec/WorkSpace/km1_arm_ws/docs")
from kinematics import Km1Kinematics  # noqa: E402

from .control_geometry import (
    MODIFIED_TOOL_LENGTH_MM,
    PAPER_SURFACE_Z_MM,
    paper_to_robot,
)
from .vertical_planner import (
    build_highest_vertical_control_plan,
    save_vertical_plan_artifacts,
)


MAGNET_ON = 1100
MAGNET_OFF = 1500
TOOL_ROTATION_ID = 4
TOOL_CENTER_PWM = 1500
TOOL_PWM_PER_DEG = 2000.0 / 270.0
TOOL_MIN_PWM = 550
TOOL_MAX_PWM = 2450
PARK_PWMS = (1500, 2000, 2000, 850, 1500)
SAFE_STAGING_X_MM = 0.0
SAFE_STAGING_Y_MM = 150.0


class ArmController(Node):
    def __init__(self):
        super().__init__("km1_arm_controller")
        self.declare_parameter("enable_automatic_motion", False)
        self.declare_parameter("paper_surface_z_mm", PAPER_SURFACE_Z_MM)
        # Verified collision-free height on the raised fixture.  The apparent
        # drop in the latest run was caused by magnet-signal instability, not
        # by insufficient pickup height.
        self.declare_parameter("pick_clearance_mm", 30.0)
        self.declare_parameter("tool_length_mm", MODIFIED_TOOL_LENGTH_MM)
        self.declare_parameter("travel_clearance_mm", 150.0)
        self.declare_parameter("min_travel_clearance_mm", 70.0)
        self.declare_parameter("travel_search_step_mm", 5.0)
        self.declare_parameter("drop_clearance_mm", 25.0)
        self.declare_parameter("layout_edge_margin_mm", 2.0)
        self.declare_parameter("layout_search_step_mm", 1.0)
        self.declare_parameter("min_pwm_margin_us", 50)
        self.declare_parameter("layout_center_weight", 20.0)
        self.declare_parameter("max_run_time_s", 120.0)
        self.declare_parameter("move_time_ms", 1000)
        self.declare_parameter("magnet_dwell_ms", 350)

        self.enable_automatic_motion = bool(
            self.get_parameter("enable_automatic_motion").value
        )
        self.paper_surface_z_mm = float(
            self.get_parameter("paper_surface_z_mm").value
        )
        self.pick_clearance_mm = float(
            self.get_parameter("pick_clearance_mm").value
        )
        self.travel_clearance_mm = float(
            self.get_parameter("travel_clearance_mm").value
        )
        self.min_travel_clearance_mm = float(
            self.get_parameter("min_travel_clearance_mm").value
        )
        self.travel_search_step_mm = float(
            self.get_parameter("travel_search_step_mm").value
        )
        self.drop_clearance_mm = float(
            self.get_parameter("drop_clearance_mm").value
        )
        self.layout_edge_margin_mm = float(
            self.get_parameter("layout_edge_margin_mm").value
        )
        self.layout_search_step_mm = float(
            self.get_parameter("layout_search_step_mm").value
        )
        self.min_pwm_margin_us = int(
            self.get_parameter("min_pwm_margin_us").value
        )
        self.layout_center_weight = float(
            self.get_parameter("layout_center_weight").value
        )
        self.max_run_time_s = float(
            self.get_parameter("max_run_time_s").value
        )
        self.move_time_ms = int(self.get_parameter("move_time_ms").value)
        self.magnet_dwell_ms = int(
            self.get_parameter("magnet_dwell_ms").value
        )
        self.ik = Km1Kinematics(
            tool_length_mm=float(
                self.get_parameter("tool_length_mm").value
            )
        )

        self.executing = False
        self.lock = threading.Lock()
        self.pub_raw = self.create_publisher(
            String,
            "/km1/raw_command",
            10,
        )
        self.pub_status = self.create_publisher(
            String,
            "/km1/control_status",
            10,
        )
        self.create_subscription(
            String,
            "/km1/control_plan",
            self.plan_callback,
            10,
        )

        state = "ENABLED" if self.enable_automatic_motion else "LOCKED"
        self.get_logger().info(
            "Vertical electromagnet controller ready; "
            f"automatic_motion={state}, "
            f"surface_z={self.paper_surface_z_mm:.1f} mm, "
            f"travel={self.travel_clearance_mm:.1f} mm, "
            f"drop={self.drop_clearance_mm:.1f} mm"
        )

    def _publish_status(self, payload: dict) -> None:
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False)
        self.pub_status.publish(message)

    def plan_callback(self, message: String) -> None:
        if not self.enable_automatic_motion:
            self.get_logger().warning(
                "Control envelope ignored: enable_automatic_motion is false"
            )
            return
        if self.executing:
            self.get_logger().warning("Controller is busy; envelope ignored")
            return
        try:
            envelope = json.loads(message.data)
            if not isinstance(envelope, dict):
                raise ValueError("legacy list plans are no longer executable")
            if int(envelope.get("schema_version", 0)) != 2:
                raise ValueError("schema_version=2 control envelope is required")
            run_dir = str(envelope["run_dir"])
            plan = build_highest_vertical_control_plan(
                envelope,
                self.ik,
                paper_surface_z_mm=self.paper_surface_z_mm,
                pick_clearance_mm=self.pick_clearance_mm,
                max_travel_clearance_mm=self.travel_clearance_mm,
                min_travel_clearance_mm=self.min_travel_clearance_mm,
                travel_search_step_mm=self.travel_search_step_mm,
                drop_clearance_mm=self.drop_clearance_mm,
                layout_edge_margin_mm=self.layout_edge_margin_mm,
                layout_search_step_mm=self.layout_search_step_mm,
                min_pwm_margin_us=self.min_pwm_margin_us,
                layout_center_weight=self.layout_center_weight,
            )
            save_vertical_plan_artifacts(
                run_dir,
                plan,
                pixels_per_mm=float(envelope["pixels_per_mm"]),
            )
            self.get_logger().info(
                f"Accepted schema-v2 envelope: {len(plan)} pieces; "
                f"pick={plan[0]['pick_z_mm']:.1f} mm; "
                f"selected_travel={plan[0]['travel_z_mm']:.1f} mm; "
                f"drop={plan[0]['drop_z_mm']:.1f} mm; "
                f"artifacts={run_dir}"
            )
            threading.Thread(
                target=self.execute_plan,
                args=(
                    plan,
                    run_dir,
                    float(envelope.get("run_started_unix_s", time.time())),
                ),
                daemon=True,
            ).start()
        except Exception as exc:
            self.get_logger().error(f"Control envelope rejected: {exc}")
            self._publish_status(
                {
                    "event": "rejected",
                    "error": str(exc),
                }
            )

    def execute_plan(
        self,
        plan: list[dict],
        run_dir: str,
        run_started_unix_s: float,
    ) -> None:
        with self.lock:
            self.executing = True
        timing_events: list[dict] = []

        def record(event: str, **extra) -> float:
            elapsed = time.time() - run_started_unix_s
            timing_events.append(
                {
                    "event": event,
                    "elapsed_s": round(elapsed, 3),
                    **extra,
                }
            )
            return elapsed

        try:
            record("controller_started", pieces=len(plan))
            self.magnet_off()
            self.rotate_tool(0.0, 500)
            time.sleep(0.65)
            selected_travel_z = float(plan[0]["travel_z_mm"])
            self.move_xyz_vertical(
                SAFE_STAGING_X_MM,
                SAFE_STAGING_Y_MM,
                selected_travel_z,
                self.move_time_ms,
            )
            self._wait_move()
            self._publish_status(
                {
                    "event": "started",
                    "run_dir": run_dir,
                    "pieces": len(plan),
                }
            )

            for index, command in enumerate(plan):
                piece_id = int(command["piece_id"])
                pick = command["pick"]
                place = command["place"]
                place_command = command.get("place_command", place)
                rotation = float(command["rotation_delta_deg"])
                pick_tool_yaw = float(command["pick_tool_yaw_deg"])
                place_tool_yaw = float(command["place_tool_yaw_deg"])
                pick_x, pick_y = paper_to_robot(*pick)
                # The planner resolves the generic placement calibration once
                # and embeds the compensated paper command in the executable
                # plan.  Pickup remains on the raw vision coordinate.
                place_x, place_y = paper_to_robot(*place_command)
                pick_z = float(command["pick_z_mm"])
                travel_z = float(command["travel_z_mm"])
                drop_z = float(command["drop_z_mm"])

                self.get_logger().info(
                    f"P{piece_id} ({index + 1}/{len(plan)}): "
                    f"pick={pick}, place={place}, "
                    f"place_cmd={place_command}, "
                    f"inset={command['grasp_inset_mm']:.1f} mm, "
                    f"piece_yaw={rotation:.1f} deg, "
                    f"tool={pick_tool_yaw:.1f}->{place_tool_yaw:.1f} deg"
                )

                # PICK: vertical tool, no XY movement below travel height.
                self.rotate_tool(0.0, 500)
                self.move_xyz_vertical(
                    pick_x,
                    pick_y,
                    travel_z,
                    self.move_time_ms,
                )
                self._wait_move()
                self.rotate_tool(pick_tool_yaw, self.move_time_ms)
                self._wait_move()
                # Keep the magnet disabled throughout high travel and descent.
                # Early energising can pull a fragment sideways before the
                # tool reaches the vision-derived grasp point, shifting its
                # effective centroid.
                self.magnet_off()
                self.move_xyz_vertical(
                    pick_x,
                    pick_y,
                    pick_z,
                    self.move_time_ms,
                )
                self._wait_move()
                self.magnet_on()
                time.sleep(self.magnet_dwell_ms / 1000.0)
                self.move_xyz_vertical(
                    pick_x,
                    pick_y,
                    travel_z,
                    self.move_time_ms,
                )
                self._wait_move()

                # Yaw only after the magnet is safely above the paper.
                self.rotate_tool(place_tool_yaw, self.move_time_ms)
                self._wait_move()

                # PLACE: vertical approach to the configured clearance above
                # the raised paper, then disable the magnet for a free fall.
                self.move_xyz_vertical(
                    place_x,
                    place_y,
                    travel_z,
                    self.move_time_ms,
                )
                self._wait_move()
                self.move_xyz_vertical(
                    place_x,
                    place_y,
                    drop_z,
                    self.move_time_ms,
                )
                self._wait_move()
                self.magnet_off()
                time.sleep(self.magnet_dwell_ms / 1000.0)
                self.move_xyz_vertical(
                    place_x,
                    place_y,
                    travel_z,
                    self.move_time_ms,
                )
                self._wait_move()
                self.rotate_tool(0.0, 500)
                time.sleep(0.65)

                self._publish_status(
                    {
                        "event": "piece_done",
                        "run_dir": run_dir,
                        "piece_id": piece_id,
                        "index": index + 1,
                        "count": len(plan),
                        "elapsed_s": round(
                            record("piece_done", piece_id=piece_id),
                            3,
                        ),
                    }
                )

            self.magnet_off()
            self.rotate_tool(0.0, 500)
            time.sleep(0.65)
            self.move_xyz_vertical(
                SAFE_STAGING_X_MM,
                SAFE_STAGING_Y_MM,
                selected_travel_z,
                self.move_time_ms,
            )
            self._wait_move()
            self.park(1800)
            time.sleep(2.0)
            self._publish_status(
                {
                    "event": "all_done",
                    "run_dir": run_dir,
                    "pieces": len(plan),
                    "elapsed_s": round(record("all_done"), 3),
                    "within_120_s": (
                        time.time() - run_started_unix_s
                        <= self.max_run_time_s
                    ),
                }
            )
            Path(run_dir, "13_timing.json").write_text(
                json.dumps(
                    {
                        "run_started_unix_s": run_started_unix_s,
                        "max_run_time_s": self.max_run_time_s,
                        "total_elapsed_s": round(
                            time.time() - run_started_unix_s,
                            3,
                        ),
                        "within_limit": (
                            time.time() - run_started_unix_s
                            <= self.max_run_time_s
                        ),
                        "events": timing_events,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            self.get_logger().info(
                "All pieces completed; arm parked; final camera comparison "
                "requested"
            )
        except Exception as exc:
            self.get_logger().error(f"Execution failed: {exc}")
            self.magnet_off()
            self._publish_status(
                {
                    "event": "failed",
                    "run_dir": run_dir,
                    "error": str(exc),
                }
            )
        finally:
            self.executing = False

    def _wait_move(self) -> None:
        time.sleep(self.move_time_ms / 1000.0 + 0.15)

    def move_xyz_vertical(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        time_ms: int,
    ) -> None:
        pwms = self.ik.solve_vertical(x_mm, y_mm, z_mm)
        if pwms is None:
            raise RuntimeError(
                "Vertical IK failed at "
                f"({x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f})"
            )
        if any(pwm < 500 or pwm > 2500 for pwm in pwms):
            raise RuntimeError(f"Vertical IK PWM out of range: {pwms}")
        frame = self.ik.build_frame(pwms, time_ms)
        if frame is None:
            raise RuntimeError("Unable to build vertical IK frame")
        self.send_raw(frame)

    def rotate_tool(self, angle_deg: float, time_ms: int) -> None:
        pwm = int(
            round(
                TOOL_CENTER_PWM
                + TOOL_PWM_PER_DEG * float(angle_deg)
            )
        )
        if pwm < TOOL_MIN_PWM or pwm > TOOL_MAX_PWM:
            raise RuntimeError(
                f"Tool yaw {angle_deg:.1f} deg -> {pwm} us outside "
                f"[{TOOL_MIN_PWM}, {TOOL_MAX_PWM}]"
            )
        self.send_raw(
            f"#{TOOL_ROTATION_ID:03d}P{pwm:04d}T{int(time_ms):04d}!"
        )

    def magnet_on(self) -> None:
        self.send_raw(f"#005P{MAGNET_ON:04d}T0100!")

    def magnet_off(self) -> None:
        self.send_raw(f"#005P{MAGNET_OFF:04d}T0100!")

    def park(self, time_ms: int) -> None:
        parts = [
            f"#{servo_id:03d}P{pwm:04d}T{int(time_ms):04d}!"
            for servo_id, pwm in enumerate(PARK_PWMS)
        ]
        self.send_raw("{" + "".join(parts) + "}")

    def send_raw(self, command: str) -> None:
        message = String()
        message.data = command
        self.pub_raw.publish(message)


def main():
    rclpy.init()
    node = ArmController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
