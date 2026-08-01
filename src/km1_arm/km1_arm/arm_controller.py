#!/usr/bin/env python3
"""KM1 phase-aware electromagnet pick/place controller.

Production data flow:

    vision_bridge -> /km1/control_plan -> this node -> serial_driver

The controller accepts only the complete schema-v2 vision envelope.  It
chooses reachable grasp points from the detected polygons, keeps pickup and
release vertical whenever IK permits, allows the minimum necessary pitch at
far-edge poses, and performs XY travel at a clearance height.
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

from .control_geometry import MODIFIED_TOOL_LENGTH_MM, paper_to_robot
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
        self.declare_parameter("paper_surface_z_mm", 0.0)
        self.declare_parameter("pick_clearance_mm", 20.0)
        self.declare_parameter("tool_length_mm", MODIFIED_TOOL_LENGTH_MM)
        self.declare_parameter("travel_clearance_mm", 150.0)
        self.declare_parameter("min_travel_clearance_mm", 70.0)
        self.declare_parameter("travel_search_step_mm", 5.0)
        self.declare_parameter("drop_clearance_mm", 25.0)
        self.declare_parameter("layout_edge_margin_mm", 2.0)
        self.declare_parameter("layout_near_edge_margin_mm", 2.0)
        self.declare_parameter("layout_search_step_mm", 1.0)
        self.declare_parameter("min_pwm_margin_us", 50)
        self.declare_parameter("layout_center_weight", 20.0)
        self.declare_parameter("max_run_time_s", 120.0)
        self.declare_parameter("move_time_ms", 1000)
        self.declare_parameter("magnet_dwell_ms", 350)
        self.declare_parameter("event_capture_hold_s", 0.75)
        self.declare_parameter("completion_beep_count", 2)
        self.declare_parameter("completion_beep_gap_s", 0.35)

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
        self.layout_near_edge_margin_mm = float(
            self.get_parameter("layout_near_edge_margin_mm").value
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
        self.event_capture_hold_s = max(
            0.0,
            float(self.get_parameter("event_capture_hold_s").value),
        )
        self.completion_beep_count = max(
            0,
            int(self.get_parameter("completion_beep_count").value),
        )
        self.completion_beep_gap_s = max(
            0.0,
            float(self.get_parameter("completion_beep_gap_s").value),
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
            f"drop={self.drop_clearance_mm:.1f} mm, "
            f"capture_hold={self.event_capture_hold_s:.2f} s"
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
                layout_near_edge_margin_mm=(
                    self.layout_near_edge_margin_mm
                ),
                layout_search_step_mm=self.layout_search_step_mm,
                min_pwm_margin_us=self.min_pwm_margin_us,
                layout_center_weight=self.layout_center_weight,
            )
            save_vertical_plan_artifacts(
                run_dir,
                plan,
                pixels_per_mm=float(envelope["pixels_per_mm"]),
            )
            total_piece_count = int(
                plan[0].get("total_piece_count", len(plan))
            )
            skipped_piece_ids = list(plan[0].get("skipped_piece_ids", []))
            if skipped_piece_ids:
                self.get_logger().warning(
                    "Partial-score execution: skipping unreachable pieces "
                    f"{skipped_piece_ids}"
                )
            self.get_logger().info(
                "Accepted schema-v2 envelope: "
                f"planned={len(plan)}/{total_piece_count} pieces; "
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
            staging_z = self._select_reachable_staging_z(selected_travel_z)
            if staging_z < selected_travel_z - 1e-6:
                self.get_logger().warning(
                    "Safe staging Z lowered from "
                    f"{selected_travel_z:.1f} to {staging_z:.1f} mm; "
                    "piece travel poses keep their planned height"
                )
            self.move_xyz_vertical(
                SAFE_STAGING_X_MM,
                SAFE_STAGING_Y_MM,
                staging_z,
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
                rotation = float(command["rotation_delta_deg"])
                pick_tool_yaw = float(command["pick_tool_yaw_deg"])
                place_tool_yaw = float(command["place_tool_yaw_deg"])
                pick_x, pick_y = paper_to_robot(*pick)
                place_x, place_y = paper_to_robot(*place)
                pick_z = float(command["pick_z_mm"])
                travel_z = float(command["travel_z_mm"])
                drop_z = float(command["drop_z_mm"])
                pick_travel_alpha = float(
                    command.get("pick_travel_alpha_deg", -90.0)
                )
                pick_contact_alpha = float(
                    command.get("pick_contact_alpha_deg", -90.0)
                )
                place_travel_alpha = float(
                    command.get("place_travel_alpha_deg", -90.0)
                )
                place_drop_alpha = float(
                    command.get("place_drop_alpha_deg", -90.0)
                )

                self.get_logger().info(
                    f"P{piece_id} ({index + 1}/{len(plan)}): "
                    f"pick={pick}, place={place}, "
                    f"inset={command['grasp_inset_mm']:.1f} mm, "
                    f"piece_yaw={rotation:.1f} deg, "
                    f"tool={pick_tool_yaw:.1f}->{place_tool_yaw:.1f} deg, "
                    f"pitch=({pick_travel_alpha:.1f},"
                    f"{pick_contact_alpha:.1f})->"
                    f"({place_travel_alpha:.1f},"
                    f"{place_drop_alpha:.1f}) deg"
                )

                # PICK: use the steepest reachable pitch.  The planner keeps
                # contact vertical unless the real far-edge workspace needs a
                # small tilt, as confirmed by manual pickup.
                self.rotate_tool(0.0, 500)
                self.move_precomputed_pwms(
                    command["pwms"]["pick_travel"],
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
                self.move_precomputed_pwms(
                    command["pwms"]["pick_contact"],
                    self.move_time_ms,
                )
                self._wait_move()
                self.magnet_on()
                time.sleep(self.magnet_dwell_ms / 1000.0)
                self._publish_status(
                    {
                        "event": "pick_attached",
                        "run_dir": run_dir,
                        "piece_id": piece_id,
                        "index": index + 1,
                        "count": len(plan),
                    }
                )
                record("pick_attached", piece_id=piece_id)
                time.sleep(self.event_capture_hold_s)
                self.move_precomputed_pwms(
                    command["pwms"]["pick_travel"],
                    self.move_time_ms,
                )
                self._wait_move()

                # Yaw only after the magnet is safely above the paper.
                self.rotate_tool(place_tool_yaw, self.move_time_ms)
                self._wait_move()

                # PLACE: both high travel and release remain vertical.  The
                # +/-8 degree fallback is exclusive to pickup contact.
                self.move_precomputed_pwms(
                    command["pwms"]["place_travel"],
                    self.move_time_ms,
                )
                self._wait_move()
                self.move_precomputed_pwms(
                    command["pwms"]["place_drop"],
                    self.move_time_ms,
                )
                self._wait_move()
                self.magnet_off()
                time.sleep(self.magnet_dwell_ms / 1000.0)
                self._publish_status(
                    {
                        "event": "place_released",
                        "run_dir": run_dir,
                        "piece_id": piece_id,
                        "index": index + 1,
                        "count": len(plan),
                    }
                )
                record("place_released", piece_id=piece_id)
                time.sleep(self.event_capture_hold_s)
                self.move_precomputed_pwms(
                    command["pwms"]["place_travel"],
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
                staging_z,
                self.move_time_ms,
            )
            self._wait_move()
            self.park(1800)
            time.sleep(2.0)
            for beep_index in range(self.completion_beep_count):
                self.beep()
                if beep_index + 1 < self.completion_beep_count:
                    time.sleep(self.completion_beep_gap_s)
            if self.completion_beep_count:
                record(
                    "completion_beep",
                    count=self.completion_beep_count,
                )
                self.get_logger().info(
                    "Completion beeper sounded "
                    f"{self.completion_beep_count} times"
                )
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

    def _select_reachable_staging_z(self, requested_z_mm: float) -> float:
        """Use the highest reachable Z at the fixed safe staging point."""

        minimum_z = self.paper_surface_z_mm + self.min_travel_clearance_mm
        step = max(1.0, self.travel_search_step_mm)
        candidate = float(requested_z_mm)
        while candidate >= minimum_z - 1e-6:
            pwms = self.ik.solve_vertical(
                SAFE_STAGING_X_MM,
                SAFE_STAGING_Y_MM,
                candidate,
            )
            if pwms is not None and all(500 <= pwm <= 2500 for pwm in pwms):
                return candidate
            candidate -= step
        raise RuntimeError(
            "No reachable safe staging pose between "
            f"{minimum_z:.1f} and {requested_z_mm:.1f} mm"
        )

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

    def move_precomputed_pwms(self, pwms, time_ms: int) -> None:
        """Execute the exact IK result saved by the one-shot planner."""

        values = tuple(int(pwm) for pwm in pwms)
        if len(values) != 4:
            raise RuntimeError(
                f"Expected four precomputed joint PWMs, got {values}"
            )
        if any(pwm < 500 or pwm > 2500 for pwm in values):
            raise RuntimeError(f"Precomputed PWM out of range: {values}")
        frame = self.ik.build_frame(values, time_ms)
        if frame is None:
            raise RuntimeError("Unable to build precomputed IK frame")
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

    def beep(self) -> None:
        """Sound the controller-board passive buzzer once."""

        self.send_raw("$BEEP!\n")

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
