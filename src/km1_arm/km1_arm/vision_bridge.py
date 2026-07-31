#!/usr/bin/env python3
"""ROS 2 bridge from the real USB camera to a guarded KM1 control plan.

The bridge keeps control output disabled by default.  Every trigger still
captures, solves and archives diagnostics, but `/km1/control_plan` is only
published when the competition scene checks pass and
`enable_control_output:=true` was explicitly supplied.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


VISION_PATH = os.environ.get("PUZZLE_VISION_PATH", "")
if VISION_PATH:
    sys.path.insert(0, VISION_PATH)

from puzzle_vision import VisionConfig, rectify_a4, run_pipeline  # noqa: E402


class VisionBridge(Node):
    def __init__(self):
        super().__init__("km1_vision_bridge")
        default_config = (
            str(Path(VISION_PATH) / "config.json")
            if VISION_PATH and (Path(VISION_PATH) / "config.json").exists()
            else ""
        )
        default_log_dir = str(
            Path(VISION_PATH).parent / "vision_logs"
            if VISION_PATH
            else Path.cwd() / "vision_logs"
        )

        self.declare_parameter("camera_index", 0)
        self.declare_parameter("camera_width", 1920)
        self.declare_parameter("camera_height", 1080)
        self.declare_parameter("camera_fps", 30)
        self.declare_parameter("warmup_frames", 20)
        self.declare_parameter("median_frames", 3)
        self.declare_parameter("auto_trigger", False)
        self.declare_parameter("auto_trigger_delay_s", 7.0)
        self.declare_parameter("run_started_unix_s", 0.0)
        self.declare_parameter("mode", "auto")
        self.declare_parameter("competition_task", 2)
        self.declare_parameter("layout", "bench_right_to_left")
        self.declare_parameter("config_json", default_config)
        self.declare_parameter("diagnostic_dir", default_log_dir)
        self.declare_parameter("enable_control_output", False)

        self.cam_idx = int(self.get_parameter("camera_index").value)
        self.camera_width = int(self.get_parameter("camera_width").value)
        self.camera_height = int(self.get_parameter("camera_height").value)
        self.camera_fps = int(self.get_parameter("camera_fps").value)
        self.warmup_frames = int(self.get_parameter("warmup_frames").value)
        self.median_frames = max(1, int(self.get_parameter("median_frames").value))
        self.auto_trigger = bool(self.get_parameter("auto_trigger").value)
        self.auto_trigger_delay_s = float(
            self.get_parameter("auto_trigger_delay_s").value
        )
        configured_start = float(
            self.get_parameter("run_started_unix_s").value
        )
        self.run_started_unix_s = (
            configured_start if configured_start > 0.0 else time.time()
        )
        self.mode = str(self.get_parameter("mode").value)
        self.competition_task = int(
            self.get_parameter("competition_task").value
        )
        if self.competition_task not in (1, 2):
            raise ValueError("competition_task must be 1 or 2")
        self.layout = str(self.get_parameter("layout").value)
        self.enable_control_output = bool(
            self.get_parameter("enable_control_output").value
        )
        config_path = str(self.get_parameter("config_json").value)
        self.diagnostic_dir = Path(
            str(self.get_parameter("diagnostic_dir").value)
        )

        self.config = VisionConfig.from_json(config_path if config_path else None)
        if self.layout == "competition_portrait":
            self.config.require_portrait_input = True
        elif self.layout == "bench_right_to_left":
            self.config.require_portrait_input = False
            self.config.landscape_source_side = "right"
        else:
            raise ValueError(
                "layout must be 'competition_portrait' or "
                "'bench_right_to_left'"
            )
        self.pub_plan = self.create_publisher(String, "/km1/control_plan", 10)
        self.pub_status = self.create_publisher(String, "/km1/vision_status", 10)
        self.create_subscription(
            String,
            "/km1/vision_trigger",
            self.trigger_cb,
            10,
        )
        self.create_subscription(
            String,
            "/km1/control_status",
            self.control_status_cb,
            10,
        )

        self.capture_lock = threading.Lock()
        self.final_capture_lock = threading.Lock()
        self.event_capture_lock = threading.Lock()
        self.cap = None
        self.auto_triggered = False
        self.auto_timer = None
        self._open_camera()
        if self.auto_trigger:
            self.auto_timer = self.create_timer(
                max(0.5, self.auto_trigger_delay_s),
                self.auto_run,
            )

        state = "ENABLED" if self.enable_control_output else "LOCKED"
        self.get_logger().info(
            f"Vision bridge ready: camera={self.cam_idx}, "
            f"{self.camera_width}x{self.camera_height}@{self.camera_fps} MJPG, "
            f"task={self.competition_task}, "
            f"layout={self.layout}, "
            f"control_output={state}"
        )

    def _open_camera(self) -> bool:
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.cam_idx, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error(f"Cannot open camera {self.cam_idx}")
            return False
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(max(1, self.warmup_frames)):
            self.cap.read()
        return True

    def _capture_median(self) -> np.ndarray:
        if self.cap is None or not self.cap.isOpened():
            if not self._open_camera():
                raise RuntimeError(f"无法打开相机{self.cam_idx}")

        frames: list[np.ndarray] = []
        for _ in range(self.median_frames):
            ok, frame = self.cap.read()
            if ok and frame is not None:
                frames.append(frame)
        if not frames:
            raise RuntimeError("相机未返回有效帧")
        if len(frames) == 1:
            return frames[0]
        return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)

    def _new_run_dir(self) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = self.diagnostic_dir / stamp
        suffix = 1
        while run_dir.exists():
            run_dir = self.diagnostic_dir / f"{stamp}_{suffix:02d}"
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _publish_status(self, payload: dict) -> None:
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False)
        self.pub_status.publish(message)

    def _save_success(self, run_dir: Path, frame: np.ndarray, result: dict) -> None:
        cv2.imwrite(str(run_dir / "00_input.jpg"), frame)
        cv2.imwrite(str(run_dir / "00_input.png"), frame)
        cv2.imwrite(str(run_dir / "01_quad.jpg"), result["quad_debug"])
        cv2.imwrite(str(run_dir / "02_rectified.jpg"), result["rectified"])
        cv2.imwrite(str(run_dir / "02_rectified.png"), result["rectified"])
        cv2.imwrite(str(run_dir / "03_segmentation.png"), result["segmentation_mask"])
        cv2.imwrite(str(run_dir / "04_detection.jpg"), result["detection_overlay"])
        cv2.imwrite(str(run_dir / "05_solution.jpg"), result["solution_overlay"])
        cv2.imwrite(
            str(run_dir / "06_reconstructed_texture.png"),
            result["reconstructed_texture"],
        )
        data = {
            "competition_task": result["competition_task"],
            "scene_quality": result["scene_quality"],
            "vision_adaptation": result.get("vision_adaptation", {}),
            "pieces": [piece.to_summary() for piece in result["pieces"]],
            "solution": result["solution"].to_dict(result["pieces"]),
            "control_plan": result["control_plan"],
        }
        (run_dir / "result.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def trigger_cb(self, _msg):
        self.get_logger().info("Vision triggered")
        self.capture_and_solve()

    def auto_run(self):
        if self.auto_triggered:
            return
        self.auto_triggered = True
        if self.auto_timer is not None:
            self.auto_timer.cancel()
        self.capture_and_solve()

    def capture_and_solve(self):
        if not self.capture_lock.acquire(blocking=False):
            self.get_logger().warning("Vision is busy; trigger ignored")
            return

        run_dir = self._new_run_dir()
        frame = None
        try:
            frame = self._capture_median()
            self.get_logger().info(
                f"Captured {frame.shape[1]}x{frame.shape[0]}; running guarded pipeline"
            )
            result = run_pipeline(
                frame,
                self.config,
                mode=self.mode,
                competition_task=self.competition_task,
            )
            self._save_success(run_dir, frame, result)

            plan = result.get("control_plan", [])
            if not plan:
                raise RuntimeError("视觉成功但没有生成控制计划")

            status = {
                "ok": True,
                "published": self.enable_control_output,
                "competition_task": self.competition_task,
                "pieces": len(plan),
                "scene_quality": result["scene_quality"],
                "diagnostic_dir": str(run_dir),
                "elapsed_s": round(time.time() - self.run_started_unix_s, 3),
            }
            self._publish_status(status)

            if not self.enable_control_output:
                self.get_logger().warning(
                    f"Plan passed but control output is locked; diagnostics: {run_dir}"
                )
                return

            solution_data = result["solution"].to_dict(result["pieces"])
            envelope = {
                "schema_version": 2,
                "run_dir": str(run_dir),
                "run_started_unix_s": self.run_started_unix_s,
                "competition_task": self.competition_task,
                "pixels_per_mm": float(self.config.pixels_per_mm),
                "scene_quality": result["scene_quality"],
                "pieces": [
                    piece.to_summary() for piece in result["pieces"]
                ],
                "solution": solution_data,
                "control_plan": plan,
            }
            message = String()
            message.data = json.dumps(envelope, ensure_ascii=False)
            self.pub_plan.publish(message)
            self.get_logger().info(
                f"Published complete schema-v2 control envelope for "
                f"{len(plan)} pieces"
            )
        except Exception as exc:
            error = str(exc)
            if frame is not None:
                cv2.imwrite(str(run_dir / "00_input.jpg"), frame)
                cv2.imwrite(str(run_dir / "00_input.png"), frame)
            (run_dir / "error.txt").write_text(error + "\n", encoding="utf-8")
            self._publish_status(
                {
                    "ok": False,
                    "published": False,
                    "competition_task": self.competition_task,
                    "error": error,
                    "diagnostic_dir": str(run_dir),
                    "elapsed_s": round(
                        time.time() - self.run_started_unix_s,
                        3,
                    ),
                }
            )
            self.get_logger().error(f"Pipeline rejected scene: {error}")
        finally:
            self.capture_lock.release()

    def control_status_cb(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except json.JSONDecodeError:
            return
        event = status.get("event")
        if event in {"pick_attached", "place_released"}:
            run_dir = status.get("run_dir")
            if not run_dir:
                self.get_logger().error(
                    f"{event} status did not include run_dir"
                )
                return
            threading.Thread(
                target=self._capture_motion_event,
                args=(
                    Path(run_dir),
                    int(status.get("piece_id", -1)),
                    int(status.get("index", 0)),
                    str(event),
                ),
                daemon=True,
            ).start()
            return
        if event != "all_done":
            return
        run_dir = status.get("run_dir")
        if not run_dir:
            self.get_logger().error("all_done status did not include run_dir")
            return
        threading.Thread(
            target=self._capture_final_comparison,
            args=(Path(run_dir),),
            daemon=True,
        ).start()

    def _capture_motion_event(
        self,
        run_dir: Path,
        piece_id: int,
        index: int,
        phase: str,
    ) -> None:
        with self.event_capture_lock:
            try:
                with self.capture_lock:
                    frame = self._capture_median()

                phase_offset = 0 if phase == "pick_attached" else 1
                artifact_number = 18 + 2 * index + phase_offset
                stem = (
                    f"{artifact_number:02d}_step{index:02d}_p{piece_id}_"
                    f"{phase}"
                )
                full_path = run_dir / f"{stem}.jpg"
                cv2.imwrite(str(full_path), frame)

                try:
                    rectified, _ = rectify_a4(frame, self.config)
                    cv2.imwrite(
                        str(run_dir / f"{stem}_rectified.jpg"),
                        rectified,
                    )
                except Exception as error:
                    self.get_logger().warning(
                        f"P{piece_id} {phase} rectification skipped: {error}"
                    )

                comparison_path = None
                if phase == "place_released":
                    pick_number = 18 + 2 * index
                    pick_path = run_dir / (
                        f"{pick_number:02d}_step{index:02d}_p{piece_id}_"
                        "pick_attached.jpg"
                    )
                    pick_frame = cv2.imread(str(pick_path))
                    if pick_frame is not None:
                        if pick_frame.shape[:2] != frame.shape[:2]:
                            pick_frame = cv2.resize(
                                pick_frame,
                                (frame.shape[1], frame.shape[0]),
                                interpolation=cv2.INTER_LINEAR,
                            )
                        comparison = np.hstack(
                            [
                                self._label_panel(
                                    pick_frame,
                                    f"P{piece_id} PICK ATTACHED",
                                ),
                                self._label_panel(
                                    frame,
                                    f"P{piece_id} PLACE RELEASED",
                                ),
                            ]
                        )
                        comparison_path = run_dir / (
                            f"{artifact_number:02d}_step{index:02d}_p{piece_id}_"
                            "pick_vs_place.jpg"
                        )
                        cv2.imwrite(str(comparison_path), comparison)

                self._publish_status(
                    {
                        "ok": True,
                        "event": "motion_capture_ready",
                        "phase": phase,
                        "piece_id": piece_id,
                        "index": index,
                        "image": str(full_path),
                        "comparison": (
                            str(comparison_path)
                            if comparison_path is not None
                            else None
                        ),
                    }
                )
                self.get_logger().info(
                    f"Saved P{piece_id} {phase} capture: {full_path.name}"
                )
            except Exception as error:
                self.get_logger().error(
                    f"P{piece_id} {phase} capture failed: {error}"
                )

    @staticmethod
    def _label_panel(image: np.ndarray, text: str) -> np.ndarray:
        panel = image.copy()
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 58), (20, 20, 20), -1)
        cv2.putText(
            panel,
            text,
            (18, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.15,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        return panel

    def _capture_final_comparison(self, run_dir: Path) -> None:
        if not self.final_capture_lock.acquire(blocking=False):
            return
        try:
            # arm_controller emits all_done only after the park motion has
            # completed, so this frame is both the verification image and the
            # unobstructed final comparison.
            with self.capture_lock:
                frame = self._capture_median()
            cv2.imwrite(str(run_dir / "10_final_input.png"), frame)
            final_rectified, _ = rectify_a4(frame, self.config)
            cv2.imwrite(
                str(run_dir / "11_final_rectified.png"),
                final_rectified,
            )

            theoretical = cv2.imread(
                str(run_dir / "09_theoretical_target.png")
            )
            if theoretical is None:
                raise RuntimeError("09_theoretical_target.png is missing")
            if final_rectified.shape[:2] != theoretical.shape[:2]:
                final_rectified = cv2.resize(
                    final_rectified,
                    (theoretical.shape[1], theoretical.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            comparison = np.hstack(
                [
                    self._label_panel(theoretical, "THEORETICAL TARGET"),
                    self._label_panel(final_rectified, "FINAL CAMERA RESULT"),
                ]
            )
            cv2.imwrite(
                str(run_dir / "12_target_vs_actual.png"),
                comparison,
            )
            self._publish_status(
                {
                    "ok": True,
                    "event": "comparison_ready",
                    "diagnostic_dir": str(run_dir),
                    "comparison": str(
                        run_dir / "12_target_vs_actual.png"
                    ),
                    "elapsed_s": round(
                        time.time() - self.run_started_unix_s,
                        3,
                    ),
                }
            )
            self.get_logger().info(
                f"Final comparison saved: "
                f"{run_dir / '12_target_vs_actual.png'}"
            )
        except Exception as exc:
            self.get_logger().error(f"Final comparison failed: {exc}")
        finally:
            self.final_capture_lock.release()

    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        return super().destroy_node()


def main():
    rclpy.init()
    node = VisionBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
