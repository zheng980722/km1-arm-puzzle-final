"""Run one complete camera-to-arm KM1 puzzle attempt."""

import time

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    workspace = "/home/wheeltec/WorkSpace/km1_arm_ws"
    puzzle_vision = f"{workspace}/puzzle_vision"
    run_started_unix_s = float(time.time())

    start_serial = LaunchConfiguration("start_serial_driver")
    enable_motion = LaunchConfiguration("enable_motion")
    camera_index = LaunchConfiguration("camera_index")
    serial_port = LaunchConfiguration("serial_port")
    diagnostic_dir = LaunchConfiguration("diagnostic_dir")

    return LaunchDescription(
        [
            DeclareLaunchArgument("start_serial_driver", default_value="true"),
            DeclareLaunchArgument("enable_motion", default_value="true"),
            DeclareLaunchArgument("camera_index", default_value="0"),
            DeclareLaunchArgument(
                "serial_port",
                default_value="/dev/ttyCH341USB0",
            ),
            DeclareLaunchArgument(
                "diagnostic_dir",
                default_value=f"{workspace}/vertical_runs",
            ),
            SetEnvironmentVariable("PUZZLE_VISION_PATH", puzzle_vision),
            Node(
                package="km1_arm",
                executable="serial_driver",
                name="km1_serial_driver",
                output="screen",
                condition=IfCondition(start_serial),
                parameters=[
                    {
                        "port": serial_port,
                        "baud_rate": 115200,
                        "boot_wait": 5.0,
                        "release_on_shutdown": False,
                    }
                ],
            ),
            Node(
                package="km1_arm",
                executable="arm_controller",
                name="km1_arm_controller",
                output="screen",
                parameters=[
                    {
                        "enable_automatic_motion": enable_motion,
                        # Raised paper is 30 mm above the floor while the
                        # robot base board is 10 mm above the floor.
                        "paper_surface_z_mm": 20.0,
                        "pick_clearance_mm": 20.0,
                        "tool_length_mm": 50.0,
                        "travel_clearance_mm": 150.0,
                        "min_travel_clearance_mm": 70.0,
                        "travel_search_step_mm": 5.0,
                        "drop_clearance_mm": 25.0,
                        "layout_edge_margin_mm": 2.0,
                        "layout_search_step_mm": 1.0,
                        "min_pwm_margin_us": 50,
                        "layout_center_weight": 20.0,
                        "max_run_time_s": 120.0,
                        "move_time_ms": 1000,
                        "magnet_dwell_ms": 350,
                    }
                ],
            ),
            Node(
                package="km1_arm",
                executable="vision_bridge",
                name="km1_vision_bridge",
                output="screen",
                parameters=[
                    {
                        "camera_index": camera_index,
                        "camera_width": 1920,
                        "camera_height": 1080,
                        "camera_fps": 30,
                        "auto_trigger": True,
                        "auto_trigger_delay_s": 7.0,
                        "run_started_unix_s": run_started_unix_s,
                        "mode": "auto",
                        "layout": "bench_right_to_left",
                        "config_json": f"{puzzle_vision}/config.json",
                        "diagnostic_dir": diagnostic_dir,
                        "enable_control_output": enable_motion,
                    }
                ],
            ),
        ]
    )
