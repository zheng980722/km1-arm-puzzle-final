"""Vertical-electromagnet motion planning for the KM1 puzzle workflow.

This module converts the complete vision result into executable grasp poses.
It deliberately lives in the existing ROS package so the production chain is:

    vision_bridge -> /km1/control_plan -> arm_controller -> serial_driver

No image-space point is estimated by the controller.  Every pose originates
from the millimetre-scale A4 result published by ``vision_bridge``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .control_geometry import (
    PAPER_DEPTH_MM,
    PAPER_WIDTH_MM,
    paper_to_robot,
    placement_command_paper,
    placement_to_robot,
)


PWM_MIN = 500
PWM_MAX = 2500
SOURCE_TARGET_DIVIDER_MM = PAPER_WIDTH_MM / 2.0
TOOL_YAW_LIMIT_DEG = 120.0
DEFAULT_LAYOUT_EDGE_MARGIN_MM = 2.0
DEFAULT_LAYOUT_SEARCH_STEP_MM = 1.0
DEFAULT_MIN_PWM_MARGIN_US = 50
DEFAULT_LAYOUT_CENTER_WEIGHT = 20.0


def _rotation_matrix(angle_deg: float) -> np.ndarray:
    angle = math.radians(float(angle_deg))
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, -s], [s, c]], dtype=np.float64)


def _normalise_angle(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def _vertical_pwms(
    ik,
    paper_point: np.ndarray,
    z_mm: float,
    *,
    placement: bool = False,
):
    transform = placement_to_robot if placement else paper_to_robot
    robot_x, robot_y = transform(*paper_point)
    pwms = ik.solve_vertical(robot_x, robot_y, z_mm)
    if pwms is None:
        return None
    if any(pwm < PWM_MIN or pwm > PWM_MAX for pwm in pwms):
        return None
    return tuple(int(pwm) for pwm in pwms)


def _select_reachable_grasp(
    polygon_mm: np.ndarray,
    center_mm: np.ndarray,
    ik,
    *,
    pick_z_mm: float,
    travel_z_mm: float,
    grid_step_mm: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Choose the deepest reachable point inside a source polygon."""

    polygon = np.asarray(polygon_mm, dtype=np.float32)
    center = np.asarray(center_mm, dtype=np.float64)
    minimum = np.floor(np.min(polygon, axis=0))
    maximum = np.ceil(np.max(polygon, axis=0))
    candidates: list[tuple[float, np.ndarray, float]] = []

    # The contour-area centroid best balances a thin steel fragment under the
    # circular magnet.  Once it has at least 4 mm of contour clearance, keep it
    # instead of moving far away merely to gain a few millimetres of inset.
    center_inset = float(
        cv2.pointPolygonTest(polygon, tuple(center.astype(float)), True)
    )
    if (
        center_inset >= 4.0
        and _vertical_pwms(ik, center, pick_z_mm) is not None
        and _vertical_pwms(ik, center, travel_z_mm) is not None
    ):
        return center.copy(), center_inset

    x_values = np.arange(
        minimum[0],
        maximum[0] + 0.5 * grid_step_mm,
        grid_step_mm,
    )
    y_values = np.arange(
        minimum[1],
        maximum[1] + 0.5 * grid_step_mm,
        grid_step_mm,
    )
    for x_mm in x_values:
        for y_mm in y_values:
            point = np.asarray([x_mm, y_mm], dtype=np.float64)
            inset = float(
                cv2.pointPolygonTest(
                    polygon,
                    (float(x_mm), float(y_mm)),
                    True,
                )
            )
            if inset < 2.0:
                continue
            if _vertical_pwms(ik, point, pick_z_mm) is None:
                continue
            if _vertical_pwms(ik, point, travel_z_mm) is None:
                continue
            # Maximise steel coverage under the 20 mm radius magnet, then
            # mildly prefer the vision centroid when inset is comparable.
            # Fallback is used only when the exact centroid is unreachable.
            # Keep the point as close as possible to the centroid while
            # retaining at least 2 mm of contour clearance.
            score = (
                -float(np.linalg.norm(point - center))
                + 0.02 * inset
            )
            candidates.append((score, point, inset))

    if not candidates:
        raise RuntimeError(
            "No vertically reachable point exists inside source polygon "
            f"{np.round(polygon, 2).tolist()}"
        )
    _, point, inset = max(candidates, key=lambda item: item[0])
    return point, float(inset)


def _select_reachable_layout_translation(
    drafts: list[dict[str, Any]],
    ik,
    *,
    travel_z_mm: float,
    drop_z_mm: float,
    edge_margin_mm: float,
    search_step_mm: float,
    min_pwm_margin_us: int,
    center_weight: float,
) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    """Find one rigid target-layout translation that is reachable for all pieces."""

    target_points = np.vstack(
        [np.asarray(draft["target_polygon"], dtype=np.float64) for draft in drafts]
    )
    minimum = np.min(target_points, axis=0)
    maximum = np.max(target_points, axis=0)
    preferred_center = np.asarray(
        [
            PAPER_DEPTH_MM / 2.0,
            0.5 * (SOURCE_TARGET_DIVIDER_MM + PAPER_WIDTH_MM),
        ],
        dtype=np.float64,
    )
    margin = max(0.0, float(edge_margin_mm))
    lower_translation = np.asarray(
        [
            margin - minimum[0],
            SOURCE_TARGET_DIVIDER_MM + margin - minimum[1],
        ],
        dtype=np.float64,
    )
    upper_translation = np.asarray(
        [
            PAPER_DEPTH_MM - margin - maximum[0],
            PAPER_WIDTH_MM - margin - maximum[1],
        ],
        dtype=np.float64,
    )
    if np.any(lower_translation > upper_translation):
        raise RuntimeError(
            "Solved target assembly does not fit inside the destination half: "
            f"bounds={np.round(minimum, 2).tolist()}.."
            f"{np.round(maximum, 2).tolist()}"
        )

    step = max(0.5, float(search_step_mm))
    starts = np.ceil(lower_translation / step) * step
    stops = np.floor(upper_translation / step) * step
    x_values = np.arange(starts[0], stops[0] + 0.5 * step, step)
    y_values = np.arange(starts[1], stops[1] + 0.5 * step, step)
    best: tuple[
        float,
        np.ndarray,
        dict[int, dict[str, Any]],
    ] | None = None

    for shift_x in x_values:
        for shift_y in y_values:
            translation = np.asarray([shift_x, shift_y], dtype=np.float64)
            solved_by_piece: dict[int, dict[str, Any]] = {}
            pulse_margin = float("inf")
            reachable = True
            for draft in drafts:
                point = np.asarray(
                    draft["destination_grasp"],
                    dtype=np.float64,
                ) + translation
                travel_pwms = _vertical_pwms(
                    ik,
                    point,
                    travel_z_mm,
                    placement=True,
                )
                drop_pwms = _vertical_pwms(
                    ik,
                    point,
                    drop_z_mm,
                    placement=True,
                )
                if travel_pwms is None or drop_pwms is None:
                    reachable = False
                    break
                for pwm in (*travel_pwms, *drop_pwms):
                    pulse_margin = min(
                        pulse_margin,
                        float(pwm - PWM_MIN),
                        float(PWM_MAX - pwm),
                    )
                pick_robot_x, pick_robot_y = paper_to_robot(
                    *draft["source_grasp"]
                )
                place_robot_x, place_robot_y = placement_to_robot(*point)
                place_command = placement_command_paper(*point)
                pick_base_deg = math.degrees(
                    math.atan2(pick_robot_x, pick_robot_y)
                )
                place_base_deg = math.degrees(
                    math.atan2(place_robot_x, place_robot_y)
                )
                base_rotation_deg = _normalise_angle(
                    place_base_deg - pick_base_deg
                )
                required_tool_delta_deg = _normalise_angle(
                    float(draft["rotation_delta_deg"]) - base_rotation_deg
                )
                # Splitting the relative tool motion evenly between pickup
                # and placement minimises the largest absolute ID4 angle.
                # Since the required relative rotation is normalised to
                # ±180 degrees, both endpoints remain inside ±90 degrees.
                pick_tool_yaw_deg = -0.5 * required_tool_delta_deg
                place_tool_yaw_deg = 0.5 * required_tool_delta_deg
                if (
                    abs(pick_tool_yaw_deg) > TOOL_YAW_LIMIT_DEG
                    or abs(place_tool_yaw_deg) > TOOL_YAW_LIMIT_DEG
                ):
                    reachable = False
                    break
                for yaw_deg in (pick_tool_yaw_deg, place_tool_yaw_deg):
                    yaw_pwm = 1500.0 + (2000.0 / 270.0) * yaw_deg
                    pulse_margin = min(
                        pulse_margin,
                        yaw_pwm - PWM_MIN,
                        PWM_MAX - yaw_pwm,
                    )
                solved_by_piece[int(draft["piece_id"])] = {
                    "place_travel": list(travel_pwms),
                    "place_drop": list(drop_pwms),
                    "place_command_paper": list(place_command),
                    "pick_tool_yaw_deg": pick_tool_yaw_deg,
                    "place_tool_yaw_deg": place_tool_yaw_deg,
                    "base_rotation_deg": base_rotation_deg,
                }
            if not reachable or pulse_margin < float(min_pwm_margin_us):
                continue

            shifted_minimum = minimum + translation
            shifted_maximum = maximum + translation
            boundary_clearance = min(
                shifted_minimum[0],
                shifted_minimum[1] - SOURCE_TARGET_DIVIDER_MM,
                PAPER_DEPTH_MM - shifted_maximum[0],
                PAPER_WIDTH_MM - shifted_maximum[1],
            )
            shifted_center = 0.5 * (
                shifted_minimum + shifted_maximum
            )
            center_distance = float(
                np.linalg.norm(shifted_center - preferred_center)
            )
            # Keep the pulse and edge guards, but prefer the centre of the
            # destination half so the assembly is not pushed unnecessarily
            # toward the near/bottom A4 edge.
            score = (
                pulse_margin
                + 2.0 * boundary_clearance
                - float(center_weight) * center_distance
            )
            if best is None or score > best[0]:
                best = (score, translation, solved_by_piece)

    if best is None:
        raise RuntimeError(
            "No common target translation is vertically reachable for every "
            "piece inside the destination half"
        )
    return best[1], best[2]


def build_vertical_control_plan(
    envelope: dict[str, Any],
    ik,
    *,
    paper_surface_z_mm: float,
    pick_clearance_mm: float = 0.0,
    travel_clearance_mm: float,
    drop_clearance_mm: float,
    grasp_validation_clearance_mm: float = 70.0,
    layout_edge_margin_mm: float = DEFAULT_LAYOUT_EDGE_MARGIN_MM,
    layout_search_step_mm: float = DEFAULT_LAYOUT_SEARCH_STEP_MM,
    min_pwm_margin_us: int = DEFAULT_MIN_PWM_MARGIN_US,
    layout_center_weight: float = DEFAULT_LAYOUT_CENTER_WEIGHT,
) -> list[dict[str, Any]]:
    """Build a complete, checked vertical-tool plan from a vision envelope."""

    pieces = {
        int(piece["piece_id"]): piece for piece in envelope["pieces"]
    }
    placements = {
        int(item["piece_id"]): item
        for item in envelope["solution"]["placements"]
    }
    sequence = [
        int(item["piece_id"]) for item in envelope["control_plan"]
    ]
    surface_z = float(paper_surface_z_mm)
    pick_z = surface_z + float(pick_clearance_mm)
    travel_z = surface_z + float(travel_clearance_mm)
    drop_z = surface_z + float(drop_clearance_mm)
    grasp_validation_z = surface_z + min(
        float(travel_clearance_mm),
        float(grasp_validation_clearance_mm),
    )
    drafts: list[dict[str, Any]] = []

    for piece_id in sequence:
        piece = pieces[piece_id]
        placement = placements[piece_id]
        source_center = np.asarray(piece["center_mm"], dtype=np.float64)
        source_polygon = np.asarray(piece["vertices_mm"], dtype=np.float64)
        source_grasp, inset = _select_reachable_grasp(
            source_polygon,
            source_center,
            ik,
            pick_z_mm=pick_z,
            travel_z_mm=grasp_validation_z,
        )

        rotation_delta = _normalise_angle(
            float(placement["rotation_delta_deg"])
        )

        target_center = (
            np.asarray(
                [
                    placement["target_pose"]["x_mm"],
                    placement["target_pose"]["y_mm"],
                ],
                dtype=np.float64,
            )
        )
        destination_grasp = (
            target_center
            + _rotation_matrix(rotation_delta)
            @ (source_grasp - source_center)
        )
        solved_pick: dict[str, list[int]] = {}
        for name, z_mm in (
            ("pick_travel", travel_z),
            ("pick_contact", pick_z),
        ):
            point = source_grasp
            pwms = _vertical_pwms(ik, point, z_mm)
            if pwms is None:
                raise RuntimeError(
                    f"Piece {piece_id} vertical pose {name} is unreachable: "
                    f"paper={np.round(point, 2).tolist()}, z={z_mm:.1f}"
                )
            solved_pick[name] = list(pwms)

        drafts.append(
            {
                "piece_id": piece_id,
                "source_grasp": source_grasp,
                "destination_grasp": destination_grasp,
                "source_center": source_center,
                "target_center": target_center,
                "source_polygon": source_polygon,
                "target_polygon": np.asarray(
                    placement["target_polygon_mm"],
                    dtype=np.float64,
                ),
                "grasp_inset_mm": inset,
                "rotation_delta_deg": rotation_delta,
                "pick_pwms": solved_pick,
            }
        )

    layout_translation, place_pwms = _select_reachable_layout_translation(
        drafts,
        ik,
        travel_z_mm=travel_z,
        drop_z_mm=drop_z,
        edge_margin_mm=layout_edge_margin_mm,
        search_step_mm=layout_search_step_mm,
        min_pwm_margin_us=min_pwm_margin_us,
        center_weight=layout_center_weight,
    )
    plan: list[dict[str, Any]] = []
    for draft in drafts:
        piece_id = int(draft["piece_id"])
        destination_grasp = (
            np.asarray(draft["destination_grasp"]) + layout_translation
        )
        destination_polygon = (
            np.asarray(draft["target_polygon"]) + layout_translation
        )
        target_center = np.asarray(draft["target_center"]) + layout_translation
        plan.append(
            {
                "piece_id": piece_id,
                "pick": np.round(draft["source_grasp"], 3).tolist(),
                "place": np.round(destination_grasp, 3).tolist(),
                "place_command": np.round(
                    place_pwms[piece_id]["place_command_paper"],
                    3,
                ).tolist(),
                "placement_compensation_mm": np.round(
                    np.asarray(
                        place_pwms[piece_id]["place_command_paper"],
                        dtype=np.float64,
                    )
                    - destination_grasp,
                    3,
                ).tolist(),
                "source_center": np.round(draft["source_center"], 3).tolist(),
                "target_center": np.round(target_center, 3).tolist(),
                "source_polygon": np.round(draft["source_polygon"], 3).tolist(),
                "target_polygon": np.round(destination_polygon, 3).tolist(),
                "layout_translation_mm": np.round(
                    layout_translation,
                    3,
                ).tolist(),
                "grasp_inset_mm": round(
                    float(draft["grasp_inset_mm"]),
                    3,
                ),
                "rotation_delta_deg": round(
                    float(draft["rotation_delta_deg"]),
                    3,
                ),
                "pick_tool_yaw_deg": round(
                    float(place_pwms[piece_id]["pick_tool_yaw_deg"]),
                    3,
                ),
                "place_tool_yaw_deg": round(
                    float(place_pwms[piece_id]["place_tool_yaw_deg"]),
                    3,
                ),
                "base_rotation_deg": round(
                    float(place_pwms[piece_id]["base_rotation_deg"]),
                    3,
                ),
                "pick_z_mm": pick_z,
                "travel_z_mm": travel_z,
                "drop_z_mm": drop_z,
                "vertical_alpha_deg": -90.0,
                "pwms": {
                    **draft["pick_pwms"],
                    "place_travel": place_pwms[piece_id]["place_travel"],
                    "place_drop": place_pwms[piece_id]["place_drop"],
                },
            }
        )
    return plan


def build_highest_vertical_control_plan(
    envelope: dict[str, Any],
    ik,
    *,
    paper_surface_z_mm: float,
    pick_clearance_mm: float,
    max_travel_clearance_mm: float,
    min_travel_clearance_mm: float,
    travel_search_step_mm: float,
    drop_clearance_mm: float,
    layout_edge_margin_mm: float = DEFAULT_LAYOUT_EDGE_MARGIN_MM,
    layout_search_step_mm: float = DEFAULT_LAYOUT_SEARCH_STEP_MM,
    min_pwm_margin_us: int = DEFAULT_MIN_PWM_MARGIN_US,
    layout_center_weight: float = DEFAULT_LAYOUT_CENTER_WEIGHT,
) -> list[dict[str, Any]]:
    """Return the highest complete plan reachable by every pick and place pose."""

    maximum = float(max_travel_clearance_mm)
    minimum = float(min_travel_clearance_mm)
    if maximum < minimum:
        maximum, minimum = minimum, maximum
    step = max(1.0, float(travel_search_step_mm))
    clearances = list(np.arange(maximum, minimum - 0.5 * step, -step))
    if not clearances or clearances[-1] > minimum + 1e-6:
        clearances.append(minimum)

    errors: list[str] = []
    for clearance in clearances:
        try:
            return build_vertical_control_plan(
                envelope,
                ik,
                paper_surface_z_mm=paper_surface_z_mm,
                pick_clearance_mm=pick_clearance_mm,
                travel_clearance_mm=float(clearance),
                drop_clearance_mm=drop_clearance_mm,
                grasp_validation_clearance_mm=minimum,
                layout_edge_margin_mm=layout_edge_margin_mm,
                layout_search_step_mm=layout_search_step_mm,
                min_pwm_margin_us=min_pwm_margin_us,
                layout_center_weight=layout_center_weight,
            )
        except RuntimeError as error:
            errors.append(f"{clearance:.1f} mm: {error}")
    raise RuntimeError(
        "No complete vertical plan is reachable between "
        f"{minimum:.1f} and {maximum:.1f} mm. "
        + " | ".join(errors)
    )


def save_vertical_plan_artifacts(
    run_dir: str | Path,
    plan: list[dict[str, Any]],
    *,
    pixels_per_mm: float,
) -> None:
    """Save the executable plan and target visualisations into the run log."""

    run_path = Path(run_dir)
    (run_path / "07_vertical_control_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rectified = cv2.imread(str(run_path / "02_rectified.png"))
    if rectified is None:
        rectified = cv2.imread(str(run_path / "02_rectified.jpg"))
    if rectified is None:
        rectified = cv2.imread(str(run_path / "01_rectified.png"))
    reconstructed = cv2.imread(str(run_path / "06_reconstructed_texture.png"))
    if reconstructed is None:
        reconstructed = cv2.imread(
            str(run_path / "05_reconstructed_texture.png")
        )
    if rectified is None or reconstructed is None:
        raise RuntimeError("Vision artifacts needed for plan rendering are missing")

    scale = float(pixels_per_mm)
    overlay = rectified.copy()
    colours = [
        (0, 90, 255),
        (255, 80, 0),
        (0, 180, 255),
        (180, 0, 255),
    ]
    for index, command in enumerate(plan):
        colour = colours[index % len(colours)]
        source = np.round(
            np.asarray(command["source_polygon"]) * scale
        ).astype(np.int32)
        target = np.round(
            np.asarray(command["target_polygon"]) * scale
        ).astype(np.int32)
        pick = tuple(
            np.round(np.asarray(command["pick"]) * scale).astype(int)
        )
        place = tuple(
            np.round(np.asarray(command["place"]) * scale).astype(int)
        )
        cv2.polylines(overlay, [source], True, colour, 4, cv2.LINE_AA)
        cv2.polylines(overlay, [target], True, colour, 4, cv2.LINE_AA)
        cv2.circle(overlay, pick, 14, colour, -1, cv2.LINE_AA)
        cv2.circle(overlay, place, 14, colour, -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"P{command['piece_id']} PICK",
            (pick[0] + 10, pick[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            colour,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"P{command['piece_id']} DROP",
            (place[0] + 10, place[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            colour,
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(run_path / "08_vertical_plan.png"), overlay)

    # Shift only the theoretical target half by the rigid translation selected
    # from this run's target dimensions and common vertical-IK reachable set.
    split_row = int(round(SOURCE_TARGET_DIVIDER_MM * scale))
    translation = np.asarray(
        plan[0].get("layout_translation_mm", [0.0, 0.0])
        if plan
        else [0.0, 0.0],
        dtype=np.float64,
    )
    shift_x_px = int(round(float(translation[0]) * scale))
    shift_y_px = int(round(float(translation[1]) * scale))
    theoretical = reconstructed.copy()
    lower = reconstructed[split_row:, :]
    shifted_lower = cv2.warpAffine(
        lower,
        np.asarray(
            [[1.0, 0.0, shift_x_px], [0.0, 1.0, shift_y_px]],
            dtype=np.float64,
        ),
        (lower.shape[1], lower.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    theoretical[split_row:, :] = shifted_lower
    cv2.imwrite(str(run_path / "09_theoretical_target.png"), theoretical)
