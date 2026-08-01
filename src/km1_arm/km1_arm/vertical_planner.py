"""Vertical-electromagnet motion planning for the KM1 puzzle workflow.

This module converts the complete vision result into executable grasp poses.
It deliberately lives in the existing ROS package so the production chain is:

    vision_bridge -> /km1/control_plan -> arm_controller -> serial_driver

No image-space point is estimated by the controller.  Every pose originates
from the millimetre-scale A4 result published by ``vision_bridge``.
"""

from __future__ import annotations

import copy
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
)


PWM_MIN = 500
PWM_MAX = 2500
SOURCE_TARGET_DIVIDER_MM = PAPER_WIDTH_MM / 2.0
TOOL_YAW_LIMIT_DEG = 120.0
DEFAULT_LAYOUT_EDGE_MARGIN_MM = 2.0
DEFAULT_LAYOUT_NEAR_EDGE_MARGIN_MM = DEFAULT_LAYOUT_EDGE_MARGIN_MM
DEFAULT_LAYOUT_SEARCH_STEP_MM = 1.0
DEFAULT_MIN_PWM_MARGIN_US = 50
DEFAULT_LAYOUT_CENTER_WEIGHT = 20.0
# Preserve the vision orientation first, then try quarter-turn alternatives.
# A rigid rotation keeps every inter-piece gap unchanged while allowing a long
# reconstruction to follow the arm's radial workspace instead of crossing it.
LAYOUT_ROTATION_CANDIDATES_DEG = (0.0, -90.0, 90.0, 180.0)
COARSE_LAYOUT_SEARCH_STEP_MM = 4.0
PREFLIGHT_LAYOUT_SEARCH_STEP_MM = 2.0
VERTICAL_ALPHA_DEG = -90.0
# Pickup contact may deviate by at most +/-8 degrees from vertical.  The order
# is deliberate: exact vertical first, then the smallest symmetric deviation.
PICK_CONTACT_ALPHA_OFFSETS_DEG = tuple(
    offset
    for deviation in range(0, 9)
    for offset in ((0,) if deviation == 0 else (deviation, -deviation))
)
# High pickup approach/retreat poses do not determine the contact point, so
# they may use the documented KM1 downward working range.  Placement travel
# and release remain strictly vertical.
TRAVEL_FLATTEST_ALPHA_DEG = -25.0
ALPHA_SEARCH_STEP_DEG = 1.0
# One degree of contact tilt costs 0.75 mm in the grasp score.  This avoids
# trading a well-balanced centroid grasp for a nearly vertical point at the
# extreme fragment edge.
CONTACT_TILT_SCORE_WEIGHT = 0.75


def _rotation_matrix(angle_deg: float) -> np.ndarray:
    angle = math.radians(float(angle_deg))
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, -s], [s, c]], dtype=np.float64)


def _normalise_angle(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def _rotate_layout_envelope(
    envelope: dict[str, Any],
    angle_deg: float,
) -> dict[str, Any]:
    """Return a copy whose complete solved layout is rigidly rotated.

    Rotation is around the solved assembly's bounding-box centre.  The later
    translation search remains responsible for placing that assembly inside
    the destination half and the arm workspace.
    """

    rotated = copy.deepcopy(envelope)
    placements = rotated.get("solution", {}).get("placements", [])
    polygons = [
        np.asarray(item["target_polygon_mm"], dtype=np.float64)
        for item in placements
        if item.get("target_polygon_mm")
    ]
    if not polygons:
        rotated["layout_rotation_deg"] = float(angle_deg)
        return rotated

    all_points = np.vstack(polygons)
    pivot = 0.5 * (
        np.min(all_points, axis=0) + np.max(all_points, axis=0)
    )
    matrix = _rotation_matrix(angle_deg)
    for placement in placements:
        polygon = np.asarray(
            placement["target_polygon_mm"],
            dtype=np.float64,
        )
        target_pose = placement["target_pose"]
        center = np.asarray(
            [target_pose["x_mm"], target_pose["y_mm"]],
            dtype=np.float64,
        )
        rotated_polygon = (polygon - pivot) @ matrix.T + pivot
        rotated_center = matrix @ (center - pivot) + pivot
        placement["target_polygon_mm"] = rotated_polygon.tolist()
        target_pose["x_mm"] = float(rotated_center[0])
        target_pose["y_mm"] = float(rotated_center[1])
        if "theta_deg" in target_pose:
            target_pose["theta_deg"] = _normalise_angle(
                float(target_pose["theta_deg"]) + float(angle_deg)
            )
        placement["rotation_delta_deg"] = _normalise_angle(
            float(placement["rotation_delta_deg"]) + float(angle_deg)
        )
    rotated["layout_rotation_deg"] = float(angle_deg)
    return rotated


def _pose_pwms(
    ik,
    paper_point: np.ndarray,
    z_mm: float,
    alpha_deg: float,
):
    robot_x, robot_y = paper_to_robot(*paper_point)
    pwms = ik.solve(robot_x, robot_y, z_mm, float(alpha_deg))
    if pwms is None:
        return None
    if any(pwm < PWM_MIN or pwm > PWM_MAX for pwm in pwms):
        return None
    return tuple(int(pwm) for pwm in pwms)


def _vertical_pwms(ik, paper_point: np.ndarray, z_mm: float):
    return _pose_pwms(ik, paper_point, z_mm, VERTICAL_ALPHA_DEG)


def _select_downward_pose(
    ik,
    paper_point: np.ndarray,
    z_mm: float,
    *,
    flattest_alpha_deg: float,
) -> tuple[tuple[int, ...], float] | None:
    """Return the steepest downward pose that reaches one paper point.

    Candidates start at -90 degrees, so a vertical magnet is always selected
    when possible.  Progressively flatter poses are considered only when the
    previous pose has no valid IK solution.
    """

    flattest = min(
        -1.0,
        max(VERTICAL_ALPHA_DEG, float(flattest_alpha_deg)),
    )
    step = max(0.5, float(ALPHA_SEARCH_STEP_DEG))
    candidates = list(
        np.arange(VERTICAL_ALPHA_DEG, flattest + 0.5 * step, step)
    )
    if not candidates or candidates[-1] < flattest - 1e-6:
        candidates.append(flattest)
    for alpha_deg in candidates:
        pwms = _pose_pwms(ik, paper_point, z_mm, float(alpha_deg))
        if pwms is not None:
            return pwms, float(alpha_deg)
    return None


def _select_pick_contact_pose(
    ik,
    paper_point: np.ndarray,
    z_mm: float,
) -> tuple[tuple[int, ...], float] | None:
    """Select one precomputed pickup pose inside the +/-8 degree window."""

    for offset_deg in PICK_CONTACT_ALPHA_OFFSETS_DEG:
        alpha_deg = VERTICAL_ALPHA_DEG + float(offset_deg)
        pwms = _pose_pwms(ik, paper_point, z_mm, alpha_deg)
        if pwms is not None:
            return pwms, alpha_deg
    return None


def _select_reachable_grasp(
    polygon_mm: np.ndarray,
    center_mm: np.ndarray,
    ik,
    *,
    pick_z_mm: float,
    travel_z_mm: float,
    grid_step_mm: float = 1.0,
) -> tuple[np.ndarray, float, float]:
    """Choose an inset grasp, preferring a vertical contact pose."""

    polygon = np.asarray(polygon_mm, dtype=np.float32)
    center = np.asarray(center_mm, dtype=np.float64)
    minimum = np.floor(np.min(polygon, axis=0))
    maximum = np.ceil(np.max(polygon, axis=0))
    vertical_candidates: list[tuple[float, np.ndarray, float, float]] = []
    fallback_candidates: list[tuple[float, np.ndarray, float, float]] = []

    # A vertical centroid remains the best-balanced grasp and is accepted
    # immediately.  If it is not vertically reachable, include it in the
    # fallback search so nearby, steeper contact poses can win.
    center_inset = float(
        cv2.pointPolygonTest(polygon, tuple(center.astype(float)), True)
    )
    center_contact = _select_pick_contact_pose(
        ik,
        center,
        pick_z_mm,
    )
    center_travel = _select_downward_pose(
        ik,
        center,
        travel_z_mm,
        flattest_alpha_deg=TRAVEL_FLATTEST_ALPHA_DEG,
    )
    if (
        center_inset >= 4.0
        and center_contact is not None
        and center_contact[1] == VERTICAL_ALPHA_DEG
        and center_travel is not None
    ):
        return center.copy(), center_inset, center_contact[1]

    if (
        center_inset >= 2.0
        and center_contact is not None
        and center_travel is not None
    ):
        center_tilt = abs(center_contact[1] - VERTICAL_ALPHA_DEG)
        center_score = (
            0.02 * center_inset
            - CONTACT_TILT_SCORE_WEIGHT * center_tilt
        )
        candidate = (
            center_score,
            center.copy(),
            center_inset,
            center_contact[1],
        )
        if center_contact[1] == VERTICAL_ALPHA_DEG:
            vertical_candidates.append(candidate)
        else:
            fallback_candidates.append(candidate)

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
            contact_pose = _select_pick_contact_pose(
                ik,
                point,
                pick_z_mm,
            )
            if contact_pose is None:
                continue
            if _select_downward_pose(
                ik,
                point,
                travel_z_mm,
                flattest_alpha_deg=TRAVEL_FLATTEST_ALPHA_DEG,
            ) is None:
                continue
            # Contact pitch dominates the fallback score because a steeper
            # magnet reduces lateral offset.  Distance to the contour centroid
            # then keeps the steel fragment balanced under the 20 mm magnet.
            contact_alpha = contact_pose[1]
            contact_tilt = abs(contact_alpha - VERTICAL_ALPHA_DEG)
            score = (
                -float(np.linalg.norm(point - center))
                + 0.02 * inset
                - CONTACT_TILT_SCORE_WEIGHT * contact_tilt
            )
            candidate = (score, point, inset, contact_alpha)
            if contact_alpha == VERTICAL_ALPHA_DEG:
                vertical_candidates.append(candidate)
            else:
                fallback_candidates.append(candidate)

    # Exact vertical contact is a categorical priority. A tilted centroid must
    # never outrank a slightly offset point that can be picked at -90 degrees.
    candidates = vertical_candidates or fallback_candidates
    if not candidates:
        raise RuntimeError(
            "No downward-reachable point exists inside source polygon "
            f"{np.round(polygon, 2).tolist()}"
        )
    _, point, inset, contact_alpha = max(
        candidates,
        key=lambda item: item[0],
    )
    return point, float(inset), float(contact_alpha)


def _select_reachable_layout_translation(
    drafts: list[dict[str, Any]],
    ik,
    *,
    travel_z_mm: float,
    drop_z_mm: float,
    edge_margin_mm: float,
    near_edge_margin_mm: float,
    search_step_mm: float,
    min_pwm_margin_us: int,
    center_weight: float,
) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    """Find one rigid translation that moves the largest reachable subset."""

    preferred_center = np.asarray(
        [
            PAPER_DEPTH_MM / 2.0,
            0.5 * (SOURCE_TARGET_DIVIDER_MM + PAPER_WIDTH_MM),
        ],
        dtype=np.float64,
    )
    margin = max(0.0, float(edge_margin_mm))
    near_margin = max(margin, float(near_edge_margin_mm))
    piece_bounds: dict[
        int,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    valid_intervals: list[tuple[np.ndarray, np.ndarray]] = []
    for draft in drafts:
        piece_id = int(draft["piece_id"])
        polygon = np.asarray(draft["target_polygon"], dtype=np.float64)
        minimum = np.min(polygon, axis=0)
        maximum = np.max(polygon, axis=0)
        lower = np.asarray(
            [
                margin - minimum[0],
                SOURCE_TARGET_DIVIDER_MM + margin - minimum[1],
            ],
            dtype=np.float64,
        )
        upper = np.asarray(
            [
                PAPER_DEPTH_MM - near_margin - maximum[0],
                PAPER_WIDTH_MM - margin - maximum[1],
            ],
            dtype=np.float64,
        )
        piece_bounds[piece_id] = (minimum, maximum, lower, upper)
        if np.all(lower <= upper):
            valid_intervals.append((lower, upper))

    if len(valid_intervals) != len(drafts):
        raise RuntimeError(
            "Complete target layout cannot fit inside the destination half"
        )

    # The rotation applies to the complete reconstruction, not only to pieces
    # that happen to be reachable.  Intersect every fragment's legal shift so
    # every target polygon remains inside the A4 destination half even when a
    # later IK guard can move only a subset.
    lower_translation = np.max(
        np.vstack([interval[0] for interval in valid_intervals]),
        axis=0,
    )
    upper_translation = np.min(
        np.vstack([interval[1] for interval in valid_intervals]),
        axis=0,
    )
    if np.any(lower_translation > upper_translation):
        raise RuntimeError(
            "Complete target layout has no translation that stays inside "
            "the destination half"
        )

    step = max(0.5, float(search_step_mm))
    starts = np.ceil(lower_translation / step) * step
    stops = np.floor(upper_translation / step) * step
    x_values = np.arange(starts[0], stops[0] + 0.5 * step, step)
    y_values = np.arange(starts[1], stops[1] + 0.5 * step, step)
    best: tuple[
        int,
        float,
        np.ndarray,
        dict[int, dict[str, Any]],
    ] | None = None

    for shift_x in x_values:
        for shift_y in y_values:
            translation = np.asarray([shift_x, shift_y], dtype=np.float64)
            solved_by_piece: dict[int, dict[str, Any]] = {}
            pulse_margin = float("inf")
            for draft in drafts:
                piece_id = int(draft["piece_id"])
                minimum, maximum, lower, upper = piece_bounds[piece_id]
                if np.any(translation < lower) or np.any(translation > upper):
                    continue
                point = np.asarray(
                    draft["destination_grasp"],
                    dtype=np.float64,
                ) + translation
                travel_pwms = _vertical_pwms(
                    ik,
                    point,
                    travel_z_mm,
                )
                drop_pwms = _vertical_pwms(
                    ik,
                    point,
                    drop_z_mm,
                )
                if travel_pwms is None or drop_pwms is None:
                    continue
                piece_pulse_margin = float("inf")
                for pwm in (*travel_pwms, *drop_pwms):
                    piece_pulse_margin = min(
                        piece_pulse_margin,
                        float(pwm - PWM_MIN),
                        float(PWM_MAX - pwm),
                    )
                pick_robot_x, pick_robot_y = paper_to_robot(
                    *draft["source_grasp"]
                )
                place_robot_x, place_robot_y = paper_to_robot(*point)
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
                    continue
                for yaw_deg in (pick_tool_yaw_deg, place_tool_yaw_deg):
                    yaw_pwm = 1500.0 + (2000.0 / 270.0) * yaw_deg
                    piece_pulse_margin = min(
                        piece_pulse_margin,
                        yaw_pwm - PWM_MIN,
                        PWM_MAX - yaw_pwm,
                    )
                if piece_pulse_margin < float(min_pwm_margin_us):
                    continue
                pulse_margin = min(pulse_margin, piece_pulse_margin)
                solved_by_piece[piece_id] = {
                    "place_travel": list(travel_pwms),
                    "place_drop": list(drop_pwms),
                    "place_travel_alpha_deg": VERTICAL_ALPHA_DEG,
                    "place_drop_alpha_deg": VERTICAL_ALPHA_DEG,
                    "pick_tool_yaw_deg": pick_tool_yaw_deg,
                    "place_tool_yaw_deg": place_tool_yaw_deg,
                    "base_rotation_deg": base_rotation_deg,
                }
            if not solved_by_piece:
                continue

            solved_polygons = [
                np.asarray(draft["target_polygon"], dtype=np.float64)
                + translation
                for draft in drafts
                if int(draft["piece_id"]) in solved_by_piece
            ]
            solved_points = np.vstack(solved_polygons)
            shifted_minimum = np.min(solved_points, axis=0)
            shifted_maximum = np.max(solved_points, axis=0)
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
            reachable_count = len(solved_by_piece)
            if best is None or (reachable_count, score) > (best[0], best[1]):
                best = (
                    reachable_count,
                    score,
                    translation,
                    solved_by_piece,
                )

    if best is None:
        raise RuntimeError(
            "No target translation is vertically reachable for any piece "
            "inside the destination half"
        )
    return best[2], best[3]


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
    layout_near_edge_margin_mm: float = DEFAULT_LAYOUT_NEAR_EDGE_MARGIN_MM,
    layout_search_step_mm: float = DEFAULT_LAYOUT_SEARCH_STEP_MM,
    min_pwm_margin_us: int = DEFAULT_MIN_PWM_MARGIN_US,
    layout_center_weight: float = DEFAULT_LAYOUT_CENTER_WEIGHT,
) -> list[dict[str, Any]]:
    """Build a checked plan for every individually reachable piece."""

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
    skipped_reasons: dict[int, str] = {}

    for piece_id in sequence:
        piece = pieces[piece_id]
        placement = placements[piece_id]
        source_center = np.asarray(piece["center_mm"], dtype=np.float64)
        source_polygon = np.asarray(piece["vertices_mm"], dtype=np.float64)
        try:
            source_grasp, inset, pick_contact_alpha = _select_reachable_grasp(
                source_polygon,
                source_center,
                ik,
                pick_z_mm=pick_z,
                travel_z_mm=grasp_validation_z,
            )
        except RuntimeError as error:
            skipped_reasons[piece_id] = str(error)
            continue

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
        pick_travel_pose = _select_downward_pose(
            ik,
            source_grasp,
            travel_z,
            flattest_alpha_deg=TRAVEL_FLATTEST_ALPHA_DEG,
        )
        pick_contact_pwms = _pose_pwms(
            ik,
            source_grasp,
            pick_z,
            pick_contact_alpha,
        )
        if pick_travel_pose is None or pick_contact_pwms is None:
            skipped_reasons[piece_id] = (
                f"Piece {piece_id} downward pickup poses are unreachable: "
                f"paper={np.round(source_grasp, 2).tolist()}, "
                f"pick_z={pick_z:.1f}, travel_z={travel_z:.1f}"
            )
            continue
        pick_travel_pwms, pick_travel_alpha = pick_travel_pose

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
                "pick_pwms": {
                    "pick_travel": list(pick_travel_pwms),
                    "pick_contact": list(pick_contact_pwms),
                },
                "pick_contact_alpha_deg": pick_contact_alpha,
                "pick_travel_alpha_deg": pick_travel_alpha,
            }
        )

    if not drafts:
        details = " | ".join(
            f"piece {piece_id}: {reason}"
            for piece_id, reason in skipped_reasons.items()
        )
        raise RuntimeError(
            "No piece has a reachable pickup trajectory"
            + (f". {details}" if details else "")
        )

    # Move all exact-vertical pieces first. Preserve the vision order inside
    # the vertical and fallback classes.
    sequence_index = {
        piece_id: index for index, piece_id in enumerate(sequence)
    }
    drafts.sort(
        key=lambda draft: (
            abs(
                float(draft["pick_contact_alpha_deg"])
                - VERTICAL_ALPHA_DEG
            )
            > 1e-6,
            sequence_index[int(draft["piece_id"])],
        )
    )

    layout_translation, place_pwms = _select_reachable_layout_translation(
        drafts,
        ik,
        travel_z_mm=travel_z,
        drop_z_mm=drop_z,
        edge_margin_mm=layout_edge_margin_mm,
        near_edge_margin_mm=layout_near_edge_margin_mm,
        search_step_mm=layout_search_step_mm,
        min_pwm_margin_us=min_pwm_margin_us,
        center_weight=layout_center_weight,
    )
    for draft in drafts:
        piece_id = int(draft["piece_id"])
        if piece_id not in place_pwms:
            skipped_reasons[piece_id] = (
                "No destination trajectory satisfies the layout and PWM guards"
            )

    planned_piece_ids = [
        int(draft["piece_id"])
        for draft in drafts
        if int(draft["piece_id"]) in place_pwms
    ]
    skipped_piece_ids = [
        piece_id for piece_id in sequence if piece_id not in planned_piece_ids
    ]
    plan: list[dict[str, Any]] = []
    for draft in drafts:
        piece_id = int(draft["piece_id"])
        if piece_id not in place_pwms:
            continue
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
                "source_center": np.round(draft["source_center"], 3).tolist(),
                "target_center": np.round(target_center, 3).tolist(),
                "source_polygon": np.round(draft["source_polygon"], 3).tolist(),
                "target_polygon": np.round(destination_polygon, 3).tolist(),
                "layout_translation_mm": np.round(
                    layout_translation,
                    3,
                ).tolist(),
                "layout_rotation_deg": round(
                    float(envelope.get("layout_rotation_deg", 0.0)),
                    3,
                ),
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
                "pick_travel_alpha_deg": round(
                    float(draft["pick_travel_alpha_deg"]),
                    3,
                ),
                "pick_contact_alpha_deg": round(
                    float(draft["pick_contact_alpha_deg"]),
                    3,
                ),
                "place_travel_alpha_deg": round(
                    float(
                        place_pwms[piece_id]["place_travel_alpha_deg"]
                    ),
                    3,
                ),
                "place_drop_alpha_deg": round(
                    float(place_pwms[piece_id]["place_drop_alpha_deg"]),
                    3,
                ),
                "pick_z_mm": pick_z,
                "travel_z_mm": travel_z,
                "drop_z_mm": drop_z,
                "total_piece_count": len(sequence),
                "planned_piece_count": len(planned_piece_ids),
                "skipped_piece_ids": skipped_piece_ids,
                "skipped_reasons": {
                    str(skipped_id): skipped_reasons.get(
                        skipped_id,
                        "Unreachable",
                    )
                    for skipped_id in skipped_piece_ids
                },
                # Retained for old artifact readers.  Per-phase fields above
                # are authoritative for execution.
                "vertical_alpha_deg": VERTICAL_ALPHA_DEG,
                "pwms": {
                    **draft["pick_pwms"],
                    "place_travel": place_pwms[piece_id]["place_travel"],
                    "place_drop": place_pwms[piece_id]["place_drop"],
                },
            }
        )
    if not plan:
        raise RuntimeError("No piece has both reachable pickup and placement poses")
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
    layout_near_edge_margin_mm: float = DEFAULT_LAYOUT_NEAR_EDGE_MARGIN_MM,
    layout_search_step_mm: float = DEFAULT_LAYOUT_SEARCH_STEP_MM,
    min_pwm_margin_us: int = DEFAULT_MIN_PWM_MARGIN_US,
    layout_center_weight: float = DEFAULT_LAYOUT_CENTER_WEIGHT,
) -> list[dict[str, Any]]:
    """Return the most complete plan, then maximise travel clearance.

    The minimum-clearance preflight identifies which rigid quarter-turn
    layouts can move the most pieces.  Only those candidates are searched at
    higher clearances.  A 4 mm grid keeps this phase fast; the selected
    clearance and rotation are always recomputed on the requested fine grid
    before any command can be executed.
    """

    maximum = float(max_travel_clearance_mm)
    minimum = float(min_travel_clearance_mm)
    if maximum < minimum:
        maximum, minimum = minimum, maximum
    step = max(1.0, float(travel_search_step_mm))
    clearances = list(np.arange(maximum, minimum - 0.5 * step, -step))
    if not clearances or clearances[-1] > minimum + 1e-6:
        clearances.append(minimum)

    errors: list[str] = []
    fine_step = max(0.5, float(layout_search_step_mm))
    preflight_step = max(fine_step, PREFLIGHT_LAYOUT_SEARCH_STEP_MM)
    coarse_step = max(fine_step, COARSE_LAYOUT_SEARCH_STEP_MM)
    rotated_candidates = [
        (
            angle_deg,
            _rotate_layout_envelope(envelope, angle_deg),
        )
        for angle_deg in LAYOUT_ROTATION_CANDIDATES_DEG
    ]

    def build_candidate(
        candidate_envelope: dict[str, Any],
        clearance_mm: float,
        search_step_mm: float,
    ) -> list[dict[str, Any]]:
        return build_vertical_control_plan(
            candidate_envelope,
            ik,
            paper_surface_z_mm=paper_surface_z_mm,
            pick_clearance_mm=pick_clearance_mm,
            travel_clearance_mm=float(clearance_mm),
            drop_clearance_mm=drop_clearance_mm,
            grasp_validation_clearance_mm=minimum,
            layout_edge_margin_mm=layout_edge_margin_mm,
            layout_near_edge_margin_mm=layout_near_edge_margin_mm,
            layout_search_step_mm=search_step_mm,
            min_pwm_margin_us=min_pwm_margin_us,
            layout_center_weight=layout_center_weight,
        )

    # At minimum travel height the workspace is widest.  Its best piece count
    # is therefore the reachability ceiling that higher clearances must match.
    preflight: list[tuple[float, dict[str, Any], list[dict[str, Any]]]] = []
    best_piece_count = 0
    for angle_deg, candidate_envelope in rotated_candidates:
        try:
            plan = build_candidate(
                candidate_envelope,
                minimum,
                preflight_step,
            )
            count = len(plan)
            if count > best_piece_count:
                best_piece_count = count
                preflight = [(angle_deg, candidate_envelope, plan)]
            elif count == best_piece_count:
                preflight.append((angle_deg, candidate_envelope, plan))
        except RuntimeError as error:
            errors.append(
                f"preflight {angle_deg:+.0f} deg: {error}"
            )

    if not preflight or best_piece_count <= 0:
        raise RuntimeError(
            "No phase-aware plan can move any piece at minimum clearance. "
            + " | ".join(errors)
        )

    # Clearance is the primary safety preference once the maximum movable
    # piece count is known.  Candidate order keeps the original orientation
    # whenever it is equally capable.
    for clearance in clearances:
        for angle_deg, candidate_envelope, _ in preflight:
            try:
                coarse_plan = build_candidate(
                    candidate_envelope,
                    float(clearance),
                    coarse_step,
                )
                if len(coarse_plan) < best_piece_count:
                    continue
                if abs(coarse_step - fine_step) < 1e-9:
                    return coarse_plan
                fine_plan = build_candidate(
                    candidate_envelope,
                    float(clearance),
                    fine_step,
                )
                if len(fine_plan) >= best_piece_count:
                    return fine_plan
            except RuntimeError as error:
                errors.append(
                    f"{clearance:.1f} mm, {angle_deg:+.0f} deg: {error}"
                )

    # A coarse grid can miss a very narrow IK boundary.  Exact fine-grid
    # preflight is the deterministic fallback and still runs only once per
    # surviving rotation candidate.
    best_fine: list[dict[str, Any]] | None = None
    for angle_deg, candidate_envelope, preflight_plan in preflight:
        try:
            fine_plan = build_candidate(
                candidate_envelope,
                minimum,
                fine_step,
            )
        except RuntimeError as error:
            errors.append(
                f"fine fallback {angle_deg:+.0f} deg: {error}"
            )
            fine_plan = preflight_plan
        if best_fine is None or len(fine_plan) > len(best_fine):
            best_fine = fine_plan
    if best_fine:
        return best_fine
    raise RuntimeError(
        "No phase-aware plan can move any piece between "
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
