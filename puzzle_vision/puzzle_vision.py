"""Vision baseline for the 2026 E-problem puzzle device.

Coordinate convention
---------------------
Paper coordinates use millimetres.  The origin is the top-left corner of the
A4 sheet, +x points right, +y points down, and positive angles are clockwise
in the image/paper plane.

This module intentionally keeps hardware control behind a placeholder
function.  The vision result is a list of source/target poses that can later
be converted to UART, CAN, ROS 2, or a vendor-specific arm command.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import cv2
import numpy as np


def normalize_angle_deg(angle: float) -> float:
    """Normalize an angle to [-180, 180)."""

    return (float(angle) + 180.0) % 360.0 - 180.0


def rotation_matrix(angle_deg: float) -> np.ndarray:
    """2-D rotation for the paper coordinate system (+y points down)."""

    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def polygon_signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def ensure_consistent_winding(points: np.ndarray) -> np.ndarray:
    """Return all polygons with the same winding in image coordinates."""

    points = np.asarray(points, dtype=np.float64)
    if polygon_signed_area(points) < 0:
        return points[::-1].copy()
    return points.copy()


def polygon_area(points: np.ndarray) -> float:
    return abs(polygon_signed_area(points))


def simplify_polygon_vertices(
    points: np.ndarray,
    *,
    min_edge_mm: float = 8.0,
    collinear_distance_mm: float = 0.8,
) -> np.ndarray:
    """Remove tiny anti-aliasing edges and nearly collinear intermediate points."""

    polygon = [np.asarray(point, dtype=np.float64) for point in points]
    changed = True
    while changed and len(polygon) > 3:
        changed = False
        count = len(polygon)
        for index in range(count):
            previous = polygon[(index - 1) % count]
            current = polygon[index]
            following = polygon[(index + 1) % count]
            previous_length = float(np.linalg.norm(current - previous))
            next_length = float(np.linalg.norm(following - current))

            baseline = following - previous
            baseline_length = float(np.linalg.norm(baseline))
            if baseline_length > 1e-6:
                cross = baseline[0] * (current - previous)[1] - baseline[1] * (
                    current - previous
                )[0]
                distance = abs(float(cross)) / baseline_length
            else:
                distance = 0.0

            # A short edge alone is not enough reason to delete a vertex:
            # an anti-aliased intermediate point can sit close to a genuine
            # sharp corner.  Remove short-edge vertices only when they are
            # also close to the neighbouring baseline.
            short_and_nearly_collinear = (
                previous_length < min_edge_mm or next_length < min_edge_mm
            ) and distance < 2.0 * collinear_distance_mm
            if distance < collinear_distance_mm or short_and_nearly_collinear:
                polygon.pop(index)
                changed = True
                break
    return np.asarray(polygon, dtype=np.float64)


def transform_points(points: np.ndarray, angle_deg: float, translation: np.ndarray) -> np.ndarray:
    return points @ rotation_matrix(angle_deg).T + np.asarray(translation, dtype=np.float64)


def order_quad(points: np.ndarray) -> np.ndarray:
    """Order four image points as top-left, top-right, bottom-right, bottom-left."""

    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def pca_orientation_deg(points: np.ndarray) -> float:
    centered = points - np.mean(points, axis=0, keepdims=True)
    covariance = centered.T @ centered
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    angle = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    # A principal axis is directional only modulo 180 degrees.
    return normalize_angle_deg(angle if angle < 90.0 else angle - 180.0)


@dataclass
class VisionConfig:
    paper_width_mm: float = 210.0
    paper_height_mm: float = 297.0
    divider_y_mm: float = 148.5
    pixels_per_mm: float = 6.0

    target_center_x_mm: float = 105.0
    target_center_y_mm: float = 223.0
    placement_gap_mm: float = 5.0
    target_vertex_gap_mm: float = 5.0
    target_vertex_gap_tolerance_mm: float = 0.25
    max_allowed_vertex_gap_mm: float = 12.0
    max_post_placement_overlap_mm2: float = 0.0

    # OpenCV HSV values (H uses [0, 179]).
    paper_hsv_low: tuple[int, int, int] = (45, 65, 150)
    paper_hsv_high: tuple[int, int, int] = (65, 160, 240)
    background_lab_distance: float = 24.0

    border_ignore_mm: float = 2.0
    divider_ignore_mm: float = 2.5
    min_piece_area_mm2: float = 120.0
    min_piece_count: int = 2
    max_piece_count: int = 4
    max_piece_vertices: int = 5
    polygon_epsilon_mm: float = 0.5
    low_resolution_polygon_epsilon_mm: float = 5.0
    low_resolution_paper_short_edge_px: float = 400.0
    morphology_radius_mm: float = 0.30
    use_convex_hull_for_polygon: bool = False

    # Competition scene guards.  These reject an unsafe scene before a
    # control plan can be published.  The rules require a portrait A4 sheet,
    # source pieces in the upper half and an empty lower assembly area.
    require_portrait_input: bool = False
    landscape_source_side: str = "right"
    min_paper_area_ratio: float = 0.04
    min_paper_short_edge_px: float = 280.0
    min_lower_green_ratio: float = 0.78
    min_total_piece_area_mm2: float = 2200.0
    max_total_piece_area_mm2: float = 13000.0

    edge_length_tolerance_mm: float = 3.0
    edge_length_tolerance_ratio: float = 0.15
    overlap_tolerance_mm2: float = 12.0
    beam_width: int = 200
    dedupe_translation_mm: float = 3.0
    dedupe_angle_deg: float = 3.0
    allow_partial_edge_matching: bool = True
    partial_edge_min_mm: float = 10.0
    partial_edge_max_ratio: float = 3.5
    partial_match_heuristic_penalty: float = 0.35
    search_overlap_scale: float = 1.5
    max_rectangle_fill_error: float = 0.15
    outer_edge_tolerance_mm: float = 6.0
    max_solution_score: float = 12.0

    self_target_long_mm: float = 100.0
    self_target_short_mm: float = 60.0
    self_target_size_tolerance_mm: float = 5.0
    target_long_min_mm: float = 90.0
    target_long_max_mm: float = 120.0
    target_short_min_mm: float = 50.0
    target_short_max_mm: float = 90.0

    self_saturation_threshold: float = 55.0
    poker_texture_ratio_threshold: float = 0.025
    texture_weight: float = 1.4
    texture_sample_offset_mm: float = 1.1

    @property
    def rectified_width_px(self) -> int:
        return int(round(self.paper_width_mm * self.pixels_per_mm))

    @property
    def rectified_height_px(self) -> int:
        return int(round(self.paper_height_mm * self.pixels_per_mm))

    @classmethod
    def from_json(cls, path: Optional[str | Path]) -> "VisionConfig":
        config = cls()
        if path is None:
            return config
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for key, value in data.items():
            if not hasattr(config, key):
                raise ValueError(f"Unknown configuration field: {key}")
            current = getattr(config, key)
            if isinstance(current, tuple):
                value = tuple(value)
            setattr(config, key, value)
        return config


@dataclass
class PieceObservation:
    piece_id: int
    contour_px: np.ndarray
    polygon_mm: np.ndarray
    center_mm: np.ndarray
    local_polygon_mm: np.ndarray
    area_mm2: float
    orientation_deg: float
    median_saturation: float
    texture_ratio: float

    def edge(self, edge_index: int) -> tuple[np.ndarray, np.ndarray]:
        polygon = self.local_polygon_mm
        return polygon[edge_index], polygon[(edge_index + 1) % len(polygon)]

    def edge_length(self, edge_index: int) -> float:
        p0, p1 = self.edge(edge_index)
        return float(np.linalg.norm(p1 - p0))

    def to_summary(self) -> dict[str, Any]:
        return {
            "piece_id": self.piece_id,
            "center_mm": np.round(self.center_mm, 3).tolist(),
            "orientation_deg": round(self.orientation_deg, 3),
            "area_mm2": round(self.area_mm2, 3),
            "vertex_count": int(len(self.local_polygon_mm)),
            "vertices_mm": np.round(self.polygon_mm, 3).tolist(),
            "median_saturation": round(self.median_saturation, 3),
            "texture_ratio": round(self.texture_ratio, 5),
        }


@dataclass
class RigidTransform:
    angle_deg: float
    translation_mm: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        return transform_points(points, self.angle_deg, self.translation_mm)


@dataclass
class EdgeMatch:
    piece_a: int
    edge_a: int
    piece_b: int
    edge_b: int
    normalized_length_error: float
    anchor_mode: str = "both"


@dataclass
class AssemblyState:
    placements: dict[int, RigidTransform] = field(default_factory=dict)
    matches: list[EdgeMatch] = field(default_factory=list)
    edge_error: float = 0.0
    overlap_area_mm2: float = 0.0
    heuristic: float = 0.0


@dataclass
class PuzzleSolution:
    mode: str
    score: float
    geometric_score: float
    texture_score: float
    rectangle_size_mm: tuple[float, float]
    placement_extent_mm: tuple[float, float]
    applied_clearance_mm: float
    max_matched_vertex_gap_mm: float
    min_pairwise_gap_mm: float
    placement_overlap_area_mm2: float
    target_polygons_mm: dict[int, np.ndarray]
    target_transforms: dict[int, RigidTransform]
    matches: list[EdgeMatch]

    def to_dict(self, pieces: list[PieceObservation]) -> dict[str, Any]:
        observations = {piece.piece_id: piece for piece in pieces}
        placements: list[dict[str, Any]] = []
        for piece_id in sorted(self.target_transforms):
            transform = self.target_transforms[piece_id]
            piece = observations[piece_id]
            delta = normalize_angle_deg(transform.angle_deg)
            source_edges = np.linalg.norm(
                np.roll(piece.local_polygon_mm, -1, axis=0)
                - piece.local_polygon_mm,
                axis=1,
            )
            target_polygon = self.target_polygons_mm[piece_id]
            target_edges = np.linalg.norm(
                np.roll(target_polygon, -1, axis=0) - target_polygon,
                axis=1,
            )
            placements.append(
                {
                    "piece_id": piece_id,
                    "source_pose": {
                        "x_mm": round(float(piece.center_mm[0]), 3),
                        "y_mm": round(float(piece.center_mm[1]), 3),
                        "theta_deg": round(float(piece.orientation_deg), 3),
                    },
                    "target_pose": {
                        "x_mm": round(float(transform.translation_mm[0]), 3),
                        "y_mm": round(float(transform.translation_mm[1]), 3),
                        "theta_deg": round(normalize_angle_deg(piece.orientation_deg + delta), 3),
                    },
                    "rotation_delta_deg": round(delta, 3),
                    "grasp_point_source_mm": np.round(piece.center_mm, 3).tolist(),
                    "target_polygon_mm": np.round(target_polygon, 3).tolist(),
                    "shape_preservation": {
                        "source_edge_lengths_mm": np.round(source_edges, 3).tolist(),
                        "target_edge_lengths_mm": np.round(target_edges, 3).tolist(),
                        "source_area_mm2": round(
                            polygon_area(piece.local_polygon_mm), 3
                        ),
                        "target_area_mm2": round(polygon_area(target_polygon), 3),
                        "max_edge_length_error_mm": round(
                            float(np.max(np.abs(source_edges - target_edges))), 6
                        ),
                    },
                }
            )

        return {
            "mode": self.mode,
            "coordinate_convention": {
                "origin": "A4 top-left corner",
                "x": "right",
                "y": "down",
                "angle": "clockwise-positive degrees",
            },
            "score": round(self.score, 6),
            "geometric_score": round(self.geometric_score, 6),
            "texture_score": round(self.texture_score, 6),
            "rectangle_size_mm": [
                round(float(self.rectangle_size_mm[0]), 3),
                round(float(self.rectangle_size_mm[1]), 3),
            ],
            "placement_extent_mm": [
                round(float(self.placement_extent_mm[0]), 3),
                round(float(self.placement_extent_mm[1]), 3),
            ],
            "applied_clearance_mm": round(
                float(self.applied_clearance_mm),
                3,
            ),
            "max_matched_vertex_gap_mm": round(
                float(self.max_matched_vertex_gap_mm), 3
            ),
            "min_pairwise_gap_mm": round(
                float(self.min_pairwise_gap_mm), 3
            ),
            "placement_overlap_area_mm2": round(
                float(self.placement_overlap_area_mm2),
                6,
            ),
            "matches": [asdict(match) for match in self.matches],
            "placements": placements,
        }


def detect_a4_quad(image_bgr: np.ndarray, config: VisionConfig) -> np.ndarray:
    """Detect the recommended high-saturation green A4 sheet.

    Robustness improvements:
    - convexHull before approxPolyDP (piece stickers can bite notches into
      the green mask contour, making it non-convex)
    - Retry with a lower saturation floor when camera auto white balance
      desaturates the green sheet after a reboot
    - Multiple epsilon values (0.02~0.10) tried sequentially
    - minAreaRect fallback when no valid 4-vertex polygon is found
    """

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    low = np.array(config.paper_hsv_low, dtype=np.uint8)
    high = np.array(config.paper_hsv_high, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))

    image_area = float(image_bgr.shape[0] * image_bgr.shape[1])
    threshold_lows = [low]
    if int(low[1]) > 30:
        fallback_low = low.copy()
        fallback_low[1] = 30
        threshold_lows.append(fallback_low)

    expected_ratio = config.paper_width_mm / config.paper_height_mm

    # Thresholds are tried in priority order.  Do not mix fallback contours
    # with configured-threshold contours: on an uncompressed camera frame the
    # low-saturation fallback can connect grey table pixels to the paper and
    # win only because its incorrect hull is larger.
    for threshold_low in threshold_lows:
        mask = cv2.inRange(hsv, threshold_low, high)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )
        trial_contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        candidates: list[tuple[float, np.ndarray]] = []
        for contour in trial_contours:
            area = float(cv2.contourArea(contour))
            if area < config.min_paper_area_ratio * image_area:
                continue

            # Convex hull fills notches caused by piece stickers on the paper edge.
            hull = cv2.convexHull(contour)
            perimeter = cv2.arcLength(hull, True)

            found_quad = None
            for eps_factor in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10):
                approx = cv2.approxPolyDP(
                    hull,
                    eps_factor * perimeter,
                    True,
                )
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    found_quad = order_quad(approx.reshape(4, 2))
                    break

            if found_quad is None:
                rect = cv2.minAreaRect(hull)
                found_quad = order_quad(
                    cv2.boxPoints(rect).astype(np.float32)
                )

            quad = found_quad
            top = np.linalg.norm(quad[1] - quad[0])
            bottom = np.linalg.norm(quad[2] - quad[3])
            left = np.linalg.norm(quad[3] - quad[0])
            right = np.linalg.norm(quad[2] - quad[1])
            width = 0.5 * (top + bottom)
            height = 0.5 * (left + right)
            ratio = min(width, height) / max(width, height)
            ratio_error = abs(ratio - expected_ratio)
            if ratio_error > 0.20:
                continue
            candidates.append(
                (area - ratio_error * image_area, quad)
            )

        if candidates:
            return max(candidates, key=lambda item: item[0])[1]

    raise RuntimeError(
        "Unable to detect the coloured A4 sheet. Use --corners, adjust the HSV "
        "range in config.json, or pass --rectified for an already rectified image."
    )


def paper_orientation(quad: np.ndarray) -> str:
    """Return the physical paper orientation visible in the camera frame."""

    ordered = order_quad(quad)
    width = 0.5 * (
        np.linalg.norm(ordered[1] - ordered[0])
        + np.linalg.norm(ordered[2] - ordered[3])
    )
    height = 0.5 * (
        np.linalg.norm(ordered[3] - ordered[0])
        + np.linalg.norm(ordered[2] - ordered[1])
    )
    return "portrait" if height >= width else "landscape"


def config_for_paper_resolution(
    config: VisionConfig,
    detected_corners: Optional[np.ndarray],
) -> VisionConfig:
    """Return a per-frame copy with low-resolution contour simplification.

    The current overhead camera sees the A4 short edge at roughly 314 native
    pixels.  Perspective rectification upsamples that image but cannot create
    edge detail, so the stronger 5 mm simplification is needed there.  Clean
    already-rectified and high-resolution regression images keep the fine
    default epsilon and therefore retain legitimate five-edge fragments.
    """

    adapted = replace(config)
    if detected_corners is None:
        return adapted
    quad = order_quad(detected_corners)
    edge_lengths = [
        float(np.linalg.norm(quad[(index + 1) % 4] - quad[index]))
        for index in range(4)
    ]
    short_edge = 0.5 * sum(sorted(edge_lengths)[:2])
    if short_edge < config.low_resolution_paper_short_edge_px:
        adapted.polygon_epsilon_mm = config.low_resolution_polygon_epsilon_mm
    return adapted


def rectify_a4(
    image_bgr: np.ndarray,
    config: VisionConfig,
    corners: Optional[np.ndarray] = None,
    already_rectified: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a millimetre-scaled top-down A4 image and its homography.

    Auto-detects paper orientation: if the detected quad is wider than tall
    (landscape in image), rotates the mapping 90° CW so the output is always
    portrait (210mm wide x 297mm tall, divider at y=148.5mm).
    """

    width = config.rectified_width_px
    height = config.rectified_height_px
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )

    if already_rectified:
        source = np.array(
            [
                [0, 0],
                [image_bgr.shape[1] - 1, 0],
                [image_bgr.shape[1] - 1, image_bgr.shape[0] - 1],
                [0, image_bgr.shape[0] - 1],
            ],
            dtype=np.float32,
        )
    else:
        source = order_quad(corners) if corners is not None else detect_a4_quad(image_bgr, config)
        # Auto-detect orientation using A4 aspect ratio (297/210 ≈ 1.414).
        # Only rotate if the image clearly shows landscape (top/left > 1.2).
        # This avoids false rotation from perspective distortion.
        top_edge = np.linalg.norm(source[1] - source[0])
        left_edge = np.linalg.norm(source[3] - source[0])
        if left_edge > 1e-6 and (top_edge / left_edge) > 1.2:
            if config.landscape_source_side == "left":
                # Physical left half -> rectified upper half.
                source = np.array(
                    [source[3], source[0], source[1], source[2]],
                    dtype=np.float32,
                )
            elif config.landscape_source_side == "right":
                # Physical right half -> rectified upper half.
                source = np.array(
                    [source[1], source[2], source[3], source[0]],
                    dtype=np.float32,
                )
            else:
                raise ValueError(
                    "landscape_source_side must be 'left' or 'right'"
                )

    homography = cv2.getPerspectiveTransform(source, destination)
    rectified = cv2.warpPerspective(
        image_bgr,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rectified, homography


def estimate_background_lab(rectified_bgr: np.ndarray, config: VisionConfig) -> np.ndarray:
    lab = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    scale = config.pixels_per_mm
    sample_mask = np.zeros((h, w), dtype=np.uint8)

    border = max(2, int(round(6.0 * scale)))
    sample_mask[:border, :] = 255
    sample_mask[-border:, :] = 255
    sample_mask[:, :border] = 255
    sample_mask[:, -border:] = 255

    lower_start = int(round((config.divider_y_mm + 8.0) * scale))
    lower_end = int(round((config.paper_height_mm - 8.0) * scale))
    sample_mask[lower_start:lower_end, border:-border] = 255

    samples = lab[sample_mask > 0]
    if len(samples) < 100:
        raise RuntimeError("Not enough background pixels for colour estimation.")

    # The median is robust against the black divider and external markers.
    return np.median(samples, axis=0)


def _piece_from_contour(
    contour: np.ndarray,
    rectified_bgr: np.ndarray,
    piece_id: int,
    config: VisionConfig,
) -> PieceObservation:
    scale = config.pixels_per_mm
    geometry_contour = (
        cv2.convexHull(contour) if config.use_convex_hull_for_polygon else contour
    )
    area_mm2 = float(cv2.contourArea(geometry_contour)) / (scale * scale)

    moments = cv2.moments(geometry_contour)
    if abs(moments["m00"]) < 1e-8:
        center_px = np.mean(geometry_contour.reshape(-1, 2), axis=0)
    else:
        center_px = np.array(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
            dtype=np.float64,
        )
    center_mm = center_px / scale

    epsilon_px = max(1.0, config.polygon_epsilon_mm * scale)
    approx = (
        cv2.approxPolyDP(geometry_contour, epsilon_px, True)
        .reshape(-1, 2)
        .astype(np.float64)
    )
    # If paper wrinkles or antialiasing create too many corners, use the
    # smallest additional epsilon that brings the observation back inside the
    # rule's per-piece vertex limit.  This avoids accepting a noisy 6/7-corner
    # contour while preserving legitimate 5-corner fragments.
    extra_epsilon = epsilon_px
    while (
        len(approx) > config.max_piece_vertices
        and extra_epsilon < 5.0 * scale
    ):
        extra_epsilon *= 1.25
        approx = (
            cv2.approxPolyDP(geometry_contour, extra_epsilon, True)
            .reshape(-1, 2)
            .astype(np.float64)
        )
    # A strong low-resolution epsilon can collapse a small legal fragment to
    # a line.  Recover with progressively finer approximations instead of
    # rejecting the entire frame because one segmentation candidate used an
    # over-large epsilon.
    recovery_epsilon = extra_epsilon
    while len(approx) < 3 and recovery_epsilon > 1.0:
        recovery_epsilon = max(1.0, recovery_epsilon * 0.5)
        approx = (
            cv2.approxPolyDP(
                geometry_contour,
                recovery_epsilon,
                True,
            )
            .reshape(-1, 2)
            .astype(np.float64)
        )
    if len(approx) < 3:
        raise RuntimeError(f"Piece {piece_id} has fewer than three polygon vertices.")

    polygon_mm = simplify_polygon_vertices(approx / scale)
    polygon_mm = ensure_consistent_winding(polygon_mm)
    local_polygon_mm = polygon_mm - center_mm
    orientation_deg = pca_orientation_deg(polygon_mm)

    mask = np.zeros(rectified_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    hsv = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2HSV)
    piece_pixels = hsv[mask > 0]
    median_saturation = float(np.median(piece_pixels[:, 1]))

    hue = piece_pixels[:, 0]
    saturation = piece_pixels[:, 1]
    value = piece_pixels[:, 2]
    dark = value < 150
    red = ((hue < 12) | (hue > 168)) & (saturation > 70) & (value > 45)
    texture_ratio = float(np.mean(dark | red))

    return PieceObservation(
        piece_id=piece_id,
        contour_px=contour,
        polygon_mm=polygon_mm,
        center_mm=center_mm,
        local_polygon_mm=local_polygon_mm,
        area_mm2=area_mm2,
        orientation_deg=orientation_deg,
        median_saturation=median_saturation,
        texture_ratio=texture_ratio,
    )


def _green_inverse_foreground(
    rectified_bgr: np.ndarray,
    config: VisionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract foreground as holes within the green paper region.

    Returns (foreground_mask, green_mask) for debugging.
    """

    hsv = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2HSV)
    low = np.array(config.paper_hsv_low, dtype=np.uint8)
    high = np.array(config.paper_hsv_high, dtype=np.uint8)
    green_mask = cv2.inRange(hsv, low, high)

    # Morphological close to bridge small gaps in the green mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    green_closed = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Find the largest contour (the paper) and fill it solid
    contours, _ = cv2.findContours(green_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(green_mask.shape, dtype=np.uint8), green_mask

    largest = max(contours, key=cv2.contourArea)
    paper_region = np.zeros_like(green_mask)
    cv2.drawContours(paper_region, [largest], -1, 255, thickness=cv2.FILLED)

    # Foreground = inside paper but NOT green → pieces
    # Use the original (non-closed) green_mask so we don't lose piece pixels
    # that were bridged by the close operation
    foreground = cv2.bitwise_and(paper_region, cv2.bitwise_not(green_mask))
    return foreground, green_mask


def segment_pieces(
    rectified_bgr: np.ndarray,
    config: VisionConfig,
) -> tuple[list[PieceObservation], np.ndarray, np.ndarray, np.ndarray]:
    """Segment pieces using the best of Lab-distance and green-inverse masks.

    Lab-distance preserves green/teal artwork printed on a card.  Green-inverse
    is more stable on the real bench when illumination gradients make a large
    part of the green paper look Lab-different.  Both candidates are evaluated
    and the rule-plausible one is selected instead of trusting the first mask
    that happens to contain a contour.

    Returns (pieces, foreground_mask, background_lab, green_mask).
    """

    scale = config.pixels_per_mm
    h, w = rectified_bgr.shape[:2]

    # Valid region: upper half only (above divider)
    valid_region = np.zeros((h, w), dtype=np.uint8)
    border = int(round(config.border_ignore_mm * scale))
    divider = int(round((config.divider_y_mm - config.divider_ignore_mm) * scale))
    valid_region[border:max(border, divider), border : w - border] = 255

    # Morphology kernel
    radius = max(1, int(round(config.morphology_radius_mm * scale)))
    kernel_size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Primary: Lab colour distance (colour-agnostic, robust to green reflection)
    lab = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    background_lab = estimate_background_lab(rectified_bgr, config)
    delta = np.linalg.norm(lab - background_lab.reshape(1, 1, 3), axis=2)
    lab_foreground = (
        (delta >= config.background_lab_distance).astype(np.uint8) * 255
    )
    # Do not subtract the HSV-green mask here.  Court-card artwork can contain
    # green/teal pixels; subtracting them bites holes into the external piece
    # contour and makes a legal 4/5-corner fragment appear to have 6+ corners.
    # Lab distance already separates the uniform paper background.  The
    # green-inverse method remains available below as a fallback only.
    green_foreground, green_mask = _green_inverse_foreground(rectified_bgr, config)

    def clean_mask(mask: np.ndarray) -> np.ndarray:
        cleaned = cv2.bitwise_and(mask, valid_region)
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )
        return cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )

    def contours_for(mask: np.ndarray) -> list[np.ndarray]:
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        filtered = [
            contour for contour in contours
            if cv2.contourArea(contour) / (scale * scale)
            >= config.min_piece_area_mm2
        ]
        filtered.sort(key=cv2.contourArea, reverse=True)
        return filtered

    # Rule-based candidate selection: a valid source scene contains 2–4
    # components with a plausible total metal area.  This makes the real
    # illumination-gradient failure (one giant Lab component) lose to the
    # stable green-inverse result, while court-card cases prefer Lab.
    candidates: list[
        tuple[float, np.ndarray, list[np.ndarray], list[PieceObservation]]
    ] = []
    lab_non_green = cv2.bitwise_and(
        lab_foreground,
        cv2.bitwise_not(green_mask),
    )
    for method_bias, candidate_mask in (
        (0.0, clean_mask(lab_foreground)),
        (0.05, clean_mask(lab_non_green)),
        (0.1, clean_mask(green_foreground)),
    ):
        candidate_contours = contours_for(candidate_mask)
        candidate_pieces: list[PieceObservation] = []
        invalid_candidate = False
        for piece_id, contour in enumerate(candidate_contours):
            try:
                candidate_pieces.append(
                    _piece_from_contour(
                        contour,
                        rectified_bgr,
                        piece_id,
                        config,
                    )
                )
            except RuntimeError:
                invalid_candidate = True
                break
        if invalid_candidate:
            continue
        count = len(candidate_pieces)
        total_area = sum(piece.area_mm2 for piece in candidate_pieces)
        count_penalty = 100.0 * (
            max(0, config.min_piece_count - count)
            + max(0, count - config.max_piece_count)
        )
        area_penalty = 0.0
        if total_area < config.min_total_piece_area_mm2:
            area_penalty = (
                config.min_total_piece_area_mm2 - total_area
            ) / 100.0
        elif total_area > config.max_total_piece_area_mm2:
            area_penalty = (
                total_area - config.max_total_piece_area_mm2
            ) / 100.0
        vertex_penalty = 20.0 * sum(
            max(0, len(piece.local_polygon_mm) - config.max_piece_vertices)
            for piece in candidate_pieces
        )
        candidates.append(
            (
                method_bias + count_penalty + area_penalty + vertex_penalty,
                candidate_mask,
                candidate_contours,
                candidate_pieces,
            )
        )

    if not candidates:
        raise RuntimeError(
            "All segmentation candidates produced degenerate piece polygons."
        )

    _, foreground, _, pieces = min(candidates, key=lambda candidate: candidate[0])

    return pieces, foreground, background_lab, green_mask


def evaluate_scene_quality(
    image_bgr: np.ndarray,
    rectified_bgr: np.ndarray,
    detected_corners: Optional[np.ndarray],
    pieces: list[PieceObservation],
    green_mask: np.ndarray,
    config: VisionConfig,
) -> dict[str, Any]:
    """Evaluate competition constraints before solving or publishing control.

    The checks deliberately prefer a clear rejection over a plausible but
    unsafe pose.  In particular, a hand/arm in the lower half or two touching
    pieces must never be interpreted as a valid fragment.
    """

    issues: list[str] = []
    orientation = "rectified"
    paper_area_ratio = 1.0
    paper_short_edge_px = float(
        min(config.rectified_width_px, config.rectified_height_px)
    )

    if detected_corners is not None:
        quad = order_quad(detected_corners)
        orientation = paper_orientation(quad)
        paper_area_ratio = abs(float(cv2.contourArea(quad))) / float(
            image_bgr.shape[0] * image_bgr.shape[1]
        )
        edge_lengths = [
            float(np.linalg.norm(quad[(index + 1) % 4] - quad[index]))
            for index in range(4)
        ]
        paper_short_edge_px = 0.5 * sum(sorted(edge_lengths)[:2])
        if config.require_portrait_input and orientation != "portrait":
            issues.append("规则要求纵向A4，当前检测为横向放置")
        if paper_area_ratio < config.min_paper_area_ratio:
            issues.append(
                f"A4占画面比例过小({paper_area_ratio:.1%})"
            )
        if paper_short_edge_px < config.min_paper_short_edge_px:
            issues.append(
                f"A4短边仅{paper_short_edge_px:.0f}px，视觉精度不足"
            )

    scale = config.pixels_per_mm
    border = max(1, int(round(max(4.0, config.border_ignore_mm) * scale)))
    lower_start = int(
        round((config.divider_y_mm + config.divider_ignore_mm) * scale)
    )
    lower_roi = green_mask[
        lower_start : rectified_bgr.shape[0] - border,
        border : rectified_bgr.shape[1] - border,
    ]
    lower_green_ratio = (
        float(np.count_nonzero(lower_roi)) / float(lower_roi.size)
        if lower_roi.size
        else 0.0
    )
    if lower_green_ratio < config.min_lower_green_ratio:
        issues.append(
            "下半拼装区被机械臂、手或杂物遮挡"
            f"(绿色可见率{lower_green_ratio:.1%})"
        )

    piece_count = len(pieces)
    if piece_count < config.min_piece_count or piece_count > config.max_piece_count:
        issues.append(
            f"碎片数={piece_count}，规则范围应为"
            f"{config.min_piece_count}~{config.max_piece_count}"
        )

    excessive_vertices = [
        piece.piece_id
        for piece in pieces
        if len(piece.local_polygon_mm) > config.max_piece_vertices
    ]
    if excessive_vertices:
        issues.append(
            "以下轮廓超过每片5边，可能有碎片贴连或遮挡: "
            + ", ".join(f"P{piece_id}" for piece_id in excessive_vertices)
        )

    total_piece_area_mm2 = float(sum(piece.area_mm2 for piece in pieces))
    if total_piece_area_mm2 < config.min_total_piece_area_mm2:
        issues.append(
            f"碎片总面积过小({total_piece_area_mm2:.0f}mm^2)"
        )
    if total_piece_area_mm2 > config.max_total_piece_area_mm2:
        issues.append(
            f"碎片总面积过大({total_piece_area_mm2:.0f}mm^2)，疑似机械臂或多片粘连"
        )

    return {
        "passed": not issues,
        "issues": issues,
        "paper_orientation": orientation,
        "paper_area_ratio": paper_area_ratio,
        "paper_short_edge_px": paper_short_edge_px,
        "lower_green_ratio": lower_green_ratio,
        "piece_count": piece_count,
        "piece_vertex_counts": [
            int(len(piece.local_polygon_mm)) for piece in pieces
        ],
        "total_piece_area_mm2": total_piece_area_mm2,
    }


def infer_mode(pieces: list[PieceObservation], config: VisionConfig) -> str:
    if not pieces:
        raise RuntimeError("No puzzle pieces were detected.")
    weighted_saturation = sum(p.median_saturation * p.area_mm2 for p in pieces) / sum(
        p.area_mm2 for p in pieces
    )
    weighted_texture = sum(p.texture_ratio * p.area_mm2 for p in pieces) / sum(
        p.area_mm2 for p in pieces
    )
    if weighted_saturation >= config.self_saturation_threshold:
        return "self"
    if weighted_texture >= config.poker_texture_ratio_threshold:
        return "poker"
    return "white"


def _edge_alignment_transform(
    fixed_start: np.ndarray,
    fixed_end: np.ndarray,
    moving_start: np.ndarray,
    moving_end: np.ndarray,
) -> RigidTransform:
    """Align moving edge to the fixed edge in the reverse direction."""

    moving_vector = moving_end - moving_start
    target_vector = fixed_start - fixed_end
    source_angle = math.degrees(math.atan2(float(moving_vector[1]), float(moving_vector[0])))
    target_angle = math.degrees(math.atan2(float(target_vector[1]), float(target_vector[0])))
    angle = normalize_angle_deg(target_angle - source_angle)
    rotated_start = rotation_matrix(angle) @ moving_start
    translation = fixed_end - rotated_start
    return RigidTransform(angle_deg=angle, translation_mm=translation)


def _edge_alignment_candidates(
    fixed_start: np.ndarray,
    fixed_end: np.ndarray,
    moving_start: np.ndarray,
    moving_end: np.ndarray,
    config: VisionConfig,
) -> list[tuple[RigidTransform, str, float]]:
    """Return full-edge or endpoint-anchored partial-edge alignments.

    A legal random cut can create a T junction: one long straight edge on one
    fragment then corresponds to two shorter collinear edges on two other
    fragments.  Full-edge-only matching cannot represent that topology.
    """

    fixed_length = float(np.linalg.norm(fixed_end - fixed_start))
    moving_length = float(np.linalg.norm(moving_end - moving_start))
    difference = abs(fixed_length - moving_length)
    full_tolerance = max(
        config.edge_length_tolerance_mm,
        config.edge_length_tolerance_ratio * max(fixed_length, moving_length),
    )
    results: list[tuple[RigidTransform, str, float]] = []
    if difference <= full_tolerance:
        results.append(
            (
                _edge_alignment_transform(
                    fixed_start,
                    fixed_end,
                    moving_start,
                    moving_end,
                ),
                "both",
                difference / max(1.0, fixed_length, moving_length),
            )
        )

    if not config.allow_partial_edge_matching:
        return results
    shorter = min(fixed_length, moving_length)
    longer = max(fixed_length, moving_length)
    if (
        shorter < config.partial_edge_min_mm
        or longer / max(shorter, 1e-6) > config.partial_edge_max_ratio
    ):
        return results

    moving_vector = moving_end - moving_start
    target_vector = fixed_start - fixed_end
    source_angle = math.degrees(
        math.atan2(float(moving_vector[1]), float(moving_vector[0]))
    )
    target_angle = math.degrees(
        math.atan2(float(target_vector[1]), float(target_vector[0]))
    )
    angle = normalize_angle_deg(target_angle - source_angle)
    rotation = rotation_matrix(angle)
    rotated_start = rotation @ moving_start
    rotated_end = rotation @ moving_end
    endpoint_candidates = [
        (
            RigidTransform(
                angle_deg=angle,
                translation_mm=fixed_end - rotated_start,
            ),
            "fixed_end_to_moving_start",
            0.0,
        ),
        (
            RigidTransform(
                angle_deg=angle,
                translation_mm=fixed_start - rotated_end,
            ),
            "fixed_start_to_moving_end",
            0.0,
        ),
    ]
    for candidate in endpoint_candidates:
        transform = candidate[0]
        if any(
            abs(normalize_angle_deg(transform.angle_deg - existing[0].angle_deg))
            < 0.2
            and np.linalg.norm(
                transform.translation_mm - existing[0].translation_mm
            )
            < 0.5
            for existing in results
        ):
            continue
        results.append(candidate)
    return results


def _raster_overlap_area(polygons: Iterable[np.ndarray], scale: float = 3.0) -> float:
    polygons = [np.asarray(polygon, dtype=np.float64) for polygon in polygons]
    all_points = np.concatenate(polygons, axis=0)
    minimum = np.floor(np.min(all_points, axis=0) - 2.0)
    maximum = np.ceil(np.max(all_points, axis=0) + 2.0)
    width = max(8, int(math.ceil((maximum[0] - minimum[0]) * scale)))
    height = max(8, int(math.ceil((maximum[1] - minimum[1]) * scale)))

    accumulated = np.zeros((height, width), dtype=np.uint16)
    for polygon in polygons:
        pixels = np.round((polygon - minimum) * scale).astype(np.int32)
        single = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(single, [pixels], 1)
        # Shared puzzle edges should not count as area overlap.  Removing one
        # raster boundary pixel also reduces false overlap from subpixel
        # contour fitting and integer rounding.
        single = cv2.erode(single, np.ones((3, 3), dtype=np.uint8), iterations=1)
        accumulated += single.astype(np.uint16)
    overlap_pixels = int(np.sum(np.maximum(accumulated.astype(np.int32) - 1, 0)))
    return overlap_pixels / (scale * scale)


def _state_polygons(state: AssemblyState, pieces: list[PieceObservation]) -> list[np.ndarray]:
    return [
        state.placements[piece_id].apply(pieces[piece_id].local_polygon_mm)
        for piece_id in sorted(state.placements)
    ]


def _state_key(state: AssemblyState, config: VisionConfig) -> tuple[Any, ...]:
    values: list[Any] = []
    for piece_id in sorted(state.placements):
        transform = state.placements[piece_id]
        values.extend(
            [
                piece_id,
                round(transform.angle_deg / config.dedupe_angle_deg),
                round(float(transform.translation_mm[0]) / config.dedupe_translation_mm),
                round(float(transform.translation_mm[1]) / config.dedupe_translation_mm),
            ]
        )
    return tuple(values)


def _partial_heuristic(
    state: AssemblyState,
    pieces: list[PieceObservation],
    config: VisionConfig,
) -> float:
    polygons = _state_polygons(state, pieces)
    all_points = np.concatenate(polygons, axis=0).astype(np.float32)
    rectangle = cv2.minAreaRect(all_points)
    width, height = rectangle[1]
    long_side = max(float(width), float(height))
    short_side = min(float(width), float(height))
    bounding_area = max(long_side * short_side, 1.0)
    piece_area = sum(polygon_area(polygon) for polygon in polygons)
    union_area = max(0.0, piece_area - state.overlap_area_mm2)
    fill_error = max(0.0, bounding_area - union_area) / bounding_area

    overlap_ratio = state.overlap_area_mm2 / max(piece_area, 1.0)
    partial_match_count = sum(
        match.anchor_mode != "both"
        for match in state.matches
    )

    return (
        0.15 * state.edge_error
        + fill_error
        + 8.0 * overlap_ratio
        + config.partial_match_heuristic_penalty * partial_match_count
    )


def enumerate_assemblies(
    pieces: list[PieceObservation],
    config: VisionConfig,
) -> list[AssemblyState]:
    """Enumerate connected assemblies by aligning compatible polygon edges."""

    if not pieces:
        return []
    if len(pieces) == 1:
        return [
            AssemblyState(
                placements={0: RigidTransform(0.0, np.zeros(2, dtype=np.float64))}
            )
        ]

    root = int(np.argmax([piece.area_mm2 for piece in pieces]))
    states = [
        AssemblyState(
            placements={root: RigidTransform(0.0, np.zeros(2, dtype=np.float64))}
        )
    ]

    while states and len(states[0].placements) < len(pieces):
        expanded: list[AssemblyState] = []
        for state in states:
            placed_ids = sorted(state.placements)
            unplaced_ids = [i for i in range(len(pieces)) if i not in state.placements]
            fixed_polygons = {
                piece_id: state.placements[piece_id].apply(pieces[piece_id].local_polygon_mm)
                for piece_id in placed_ids
            }

            for moving_id in unplaced_ids:
                moving_piece = pieces[moving_id]
                for fixed_id in placed_ids:
                    fixed_polygon = fixed_polygons[fixed_id]
                    fixed_piece = pieces[fixed_id]
                    for fixed_edge in range(len(fixed_polygon)):
                        fixed_start = fixed_polygon[fixed_edge]
                        fixed_end = fixed_polygon[(fixed_edge + 1) % len(fixed_polygon)]
                        fixed_length = float(np.linalg.norm(fixed_end - fixed_start))

                        for moving_edge in range(len(moving_piece.local_polygon_mm)):
                            moving_start, moving_end = moving_piece.edge(moving_edge)
                            moving_length = float(np.linalg.norm(moving_end - moving_start))
                            candidates = _edge_alignment_candidates(
                                fixed_start,
                                fixed_end,
                                moving_start,
                                moving_end,
                                config,
                            )
                            for transform, anchor_mode, normalized_error in candidates:
                                moving_polygon = transform.apply(
                                    moving_piece.local_polygon_mm
                                )
                                overlap = _raster_overlap_area(
                                    [*fixed_polygons.values(), moving_polygon],
                                    scale=config.search_overlap_scale,
                                )
                                if overlap > config.overlap_tolerance_mm2:
                                    continue

                                new_state = AssemblyState(
                                    placements={
                                        **state.placements,
                                        moving_id: transform,
                                    },
                                    matches=[
                                        *state.matches,
                                        EdgeMatch(
                                            piece_a=fixed_id,
                                            edge_a=fixed_edge,
                                            piece_b=moving_id,
                                            edge_b=moving_edge,
                                            normalized_length_error=normalized_error,
                                            anchor_mode=anchor_mode,
                                        ),
                                    ],
                                    edge_error=state.edge_error + normalized_error,
                                    overlap_area_mm2=overlap,
                                )
                                new_state.heuristic = _partial_heuristic(
                                    new_state,
                                    pieces,
                                    config,
                                )
                                expanded.append(new_state)

        unique: dict[tuple[Any, ...], AssemblyState] = {}
        for state in sorted(expanded, key=lambda candidate: candidate.heuristic):
            key = _state_key(state, config)
            if key not in unique:
                unique[key] = state
            if len(unique) >= config.beam_width:
                break
        states = list(unique.values())

    return states


def _assembly_geometry(
    state: AssemblyState,
    pieces: list[PieceObservation],
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Return fill, overlap, long, short, centre, and global alignment rotation."""

    polygons = _state_polygons(state, pieces)
    all_points = np.concatenate(polygons, axis=0).astype(np.float32)
    rectangle = cv2.minAreaRect(all_points)
    box = cv2.boxPoints(rectangle).astype(np.float64)

    edge_vectors = np.roll(box, -1, axis=0) - box
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    long_edge_index = int(np.argmax(edge_lengths))
    long_vector = edge_vectors[long_edge_index]
    long_angle = math.degrees(math.atan2(float(long_vector[1]), float(long_vector[0])))
    global_rotation = rotation_matrix(-long_angle)

    aligned_polygons = [polygon @ global_rotation.T for polygon in polygons]
    aligned_points = np.concatenate(aligned_polygons, axis=0)
    minimum = np.min(aligned_points, axis=0)
    maximum = np.max(aligned_points, axis=0)
    size = maximum - minimum
    long_side = max(float(size[0]), float(size[1]))
    short_side = min(float(size[0]), float(size[1]))

    rectangle_area = max(long_side * short_side, 1.0)
    sum_piece_area = sum(polygon_area(polygon) for polygon in polygons)
    overlap_area = _raster_overlap_area(polygons, scale=4.0)
    union_area = max(0.0, sum_piece_area - overlap_area)
    fill_error = max(0.0, rectangle_area - union_area) / rectangle_area
    overlap_ratio = overlap_area / max(sum_piece_area, 1.0)
    centre = 0.5 * (minimum + maximum)
    return (
        fill_error,
        overlap_ratio,
        long_side,
        short_side,
        centre,
        global_rotation,
    )


def _size_penalty(
    long_side: float,
    short_side: float,
    mode: str,
    config: VisionConfig,
) -> float:
    if mode == "self":
        long_error = (long_side - config.self_target_long_mm) / 10.0
        short_error = (short_side - config.self_target_short_mm) / 10.0
        return long_error * long_error + short_error * short_error

    # On-site fragments may form any rectangle from 90×50 mm to 120×90 mm.
    # Penalise only candidates outside that rule range; rectangular fill and
    # seam texture must distinguish candidates inside it.
    def range_error(value: float, minimum: float, maximum: float) -> float:
        if value < minimum:
            return (minimum - value) / 10.0
        if value > maximum:
            return (value - maximum) / 10.0
        return 0.0

    return (
        range_error(long_side, config.target_long_min_mm, config.target_long_max_mm) ** 2
        + range_error(short_side, config.target_short_min_mm, config.target_short_max_mm)
        ** 2
    )


def _size_is_rule_compliant(
    long_side: float,
    short_side: float,
    mode: str,
    config: VisionConfig,
) -> bool:
    """Apply the size rule to the nominal, zero-clearance assembly."""

    if mode == "self":
        tolerance = config.self_target_size_tolerance_mm
        return (
            abs(long_side - config.self_target_long_mm) <= tolerance
            and abs(short_side - config.self_target_short_mm) <= tolerance
        )
    return (
        config.target_long_min_mm <= long_side <= config.target_long_max_mm
        and config.target_short_min_mm <= short_side <= config.target_short_max_mm
    )


def _each_piece_has_outer_edge(
    state: AssemblyState,
    pieces: list[PieceObservation],
    config: VisionConfig,
) -> bool:
    """Check the rule that every fragment contributes a rectangle outer edge."""

    polygons = {
        piece_id: state.placements[piece_id].apply(
            pieces[piece_id].local_polygon_mm
        )
        for piece_id in sorted(state.placements)
    }
    all_points = np.concatenate(list(polygons.values()), axis=0).astype(np.float32)
    rectangle = cv2.minAreaRect(all_points)
    box = cv2.boxPoints(rectangle).astype(np.float64)
    vectors = np.roll(box, -1, axis=0) - box
    long_vector = vectors[int(np.argmax(np.linalg.norm(vectors, axis=1)))]
    angle = math.degrees(
        math.atan2(float(long_vector[1]), float(long_vector[0]))
    )
    alignment = rotation_matrix(-angle)
    aligned = {
        piece_id: polygon @ alignment.T
        for piece_id, polygon in polygons.items()
    }
    aligned_points = np.concatenate(list(aligned.values()), axis=0)
    minimum = np.min(aligned_points, axis=0)
    maximum = np.max(aligned_points, axis=0)
    tolerance = config.outer_edge_tolerance_mm

    for polygon in aligned.values():
        has_outer_edge = False
        for edge_index in range(len(polygon)):
            start = polygon[edge_index]
            end = polygon[(edge_index + 1) % len(polygon)]
            side_distances = (
                max(abs(start[0] - minimum[0]), abs(end[0] - minimum[0])),
                max(abs(start[0] - maximum[0]), abs(end[0] - maximum[0])),
                max(abs(start[1] - minimum[1]), abs(end[1] - minimum[1])),
                max(abs(start[1] - maximum[1]), abs(end[1] - maximum[1])),
            )
            if min(side_distances) <= tolerance:
                has_outer_edge = True
                break
        if not has_outer_edge:
            return False
    return True


def _sample_edge_profile(
    piece: PieceObservation,
    edge_index: int,
    rectified_lab: np.ndarray,
    config: VisionConfig,
    sample_count: int = 32,
) -> np.ndarray:
    """Sample a colour profile just inside one source edge."""

    local = piece.local_polygon_mm
    p0 = local[edge_index]
    p1 = local[(edge_index + 1) % len(local)]
    vector = p1 - p0
    length = max(float(np.linalg.norm(vector)), 1e-6)
    normal = np.array([-vector[1], vector[0]], dtype=np.float64) / length
    midpoint = 0.5 * (p0 + p1)
    contour_local = local.astype(np.float32)
    test_point = midpoint + normal * config.texture_sample_offset_mm
    inside = cv2.pointPolygonTest(contour_local, tuple(test_point.astype(float)), False)
    if inside < 0:
        normal *= -1.0

    parameters = np.linspace(0.06, 0.94, sample_count)
    points_local = (
        p0.reshape(1, 2)
        + parameters.reshape(-1, 1) * vector.reshape(1, 2)
        + normal.reshape(1, 2) * config.texture_sample_offset_mm
    )
    points_source_mm = points_local + piece.center_mm
    points_px = points_source_mm * config.pixels_per_mm

    map_x = points_px[:, 0].astype(np.float32).reshape(1, -1)
    map_y = points_px[:, 1].astype(np.float32).reshape(1, -1)
    profile = cv2.remap(
        rectified_lab,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return profile.reshape(-1, 3).astype(np.float32)


def _texture_score(
    state: AssemblyState,
    pieces: list[PieceObservation],
    rectified_bgr: np.ndarray,
    config: VisionConfig,
) -> float:
    if not state.matches:
        return 0.0
    lab = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    costs: list[float] = []
    for match in state.matches:
        # Full profiles are comparable only for one-to-one full-edge seams.
        # Partial T-junction seams are ranked by the global rectangle and
        # outer-boundary constraints until sub-profile sampling is available.
        if match.anchor_mode != "both":
            continue
        profile_a = _sample_edge_profile(
            pieces[match.piece_a], match.edge_a, lab, config
        )
        profile_b = _sample_edge_profile(
            pieces[match.piece_b], match.edge_b, lab, config
        )[::-1]
        colour_cost = float(np.mean(np.linalg.norm(profile_a - profile_b, axis=1))) / 100.0
        gradient_a = np.diff(profile_a, axis=0)
        gradient_b = np.diff(profile_b, axis=0)
        gradient_cost = (
            float(np.mean(np.linalg.norm(gradient_a - gradient_b, axis=1))) / 100.0
        )
        costs.append(colour_cost + 0.35 * gradient_cost)
    return float(np.mean(costs)) if costs else 0.0


def _normalise_solution_layout(
    state: AssemblyState,
    pieces: list[PieceObservation],
    config: VisionConfig,
) -> tuple[
    dict[int, RigidTransform],
    dict[int, np.ndarray],
    tuple[float, float],
    tuple[float, float],
    float,
    float,
    float,
]:
    polygons = _state_polygons(state, pieces)
    all_points = np.concatenate(polygons, axis=0).astype(np.float32)
    rectangle = cv2.minAreaRect(all_points)
    box = cv2.boxPoints(rectangle).astype(np.float64)
    vectors = np.roll(box, -1, axis=0) - box
    lengths = np.linalg.norm(vectors, axis=1)
    long_vector = vectors[int(np.argmax(lengths))]
    angle = math.degrees(math.atan2(float(long_vector[1]), float(long_vector[0])))
    align_angle = normalize_angle_deg(-angle)
    align_rotation = rotation_matrix(align_angle)

    aligned = [polygon @ align_rotation.T for polygon in polygons]
    aligned_points = np.concatenate(aligned, axis=0)
    size = np.ptp(aligned_points, axis=0)
    if size[1] > size[0]:
        extra = rotation_matrix(-90.0)
        align_rotation = extra @ align_rotation
        align_angle = normalize_angle_deg(align_angle - 90.0)
        aligned = [polygon @ extra.T for polygon in aligned]
        aligned_points = np.concatenate(aligned, axis=0)

    minimum = np.min(aligned_points, axis=0)
    maximum = np.max(aligned_points, axis=0)
    centre = 0.5 * (minimum + maximum)
    target_centre = np.array(
        [config.target_center_x_mm, config.target_center_y_mm], dtype=np.float64
    )
    global_translation = target_centre - centre

    target_transforms: dict[int, RigidTransform] = {}
    target_polygons: dict[int, np.ndarray] = {}
    for piece_id, transform in state.placements.items():
        piece_rotation = rotation_matrix(transform.angle_deg)
        combined_rotation = align_rotation @ piece_rotation
        combined_angle = math.degrees(
            math.atan2(float(combined_rotation[1, 0]), float(combined_rotation[0, 0]))
        )
        combined_translation = (
            align_rotation @ transform.translation_mm + global_translation
        )
        combined = RigidTransform(
            angle_deg=normalize_angle_deg(combined_angle),
            translation_mm=combined_translation,
        )
        target_transforms[piece_id] = combined
        target_polygons[piece_id] = combined.apply(pieces[piece_id].local_polygon_mm)

    nominal_points = np.concatenate(list(target_polygons.values()), axis=0)
    nominal_size = np.ptp(nominal_points, axis=0)

    nominal_translations = {
        piece_id: transform.translation_mm.copy()
        for piece_id, transform in target_transforms.items()
    }
    target_gap = max(0.0, float(config.target_vertex_gap_mm))
    radial_vectors = {
        piece_id: translation - target_centre
        for piece_id, translation in nominal_translations.items()
    }
    reference_radius = max(
        (
            float(np.linalg.norm(vector))
            for vector in radial_vectors.values()
        ),
        default=1.0,
    )
    reference_radius = max(reference_radius, 1.0)

    def apply_clearance(distance_mm: float) -> None:
        for piece_id, transform in target_transforms.items():
            transform.translation_mm = (
                target_centre
                + (1.0 + float(distance_mm) / reference_radius)
                * radial_vectors[piece_id]
            )
            target_polygons[piece_id] = transform.apply(
                pieces[piece_id].local_polygon_mm
            )

    def point_segment_distance(
        point: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
    ) -> float:
        vector = end - start
        denominator = float(np.dot(vector, vector))
        if denominator <= 1e-12:
            return float(np.linalg.norm(point - start))
        ratio = float(np.dot(point - start, vector)) / denominator
        ratio = max(0.0, min(1.0, ratio))
        projection = start + ratio * vector
        return float(np.linalg.norm(point - projection))

    def polygon_distance(
        polygon_a: np.ndarray,
        polygon_b: np.ndarray,
    ) -> float:
        distances: list[float] = []
        for point in polygon_a:
            for edge_index in range(len(polygon_b)):
                distances.append(
                    point_segment_distance(
                        point,
                        polygon_b[edge_index],
                        polygon_b[(edge_index + 1) % len(polygon_b)],
                    )
                )
        for point in polygon_b:
            for edge_index in range(len(polygon_a)):
                distances.append(
                    point_segment_distance(
                        point,
                        polygon_a[edge_index],
                        polygon_a[(edge_index + 1) % len(polygon_a)],
                    )
                )
        return min(distances, default=float("inf"))

    def pairwise_gaps() -> list[float]:
        piece_ids = sorted(target_polygons)
        return [
            polygon_distance(
                target_polygons[piece_ids[index_a]],
                target_polygons[piece_ids[index_b]],
            )
            for index_a in range(len(piece_ids))
            for index_b in range(index_a + 1, len(piece_ids))
        ]

    def matched_gaps() -> list[float]:
        gaps: list[float] = []
        for match in state.matches:
            polygon_a = target_polygons[match.piece_a]
            polygon_b = target_polygons[match.piece_b]
            a0 = polygon_a[match.edge_a]
            a1 = polygon_a[(match.edge_a + 1) % len(polygon_a)]
            b0 = polygon_b[match.edge_b]
            b1 = polygon_b[(match.edge_b + 1) % len(polygon_b)]
            if match.anchor_mode == "both":
                gaps.extend(
                    [
                        float(np.linalg.norm(a0 - b1)),
                        float(np.linalg.norm(a1 - b0)),
                    ]
                )
            elif match.anchor_mode == "fixed_end_to_moving_start":
                gaps.append(float(np.linalg.norm(a1 - b0)))
            elif match.anchor_mode == "fixed_start_to_moving_end":
                gaps.append(float(np.linalg.norm(a0 - b1)))
            else:
                raise ValueError(
                    f"Unknown edge anchor mode: {match.anchor_mode}"
                )
        return gaps

    # Use rigid translations only.  The configured 5 mm value is a soft
    # placement allowance for control error, not a rule gate on every pair.
    # Expanding until the largest matched endpoint gap reaches that allowance
    # reproduces the compact, previously validated layout while preserving
    # every fragment's detected size, shape and rotation.  Piece IDs and
    # external dimensions are never hard-coded.
    clearance = 0.0
    if state.matches and target_gap > 0:
        low = 0.0
        high = max(1.0, 0.5 * target_gap)
        apply_clearance(high)
        high_gap = max(matched_gaps(), default=0.0)
        while high_gap < target_gap and high < 100.0:
            high *= 1.5
            apply_clearance(high)
            high_gap = max(matched_gaps(), default=0.0)
        for _ in range(24):
            middle = 0.5 * (low + high)
            apply_clearance(middle)
            middle_gap = max(matched_gaps(), default=0.0)
            if middle_gap < target_gap:
                low = middle
            else:
                high = middle
        clearance = high

    apply_clearance(clearance)
    matched_vertex_gaps = matched_gaps()
    max_vertex_gap = max(matched_vertex_gaps, default=0.0)
    min_pairwise_gap = min(pairwise_gaps(), default=0.0)
    placement_points = np.concatenate(list(target_polygons.values()), axis=0)
    placement_size = np.ptp(placement_points, axis=0)

    return (
        target_transforms,
        target_polygons,
        (float(max(nominal_size)), float(min(nominal_size))),
        (float(max(placement_size)), float(min(placement_size))),
        max_vertex_gap,
        min_pairwise_gap,
        clearance,
    )


def solve_puzzle(
    pieces: list[PieceObservation],
    rectified_bgr: np.ndarray,
    mode: str,
    config: VisionConfig,
) -> PuzzleSolution:
    states = enumerate_assemblies(pieces, config)
    if not states:
        raise RuntimeError(
            "No valid assembly was found. Check segmentation, polygon fitting, "
            "edge tolerance, or overlap tolerance."
        )

    ranked: list[tuple[float, float, float, AssemblyState, tuple[float, float]]] = []
    for state in states:
        (
            fill_error,
            overlap_ratio,
            long_side,
            short_side,
            _,
            _,
        ) = _assembly_geometry(state, pieces)
        if fill_error > config.max_rectangle_fill_error:
            continue
        total_piece_area = max(
            sum(piece.area_mm2 for piece in pieces),
            1.0,
        )
        if overlap_ratio * total_piece_area > config.overlap_tolerance_mm2:
            continue
        if not _each_piece_has_outer_edge(state, pieces, config):
            continue

        # 矩形度约束: 凸包顶点数 + 面积比
        polygons = _state_polygons(state, pieces)
        all_pts = np.concatenate(polygons, axis=0).astype(np.float32)
        hull = cv2.convexHull(all_pts)
        hull_area = float(cv2.contourArea(hull))
        rect_area = max(long_side * short_side, 1.0)
        rectangularity = hull_area / rect_area
        # 凸包简化为多边形，数顶点
        hull_peri = cv2.arcLength(hull, True)
        hull_approx = cv2.approxPolyDP(hull, 0.02 * hull_peri, True)
        hull_vertices = len(hull_approx)
        # 惩罚: 每多一个顶点(超过4)罚25分 + 面积偏差罚80
        vertex_penalty = max(0, hull_vertices - 4) * 25.0
        area_penalty = max(0.0, 1.0 - rectangularity) * 80.0

        geometric_score = (
            40.0 * fill_error
            + 35.0 * overlap_ratio
            + vertex_penalty
            + area_penalty
            + 0.8 * state.edge_error
        )
        texture_score = (
            _texture_score(state, pieces, rectified_bgr, config)
            if mode == "poker"
            else 0.0
        )
        score = geometric_score + config.texture_weight * texture_score
        ranked.append(
            (
                score,
                geometric_score,
                texture_score,
                state,
                (long_side, short_side),
            )
        )

    if not ranked:
        raise RuntimeError(
            "No rule-compliant assembly was found: candidates must form a "
            f"rectangle with fill error <= {config.max_rectangle_fill_error:.0%}, "
            "avoid overlap, and place at least one edge of every fragment on "
            "the rectangle boundary. The assembled external dimensions are "
            "not restricted."
        )

    selected: tuple[
        float,
        float,
        float,
        AssemblyState,
        dict[int, RigidTransform],
        dict[int, np.ndarray],
        tuple[float, float],
        tuple[float, float],
        float,
        float,
        float,
        float,
    ] | None = None
    selected_soft_score = float("inf")
    rejected_overlap = 0.0
    for score, geometric, texture, state, _ in sorted(
        ranked,
        key=lambda item: item[0],
    ):
        (
            transforms,
            polygons,
            rectangle_size,
            placement_extent,
            max_vertex_gap,
            min_pairwise_gap,
            applied_clearance,
        ) = _normalise_solution_layout(state, pieces, config)
        placement_overlap = _raster_overlap_area(polygons.values(), scale=4.0)
        rejected_overlap = max(rejected_overlap, placement_overlap)
        # Clearance is a soft control allowance.  Prefer a compact candidate
        # near the requested allowance, but never reject an otherwise correct
        # random-size puzzle merely because it cannot attain that number.
        # Actual post-placement overlap remains the only blocking condition.
        if placement_overlap <= config.max_post_placement_overlap_mm2:
            clearance_soft_error = abs(
                max_vertex_gap - config.target_vertex_gap_mm
            )
            soft_score = score + clearance_soft_error
            if soft_score < selected_soft_score:
                selected_soft_score = soft_score
                selected = (
                    score,
                    geometric,
                    texture,
                    state,
                    transforms,
                    polygons,
                    rectangle_size,
                    placement_extent,
                    max_vertex_gap,
                    min_pairwise_gap,
                    applied_clearance,
                    placement_overlap,
                )
    if selected is None:
        raise RuntimeError(
            "All candidate assemblies failed the control-clearance gate: "
            f"post-placement overlap must be <= "
            f"{config.max_post_placement_overlap_mm2:.3f} mm² "
            f"(largest examined overlap {rejected_overlap:.3f} mm²). "
            f"The requested {config.target_vertex_gap_mm:.2f} mm spacing is "
            "only a soft control allowance."
        )
    (
        score,
        geometric,
        texture,
        best_state,
        transforms,
        polygons,
        rectangle_size,
        placement_extent,
        max_vertex_gap,
        min_pairwise_gap,
        applied_clearance,
        placement_overlap,
    ) = selected
    # `score` ranks already rule-filtered candidates; it is not a physical
    # safety measurement.  Keep it as a diagnostic instead of hard-rejecting
    # random sizes/cuts near an arbitrary historical threshold.
    return PuzzleSolution(
        mode=mode,
        score=float(score),
        geometric_score=float(geometric),
        texture_score=float(texture),
        rectangle_size_mm=rectangle_size,
        placement_extent_mm=placement_extent,
        applied_clearance_mm=applied_clearance,
        max_matched_vertex_gap_mm=max_vertex_gap,
        min_pairwise_gap_mm=min_pairwise_gap,
        placement_overlap_area_mm2=placement_overlap,
        target_polygons_mm=polygons,
        target_transforms=transforms,
        matches=best_state.matches,
    )


def draw_detection_overlay(
    rectified_bgr: np.ndarray,
    pieces: list[PieceObservation],
    mode: str,
    config: VisionConfig,
) -> np.ndarray:
    canvas = rectified_bgr.copy()
    scale = config.pixels_per_mm
    divider_y = int(round(config.divider_y_mm * scale))
    cv2.line(canvas, (0, divider_y), (canvas.shape[1] - 1, divider_y), (0, 0, 255), 2)

    for piece in pieces:
        colour = (0, 255, 255)
        cv2.drawContours(canvas, [piece.contour_px], -1, colour, 3)
        centre = tuple(np.round(piece.center_mm * scale).astype(int))
        cv2.circle(canvas, centre, 6, (0, 0, 255), -1)
        text = (
            f"P{piece.piece_id} V={len(piece.local_polygon_mm)} "
            f"A={piece.area_mm2:.0f}mm2"
        )
        cv2.putText(
            canvas,
            text,
            (centre[0] + 8, centre[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            text,
            (centre[0] + 8, centre[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        f"mode={mode} pieces={len(pieces)}",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"mode={mode} pieces={len(pieces)}",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return canvas


def draw_solution_overlay(
    rectified_bgr: np.ndarray,
    pieces: list[PieceObservation],
    solution: PuzzleSolution,
    config: VisionConfig,
) -> np.ndarray:
    canvas = draw_detection_overlay(rectified_bgr, pieces, solution.mode, config)
    scale = config.pixels_per_mm
    colours = [(0, 200, 0), (255, 80, 20), (180, 0, 220), (20, 180, 255)]

    for piece in pieces:
        target = solution.target_polygons_mm[piece.piece_id]
        target_px = np.round(target * scale).astype(np.int32)
        colour = colours[piece.piece_id % len(colours)]
        cv2.polylines(canvas, [target_px], True, colour, 3, cv2.LINE_AA)

        source_px = tuple(np.round(piece.center_mm * scale).astype(int))
        target_center = solution.target_transforms[piece.piece_id].translation_mm
        target_center_px = tuple(np.round(target_center * scale).astype(int))
        cv2.arrowedLine(
            canvas,
            source_px,
            target_center_px,
            colour,
            2,
            cv2.LINE_AA,
            tipLength=0.04,
        )
        cv2.putText(
            canvas,
            f"P{piece.piece_id}",
            target_center_px,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        f"score={solution.score:.3f} nominal={solution.rectangle_size_mm[0]:.1f}x"
        f"{solution.rectangle_size_mm[1]:.1f}mm "
        f"min_pair_gap={solution.min_pairwise_gap_mm:.1f}mm",
        (20, canvas.shape[0] - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"score={solution.score:.3f} nominal={solution.rectangle_size_mm[0]:.1f}x"
        f"{solution.rectangle_size_mm[1]:.1f}mm "
        f"min_pair_gap={solution.min_pairwise_gap_mm:.1f}mm",
        (20, canvas.shape[0] - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return canvas


def draw_reconstructed_texture(
    rectified_bgr: np.ndarray,
    pieces: list[PieceObservation],
    solution: PuzzleSolution,
    config: VisionConfig,
) -> np.ndarray:
    """Warp each source piece texture with the solved rigid transform."""

    canvas = rectified_bgr.copy()
    scale = config.pixels_per_mm
    for piece in pieces:
        transform = solution.target_transforms[piece.piece_id]
        rotation = rotation_matrix(transform.angle_deg)
        translation_px = scale * (
            transform.translation_mm - rotation @ piece.center_mm
        )
        affine = np.column_stack([rotation, translation_px]).astype(np.float32)

        source_mask = np.zeros(rectified_bgr.shape[:2], dtype=np.uint8)
        cv2.drawContours(
            source_mask,
            [piece.contour_px],
            -1,
            255,
            thickness=cv2.FILLED,
        )
        warped_texture = cv2.warpAffine(
            rectified_bgr,
            affine,
            (canvas.shape[1], canvas.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_mask = cv2.warpAffine(
            source_mask,
            affine,
            (canvas.shape[1], canvas.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        canvas[warped_mask > 0] = warped_texture[warped_mask > 0]

        target_px = np.round(
            solution.target_polygons_mm[piece.piece_id] * scale
        ).astype(np.int32)
        cv2.polylines(canvas, [target_px], True, (0, 0, 0), 2, cv2.LINE_AA)
        target_center_px = tuple(
            np.round(transform.translation_mm * scale).astype(int)
        )
        cv2.putText(
            canvas,
            f"P{piece.piece_id}",
            target_center_px,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return canvas


def build_control_plan(
    solution: PuzzleSolution,
    pieces: list[PieceObservation],
) -> list[dict[str, Any]]:
    """Convert a vision solution into hardware-independent control commands."""

    piece_by_id = {piece.piece_id: piece for piece in pieces}
    # Place lower/farther target pieces first as a neutral default.
    ordered_ids = sorted(
        solution.target_transforms,
        key=lambda piece_id: solution.target_transforms[piece_id].translation_mm[1],
        reverse=True,
    )

    commands: list[dict[str, Any]] = []
    for sequence, piece_id in enumerate(ordered_ids):
        piece = piece_by_id[piece_id]
        target = solution.target_transforms[piece_id]
        commands.append(
            {
                "sequence": sequence,
                "piece_id": piece_id,
                "action": "pick_rotate_place",
                "pick": {
                    "x_mm": round(float(piece.center_mm[0]), 3),
                    "y_mm": round(float(piece.center_mm[1]), 3),
                },
                "source_theta_deg": round(float(piece.orientation_deg), 3),
                "magnet_rotation_delta_deg": round(
                    normalize_angle_deg(target.angle_deg), 3
                ),
                "place": {
                    "x_mm": round(float(target.translation_mm[0]), 3),
                    "y_mm": round(float(target.translation_mm[1]), 3),
                },
                "verify_after_place": True,
            }
        )
    return commands


def send_control_command(
    command: dict[str, Any],
    transport: Optional[Callable[[dict[str, Any]], None]] = None,
) -> None:
    """Hardware control placeholder.

    Replace this function, or pass a transport callback, when the MCU protocol
    is known.  The default behaviour is deliberately safe: it prints the
    command and does not access serial ports, CAN buses, GPIO, or motors.
    """

    if transport is not None:
        transport(command)
        return
    print("[CONTROL PLACEHOLDER]", json.dumps(command, ensure_ascii=False))


def execute_control_plan(
    commands: list[dict[str, Any]],
    transport: Optional[Callable[[dict[str, Any]], None]] = None,
) -> None:
    for command in commands:
        send_control_command(command, transport=transport)


def run_pipeline(
    image_bgr: np.ndarray,
    config: VisionConfig,
    *,
    mode: str = "auto",
    corners: Optional[np.ndarray] = None,
    already_rectified: bool = False,
) -> dict[str, Any]:
    # Detect corners explicitly for debug output
    if already_rectified:
        detected_corners = None
    elif corners is not None:
        detected_corners = order_quad(corners)
    else:
        detected_corners = detect_a4_quad(image_bgr, config)

    frame_config = config_for_paper_resolution(config, detected_corners)
    rectified, homography = rectify_a4(
        image_bgr,
        frame_config,
        corners=detected_corners,
        already_rectified=already_rectified,
    )

    # A low native-resolution camera normally benefits from a strong polygon
    # simplification.  When a fragment is placed within a few millimetres of
    # the paper edge, however, that same setting can combine with the border
    # guard and delete a real corner.  Keep the original configuration as the
    # first (and usual) attempt, then retry two deterministic, stricter contour
    # fits only when the complete rule-gated solver rejects it.  A retry still
    # has to pass every scene, rectangle, overlap, size, and score gate.
    attempt_configs = [frame_config]
    for epsilon_mm in (2.5, 1.5):
        adapted = replace(frame_config)
        adapted.polygon_epsilon_mm = min(
            frame_config.polygon_epsilon_mm,
            epsilon_mm,
        )
        adapted.border_ignore_mm = min(frame_config.border_ignore_mm, 0.5)
        if not any(
            abs(candidate.polygon_epsilon_mm - adapted.polygon_epsilon_mm) < 1e-6
            and abs(candidate.border_ignore_mm - adapted.border_ignore_mm) < 1e-6
            for candidate in attempt_configs
        ):
            attempt_configs.append(adapted)

    attempt_errors: list[str] = []
    selected_attempt = -1
    for attempt_index, attempt_config in enumerate(attempt_configs):
        try:
            pieces, segmentation_mask, background_lab, green_mask = segment_pieces(
                rectified,
                attempt_config,
            )
            if not pieces:
                raise RuntimeError(
                    "No pieces were found in the upper half of the A4 sheet."
                )
            scene_quality = evaluate_scene_quality(
                image_bgr,
                rectified,
                detected_corners,
                pieces,
                green_mask,
                attempt_config,
            )
            if not scene_quality["passed"]:
                raise RuntimeError(
                    "场景质量检查未通过: " + "; ".join(scene_quality["issues"])
                )
            selected_mode = (
                infer_mode(pieces, attempt_config) if mode == "auto" else mode
            )
            if selected_mode not in {"self", "white", "poker"}:
                raise ValueError(f"Unsupported mode: {selected_mode}")
            solution = solve_puzzle(
                pieces,
                rectified,
                selected_mode,
                attempt_config,
            )
            frame_config = attempt_config
            selected_attempt = attempt_index
            break
        except RuntimeError as error:
            attempt_errors.append(
                f"epsilon={attempt_config.polygon_epsilon_mm:.2f} mm, "
                f"border={attempt_config.border_ignore_mm:.2f} mm: {error}"
            )
    else:
        raise RuntimeError(
            "All deterministic contour fits were rejected. "
            + " | ".join(attempt_errors)
        )

    control_plan = build_control_plan(solution, pieces)

    # Debug: draw detected quad on original frame
    quad_debug = image_bgr.copy()
    if detected_corners is not None:
        pts = np.round(detected_corners).astype(np.int32)
        cv2.polylines(quad_debug, [pts], True, (0, 0, 255), 3, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            cv2.circle(quad_debug, tuple(pt), 8, (0, 255, 255), -1)
            cv2.putText(
                quad_debug, f"C{i}", (pt[0] + 10, pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
            )

    return {
        "rectified": rectified,
        "segmentation_mask": segmentation_mask,
        "green_mask": green_mask,
        "detection_overlay": draw_detection_overlay(
            rectified, pieces, selected_mode, frame_config
        ),
        "solution_overlay": draw_solution_overlay(
            rectified, pieces, solution, frame_config
        ),
        "reconstructed_texture": draw_reconstructed_texture(
            rectified, pieces, solution, frame_config
        ),
        "quad_debug": quad_debug,
        "detected_corners": detected_corners,
        "homography": homography,
        "background_lab": background_lab,
        "scene_quality": scene_quality,
        "effective_polygon_epsilon_mm": frame_config.polygon_epsilon_mm,
        "vision_adaptation": {
            "selected_attempt": selected_attempt,
            "border_ignore_mm": frame_config.border_ignore_mm,
            "attempt_errors": attempt_errors,
        },
        "pieces": pieces,
        "solution": solution,
        "control_plan": control_plan,
    }
