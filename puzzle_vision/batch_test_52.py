"""Strict regression for rule-2 on-site pieces across all 52 card faces.

Every case records:
  * a random legal 90×50–120×90 mm target and a 2–4 piece cut;
  * input, segmentation, detection, solution, and reconstructed screenshots;
  * case-level timing and geometry errors;
  * per-piece source localization and rigid-transform errors;
  * per-seam endpoint gaps.

The JSON files produced here are raw/processed inputs for the Excel report.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from demo_synthetic import choose_card_template, generate_synthetic_image
from puzzle_vision import (
    VisionConfig,
    normalize_angle_deg,
    pca_orientation_deg,
    polygon_area,
    rotation_matrix,
    run_pipeline,
)


STRICT_THRESHOLDS = {
    "source_center_max_mm": 1.5,
    "source_angle_max_deg": 2.0,
    "source_vertex_chamfer_max_mm": 2.0,
    "source_area_error_max_percent": 5.0,
    "nominal_long_error_mm": 3.0,
    "nominal_short_error_mm": 3.0,
    "relative_layout_rms_mm": 2.0,
    # The control-friendly target is 10 mm; reject above the internal 12 mm
    # guard while still leaving 8 mm inside the rule's 20 mm maximum.
    "planned_vertex_gap_max_mm": 12.0,
    "planned_vertex_gap_target_error_mm": 0.25,
    # Any geometric overlap is a direct failure.  The raster measurement is
    # expected to be exactly zero after the 10 mm rigid clearance.
    "planned_overlap_area_mm2": 0.0,
    "rigid_edge_error_max_mm": 1e-5,
    "rigid_area_error_max_mm2": 1e-5,
}


def polygon_centroid(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    cross = points[:, 0] * np.roll(points[:, 1], -1) - np.roll(
        points[:, 0], -1
    ) * points[:, 1]
    denominator = 3.0 * np.sum(cross)
    if abs(denominator) < 1e-9:
        return np.mean(points, axis=0)
    x = np.sum((points[:, 0] + np.roll(points[:, 0], -1)) * cross) / denominator
    y = np.sum((points[:, 1] + np.roll(points[:, 1], -1)) * cross) / denominator
    return np.array([x, y], dtype=np.float64)


def angle_error_modulo_180(actual_deg: float, expected_deg: float) -> float:
    return abs((actual_deg - expected_deg + 90.0) % 180.0 - 90.0)


def rigid_vertex_orientation_error_deg(
    detected: np.ndarray,
    truth: np.ndarray,
) -> float:
    """Measure orientation error without PCA instability on near-square pieces."""

    detected = np.asarray(detected, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if len(detected) != len(truth):
        return angle_error_modulo_180(
            pca_orientation_deg(detected),
            pca_orientation_deg(truth),
        )

    detected_centered = detected - np.mean(detected, axis=0)
    truth_centered = truth - np.mean(truth, axis=0)
    best_rms = math.inf
    best_angle = math.inf
    for candidate in (truth_centered, truth_centered[::-1]):
        for shift in range(len(candidate)):
            ordered_truth = np.roll(candidate, shift, axis=0)
            u, _, vt = np.linalg.svd(detected_centered.T @ ordered_truth)
            row_rotation = u @ vt
            if np.linalg.det(row_rotation) < 0:
                continue
            predicted = detected_centered @ row_rotation
            rms = float(
                np.sqrt(np.mean(np.sum((predicted - ordered_truth) ** 2, axis=1)))
            )
            angle = abs(
                normalize_angle_deg(
                    math.degrees(
                        math.atan2(
                            float(row_rotation[0, 1]),
                            float(row_rotation[0, 0]),
                        )
                    )
                )
            )
            if rms < best_rms:
                best_rms = rms
                best_angle = angle
    if not math.isfinite(best_angle):
        return angle_error_modulo_180(
            pca_orientation_deg(detected),
            pca_orientation_deg(truth),
        )
    return best_angle


def symmetric_vertex_chamfer(points_a: np.ndarray, points_b: np.ndarray) -> float:
    distances = np.linalg.norm(
        points_a[:, None, :] - points_b[None, :, :],
        axis=2,
    )
    return float(
        max(
            np.max(np.min(distances, axis=1)),
            np.max(np.min(distances, axis=0)),
        )
    )


def ground_truth_source_polygons(metadata: dict[str, object]) -> list[np.ndarray]:
    polygons = metadata["target_polygons_mm"]
    source_centres = metadata["source_centres_mm"]
    source_angles = metadata["source_angles_deg"]
    transformed: list[np.ndarray] = []
    for polygon, source_centre, angle in zip(
        polygons,
        source_centres,
        source_angles,
    ):
        polygon = np.asarray(polygon, dtype=np.float64)
        reference = np.mean(polygon, axis=0)
        transformed.append(
            (polygon - reference) @ rotation_matrix(float(angle)).T
            + np.asarray(source_centre, dtype=np.float64)
        )
    return transformed


def assign_detected_to_original(
    detected_centres: list[np.ndarray],
    truth_polygons: list[np.ndarray],
) -> dict[int, int]:
    truth_centres = [polygon_centroid(polygon) for polygon in truth_polygons]
    best_cost = math.inf
    best_permutation: tuple[int, ...] | None = None
    for permutation in itertools.permutations(range(len(truth_centres))):
        cost = sum(
            float(
                np.linalg.norm(
                    detected_centres[detected_index] - truth_centres[original_index]
                )
            )
            for detected_index, original_index in enumerate(permutation)
        )
        if cost < best_cost:
            best_cost = cost
            best_permutation = permutation
    if best_permutation is None:
        raise RuntimeError("Unable to map detected pieces to synthetic ground truth")
    return {
        detected_id: original_id
        for detected_id, original_id in enumerate(best_permutation)
    }


def remove_configured_gap(
    point: np.ndarray,
    config: VisionConfig,
    applied_clearance_mm: float,
) -> np.ndarray:
    centre = np.array(
        [config.target_center_x_mm, config.target_center_y_mm],
        dtype=np.float64,
    )
    radial = point - centre
    length = float(np.linalg.norm(radial))
    if length <= 1e-9 or applied_clearance_mm <= 0:
        return point.copy()
    return point - applied_clearance_mm * radial / length


def rigid_alignment_rms(source: np.ndarray, target: np.ndarray) -> float:
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    u, _, vt = np.linalg.svd(source_centered.T @ target_centered)
    row_rotation = u @ vt
    if np.linalg.det(row_rotation) < 0:
        u[:, -1] *= -1.0
        row_rotation = u @ vt
    predicted = source_centered @ row_rotation + target_mean
    errors = np.linalg.norm(predicted - target, axis=1)
    return float(np.sqrt(np.mean(errors * errors)))


def evaluate_case(
    case_id: str,
    card_name: str,
    layout_index: int,
    layout_seed: int,
    elapsed_ms: float,
    result: dict[str, Any],
    metadata: dict[str, object],
    config: VisionConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    pieces = result["pieces"]
    solution = result["solution"]
    truth_source_polygons = ground_truth_source_polygons(metadata)
    mapping = assign_detected_to_original(
        [piece.center_mm for piece in pieces],
        truth_source_polygons,
    )

    piece_rows: list[dict[str, Any]] = []
    center_errors: list[float] = []
    angle_errors: list[float] = []
    vertex_errors: list[float] = []
    area_error_percentages: list[float] = []
    rigid_edge_errors: list[float] = []
    rigid_area_errors: list[float] = []

    target_polygons = [
        np.asarray(polygon, dtype=np.float64)
        for polygon in metadata["target_polygons_mm"]
    ]
    source_angles = [float(value) for value in metadata["source_angles_deg"]]
    expected_target_centres = [
        polygon_centroid(polygon) for polygon in target_polygons
    ]
    expected_piece_count = len(target_polygons)

    for piece in pieces:
        original_id = mapping[piece.piece_id]
        truth_source = truth_source_polygons[original_id]
        truth_center = polygon_centroid(truth_source)
        center_error = float(np.linalg.norm(piece.center_mm - truth_center))

        expected_orientation = normalize_angle_deg(
            pca_orientation_deg(target_polygons[original_id])
            + source_angles[original_id]
        )
        angle_error = rigid_vertex_orientation_error_deg(
            piece.polygon_mm,
            truth_source,
        )
        vertex_error = symmetric_vertex_chamfer(
            piece.polygon_mm,
            truth_source,
        )
        truth_area = polygon_area(truth_source)
        detected_area = polygon_area(piece.polygon_mm)
        area_error_mm2 = abs(detected_area - truth_area)
        area_error_percent = 100.0 * area_error_mm2 / max(truth_area, 1e-9)

        target_polygon = solution.target_polygons_mm[piece.piece_id]
        source_edges = np.linalg.norm(
            np.roll(piece.local_polygon_mm, -1, axis=0)
            - piece.local_polygon_mm,
            axis=1,
        )
        target_edges = np.linalg.norm(
            np.roll(target_polygon, -1, axis=0) - target_polygon,
            axis=1,
        )
        rigid_edge_error = float(np.max(np.abs(source_edges - target_edges)))
        rigid_area_error = abs(
            polygon_area(piece.local_polygon_mm)
            - polygon_area(target_polygon)
        )

        target_transform = solution.target_transforms[piece.piece_id]
        row = {
            "case_id": case_id,
            "card": card_name,
            "layout_index": layout_index,
            "layout_seed": layout_seed,
            "target_seed": int(metadata["target_seed"]),
            "expected_piece_count": expected_piece_count,
            "target_width_mm": float(metadata["target_width_mm"]),
            "target_height_mm": float(metadata["target_height_mm"]),
            "detected_piece_id": piece.piece_id,
            "original_piece_id": original_id,
            "source_x_truth_mm": float(truth_center[0]),
            "source_y_truth_mm": float(truth_center[1]),
            "source_x_detected_mm": float(piece.center_mm[0]),
            "source_y_detected_mm": float(piece.center_mm[1]),
            "source_center_error_mm": center_error,
            "source_angle_truth_deg": expected_orientation,
            "source_angle_detected_deg": float(piece.orientation_deg),
            "source_angle_error_deg": angle_error,
            "source_vertex_chamfer_mm": vertex_error,
            "source_area_truth_mm2": truth_area,
            "source_area_detected_mm2": detected_area,
            "source_area_error_mm2": area_error_mm2,
            "source_area_error_percent": area_error_percent,
            "rigid_edge_error_mm": rigid_edge_error,
            "rigid_area_error_mm2": rigid_area_error,
            "rotation_delta_deg": float(target_transform.angle_deg),
            "target_x_mm": float(target_transform.translation_mm[0]),
            "target_y_mm": float(target_transform.translation_mm[1]),
        }
        piece_rows.append(row)
        center_errors.append(center_error)
        angle_errors.append(angle_error)
        vertex_errors.append(vertex_error)
        area_error_percentages.append(area_error_percent)
        rigid_edge_errors.append(rigid_edge_error)
        rigid_area_errors.append(rigid_area_error)

    expected = np.array(
        [
            expected_target_centres[mapping[piece_id]]
            for piece_id in range(expected_piece_count)
        ],
        dtype=np.float64,
    )
    solved_nominal = np.array(
        [
            remove_configured_gap(
                solution.target_transforms[piece_id].translation_mm,
                config,
                solution.applied_clearance_mm,
            )
            for piece_id in range(expected_piece_count)
        ],
        dtype=np.float64,
    )
    layout_rms = rigid_alignment_rms(expected, solved_nominal)

    seam_rows: list[dict[str, Any]] = []
    seam_endpoint_gaps: list[float] = []
    for seam_index, match in enumerate(solution.matches):
        polygon_a = solution.target_polygons_mm[match.piece_a]
        polygon_b = solution.target_polygons_mm[match.piece_b]
        a0 = polygon_a[match.edge_a]
        a1 = polygon_a[(match.edge_a + 1) % len(polygon_a)]
        b0 = polygon_b[match.edge_b]
        b1 = polygon_b[(match.edge_b + 1) % len(polygon_b)]
        endpoint_gap_1 = float(np.linalg.norm(a0 - b1))
        endpoint_gap_2 = float(np.linalg.norm(a1 - b0))
        if match.anchor_mode == "both":
            endpoint_1_is_matched = True
            endpoint_2_is_matched = True
        elif match.anchor_mode == "fixed_start_to_moving_end":
            endpoint_1_is_matched = True
            endpoint_2_is_matched = False
        elif match.anchor_mode == "fixed_end_to_moving_start":
            endpoint_1_is_matched = False
            endpoint_2_is_matched = True
        else:
            raise RuntimeError(
                f"Unknown edge anchor mode: {match.anchor_mode}"
            )
        matched_gaps = [
            gap
            for gap, is_matched in (
                (endpoint_gap_1, endpoint_1_is_matched),
                (endpoint_gap_2, endpoint_2_is_matched),
            )
            if is_matched
        ]
        seam_max = max(matched_gaps)
        seam_endpoint_gaps.extend(matched_gaps)
        seam_rows.append(
            {
                "case_id": case_id,
                "card": card_name,
                "layout_index": layout_index,
                "layout_seed": layout_seed,
                "target_seed": int(metadata["target_seed"]),
                "expected_piece_count": expected_piece_count,
                "target_width_mm": float(metadata["target_width_mm"]),
                "target_height_mm": float(metadata["target_height_mm"]),
                "seam_index": seam_index,
                "piece_a": match.piece_a,
                "edge_a": match.edge_a,
                "piece_b": match.piece_b,
                "edge_b": match.edge_b,
                "anchor_mode": match.anchor_mode,
                "edge_length_normalized_error": match.normalized_length_error,
                "endpoint_gap_1_mm": endpoint_gap_1,
                "endpoint_gap_2_mm": endpoint_gap_2,
                "endpoint_1_is_matched": endpoint_1_is_matched,
                "endpoint_2_is_matched": endpoint_2_is_matched,
                "seam_max_vertex_gap_mm": seam_max,
            }
        )

    long_side, short_side = solution.rectangle_size_mm
    target_long_side = max(
        float(metadata["target_width_mm"]),
        float(metadata["target_height_mm"]),
    )
    target_short_side = min(
        float(metadata["target_width_mm"]),
        float(metadata["target_height_mm"]),
    )
    metrics = {
        "source_center_max_mm": max(center_errors, default=math.inf),
        "source_center_mean_mm": float(np.mean(center_errors)),
        "source_angle_max_deg": max(angle_errors, default=math.inf),
        "source_angle_mean_deg": float(np.mean(angle_errors)),
        "source_vertex_chamfer_max_mm": max(vertex_errors, default=math.inf),
        "source_vertex_chamfer_mean_mm": float(np.mean(vertex_errors)),
        "source_area_error_max_percent": max(
            area_error_percentages,
            default=math.inf,
        ),
        "source_area_error_mean_percent": float(np.mean(area_error_percentages)),
        "nominal_long_mm": float(long_side),
        "nominal_short_mm": float(short_side),
        "target_long_mm": target_long_side,
        "target_short_mm": target_short_side,
        "nominal_long_error_mm": abs(float(long_side) - target_long_side),
        "nominal_short_error_mm": abs(float(short_side) - target_short_side),
        "relative_layout_rms_mm": layout_rms,
        "planned_vertex_gap_max_mm": max(seam_endpoint_gaps, default=0.0),
        "planned_vertex_gap_mean_mm": float(np.mean(seam_endpoint_gaps)),
        "planned_vertex_gap_target_error_mm": abs(
            max(seam_endpoint_gaps, default=0.0)
            - config.target_vertex_gap_mm
        ),
        "planned_overlap_area_mm2": float(
            solution.placement_overlap_area_mm2
        ),
        "rigid_edge_error_max_mm": max(rigid_edge_errors, default=math.inf),
        "rigid_area_error_max_mm2": max(rigid_area_errors, default=math.inf),
    }
    checks = {
        "mode_is_poker": solution.mode == "poker",
        "piece_count_matches": len(pieces) == expected_piece_count,
        **{
            metric: metrics[metric] <= limit
            for metric, limit in STRICT_THRESHOLDS.items()
        },
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    case_row = {
        "case_id": case_id,
        "card": card_name,
        "rank": card_name.split("_of_")[0],
        "suit": card_name.split("_of_")[1],
        "layout_index": layout_index,
        "layout_seed": layout_seed,
        "target_seed": int(metadata["target_seed"]),
        "expected_piece_count": expected_piece_count,
        "target_width_mm": float(metadata["target_width_mm"]),
        "target_height_mm": float(metadata["target_height_mm"]),
        "passed_python": all(checks.values()),
        "failed_checks_python": "; ".join(failed_checks),
        "mode": solution.mode,
        "piece_count": len(pieces),
        "elapsed_ms": elapsed_ms,
        "solver_score": float(solution.score),
        **metrics,
    }
    return case_row, piece_rows, seam_rows


def card_paths(template_dir: Path) -> list[Path]:
    return sorted(template_dir.glob("*_of_*.png"))


def labelled_panel(image: np.ndarray, label: str, width: int = 180) -> np.ndarray:
    target_height = int(round(width * image.shape[0] / image.shape[1]))
    resized = cv2.resize(image, (width, target_height), interpolation=cv2.INTER_AREA)
    panel = cv2.copyMakeBorder(
        resized,
        28,
        2,
        2,
        2,
        cv2.BORDER_CONSTANT,
        value=(245, 245, 245),
    )
    cv2.putText(
        panel,
        label,
        (7, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (25, 25, 25),
        1,
        cv2.LINE_AA,
    )
    return panel


def write_case_images(
    directory: Path,
    image: np.ndarray,
    result: dict[str, Any] | None,
) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    panels: list[np.ndarray] = []
    images = [("input", image)]
    if result is not None:
        segmentation_bgr = cv2.cvtColor(
            result["segmentation_mask"],
            cv2.COLOR_GRAY2BGR,
        )
        images.extend(
            [
                ("segmentation", segmentation_bgr),
                ("detection", result["detection_overlay"]),
                ("solution", result["solution_overlay"]),
                ("reconstructed", result["reconstructed_texture"]),
            ]
        )

    for index, (name, current) in enumerate(images):
        path = directory / f"{index:02d}_{name}.jpg"
        cv2.imwrite(
            str(path),
            current,
            [cv2.IMWRITE_JPEG_QUALITY, 82],
        )
        paths[f"{name}_image"] = str(path.resolve())
        panels.append(labelled_panel(current, name))

    contact = cv2.hconcat(panels)
    contact_path = directory / "contact_sheet.jpg"
    cv2.imwrite(
        str(contact_path),
        contact,
        [cv2.IMWRITE_JPEG_QUALITY, 76],
    )
    paths["contact_sheet"] = str(contact_path.resolve())
    return paths


def distribution_summary(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    values = np.array(
        [float(row[field]) for row in rows if math.isfinite(float(row[field]))],
        dtype=np.float64,
    )
    if len(values) == 0:
        return {"mean": math.nan, "p95": math.nan, "max": math.nan}
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template-dir",
        default=str(Path(__file__).with_name("assets") / "card_templates"),
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.json")),
    )
    parser.add_argument("--layouts-per-card", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=20260729)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).with_name("batch_results_52_strict")),
    )
    parser.add_argument("--max-cards", type=int)
    args = parser.parse_args()

    config = VisionConfig.from_json(args.config)
    template_dir = Path(args.template_dir)
    cards = card_paths(template_dir)
    if args.max_cards is not None:
        cards = cards[: args.max_cards]
    if not cards:
        raise RuntimeError(
            "No card templates found. Run download_card_templates.py first."
        )
    if args.max_cards is None and len(cards) != 52:
        raise RuntimeError(f"Expected 52 templates, found {len(cards)}")

    output_dir = Path(args.output_dir)
    screenshot_root = output_dir / "screenshots"
    output_dir.mkdir(parents=True, exist_ok=True)
    case_rows: list[dict[str, Any]] = []
    piece_rows: list[dict[str, Any]] = []
    seam_rows: list[dict[str, Any]] = []
    total_cases = len(cards) * args.layouts_per_card
    completed_cases = 0

    for card_index, card_path in enumerate(cards):
        template, _ = choose_card_template(str(card_path), None, seed=0)
        if template is None:
            raise RuntimeError(f"Unable to load {card_path}")
        card_passes = 0

        for layout_index in range(args.layouts_per_card):
            seed = args.base_seed + card_index * 1009 + layout_index * 9176
            target_seed = seed + 48611
            expected_piece_count = 2 + ((card_index + layout_index) % 3)
            case_id = (
                f"{card_path.stem}__L{layout_index:02d}"
                f"__P{expected_piece_count}__S{seed}"
            )
            image, metadata = generate_synthetic_image(
                mode="poker",
                config=config,
                template_bgr=template,
                layout_seed=seed,
                target_seed=target_seed,
                piece_count=expected_piece_count,
            )
            result = None
            error = ""
            started = time.perf_counter()
            try:
                result = run_pipeline(
                    image,
                    config,
                    mode="auto",
                    already_rectified=True,
                )
                elapsed_ms = 1000.0 * (time.perf_counter() - started)
                case_row, case_piece_rows, case_seam_rows = evaluate_case(
                    case_id,
                    card_path.stem,
                    layout_index,
                    seed,
                    elapsed_ms,
                    result,
                    metadata,
                    config,
                )
                piece_rows.extend(case_piece_rows)
                seam_rows.extend(case_seam_rows)
            except Exception as exc:
                elapsed_ms = 1000.0 * (time.perf_counter() - started)
                error = f"{type(exc).__name__}: {exc}"
                case_row = {
                    "case_id": case_id,
                    "card": card_path.stem,
                    "rank": card_path.stem.split("_of_")[0],
                    "suit": card_path.stem.split("_of_")[1],
                    "layout_index": layout_index,
                    "layout_seed": seed,
                    "target_seed": target_seed,
                    "expected_piece_count": expected_piece_count,
                    "target_width_mm": float(metadata["target_width_mm"]),
                    "target_height_mm": float(metadata["target_height_mm"]),
                    "passed_python": False,
                    "failed_checks_python": "pipeline_exception",
                    "mode": "error",
                    "piece_count": 0,
                    "elapsed_ms": elapsed_ms,
                    "solver_score": math.nan,
                    **{
                        key: math.nan
                        for key in [
                            "source_center_max_mm",
                            "source_center_mean_mm",
                            "source_angle_max_deg",
                            "source_angle_mean_deg",
                            "source_vertex_chamfer_max_mm",
                            "source_vertex_chamfer_mean_mm",
                            "source_area_error_max_percent",
                            "source_area_error_mean_percent",
                            "nominal_long_mm",
                            "nominal_short_mm",
                            "target_long_mm",
                            "target_short_mm",
                            "nominal_long_error_mm",
                            "nominal_short_error_mm",
                            "relative_layout_rms_mm",
                            "planned_vertex_gap_max_mm",
                            "planned_vertex_gap_mean_mm",
                            "planned_vertex_gap_target_error_mm",
                            "planned_overlap_area_mm2",
                            "rigid_edge_error_max_mm",
                            "rigid_area_error_max_mm2",
                        ]
                    },
                }

            image_paths = write_case_images(
                screenshot_root / card_path.stem / f"layout_{layout_index:02d}",
                image,
                result,
            )
            case_row.update(image_paths)
            case_row["error"] = error
            case_rows.append(case_row)
            card_passes += int(bool(case_row["passed_python"]))
            completed_cases += 1

        print(
            f"[{card_index + 1:02d}/{len(cards):02d}] "
            f"{card_path.stem}: {card_passes}/{args.layouts_per_card} strict passed "
            f"(overall {completed_cases}/{total_cases})",
            flush=True,
        )

    write_csv(output_dir / "cases.csv", case_rows)
    write_csv(output_dir / "piece_errors.csv", piece_rows)
    write_csv(output_dir / "seam_errors.csv", seam_rows)
    (output_dir / "cases.json").write_text(
        json.dumps(case_rows, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    (output_dir / "piece_errors.json").write_text(
        json.dumps(piece_rows, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    (output_dir / "seam_errors.json").write_text(
        json.dumps(seam_rows, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    passed_cases = sum(int(bool(row["passed_python"])) for row in case_rows)
    metric_names = [
        "source_center_max_mm",
        "source_angle_max_deg",
        "source_vertex_chamfer_max_mm",
        "source_area_error_max_percent",
        "nominal_long_error_mm",
        "nominal_short_error_mm",
        "relative_layout_rms_mm",
        "planned_vertex_gap_max_mm",
        "planned_vertex_gap_target_error_mm",
        "planned_overlap_area_mm2",
        "rigid_edge_error_max_mm",
        "rigid_area_error_max_mm2",
        "elapsed_ms",
    ]
    summary = {
        "cards": len(cards),
        "layouts_per_card": args.layouts_per_card,
        "total_cases": len(case_rows),
        "passed_cases_python": passed_cases,
        "failed_cases_python": len(case_rows) - passed_cases,
        "pass_rate_python": passed_cases / max(1, len(case_rows)),
        "all_passed_python": passed_cases == len(case_rows),
        "strict_thresholds": STRICT_THRESHOLDS,
        "config": {
            "pixels_per_mm": config.pixels_per_mm,
            "placement_gap_mm": config.placement_gap_mm,
            "target_vertex_gap_mm": config.target_vertex_gap_mm,
            "target_vertex_gap_tolerance_mm": (
                config.target_vertex_gap_tolerance_mm
            ),
            "max_allowed_vertex_gap_mm": config.max_allowed_vertex_gap_mm,
            "max_post_placement_overlap_mm2": (
                config.max_post_placement_overlap_mm2
            ),
        },
        "piece_count_distribution": {
            str(piece_count): sum(
                int(row.get("expected_piece_count") == piece_count)
                for row in case_rows
            )
            for piece_count in (2, 3, 4)
        },
        "target_dimension_range_mm": {
            "width_min": min(float(row["target_width_mm"]) for row in case_rows),
            "width_max": max(float(row["target_width_mm"]) for row in case_rows),
            "height_min": min(float(row["target_height_mm"]) for row in case_rows),
            "height_max": max(float(row["target_height_mm"]) for row in case_rows),
        },
        "metric_distributions": {
            metric: distribution_summary(case_rows, metric)
            for metric in metric_names
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["all_passed_python"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
