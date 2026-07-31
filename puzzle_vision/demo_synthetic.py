"""Generate a rectified synthetic A4 image for a quick end-to-end test."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from puzzle_vision import VisionConfig, rotation_matrix


def heart_polygon(center: tuple[int, int], size: int) -> np.ndarray:
    cx, cy = center
    points = []
    for degree in np.linspace(0, 360, 180, endpoint=False):
        t = math.radians(float(degree))
        x = 16 * math.sin(t) ** 3
        y = (
            13 * math.cos(t)
            - 5 * math.cos(2 * t)
            - 2 * math.cos(3 * t)
            - math.cos(4 * t)
        )
        points.append([cx + x * size / 32.0, cy - y * size / 32.0])
    return np.round(points).astype(np.int32)


def build_card_texture(width_px: int, height_px: int) -> np.ndarray:
    card = np.full((height_px, width_px, 3), 248, dtype=np.uint8)
    cv2.putText(
        card,
        "A",
        (18, 62),
        cv2.FONT_HERSHEY_DUPLEX,
        1.7,
        (20, 20, 20),
        3,
        cv2.LINE_AA,
    )
    cv2.fillPoly(
        card,
        [heart_polygon((width_px // 2, height_px // 2), min(width_px, height_px) // 2)],
        (20, 20, 220),
    )
    cv2.line(
        card,
        (width_px // 5, height_px * 4 // 5),
        (width_px * 4 // 5, height_px // 5),
        (30, 30, 30),
        6,
        cv2.LINE_AA,
    )
    return card


def fit_template_to_card(
    template_bgr: np.ndarray,
    width_px: int,
    height_px: int,
) -> np.ndarray:
    """Fit a real card template to a landscape white canvas without distortion."""

    if template_bgr.shape[0] > template_bgr.shape[1]:
        template_bgr = cv2.rotate(template_bgr, cv2.ROTATE_90_CLOCKWISE)
    source_h, source_w = template_bgr.shape[:2]
    scale = min(width_px / source_w, height_px / source_h)
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    resized = cv2.resize(
        template_bgr,
        (resized_w, resized_h),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height_px, width_px, 3), 248, dtype=np.uint8)
    x0 = (width_px - resized_w) // 2
    y0 = (height_px - resized_h) // 2
    canvas[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized
    return canvas


def choose_card_template(
    explicit_template: str | None,
    template_dir: str | None,
    seed: int,
) -> tuple[np.ndarray | None, str | None]:
    candidates: list[Path] = []
    if explicit_template:
        candidates.append(Path(explicit_template))
    elif template_dir:
        directory = Path(template_dir)
        if directory.exists():
            candidates.extend(sorted(directory.glob("*.png")))
            candidates.extend(sorted(directory.glob("*.jpg")))
            candidates.extend(sorted(directory.glob("*.jpeg")))
    if not candidates:
        return None, None

    if explicit_template:
        selected = candidates[0]
    else:
        generator = np.random.default_rng(seed)
        selected = candidates[int(generator.integers(0, len(candidates)))]
    image = cv2.imread(str(selected), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Unable to read card template: {selected}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        bgr = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        white = np.full_like(bgr, 255.0)
        image = np.round(bgr * alpha + white * (1.0 - alpha)).astype(np.uint8)
    return image[:, :, :3], str(selected)


def default_target_polygons_mm() -> list[np.ndarray]:
    return [
        np.array([[0, 0], [35, 0], [45, 28], [0, 40]], dtype=np.float64),
        np.array([[35, 0], [100, 0], [100, 20], [45, 28]], dtype=np.float64),
        np.array([[45, 28], [100, 20], [100, 60], [70, 60]], dtype=np.float64),
        np.array([[0, 40], [45, 28], [70, 60], [0, 60]], dtype=np.float64),
    ]


def _target_polygons_are_rule_legal(
    polygons: list[np.ndarray],
    width_mm: float,
    height_mm: float,
    *,
    minimum_edge_mm: float = 20.0,
    minimum_vertex_deviation_mm: float = 4.0,
) -> bool:
    """Check the explicit geometry limits from requirement 2."""

    tolerance = 1e-6
    for polygon in polygons:
        if len(polygon) > 5 or len(polygon) < 3:
            return False
        lengths = np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)
        if float(np.min(lengths)) + tolerance < minimum_edge_mm:
            return False
        for index in range(len(polygon)):
            previous = polygon[(index - 1) % len(polygon)]
            current = polygon[index]
            following = polygon[(index + 1) % len(polygon)]
            baseline = following - previous
            baseline_length = float(np.linalg.norm(baseline))
            if baseline_length <= tolerance:
                return False
            cross = baseline[0] * (current - previous)[1] - baseline[1] * (
                current - previous
            )[0]
            deviation = abs(float(cross)) / baseline_length
            # The rule does not prescribe a minimum corner angle.  Synthetic
            # regression excludes near-collinear junctions that have no
            # reliably observable image corner at the available resolution.
            if deviation + tolerance < minimum_vertex_deviation_mm:
                return False

        on_left = np.isclose(polygon[:, 0], 0.0, atol=tolerance)
        on_right = np.isclose(polygon[:, 0], width_mm, atol=tolerance)
        on_top = np.isclose(polygon[:, 1], 0.0, atol=tolerance)
        on_bottom = np.isclose(polygon[:, 1], height_mm, atol=tolerance)
        boundary_edge = False
        for index in range(len(polygon)):
            next_index = (index + 1) % len(polygon)
            if (
                (on_left[index] and on_left[next_index])
                or (on_right[index] and on_right[next_index])
                or (on_top[index] and on_top[next_index])
                or (on_bottom[index] and on_bottom[next_index])
            ):
                boundary_edge = True
                break
        if not boundary_edge:
            return False
    return True


def random_rule_target_polygons_mm(
    seed: int,
    *,
    piece_count: int | None = None,
) -> tuple[list[np.ndarray], float, float]:
    """Generate a legal on-site rectangle and an unknown 2–4 piece cut.

    The generated cases follow requirement 2:
      * target width 90–120 mm and height 50–90 mm;
      * 2–4 pieces;
      * at most five edges per piece;
      * every edge at least 20 mm;
      * every piece owns at least one target-rectangle boundary edge.
    """

    generator = np.random.default_rng(seed)
    selected_count = (
        int(piece_count)
        if piece_count is not None
        else int(generator.integers(2, 5))
    )
    if selected_count not in (2, 3, 4):
        raise ValueError("piece_count must be 2, 3, or 4")

    boundary_margin = 22.0
    for _ in range(4000):
        width = float(generator.uniform(90.0, 120.0))
        height = float(generator.uniform(50.0, 90.0))

        if selected_count == 2:
            top_x = float(
                generator.uniform(boundary_margin, width - boundary_margin)
            )
            bottom_x = float(
                generator.uniform(boundary_margin, width - boundary_margin)
            )
            polygons = [
                np.array(
                    [[0.0, 0.0], [top_x, 0.0], [bottom_x, height], [0.0, height]],
                    dtype=np.float64,
                ),
                np.array(
                    [
                        [top_x, 0.0],
                        [width, 0.0],
                        [width, height],
                        [bottom_x, height],
                    ],
                    dtype=np.float64,
                ),
            ]
        elif selected_count == 3:
            top_x = float(
                generator.uniform(boundary_margin, width - boundary_margin)
            )
            left_y = float(
                generator.uniform(boundary_margin, height - boundary_margin)
            )
            right_y = float(
                generator.uniform(boundary_margin, height - boundary_margin)
            )
            centre = np.array(
                [
                    generator.uniform(0.40 * width, 0.60 * width),
                    generator.uniform(0.42 * height, 0.58 * height),
                ],
                dtype=np.float64,
            )
            top = np.array([top_x, 0.0])
            left = np.array([0.0, left_y])
            right = np.array([width, right_y])
            polygons = [
                np.array([top, [width, 0.0], right, centre], dtype=np.float64),
                np.array(
                    [
                        right,
                        [width, height],
                        [0.0, height],
                        left,
                        centre,
                    ],
                    dtype=np.float64,
                ),
                np.array([left, [0.0, 0.0], top, centre], dtype=np.float64),
            ]
        else:
            top = np.array(
                [
                    generator.uniform(boundary_margin, width - boundary_margin),
                    0.0,
                ],
                dtype=np.float64,
            )
            right = np.array(
                [
                    width,
                    generator.uniform(boundary_margin, height - boundary_margin),
                ],
                dtype=np.float64,
            )
            bottom = np.array(
                [
                    generator.uniform(boundary_margin, width - boundary_margin),
                    height,
                ],
                dtype=np.float64,
            )
            left = np.array(
                [
                    0.0,
                    generator.uniform(boundary_margin, height - boundary_margin),
                ],
                dtype=np.float64,
            )
            centre = np.array(
                [
                    generator.uniform(0.40 * width, 0.60 * width),
                    generator.uniform(0.42 * height, 0.58 * height),
                ],
                dtype=np.float64,
            )
            polygons = [
                np.array([[0.0, 0.0], top, centre, left], dtype=np.float64),
                np.array([top, [width, 0.0], right, centre], dtype=np.float64),
                np.array(
                    [centre, right, [width, height], bottom],
                    dtype=np.float64,
                ),
                np.array(
                    [left, centre, bottom, [0.0, height]],
                    dtype=np.float64,
                ),
            ]

        if _target_polygons_are_rule_legal(polygons, width, height):
            return polygons, width, height

    raise RuntimeError("Unable to generate a rule-legal random target cut")


def random_source_layout(
    target_polygons_mm: list[np.ndarray],
    config: VisionConfig,
    seed: int,
    clearance_mm: float = 2.0,
) -> tuple[list[np.ndarray], list[float]]:
    """Generate non-overlapping source poses inside the upper A4 half."""

    generator = np.random.default_rng(seed)
    scale = config.pixels_per_mm
    margin_mm = 5.0
    clearance_px = max(1, int(round(clearance_mm * scale)))
    clearance_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * clearance_px + 1, 2 * clearance_px + 1),
    )

    # Place larger pieces first to keep rejection sampling efficient.
    order = sorted(
        range(len(target_polygons_mm)),
        key=lambda index: cv2.contourArea(
            target_polygons_mm[index].astype(np.float32)
        ),
        reverse=True,
    )
    # A greedy placement can occasionally trap the final large fragment.
    # Restart the whole placement instead of declaring a valid geometry
    # impossible after one unlucky sequence.
    for _restart in range(24):
        occupied = np.zeros(
            (config.rectified_height_px, config.rectified_width_px),
            dtype=np.uint8,
        )
        poses: dict[int, tuple[np.ndarray, float]] = {}
        all_accepted = True
        for piece_index in order:
            polygon = target_polygons_mm[piece_index]
            reference_centre = np.mean(polygon, axis=0)
            local = polygon - reference_centre
            accepted = False
            for _ in range(1800):
                angle = float(generator.uniform(-180.0, 180.0))
                rotated = local @ rotation_matrix(angle).T
                minimum = np.min(rotated, axis=0)
                maximum = np.max(rotated, axis=0)
                x_low = margin_mm - minimum[0]
                x_high = config.paper_width_mm - margin_mm - maximum[0]
                y_low = margin_mm - minimum[1]
                y_high = (
                    config.divider_y_mm
                    - config.divider_ignore_mm
                    - margin_mm
                    - maximum[1]
                )
                if x_low >= x_high or y_low >= y_high:
                    continue
                centre = np.array(
                    [
                        generator.uniform(x_low, x_high),
                        generator.uniform(y_low, y_high),
                    ],
                    dtype=np.float64,
                )
                candidate = rotated + centre
                candidate_px = np.round(candidate * scale).astype(np.int32)
                mask = np.zeros_like(occupied)
                cv2.fillPoly(mask, [candidate_px], 255)
                expanded = cv2.dilate(mask, clearance_kernel, iterations=1)
                if cv2.countNonZero(cv2.bitwise_and(expanded, occupied)) > 0:
                    continue
                occupied = cv2.bitwise_or(occupied, expanded)
                poses[piece_index] = (centre, angle)
                accepted = True
                break
            if not accepted:
                all_accepted = False
                break
        if all_accepted:
            centres: list[np.ndarray] = []
            angles: list[float] = []
            for piece_index in range(len(target_polygons_mm)):
                centre, angle = poses[piece_index]
                centres.append(centre)
                angles.append(angle)
            return centres, angles

    raise RuntimeError("Unable to place all synthetic pieces without overlap")


def generate_synthetic_image(
    *,
    mode: str,
    config: VisionConfig,
    template_bgr: np.ndarray | None = None,
    layout_seed: int | None = None,
    target_seed: int | None = None,
    piece_count: int | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Generate one synthetic A4 scene and return its ground-truth metadata."""

    scale = config.pixels_per_mm
    image = np.full(
        (config.rectified_height_px, config.rectified_width_px, 3),
        (55, 155, 65),
        dtype=np.uint8,
    )
    divider_y = int(round(config.divider_y_mm * scale))
    cv2.line(image, (0, divider_y), (image.shape[1] - 1, divider_y), (20, 20, 20), 8)

    if target_seed is None:
        target_polygons_mm = default_target_polygons_mm()
        target_width_mm = 100.0
        target_height_mm = 60.0
    else:
        (
            target_polygons_mm,
            target_width_mm,
            target_height_mm,
        ) = random_rule_target_polygons_mm(
            target_seed,
            piece_count=piece_count,
        )
    if layout_seed is None:
        source_centres_mm = [
            np.array([45.0, 35.0]),
            np.array([145.0, 35.0]),
            np.array([155.0, 105.0]),
            np.array([55.0, 105.0]),
        ]
        source_angles = [22.0, -31.0, 47.0, -18.0]
    else:
        source_centres_mm, source_angles = random_source_layout(
            target_polygons_mm,
            config,
            layout_seed,
        )

    card_width_px = int(round(target_width_mm * scale))
    card_height_px = int(round(target_height_mm * scale))
    target_texture = (
        build_card_texture(card_width_px, card_height_px)
        if template_bgr is None
        else fit_template_to_card(template_bgr, card_width_px, card_height_px)
    )

    for piece_id, target_polygon in enumerate(target_polygons_mm):
        target_polygon_px = np.round(target_polygon * scale).astype(np.int32)
        piece_mask = np.zeros(target_texture.shape[:2], dtype=np.uint8)
        cv2.fillPoly(piece_mask, [target_polygon_px], 255)

        if mode == "self":
            texture = np.full_like(target_texture, (30, 220, 245))
        elif mode == "white":
            texture = np.full_like(target_texture, 245)
        else:
            texture = target_texture

        centre_mm = np.mean(target_polygon, axis=0)
        rotation = rotation_matrix(source_angles[piece_id])
        translation_px = scale * (
            source_centres_mm[piece_id] - rotation @ centre_mm
        )
        affine = np.column_stack([rotation, translation_px]).astype(np.float32)
        warped_texture = cv2.warpAffine(
            texture,
            affine,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_mask = cv2.warpAffine(
            piece_mask,
            affine,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        image[warped_mask > 0] = warped_texture[warped_mask > 0]

    metadata: dict[str, object] = {
        "target_polygons_mm": target_polygons_mm,
        "source_centres_mm": source_centres_mm,
        "source_angles_deg": source_angles,
        "layout_seed": layout_seed,
        "target_seed": target_seed,
        "target_width_mm": target_width_mm,
        "target_height_mm": target_height_mm,
        "expected_piece_count": len(target_polygons_mm),
    }
    return image, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["self", "white", "poker"],
        default="poker",
    )
    parser.add_argument(
        "--output",
        default="synthetic_puzzle.png",
    )
    parser.add_argument(
        "--template",
        help="Use one PNG/JPG playing-card face as the poker texture",
    )
    parser.add_argument(
        "--template-dir",
        default=str(Path(__file__).with_name("assets") / "card_templates"),
        help="Randomly choose a PNG/JPG playing-card face from this directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed used when choosing a template",
    )
    parser.add_argument(
        "--layout-seed",
        type=int,
        help="Randomise all four source positions and angles with this seed",
    )
    args = parser.parse_args()

    config = VisionConfig()
    external_template, selected_template = choose_card_template(
        args.template,
        args.template_dir,
        args.seed,
    )
    if external_template is None:
        selected_template = "built-in OpenCV demo pattern"
    image, _ = generate_synthetic_image(
        mode=args.mode,
        config=config,
        template_bgr=external_template,
        layout_seed=args.layout_seed,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"Unable to write {output}")
    print(output.resolve())
    print(f"template={selected_template}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
