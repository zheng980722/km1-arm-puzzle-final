"""Command-line entry point for the E-problem vision V1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from puzzle_vision import (
    VisionConfig,
    execute_control_plan,
    run_pipeline,
)


def parse_corners(value: Optional[str]) -> Optional[np.ndarray]:
    if value is None:
        return None
    points = []
    for pair in value.split(";"):
        x_text, y_text = pair.split(",")
        points.append([float(x_text), float(y_text)])
    if len(points) != 4:
        raise argparse.ArgumentTypeError(
            "--corners needs four points: x1,y1;x2,y2;x3,y3;x4,y4"
        )
    return np.asarray(points, dtype=np.float32)


def load_frame(source: str, warmup_frames: int = 20) -> np.ndarray:
    source_path = Path(source)
    if source_path.exists():
        frame = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Unable to read image: {source}")
        return frame

    try:
        camera_index = int(source)
    except ValueError as exc:
        raise FileNotFoundError(
            f"Input is neither an image path nor a camera index: {source}"
        ) from exc

    backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
    capture = cv2.VideoCapture(camera_index, backend)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}")
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    capture.set(cv2.CAP_PROP_FPS, 30)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frames: list[np.ndarray] = []
    for _ in range(max(1, warmup_frames)):
        ok, current = capture.read()
        if ok:
            frames.append(current)
            frames = frames[-3:]
    capture.release()
    if not frames:
        raise RuntimeError(f"Camera {camera_index} did not return a frame")
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


def serialise_result(result: dict, config: VisionConfig) -> dict:
    solution = result["solution"]
    pieces = result["pieces"]
    return {
        "config": {
            "paper_width_mm": config.paper_width_mm,
            "paper_height_mm": config.paper_height_mm,
            "pixels_per_mm": config.pixels_per_mm,
            "target_center_mm": [
                config.target_center_x_mm,
                config.target_center_y_mm,
            ],
            "placement_gap_mm": config.placement_gap_mm,
            "max_allowed_vertex_gap_mm": config.max_allowed_vertex_gap_mm,
        },
        "background_lab": np.round(result["background_lab"], 3).tolist(),
        "scene_quality": result["scene_quality"],
        "homography": np.round(result["homography"], 8).tolist(),
        "pieces": [piece.to_summary() for piece in pieces],
        "solution": solution.to_dict(pieces),
        "control_plan": result["control_plan"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenCV V1 for the 2026 E-problem puzzle device"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Image path or USB camera index, for example --input 0",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.json")),
        help="Path to JSON configuration",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "self", "white", "poker"],
        default="auto",
        help="Puzzle mode; auto classifies from colour and texture",
    )
    parser.add_argument(
        "--rectified",
        action="store_true",
        help="Input image is already a top-down A4 image",
    )
    parser.add_argument(
        "--corners",
        help="Manual A4 corners: x1,y1;x2,y2;x3,y3;x4,y4",
    )
    parser.add_argument(
        "--output-dir",
        default="vision_output",
        help="Directory for masks, overlays, and result.json",
    )
    parser.add_argument(
        "--print-control",
        action="store_true",
        help="Call the safe control placeholder after vision succeeds",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = VisionConfig.from_json(args.config)
    frame = load_frame(args.input)
    corners = parse_corners(args.corners)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "00_input.png"), frame)

    result = run_pipeline(
        frame,
        config,
        mode=args.mode,
        corners=corners,
        already_rectified=args.rectified,
    )

    cv2.imwrite(str(output_dir / "01_rectified.png"), result["rectified"])
    cv2.imwrite(str(output_dir / "02_segmentation.png"), result["segmentation_mask"])
    cv2.imwrite(str(output_dir / "02_green_mask.png"), result["green_mask"])
    cv2.imwrite(str(output_dir / "02_quad_debug.png"), result["quad_debug"])
    cv2.imwrite(
        str(output_dir / "03_detection_overlay.png"),
        result["detection_overlay"],
    )
    cv2.imwrite(
        str(output_dir / "04_solution_overlay.png"),
        result["solution_overlay"],
    )
    cv2.imwrite(
        str(output_dir / "05_reconstructed_texture.png"),
        result["reconstructed_texture"],
    )
    data = serialise_result(result, config)
    (output_dir / "result.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    solution = result["solution"]
    print(
        f"mode={solution.mode} pieces={len(result['pieces'])} "
        f"score={solution.score:.4f} "
        f"rectangle={solution.rectangle_size_mm[0]:.1f}x"
        f"{solution.rectangle_size_mm[1]:.1f} mm "
        f"min_pair_gap={solution.min_pairwise_gap_mm:.1f} mm"
    )
    print(f"Outputs written to: {output_dir.resolve()}")

    if args.print_control:
        execute_control_plan(result["control_plan"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
