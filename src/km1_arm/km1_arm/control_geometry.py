"""Bench geometry shared by KM1 visual planning and arm control.

The current fixture uses a landscape A4 sheet.  Its near long edge is tangent
to the front edge of the KM1 circular base, and the midpoint of both edges is
aligned.  The vision pipeline rectifies that landscape sheet into a portrait
210 x 297 mm coordinate system:

* paper x: far edge (0) -> near/base edge (210)
* paper y: physical right edge (0) -> physical left edge (297)

Robot x is positive to the physical right and robot y is positive away from
the base centre towards the sheet.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path


PAPER_DEPTH_MM = 210.0
PAPER_WIDTH_MM = 297.0
BASE_FRONT_RADIUS_MM = 55.0

# Current bench height datum (2026-07-31).  The A4 support raises the working
# plane 30 mm above the floor and the arm base board is 10 mm above the floor.
# KM1 kinematics are base-board referenced, so the paper surface is 20 mm in
# the robot Z frame.  Pick/drop clearances are added to this relative height.
PAPER_SURFACE_Z_MM = 20.0
BASE_BOARD_Z_MM = 10.0

# Camera optical-centre projection now falls directly on the midpoint of the
# A4 near long edge.  The lens is 460 mm above the floor and the raised paper
# surface is 30 mm above the floor, hence a 430 mm lens-to-paper distance.
# Positive x points to the physical right; positive y points away from the
# robot over the A4.
CAMERA_FROM_NEAR_EDGE_X_MM = 0.0
CAMERA_FROM_NEAR_EDGE_Y_MM = 0.0
CAMERA_HEIGHT_ABOVE_PAPER_MM = 430.0
CAMERA_ROBOT_X_MM = CAMERA_FROM_NEAR_EDGE_X_MM
CAMERA_ROBOT_Y_MM = BASE_FRONT_RADIUS_MM + CAMERA_FROM_NEAR_EDGE_Y_MM

# Modified end effector: rotating electromagnet cylinder.
MAGNET_RADIUS_MM = 20.0
MAGNET_HEIGHT_MM = 20.0

# Wrist-axis to magnet contact-face length calibrated from the real P3 hover
# test.  The former 80 mm estimate left the contact-face centre about 14 mm
# short in the horizontal plane at alpha=-65 degrees; 50 mm aligned the
# elevated tool projection with the vision target.
MODIFIED_TOOL_LENGTH_MM = 50.0


@dataclass(frozen=True)
class PlacementCalibration:
    """Invert a measured command-to-actual affine placement model.

    The calibration is based on paper coordinates and is independent of
    fragment IDs, dimensions and puzzle layouts.  Pickup coordinates remain
    untouched; this model compensates only destination placement commands.
    """

    enabled: bool = False
    matrix: tuple[tuple[float, float], tuple[float, float]] = (
        (1.0, 0.0),
        (0.0, 1.0),
    )
    offset_mm: tuple[float, float] = (0.0, 0.0)
    max_compensation_mm: float = 0.0
    source: str = "identity"

    @classmethod
    def from_json(cls, path: str | Path) -> "PlacementCalibration":
        calibration_path = Path(path)
        data = json.loads(calibration_path.read_text(encoding="utf-8"))
        matrix_data = data.get("command_to_actual_matrix")
        offset_data = data.get("command_to_actual_offset_mm")
        if (
            not isinstance(matrix_data, list)
            or len(matrix_data) != 2
            or any(not isinstance(row, list) or len(row) != 2 for row in matrix_data)
        ):
            raise ValueError(
                "placement calibration matrix must be a 2x2 JSON array"
            )
        if not isinstance(offset_data, list) or len(offset_data) != 2:
            raise ValueError(
                "placement calibration offset must contain two values"
            )
        matrix = (
            (float(matrix_data[0][0]), float(matrix_data[0][1])),
            (float(matrix_data[1][0]), float(matrix_data[1][1])),
        )
        determinant = (
            matrix[0][0] * matrix[1][1]
            - matrix[0][1] * matrix[1][0]
        )
        if abs(determinant) < 1e-6:
            raise ValueError("placement calibration matrix is singular")
        return cls(
            enabled=bool(data.get("enabled", False)),
            matrix=matrix,
            offset_mm=(float(offset_data[0]), float(offset_data[1])),
            max_compensation_mm=max(
                0.0,
                float(data.get("max_compensation_mm", 0.0)),
            ),
            source=str(calibration_path),
        )

    def expected_actual(
        self,
        command_x_mm: float,
        command_y_mm: float,
    ) -> tuple[float, float]:
        """Predict the observed paper point for an uncompensated command."""

        x = float(command_x_mm)
        y = float(command_y_mm)
        return (
            self.matrix[0][0] * x
            + self.matrix[0][1] * y
            + self.offset_mm[0],
            self.matrix[1][0] * x
            + self.matrix[1][1] * y
            + self.offset_mm[1],
        )

    def command_for_desired(
        self,
        desired_x_mm: float,
        desired_y_mm: float,
    ) -> tuple[float, float]:
        """Return the pre-distorted command expected to land at the target."""

        desired = (float(desired_x_mm), float(desired_y_mm))
        if not self.enabled:
            return desired

        a11, a12 = self.matrix[0]
        a21, a22 = self.matrix[1]
        determinant = a11 * a22 - a12 * a21
        residual_x = desired[0] - self.offset_mm[0]
        residual_y = desired[1] - self.offset_mm[1]
        command_x = (
            a22 * residual_x - a12 * residual_y
        ) / determinant
        command_y = (
            -a21 * residual_x + a11 * residual_y
        ) / determinant

        delta_x = command_x - desired[0]
        delta_y = command_y - desired[1]
        magnitude = math.hypot(delta_x, delta_y)
        if (
            self.max_compensation_mm > 0.0
            and magnitude > self.max_compensation_mm
        ):
            scale = self.max_compensation_mm / magnitude
            command_x = desired[0] + scale * delta_x
            command_y = desired[1] + scale * delta_y
        return command_x, command_y


def _load_placement_calibration() -> PlacementCalibration:
    path = os.environ.get("KM1_PLACEMENT_CALIBRATION_FILE", "").strip()
    if not path:
        return PlacementCalibration()
    return PlacementCalibration.from_json(path)


PLACEMENT_CALIBRATION = _load_placement_calibration()


def paper_to_robot(paper_x_mm: float, paper_y_mm: float) -> tuple[float, float]:
    """Map rectified vision millimetres to the current KM1 bench frame."""

    robot_x_mm = PAPER_WIDTH_MM / 2.0 - float(paper_y_mm)
    robot_y_mm = (
        BASE_FRONT_RADIUS_MM + PAPER_DEPTH_MM - float(paper_x_mm)
    )
    return robot_x_mm, robot_y_mm


def placement_command_paper(
    desired_x_mm: float,
    desired_y_mm: float,
) -> tuple[float, float]:
    """Pre-distort a desired placement point using the active calibration."""

    return PLACEMENT_CALIBRATION.command_for_desired(
        desired_x_mm,
        desired_y_mm,
    )


def placement_to_robot(
    desired_x_mm: float,
    desired_y_mm: float,
) -> tuple[float, float]:
    """Map a desired placement point through calibration into robot XY."""

    command_x, command_y = placement_command_paper(
        desired_x_mm,
        desired_y_mm,
    )
    return paper_to_robot(command_x, command_y)


def robot_to_paper(robot_x_mm: float, robot_y_mm: float) -> tuple[float, float]:
    """Inverse of :func:`paper_to_robot`, useful during calibration."""

    paper_x_mm = (
        BASE_FRONT_RADIUS_MM + PAPER_DEPTH_MM - float(robot_y_mm)
    )
    paper_y_mm = PAPER_WIDTH_MM / 2.0 - float(robot_x_mm)
    return paper_x_mm, paper_y_mm


def validate_paper_point(
    paper_x_mm: float,
    paper_y_mm: float,
    edge_margin_mm: float = MAGNET_RADIUS_MM,
) -> None:
    """Reject a test centre whose magnet footprint crosses the A4 boundary."""

    x = float(paper_x_mm)
    y = float(paper_y_mm)
    margin = max(0.0, float(edge_margin_mm))
    if not margin <= x <= PAPER_DEPTH_MM - margin:
        raise ValueError(
            f"paper_x={x:.1f} mm is outside the safe centre range "
            f"[{margin:.1f}, {PAPER_DEPTH_MM - margin:.1f}]"
        )
    if not margin <= y <= PAPER_WIDTH_MM - margin:
        raise ValueError(
            f"paper_y={y:.1f} mm is outside the safe centre range "
            f"[{margin:.1f}, {PAPER_WIDTH_MM - margin:.1f}]"
        )
