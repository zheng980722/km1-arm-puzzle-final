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


PAPER_DEPTH_MM = 210.0
PAPER_WIDTH_MM = 297.0
BASE_FRONT_RADIUS_MM = 55.0

# Camera optical-centre projection measured from the coincident midpoint of
# the A4 near long edge and the circular-base front edge.  Positive x points
# to the physical right; positive y points away from the robot over the A4.
CAMERA_FROM_NEAR_EDGE_X_MM = 25.0
CAMERA_FROM_NEAR_EDGE_Y_MM = 30.0
CAMERA_HEIGHT_ABOVE_PAPER_MM = 600.0
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


def paper_to_robot(paper_x_mm: float, paper_y_mm: float) -> tuple[float, float]:
    """Map rectified vision millimetres to the current KM1 bench frame."""

    robot_x_mm = PAPER_WIDTH_MM / 2.0 - float(paper_y_mm)
    robot_y_mm = (
        BASE_FRONT_RADIUS_MM + PAPER_DEPTH_MM - float(paper_x_mm)
    )
    return robot_x_mm, robot_y_mm


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
