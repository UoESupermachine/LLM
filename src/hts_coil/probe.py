from __future__ import annotations

from dataclasses import dataclass
import math

from .field import biot_savart_field


Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class PancakeFieldModel:
    """Configurable pancake coil model for point-wise magnetic field probing.

    Geometry parameters:
    - inner_radius (m)
    - radial_thickness (m): tape thickness in radial direction
    - radial_turns: number of radial turns
    - radial_gap (m): radial gap between turns
    - axial_width (m): used as axial layer pitch
    - axial_layers: number of axial layers
    - center: geometric center (x, y, z)
    """

    inner_radius: float = 0.025
    radial_thickness: float = 0.0003
    radial_turns: int = 20
    radial_gap: float = 0.0003
    axial_width: float = 0.012
    axial_layers: int = 1
    center: Vec3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.inner_radius <= 0:
            raise ValueError("inner_radius must be > 0")
        if self.radial_thickness <= 0:
            raise ValueError("radial_thickness must be > 0")
        if self.radial_turns < 1:
            raise ValueError("radial_turns must be >= 1")
        if self.radial_gap < 0:
            raise ValueError("radial_gap must be >= 0")
        if self.axial_width <= 0:
            raise ValueError("axial_width must be > 0")
        if self.axial_layers < 1:
            raise ValueError("axial_layers must be >= 1")

    def layer_z_positions(self) -> list[float]:
        zc = self.center[2]
        if self.axial_layers == 1:
            return [zc]

        offsets = [
            (i - (self.axial_layers - 1) / 2.0) * self.axial_width
            for i in range(self.axial_layers)
        ]
        return [zc + dz for dz in offsets]

    def turn_radii(self) -> list[float]:
        return [
            self.inner_radius + (i + 0.5) * self.radial_thickness + i * self.radial_gap
            for i in range(self.radial_turns)
        ]

    def build_current_elements(self, current_a: float, n_theta: int = 240) -> tuple[list[Vec3], list[Vec3], list[float]]:
        if n_theta < 8:
            raise ValueError("n_theta must be >= 8")

        dtheta = 2.0 * math.pi / n_theta
        path_count = self.radial_turns * self.axial_layers
        path_current = current_a / path_count

        x0, y0, _ = self.center
        positions: list[Vec3] = []
        dls: list[Vec3] = []
        currents: list[float] = []

        for z in self.layer_z_positions():
            for r in self.turn_radii():
                for i in range(n_theta):
                    th = (i + 0.5) * dtheta
                    c = math.cos(th)
                    s = math.sin(th)

                    pos = (x0 + r * c, y0 + r * s, z)
                    dl = (-s * r * dtheta, c * r * dtheta, 0.0)
                    positions.append(pos)
                    dls.append(dl)
                    currents.append(path_current)

        return positions, dls, currents


def solve_point_fields(
    model: PancakeFieldModel,
    points: list[Vec3],
    current_a: float,
    n_theta: int = 240,
) -> list[dict[str, float | Vec3]]:
    """Solve B at arbitrary points for a DC current.

    Returns per point:
    - Bxyz
    - B_mag
    - B_parallel: parallel to pancake plane (sqrt(Bx^2 + By^2))
    - B_perpendicular: perpendicular to pancake plane (|Bz|)
    """
    src_pos, src_dl, src_i = model.build_current_elements(current_a=current_a, n_theta=n_theta)
    B = biot_savart_field(points, src_pos, src_dl, src_i)

    out: list[dict[str, float | Vec3]] = []
    for p, b in zip(points, B):
        bx, by, bz = b
        b_par = math.sqrt(bx * bx + by * by)
        b_perp = abs(bz)
        b_mag = math.sqrt(bx * bx + by * by + bz * bz)
        out.append(
            {
                "x": p[0],
                "y": p[1],
                "z": p[2],
                "Bxyz": b,
                "B_mag": b_mag,
                "B_parallel": b_par,
                "B_perpendicular": b_perp,
            }
        )
    return out
