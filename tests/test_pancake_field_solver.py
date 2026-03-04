from __future__ import annotations

from dataclasses import dataclass
import math

from hts_coil.field import biot_savart_field


Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class PancakeCoilModel:
    """Configurable pancake coil model for B-field probing.

    Parameters map to user-provided geometry:
    - inner_radius: inr
    - radial_thickness: th1 (tape radial thickness)
    - radial_turns: N1
    - radial_gap: g1
    - axial_width: w1 (used for axial layer positioning)
    - axial_layers: N2
    - center: coil geometric center (x, y, z)
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

    @property
    def radial_pitch(self) -> float:
        return self.radial_thickness + self.radial_gap

    def layer_z_positions(self) -> list[float]:
        # Symmetric placement about center z. Using axial_width as layer-to-layer pitch.
        zc = self.center[2]
        if self.axial_layers == 1:
            return [zc]

        offsets = [
            (i - (self.axial_layers - 1) / 2.0) * self.axial_width
            for i in range(self.axial_layers)
        ]
        return [zc + dz for dz in offsets]

    def turn_radii(self) -> list[float]:
        # Use centerline radius for each radial turn stripe.
        return [
            self.inner_radius + (i + 0.5) * self.radial_thickness + i * self.radial_gap
            for i in range(self.radial_turns)
        ]

    def build_current_elements(self, total_current: float, n_theta: int = 240) -> tuple[list[Vec3], list[Vec3], list[float]]:
        """Generate discrete line elements for all turns/layers.

        Current is uniformly distributed to each turn-layer path.
        """
        if n_theta < 8:
            raise ValueError("n_theta must be >= 8")

        dtheta = 2.0 * math.pi / n_theta
        paths = self.radial_turns * self.axial_layers
        i_path = total_current / paths

        src_pos: list[Vec3] = []
        src_dl: list[Vec3] = []
        src_i: list[float] = []

        x0, y0, _ = self.center
        for z in self.layer_z_positions():
            for r in self.turn_radii():
                for k in range(n_theta):
                    th = (k + 0.5) * dtheta
                    c = math.cos(th)
                    s = math.sin(th)

                    pos = (x0 + r * c, y0 + r * s, z)
                    # azimuthal tangent * arc length
                    dl = (-s * r * dtheta, c * r * dtheta, 0.0)
                    src_pos.append(pos)
                    src_dl.append(dl)
                    src_i.append(i_path)

        return src_pos, src_dl, src_i


def solve_field_at_points(
    model: PancakeCoilModel,
    points: list[Vec3],
    total_current: float,
    n_theta: int = 240,
) -> list[dict[str, float | Vec3]]:
    """Compute B at arbitrary points.

    Returns for each point:
    - Bxyz: (Bx, By, Bz)
    - B_mag: |B|
    - B_parallel: sqrt(Bx^2 + By^2)  (parallel to coil plane)
    - B_perpendicular: |Bz|           (perpendicular to coil plane)
    """
    src_pos, src_dl, src_i = model.build_current_elements(total_current=total_current, n_theta=n_theta)
    bvals = biot_savart_field(points, src_pos, src_dl, src_i)

    out: list[dict[str, float | Vec3]] = []
    for p, b in zip(points, bvals):
        bx, by, bz = b
        b_parallel = math.sqrt(bx * bx + by * by)
        b_perp = abs(bz)
        b_mag = math.sqrt(bx * bx + by * by + bz * bz)
        out.append(
            {
                "x": p[0],
                "y": p[1],
                "z": p[2],
                "Bxyz": b,
                "B_mag": b_mag,
                "B_parallel": b_parallel,
                "B_perpendicular": b_perp,
            }
        )
    return out


def test_default_geometry_center_axis_field() -> None:
    model = PancakeCoilModel(
        inner_radius=0.025,
        radial_thickness=0.0003,
        radial_turns=20,
        radial_gap=0.0003,
        axial_width=0.012,
        axial_layers=1,
        center=(0.0, 0.0, 0.0),
    )

    results = solve_field_at_points(
        model=model,
        points=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.01)],
        total_current=120.0,
        n_theta=180,
    )

    # On the axis, in-plane component should be close to zero by symmetry.
    assert results[0]["B_parallel"] < 1e-6
    assert results[1]["B_parallel"] < 1e-6
    assert results[0]["B_perpendicular"] > 0.0
    assert results[1]["B_perpendicular"] > 0.0


if __name__ == "__main__":
    # Quick manual probe for FEM cross-check input/output.
    model = PancakeCoilModel(
        inner_radius=0.025,
        radial_thickness=0.0003,
        radial_turns=20,
        radial_gap=0.0003,
        axial_width=0.012,
        axial_layers=1,
        center=(0.0, 0.0, 0.0),
    )
    probe_points = [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.01),
        (0.03, 0.0, 0.0),
    ]
    ans = solve_field_at_points(model, probe_points, total_current=120.0, n_theta=180)
    for row in ans:
        print(row)
