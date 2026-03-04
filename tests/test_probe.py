from __future__ import annotations

from hts_coil.probe import PancakeFieldModel, solve_point_fields


def test_can_solve_field_for_100a_at_arbitrary_points() -> None:
    model = PancakeFieldModel(
        inner_radius=0.025,
        radial_thickness=0.0003,
        radial_turns=20,
        radial_gap=0.0003,
        axial_width=0.012,
        axial_layers=1,
        center=(0.0, 0.0, 0.0),
    )

    points = [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.01),
        (0.03, 0.0, 0.0),
    ]

    out = solve_point_fields(model=model, points=points, current_a=100.0, n_theta=180)

    assert len(out) == 3
    assert out[0]["B_mag"] > 0.0
    assert out[1]["B_mag"] > 0.0
    assert out[2]["B_mag"] > 0.0

    # On axis, in-plane component should be near zero by symmetry.
    assert out[0]["B_parallel"] < 1e-6
    assert out[1]["B_parallel"] < 1e-6
