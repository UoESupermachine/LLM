from __future__ import annotations

import math

from .utils import Vec3, v_add, v_cross, v_scale, v_sub

MU0 = 4.0e-7 * math.pi


def biot_savart_field(
    eval_points: list[Vec3],
    element_positions: list[Vec3],
    element_dls: list[Vec3],
    element_currents: list[float],
    mu0: float = MU0,
    eps: float = 1e-12,
) -> list[Vec3]:
    """Compute B field by discrete Biot-Savart superposition."""
    prefactor = mu0 / (4.0 * math.pi)
    out: list[Vec3] = []

    for p in eval_points:
        b: Vec3 = (0.0, 0.0, 0.0)
        for src, dl, cur in zip(element_positions, element_dls, element_currents):
            r = v_sub(p, src)
            r_norm = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
            if r_norm < eps:
                r_norm = eps
            cross = v_cross(dl, r)
            scale = prefactor * cur / (r_norm**3)
            b = v_add(b, v_scale(cross, scale))
        out.append(b)

    return out
