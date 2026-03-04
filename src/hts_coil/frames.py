from __future__ import annotations

import math

from .utils import Vec3, v_dot


def decompose_B(Bxyz: Vec3, e_t: Vec3, e_w: Vec3, e_n: Vec3) -> tuple[float, float, float, float]:
    bt = v_dot(Bxyz, e_t)
    bw = v_dot(Bxyz, e_w)
    bn = v_dot(Bxyz, e_n)

    B_parallel = math.sqrt(bt * bt + bw * bw)
    B_perpendicular = abs(bn)
    B_mag = math.sqrt(v_dot(Bxyz, Bxyz))
    theta = math.atan2(B_perpendicular, B_parallel if B_parallel > 1e-15 else 1e-15)
    return B_mag, B_parallel, B_perpendicular, theta
