from __future__ import annotations

import math
from typing import Iterable


Vec3 = tuple[float, float, float]


def v_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_scale(a: Vec3, k: float) -> Vec3:
    return (a[0] * k, a[1] * k, a[2] * k)


def v_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def v_norm(a: Vec3) -> float:
    return math.sqrt(v_dot(a, a))


def v_unit(a: Vec3) -> Vec3:
    n = v_norm(a)
    if n == 0.0:
        raise ValueError("zero norm vector")
    return (a[0] / n, a[1] / n, a[2] / n)


def v_sum(vectors: Iterable[Vec3]) -> Vec3:
    sx, sy, sz = 0.0, 0.0, 0.0
    for x, y, z in vectors:
        sx += x
        sy += y
        sz += z
    return (sx, sy, sz)
