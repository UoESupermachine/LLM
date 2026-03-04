from __future__ import annotations

from dataclasses import dataclass
import math

from .geometry import PancakeCoilSpec
from .utils import Vec3, v_cross, v_scale, v_unit


@dataclass(frozen=True)
class Segment:
    position: Vec3
    dl: Vec3
    current: float
    e_t: Vec3
    e_w: Vec3
    e_n: Vec3
    turn_index: int
    width_index: int
    theta_index: int


def discretize_turns(
    coil: PancakeCoilSpec,
    total_current: float,
    n_theta: int = 180,
    n_width: int = 5,
) -> list[Segment]:
    if n_theta < 3:
        raise ValueError("n_theta must be >= 3")
    if n_width < 1:
        raise ValueError("n_width must be >= 1")

    tape_width = coil.tape.width
    dtheta = 2.0 * math.pi / n_theta
    lane_current = total_current / (coil.turn_count * n_width)
    segments: list[Segment] = []

    for i_turn in range(coil.turn_count):
        base_r = coil.turn_radius(i_turn)
        if n_width == 1:
            u_values = [0.0]
        else:
            step = tape_width / (n_width - 1)
            u_values = [(-tape_width / 2.0) + i * step for i in range(n_width)]

        for i_w, u in enumerate(u_values):
            r = base_r + u
            if r <= 0:
                raise ValueError("Invalid geometry: width lane radius <= 0.")

            for i_t in range(n_theta):
                theta_mid = (i_t + 0.5) * dtheta
                c, s = math.cos(theta_mid), math.sin(theta_mid)

                pos: Vec3 = (r * c, r * s, coil.z0)
                e_r: Vec3 = (c, s, 0.0)
                e_t: Vec3 = (-s, c, 0.0)
                e_w = v_unit(e_r)
                e_n = v_unit(v_cross(e_t, e_w))

                arc = r * dtheta
                dl = v_scale(e_t, arc)

                segments.append(
                    Segment(
                        position=pos,
                        dl=dl,
                        current=lane_current,
                        e_t=e_t,
                        e_w=e_w,
                        e_n=e_n,
                        turn_index=i_turn,
                        width_index=i_w,
                        theta_index=i_t,
                    )
                )

    return segments


def build_current_elements(segments: list[Segment]) -> tuple[list[Vec3], list[Vec3], list[float]]:
    positions = [s.position for s in segments]
    dls = [s.dl for s in segments]
    currents = [s.current for s in segments]
    return positions, dls, currents
