"""HTS coil discretization utilities.

This module provides geometry-to-segment conversion for Biot-Savart integration.
"""

from __future__ import annotations

import math
from typing import Any


TAU = 2.0 * math.pi


def _normalize(vx: float, vy: float, vz: float) -> tuple[float, float, float]:
    norm = math.sqrt(vx * vx + vy * vy + vz * vz)
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return (vx / norm, vy / norm, vz / norm)


def _get_turn_radii(coil_spec: dict[str, Any]) -> list[float]:
    turn_count = int(coil_spec["turn_count"])
    if "turn_radii" in coil_spec:
        radii = [float(r) for r in coil_spec["turn_radii"]]
        if len(radii) != turn_count:
            raise ValueError("'turn_radii' length must equal turn_count.")
        return radii

    r_inner = float(coil_spec.get("r_inner", 0.0))
    r_pitch = float(coil_spec.get("turn_pitch_radial", 0.0))
    return [r_inner + i * r_pitch for i in range(turn_count)]


def _get_turn_z_positions(coil_spec: dict[str, Any]) -> list[float]:
    turn_count = int(coil_spec["turn_count"])
    if "turn_z" in coil_spec:
        z_values = [float(z) for z in coil_spec["turn_z"]]
        if len(z_values) != turn_count:
            raise ValueError("'turn_z' length must equal turn_count.")
        return z_values

    z0 = float(coil_spec.get("z0", 0.0))
    z_pitch = float(coil_spec.get("turn_pitch_axial", 0.0))
    return [z0 + i * z_pitch for i in range(turn_count)]


def discretize_turns(
    coil_spec: dict[str, Any],
    n_theta: int,
    n_width: int,
) -> list[dict[str, Any]]:
    """Discretize each turn into angular and width-direction segments.

    Parameters
    ----------
    coil_spec:
        Coil configuration dictionary. Required keys are:
        - ``turn_count``: number of turns
        - ``I_total``: total transport current
        And one of:
        - ``tape_width`` or ``width``

        Optional geometry keys:
        - ``turn_radii`` (list) OR ``r_inner`` + ``turn_pitch_radial``
        - ``turn_z`` (list) OR ``z0`` + ``turn_pitch_axial``
        - ``theta0`` (default 0), ``theta_span`` (default 2*pi)
        - ``winding_sign`` (+1 default, -1 for reverse current direction)
    n_theta:
        Number of segments along theta direction per turn.
    n_width:
        Number of width strips. Must be > 1 to represent width effects.

    Returns
    -------
    list[dict[str, Any]]
        One dictionary per discretized element with geometric basis vectors and
        assigned current.

    Notes
    -----
    Current version assumes *uniform current sharing* across turns and width
    strips: each element gets ``I_total / (turn_count * n_width)``. This can be
    replaced later by a non-uniform distribution model.
    """
    if n_theta < 1:
        raise ValueError("n_theta must be >= 1.")
    if n_width <= 1:
        raise ValueError("n_width must be > 1 to capture width effects.")

    turn_count = int(coil_spec["turn_count"])
    if turn_count < 1:
        raise ValueError("turn_count must be >= 1.")

    I_total = float(coil_spec["I_total"])
    tape_width = float(coil_spec.get("tape_width", coil_spec.get("width", 0.0)))
    if tape_width <= 0.0:
        raise ValueError("coil_spec requires a positive 'tape_width' or 'width'.")

    radii = _get_turn_radii(coil_spec)
    z_values = _get_turn_z_positions(coil_spec)

    theta0 = float(coil_spec.get("theta0", 0.0))
    theta_span = float(coil_spec.get("theta_span", TAU))
    winding_sign = 1.0 if float(coil_spec.get("winding_sign", 1.0)) >= 0.0 else -1.0

    d_theta = theta_span / n_theta
    d_width = tape_width / n_width
    I_segment = I_total / (turn_count * n_width)

    segments: list[dict[str, Any]] = []

    for turn_idx, (r_center, z_center) in enumerate(zip(radii, z_values)):
        for theta_idx in range(n_theta):
            theta = theta0 + (theta_idx + 0.5) * d_theta
            c = math.cos(theta)
            s = math.sin(theta)

            # Cylindrical basis at segment centerline.
            r_hat = _normalize(c, s, 0.0)
            z_hat = (0.0, 0.0, 1.0)
            t_hat = _normalize(winding_sign * -s, winding_sign * c, 0.0)

            # In this model, width direction is radial and local normal is +z.
            w_hat = r_hat
            n_hat = z_hat

            for width_idx in range(n_width):
                w_offset = (width_idx + 0.5 - n_width / 2.0) * d_width
                r_local = r_center + w_offset
                if r_local <= 0.0:
                    raise ValueError(
                        "Non-positive local radius encountered; adjust radii/width discretization."
                    )

                x = r_local * c
                y = r_local * s
                z = z_center
                ds = abs(r_local * d_theta)

                segments.append(
                    {
                        "turn_index": turn_idx,
                        "theta_index": theta_idx,
                        "width_index": width_idx,
                        "x": x,
                        "y": y,
                        "z": z,
                        "t_hat": t_hat,
                        "n_hat": n_hat,
                        "w_hat": w_hat,
                        "length": ds,
                        "I_segment": I_segment,
                    }
                )

    return segments


def build_current_elements(
    coil_spec: dict[str, Any],
    n_theta: int,
    n_width: int,
) -> list[dict[str, Any]]:
    """Build Biot-Savart-ready line current elements.

    Returns a list of dictionaries with position, direction, element length,
    and current.

    Notes
    -----
    Current version assumes *uniform current sharing* among all turns and width
    strips. A future non-uniform current-distribution model can replace this
    assignment while keeping the same output structure.
    """
    segments = discretize_turns(coil_spec=coil_spec, n_theta=n_theta, n_width=n_width)

    elements: list[dict[str, Any]] = []
    for seg in segments:
        elements.append(
            {
                "position": (seg["x"], seg["y"], seg["z"]),
                "direction": seg["t_hat"],
                "length": seg["length"],
                "current": seg["I_segment"],
            }
        )

    return elements
