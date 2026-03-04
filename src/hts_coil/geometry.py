"""Core geometry data structures for HTS pancake coils.

This module models each turn as a circular centerline and uses a thin-tape
approximation for the conductor cross-section: tape width is preserved for
spatial discretization, while tape thickness is optional metadata and ignored
in field integrals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class TapeSpec:
    """HTS tape specification.

    Args:
        width: Tape width (meters). Must be strictly positive and is used in
            geometry discretization and local coordinate definitions.
        thickness: Optional tape thickness (meters). This value is stored for
            metadata/recording only and is intentionally ignored in field
            integration under the thin-tape approximation.
    """

    width: float
    thickness: float | None = None

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("TapeSpec.width must be > 0.")


@dataclass(frozen=True)
class PancakeCoilSpec:
    """Pancake coil geometry specification.

    Args:
        inner_radius: Radius of the first turn centerline.
        turn_count: Number of turns in the pancake.
        turn_pitch: Radial pitch between adjacent turn centerlines.
        tape_spec: Tape geometry specification.
        z0: Axial position of this pancake's center plane.
    """

    inner_radius: float
    turn_count: int
    turn_pitch: float
    tape_spec: TapeSpec
    z0: float = 0.0

    def __post_init__(self) -> None:
        if self.turn_count <= 0:
            raise ValueError("PancakeCoilSpec.turn_count must be > 0.")

    def turn_radius(self, turn_index: int) -> float:
        """Return centerline radius for turn ``turn_index``.

        Radius definition:
            r_i = inner_radius + i * turn_pitch
        """
        if not 0 <= turn_index < self.turn_count:
            raise IndexError(
                f"turn_index {turn_index} out of bounds for turn_count={self.turn_count}."
            )
        return self.inner_radius + turn_index * self.turn_pitch

    def turn_paths(self) -> Iterator["TurnPath"]:
        """Yield a circular centerline path generator for each turn."""
        for i in range(self.turn_count):
            yield TurnPath(radius=self.turn_radius(i), z=self.z0, tape_spec=self.tape_spec)


@dataclass(frozen=True)
class TurnPath:
    """Circular centerline path and local width parameterization for one turn.

    The centerline is a ring of fixed radius and fixed axial coordinate ``z``.
    Width direction uses a local radial coordinate
    ``u in [-width/2, width/2]`` to sample actual tape-surface points.
    """

    radius: float
    z: float
    tape_spec: TapeSpec

    def centerline_point(self, phi: float) -> tuple[float, float, float]:
        """Return Cartesian point on the turn centerline at azimuth ``phi``."""
        from math import cos, sin

        return (self.radius * cos(phi), self.radius * sin(phi), self.z)

    @property
    def width_bounds(self) -> tuple[float, float]:
        """Return local radial width parameter bounds ``(-width/2, width/2)``."""
        half_w = self.tape_spec.width / 2.0
        return (-half_w, half_w)

    def surface_point(self, phi: float, u: float) -> tuple[float, float, float]:
        """Return Cartesian point on tape surface sampled by ``(phi, u)``.

        Args:
            phi: Azimuthal angle around the coil axis.
            u: Local radial width coordinate in ``[-width/2, width/2]``.

        Raises:
            ValueError: If ``u`` is outside valid width bounds.
        """
        from math import cos, sin

        u_min, u_max = self.width_bounds
        if not (u_min <= u <= u_max):
            raise ValueError(f"u={u} outside width bounds [{u_min}, {u_max}].")

        local_r = self.radius + u
        return (local_r * cos(phi), local_r * sin(phi), self.z)
