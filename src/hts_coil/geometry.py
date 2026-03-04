from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TapeSpec:
    """HTS tape specification.

    Thickness is optional and currently unused (thin-tape approximation).
    Width is mandatory and used in discretization.
    """

    width: float
    thickness: float | None = None

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("Tape width must be > 0.")


@dataclass(frozen=True)
class PancakeCoilSpec:
    inner_radius: float
    turn_count: int
    turn_pitch: float
    tape: TapeSpec
    z0: float = 0.0

    def __post_init__(self) -> None:
        if self.inner_radius <= 0:
            raise ValueError("inner_radius must be > 0.")
        if self.turn_count <= 0:
            raise ValueError("turn_count must be >= 1.")
        if self.turn_pitch <= 0:
            raise ValueError("turn_pitch must be > 0.")

    def turn_radius(self, i_turn: int) -> float:
        if not (0 <= i_turn < self.turn_count):
            raise IndexError("turn index out of range")
        return self.inner_radius + i_turn * self.turn_pitch

    def turn_radii(self) -> list[float]:
        return [self.inner_radius + self.turn_pitch * i for i in range(self.turn_count)]
