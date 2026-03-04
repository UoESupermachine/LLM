from __future__ import annotations

from typing import Protocol


class JcModel(Protocol):
    """Input features row: [temperature, B_mag, B_parallel, B_perpendicular]."""

    def predict(self, features: list[list[float]]) -> list[float]:
        ...


class DummyJcModel:
    def __init__(self, jc_constant: float = 2.0e10) -> None:
        self.jc_constant = jc_constant

    def predict(self, features: list[list[float]]) -> list[float]:
        return [self.jc_constant for _ in features]
