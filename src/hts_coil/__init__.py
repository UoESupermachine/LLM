"""HTS pancake coil physics-guided evaluation toolkit."""

from .geometry import PancakeCoilSpec, TapeSpec
from .jc_interface import DummyJcModel, JcModel
from .pipeline import estimate_coil_critical_current, evaluate_segments
from .probe import PancakeFieldModel, solve_point_fields

__all__ = [
    "TapeSpec",
    "PancakeCoilSpec",
    "JcModel",
    "DummyJcModel",
    "evaluate_segments",
    "estimate_coil_critical_current",
    "PancakeFieldModel",
    "solve_point_fields",
]
