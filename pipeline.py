"""Pipeline helpers for generating Jc model input features."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from src.hts_coil.frames import decompose_B, local_frame_at_segment


def _to_float_list(values: Sequence[Any], *, name: str) -> list[float]:
    try:
        return [float(v) for v in values]
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"invalid {name}") from exc


def build_jc_features(
    segments: Iterable[Any],
    Bxyz_per_segment: Sequence[Sequence[Any]],
    temperature: Any,
) -> list[list[float]]:
    """Build Jc input feature list for all segments.

    Output order per segment: [temperature, |B|, B_parallel, B_perpendicular].
    """

    segment_list = list(segments)
    if len(Bxyz_per_segment) != len(segment_list):
        raise ValueError(
            "Bxyz_per_segment length must match number of segments, "
            f"got {len(Bxyz_per_segment)} vs {len(segment_list)}"
        )

    if isinstance(temperature, (int, float)):
        temp_per_segment = [float(temperature)] * len(segment_list)
    else:
        temp_per_segment = _to_float_list(temperature, name="temperature")
        if len(temp_per_segment) != len(segment_list):
            raise ValueError(
                "temperature must be scalar or per-segment list with matching length"
            )

    features: list[list[float]] = []
    for segment, Bxyz, temp in zip(segment_list, Bxyz_per_segment, temp_per_segment):
        frame = local_frame_at_segment(segment)
        comp = decompose_B(Bxyz, frame)
        features.append(
            [
                temp,
                comp["B_magnitude"],
                comp["B_parallel"],
                comp["B_perpendicular"],
            ]
        )

    return features
