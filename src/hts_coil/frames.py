"""Local tape frame utilities for HTS coil segments."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, sqrt
from typing import Any, Mapping, Sequence


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class LocalFrame:
    """Orthonormal local frame at a coil segment."""

    e_t: Vector3
    e_w: Vector3
    e_n: Vector3


def _as_vector3(vec: Any, *, name: str) -> Vector3:
    if not isinstance(vec, Sequence) or len(vec) != 3:
        raise ValueError(f"{name} must be a 3-vector")
    return (float(vec[0]), float(vec[1]), float(vec[2]))


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v: Vector3) -> float:
    return sqrt(_dot(v, v))


def _scale(v: Vector3, c: float) -> Vector3:
    return (v[0] * c, v[1] * c, v[2] * c)


def _normalize(vec: Vector3, *, name: str) -> Vector3:
    n = _norm(vec)
    if n == 0.0:
        raise ValueError(f"{name} must be non-zero")
    return _scale(vec, 1.0 / n)


def _segment_get(segment: Any, key: str) -> Any:
    if isinstance(segment, Mapping) and key in segment:
        return segment[key]
    if hasattr(segment, key):
        return getattr(segment, key)
    return None


def local_frame_at_segment(segment: Any) -> LocalFrame:
    """Compute local tape frame: e_t (length), e_w (width), e_n=e_t×e_w."""

    tangent = _segment_get(segment, "tangent")
    if tangent is None:
        start = _segment_get(segment, "start")
        end = _segment_get(segment, "end")
        if start is None or end is None:
            raise KeyError("segment must provide tangent, or start/end")
        tangent = _sub(_as_vector3(end, name="end"), _as_vector3(start, name="start"))

    e_t = _normalize(_as_vector3(tangent, name="tangent"), name="tangent")

    width = _segment_get(segment, "width")
    if width is not None:
        w_raw = _as_vector3(width, name="width")
        w_raw = _sub(w_raw, _scale(e_t, _dot(w_raw, e_t)))
        e_w = _normalize(w_raw, name="width (orthogonalized)")
    else:
        normal = _segment_get(segment, "normal")
        if normal is None:
            raise KeyError("segment must provide width, or normal")
        n_raw = _as_vector3(normal, name="normal")
        n_raw = _sub(n_raw, _scale(e_t, _dot(n_raw, e_t)))
        e_n_hint = _normalize(n_raw, name="normal (orthogonalized)")
        e_w = _normalize(_cross(e_n_hint, e_t), name="width (from normal x tangent)")

    e_n = _normalize(_cross(e_t, e_w), name="normal")
    return LocalFrame(e_t=e_t, e_w=e_w, e_n=e_n)


def decompose_B(Bxyz: Any, frame: LocalFrame | Mapping[str, Any]) -> dict[str, float]:
    """Decompose B into in-plane/normal components and angle theta."""

    if isinstance(frame, Mapping):
        e_t = _as_vector3(frame["e_t"], name="frame.e_t")
        e_w = _as_vector3(frame["e_w"], name="frame.e_w")
        e_n = _as_vector3(frame["e_n"], name="frame.e_n")
    else:
        e_t, e_w, e_n = frame.e_t, frame.e_w, frame.e_n

    B = _as_vector3(Bxyz, name="Bxyz")

    Bt = _dot(B, e_t)
    Bw = _dot(B, e_w)
    Bn = _dot(B, e_n)

    B_parallel = sqrt(Bt * Bt + Bw * Bw)
    B_perpendicular = abs(Bn)
    B_magnitude = _norm(B)
    theta = atan2(B_perpendicular, B_parallel)

    return {
        "B_magnitude": B_magnitude,
        "B_parallel": B_parallel,
        "B_perpendicular": B_perpendicular,
        "theta": theta,
    }
