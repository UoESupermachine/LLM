from __future__ import annotations

from dataclasses import dataclass

from .field import biot_savart_field
from .frames import decompose_B
from .geometry import PancakeCoilSpec
from .jc_interface import JcModel
from .segments import build_current_elements, discretize_turns
from .utils import Vec3


@dataclass
class SegmentEvaluation:
    position: Vec3
    turn_index: int
    width_index: int
    theta_index: int
    Bxyz: Vec3
    B_mag: float
    B_parallel: float
    B_perpendicular: float
    theta: float
    Jc: float
    Ic_local: float


def evaluate_segments(
    coil: PancakeCoilSpec,
    total_current: float,
    temperature: float,
    jc_model: JcModel,
    n_theta: int = 180,
    n_width: int = 5,
) -> list[SegmentEvaluation]:
    segments = discretize_turns(coil, total_current, n_theta=n_theta, n_width=n_width)
    src_pos, src_dls, src_I = build_current_elements(segments)
    B_all = biot_savart_field(src_pos, src_pos, src_dls, src_I)

    features: list[list[float]] = []
    raw_components: list[tuple[float, float, float, float]] = []
    for seg, B in zip(segments, B_all):
        B_mag, B_par, B_perp, theta = decompose_B(B, seg.e_t, seg.e_w, seg.e_n)
        features.append([temperature, B_mag, B_par, B_perp])
        raw_components.append((B_mag, B_par, B_perp, theta))

    Jc = jc_model.predict(features)
    if len(Jc) != len(segments):
        raise ValueError("jc_model.predict must return same length as features")

    lane_width = coil.tape.width / n_width
    effective_area = lane_width * 1.0  # placeholder (thin-tape interface)
    Ic_local = [jc * effective_area for jc in Jc]

    out: list[SegmentEvaluation] = []
    for i, seg in enumerate(segments):
        B_mag, B_par, B_perp, theta = raw_components[i]
        out.append(
            SegmentEvaluation(
                position=seg.position,
                turn_index=seg.turn_index,
                width_index=seg.width_index,
                theta_index=seg.theta_index,
                Bxyz=B_all[i],
                B_mag=B_mag,
                B_parallel=B_par,
                B_perpendicular=B_perp,
                theta=theta,
                Jc=float(Jc[i]),
                Ic_local=float(Ic_local[i]),
            )
        )
    return out


def estimate_coil_critical_current(
    coil: PancakeCoilSpec,
    temperature: float,
    jc_model: JcModel,
    n_theta: int = 180,
    n_width: int = 5,
    current_guess: float = 100.0,
) -> float:
    results = evaluate_segments(
        coil=coil,
        total_current=current_guess,
        temperature=temperature,
        jc_model=jc_model,
        n_theta=n_theta,
        n_width=n_width,
    )
    return min(r.Ic_local for r in results)
