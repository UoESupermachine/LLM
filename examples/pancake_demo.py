from __future__ import annotations

from pathlib import Path
import csv

from hts_coil.geometry import PancakeCoilSpec, TapeSpec
from hts_coil.jc_interface import DummyJcModel
from hts_coil.pipeline import estimate_coil_critical_current, evaluate_segments


def stats(values: list[float]) -> tuple[float, float, float]:
    return min(values), sum(values) / len(values), max(values)


def main() -> None:
    tape = TapeSpec(width=4e-3, thickness=None)
    coil = PancakeCoilSpec(
        inner_radius=0.03,
        turn_count=10,
        turn_pitch=4.2e-3,
        tape=tape,
        z0=0.0,
    )

    temperature = 77.0
    current = 120.0
    n_theta = 36
    n_width = 3

    model = DummyJcModel(jc_constant=1.8e10)
    results = evaluate_segments(
        coil=coil,
        total_current=current,
        temperature=temperature,
        jc_model=model,
        n_theta=n_theta,
        n_width=n_width,
    )

    bmag = [r.B_mag for r in results]
    bpar = [r.B_parallel for r in results]
    bperp = [r.B_perpendicular for r in results]

    bmag_s = stats(bmag)
    bpar_s = stats(bpar)
    bperp_s = stats(bperp)

    print("=== HTS Pancake Demo ===")
    print(f"Segments: {len(results)}")
    print(f"B_mag [T] min/mean/max: {bmag_s[0]:.6f} / {bmag_s[1]:.6f} / {bmag_s[2]:.6f}")
    print(f"B_parallel [T] min/mean/max: {bpar_s[0]:.6f} / {bpar_s[1]:.6f} / {bpar_s[2]:.6f}")
    print(f"B_perpendicular [T] min/mean/max: {bperp_s[0]:.6f} / {bperp_s[1]:.6f} / {bperp_s[2]:.6f}")

    ic_est = estimate_coil_critical_current(
        coil=coil,
        temperature=temperature,
        jc_model=model,
        n_theta=n_theta,
        n_width=n_width,
        current_guess=current,
    )
    print(f"Estimated coil Ic (placeholder criterion): {ic_est:.3f} A")

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "segment_fields.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "x", "y", "z",
                "turn", "w_idx", "theta_idx",
                "Bx", "By", "Bz",
                "B_mag", "B_parallel", "B_perpendicular", "theta",
                "Jc", "Ic_local",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.position[0], r.position[1], r.position[2],
                    r.turn_index, r.width_index, r.theta_index,
                    r.Bxyz[0], r.Bxyz[1], r.Bxyz[2],
                    r.B_mag, r.B_parallel, r.B_perpendicular, r.theta,
                    r.Jc, r.Ic_local,
                ]
            )

    print(f"Saved segment data: {out_csv}")


if __name__ == "__main__":
    main()
