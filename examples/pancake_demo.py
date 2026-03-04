"""Demo: build one pancake coil and compute field on all discretized segments."""

from __future__ import annotations

import numpy as np

from src.hts_coil.field import compute_coil_self_field_on_segments


def build_single_pancake(radius=0.05, turns=20, z0=0.0, n_per_turn=100, current=100.0):
    """Build discretized current elements for one pancake coil."""
    n_total = turns * n_per_turn
    theta = np.linspace(0.0, 2.0 * np.pi * turns, n_total, endpoint=False)

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.full_like(theta, z0)
    positions = np.stack([x, y, z], axis=1)

    # Tangential line element for each segment.
    dtheta = (2.0 * np.pi * turns) / n_total
    dlx = -radius * np.sin(theta) * dtheta
    dly = radius * np.cos(theta) * dtheta
    dlz = np.zeros_like(theta)
    dl = np.stack([dlx, dly, dlz], axis=1)

    return {
        "positions": positions,
        "dl": dl,
        "current": np.full(n_total, current),
    }


def main():
    I = 120.0
    current_elements = build_single_pancake(current=I)

    # Here we evaluate self-field at all segment centers.
    Bxyz = compute_coil_self_field_on_segments(
        segments=current_elements["positions"],
        current_elements=current_elements,
        eps=1e-9,
    )

    print(f"Pancake segments: {len(Bxyz)}")
    print("First 5 B vectors [T]:")
    print(Bxyz[:5])


if __name__ == "__main__":
    main()
