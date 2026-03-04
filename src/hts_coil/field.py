"""Magnetic field utilities based on discretized Biot-Savart integration."""

from __future__ import annotations

import numpy as np


def _normalize_current_elements(current_elements):
    """Normalize current-element input into (positions, dl, current)."""
    if isinstance(current_elements, dict):
        positions = np.asarray(current_elements["positions"], dtype=float)
        dl = np.asarray(current_elements["dl"], dtype=float)
        current = np.asarray(current_elements["current"], dtype=float)
    else:
        arr = np.asarray(current_elements, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 7:
            raise ValueError(
                "current_elements must be a dict with keys positions/dl/current "
                "or an array with shape (M, 7): [x, y, z, dlx, dly, dlz, I]."
            )
        positions = arr[:, :3]
        dl = arr[:, 3:6]
        current = arr[:, 6]

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (M, 3).")
    if dl.shape != positions.shape:
        raise ValueError("dl must have shape (M, 3), matching positions.")
    if current.ndim == 0:
        current = np.full((positions.shape[0],), float(current))
    elif current.ndim == 1 and current.shape[0] == positions.shape[0]:
        pass
    else:
        raise ValueError("current must be a scalar or an array of shape (M,).")

    return positions, dl, current


def biot_savart_field(
    eval_points,
    current_elements,
    mu0: float = 4 * np.pi * 1e-7,
    eps: float = 1e-12,
):
    """Compute B field at evaluation points from discretized current elements.

    Parameters
    ----------
    eval_points : array-like, shape (N, 3)
        Cartesian coordinates where magnetic field is evaluated.
    current_elements : dict or ndarray
        Either a dict with keys: ``positions`` (M,3), ``dl`` (M,3), ``current`` (M,)
        or an array of shape (M,7): ``[x, y, z, dlx, dly, dlz, I]``.
    mu0 : float
        Vacuum permeability.
    eps : float
        Minimum cutoff on |r| for numerical stability.

    Returns
    -------
    ndarray, shape (N, 3)
        Magnetic field components [Bx, By, Bz] at each evaluation point.
    """
    points = np.asarray(eval_points, dtype=float)
    if points.ndim == 1:
        points = points[np.newaxis, :]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("eval_points must have shape (N, 3).")

    positions, dl, current = _normalize_current_elements(current_elements)

    # r_ij = point_i - element_j, shape (N, M, 3)
    r = points[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r_norm = np.linalg.norm(r, axis=-1)
    r_norm = np.maximum(r_norm, eps)

    # cross(dl, r), shape (N, M, 3)
    cross = np.cross(dl[np.newaxis, :, :], r)
    weight = current[np.newaxis, :] / (r_norm**3)
    dB = (mu0 / (4 * np.pi)) * cross * weight[..., np.newaxis]

    return np.sum(dB, axis=1)


def compute_coil_self_field_on_segments(segments, current_elements, **kwargs):
    """Compute coil self-field Bxyz on each segment position."""
    if isinstance(segments, dict):
        eval_points = np.asarray(segments["positions"], dtype=float)
    else:
        eval_points = np.asarray(segments, dtype=float)

    return biot_savart_field(eval_points=eval_points, current_elements=current_elements, **kwargs)
