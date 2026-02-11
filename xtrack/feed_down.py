"""
Vectorized multipole feed-down computation with transverse offset, tilt, and
optional expansion around a closed orbit (x0, y0).

Inputs are arrays of shape (n_elem, n_order), where n_order = max multipole
order + 1 (dipole is index 0). All arrays broadcast along n_elem.
"""

from __future__ import annotations

import numpy as np
from math import factorial
from typing import Tuple


def _as_array(x):
    return np.atleast_1d(np.asarray(x, dtype=np.float64))


def feed_down(
    kn: np.ndarray,
    kskew: np.ndarray,
    shift_x: np.ndarray,
    shift_y: np.ndarray,
    psi: np.ndarray,
    x0: np.ndarray = 0.0,
    y0: np.ndarray = 0.0,
    max_output_order: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute effective normal/skew multipoles with feed-down and tilt.

    Parameters
    ----------
    kn, kskew : array_like, shape (n_elem, n_order)
        Normal and skew multipole strengths (K_n, K_hat_n).
    shift_x, shift_y : array_like, shape (n_elem,)
        Transverse shift of the magnet (Delta x, Delta y) in the lab frame.
    psi : array_like, shape (n_elem,)
        Tilt angle of each magnet (radians), positive = CCW.
    x0, y0 : array_like, shape (n_elem,), optional
        Closed orbit coordinates; default is 0.
    max_output_order : int, optional
        If provided, truncate output to orders [0, max_output_order].

    Returns
    -------
    kn_eff, kskew_eff : np.ndarray
        Effective normal and skew multipoles including feed-down, shape
        (n_elem, n_out), where n_out = max_output_order+1 or input order.
    """
    kn = np.asarray(kn, dtype=np.float64)
    kskew = np.asarray(kskew, dtype=np.float64)
    if kn.shape != kskew.shape:
        raise ValueError("kn and kskew must have the same shape (n_elem, n_order).")

    n_elem, n_order = kn.shape
    if max_output_order is None:
        n_out = n_order
    else:
        n_out = min(n_order, int(max_output_order) + 1)

    shift_x = _as_array(shift_x)
    shift_y = _as_array(shift_y)
    psi = _as_array(psi)
    x0 = _as_array(x0)
    y0 = _as_array(y0)

    # Broadcast to n_elem
    for arr_name, arr in {
        "shift_x": shift_x,
        "shift_y": shift_y,
        "psi": psi,
        "x0": x0,
        "y0": y0,
    }.items():
        if arr.shape not in [(n_elem,), (1,)]:
            raise ValueError(f"{arr_name} must broadcast to shape ({n_elem},)")

    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    psi = psi.reshape(-1)
    x0 = x0.reshape(-1)
    y0 = y0.reshape(-1)

    k_complex = kn + 1j * kskew  # shape (n_elem, n_order)

    z0 = x0 + 1j * y0
    delta = shift_x + 1j * shift_y
    rot = np.exp(-1j * psi)
    # Offset seen in the magnet frame, including orbit displacement.
    b_tilde = rot * (z0 - delta)

    # Precompute factorials up to n_order-1
    factorials = np.array([factorial(p) for p in range(n_order)], dtype=np.float64)

    out = np.zeros((n_elem, n_out), dtype=np.complex128)

    for m in range(n_out):
        p_max = n_order - m
        p_idx = np.arange(p_max)
        coeff = k_complex[:, m : m + p_max] / factorials[p_idx]  # (n_elem, p_max)
        power = (b_tilde)[:, None] ** p_idx  # (n_elem, p_max)
        series_sum = np.sum(coeff * power, axis=1)
        out[:, m] = np.exp(-1j * (m + 1) * psi) * series_sum

    kn_eff = out.real
    kskew_eff = out.imag
    return kn_eff, kskew_eff


if __name__ == "__main__":
    # Minimal example: single element, quadrupole with offset.
    kn = np.array([[0.0, 1.0, 0.0]])  # dipole, quad, sext
    kskew = np.zeros_like(kn)
    shift_x = np.array([1e-3])
    shift_y = np.array([0.0])
    psi = np.array([0.0])
    x0 = np.array([0.0])
    y0 = np.array([0.0])

    kn_eff, kskew_eff = feed_down(kn, kskew, shift_x, shift_y, psi, x0, y0)
    print("Effective normal multipoles:", kn_eff)
    print("Effective skew multipoles:", kskew_eff)
