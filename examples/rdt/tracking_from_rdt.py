from __future__ import annotations

import re
from typing import Mapping, Tuple

import numpy as np


_KEY_PATTERN = re.compile(r"f(\d+)(\d+)(\d+)(\d+)", re.IGNORECASE)


def _parse_rdt_key(key: str) -> Tuple[int, int, int, int]:
    """
    Extract (p, q, r, t) indices from an RDT key like ``f1020``.
    """
    match = _KEY_PATTERN.fullmatch(key.strip())
    if match is None:
        raise ValueError(f"RDT key '{key}' must match pattern fPQR T (e.g., f1020).")
    return tuple(int(g) for g in match.groups())


def tracking_from_rdt(
    rdts: Mapping[str, complex],
    Ix: float | np.ndarray,
    Iy: float | np.ndarray,
    Qx: float | np.ndarray,
    Qy: float | np.ndarray,
    psi_x0: float | np.ndarray = 0.0,
    psi_y0: float | np.ndarray = 0.0,
    N: float | np.ndarray = 0.0,
):
    """
    Compute the complex Courant-Snyder variables h_{x,-} and h_{y,-} from RDTs.

    The implementation follows the relations:

        h_{x,-} = sqrt(2Ix) e^{i(2pi Qx N + psi_x0)}
                  - 2i sum_{pqrt} p f_{pqrt} (2 Ix)^{(p+q-1)/2} (2 Iy)^{(r+t)/2}
                      e^{ i[(1-p+q)(2pi Qx N + psi_x0) + (t-r)(2pi Qy N + psi_y0)] }

        h_{y,-} = sqrt(2Iy) e^{i(2pi Qy N + psi_y0)}
                  - 2i sum_{pqrt} r f_{pqrt} (2 Ix)^{(p+q)/2} (2 Iy)^{(r+t-1)/2}
                      e^{ i[(q-p)(2pi Qx N + psi_x0) + (1-r+t)(2pi Qy N + psi_y0)] }

    Parameters
    ----------
    rdts : mapping
        Dictionary of resonance driving terms keyed by strings like ``"f1020"``.
    Ix, Iy : float or array_like
        Actions in the two planes.
    Qx, Qy : float or array_like
        Tunes in the two planes.
    psi_x0, psi_y0 : float or array_like, optional
        Initial phases at the observation point.
    N : float or array_like, optional
        Turn index (or array of turns).

    Returns
    -------
    hx_minus, hy_minus : np.ndarray
        Complex Courant-Snyder variables for the horizontal and vertical planes.
        Arrays broadcast to the common shape of the input parameters.
    """
    Ix = np.asarray(Ix, dtype=np.float64)
    Iy = np.asarray(Iy, dtype=np.float64)
    Qx = np.asarray(Qx, dtype=np.float64)
    Qy = np.asarray(Qy, dtype=np.float64)
    psi_x0 = np.asarray(psi_x0, dtype=np.float64)
    psi_y0 = np.asarray(psi_y0, dtype=np.float64)
    N = np.asarray(N, dtype=np.float64)

    Ix, Iy, Qx, Qy, psi_x0, psi_y0, N = np.broadcast_arrays(
        Ix, Iy, Qx, Qy, psi_x0, psi_y0, N
    )

    phase_x = 2.0 * np.pi * Qx * N + psi_x0
    phase_y = 2.0 * np.pi * Qy * N + psi_y0

    hx_minus = np.sqrt(2.0 * Ix) * np.exp(1j * phase_x)
    hy_minus = np.sqrt(2.0 * Iy) * np.exp(1j * phase_y)

    sum_hx = np.zeros_like(hx_minus, dtype=np.complex128)
    sum_hy = np.zeros_like(hy_minus, dtype=np.complex128)

    for key, value in rdts.items():
        p, q, r, t = _parse_rdt_key(key)
        f_pqrt = complex(value)

        if p != 0:
            amp_x = (2.0 * Ix) ** ((p + q - 1) / 2.0) * (2.0 * Iy) ** ((r + t) / 2.0)
            phase = np.exp(
                1j * ((1 - p + q) * phase_x + (t - r) * phase_y)
            )
            sum_hx += p * f_pqrt * amp_x * phase

        if r != 0:
            amp_y = (2.0 * Ix) ** ((p + q) / 2.0) * (2.0 * Iy) ** ((r + t - 1) / 2.0)
            phase = np.exp(
                1j * ((q - p) * phase_x + (1 - r + t) * phase_y)
            )
            sum_hy += r * f_pqrt * amp_y * phase

    hx_minus = hx_minus - 2j * sum_hx
    hy_minus = hy_minus - 2j * sum_hy

    return hx_minus, hy_minus
