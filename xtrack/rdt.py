import numpy as np
from math import factorial, pi as PI

from .feed_down import feed_down as feed_down_computation
import xtrack as xt


def _parse_rdt_key(key: str):

    """
    Extract (p, q, r, t) indices from an RDT key like ``f1020``.
    """
    key = key.strip()
    if len(key) != 5 or key[0].lower() != "f" or not key[1:].isdigit():
        raise ValueError("RDT key must look like 'f1020' (one letter f + four digits).")
    return int(key[1]), int(key[2]), int(key[3]), int(key[4])


def rdt_first_order_perturbation(rdt,
                                 twiss=None,
                                 strengths=None,
                                 feed_down=True,
                                 orbit=None):

    if orbit is None:
        orbit = twiss

    assert len(orbit) == len(twiss), \
        "table_orbit and twiss must have the same length."
    assert len(strengths) == len(twiss), \
        "strengths and twiss must have the same length."
    assert np.all(orbit.name == twiss.name), \
        "table_orbit and twiss must have the same element names."
    assert np.all(strengths.name == twiss.name), \
        "strengths table must have the name column."

    # Compile strengths with feed down from orbit and misalignments
    if feed_down:
        knl = np.zeros(shape=(len(strengths),6))
        ksl = np.zeros(shape=(len(strengths),6))
        for ii in range(6): # up to dodecapole
            knl[:, ii] = strengths[f'k{ii}l']
            ksl[:, ii] = strengths[f'k{ii}sl']

        knl_eff, kskew_eff = feed_down_computation(
            kn=knl,
            kskew=ksl,
            shift_x=getattr(strengths, 'shift_x', 0),
            shift_y=getattr(strengths, 'shift_y', 0),
            psi=getattr(strengths, 'rot_s_rad', 0),
            x0=orbit.x,
            y0=orbit.y,
            max_output_order=None,
        )
        str_dict = {}
        for ii in range(6):
            str_dict[f'k{ii}l'] = knl_eff[:, ii]
            str_dict[f'k{ii}sl'] = kskew_eff[:, ii]
        str_dict['name'] = strengths.name
        strengths_with_fd = xt.Table(data=str_dict)
    else:
        strengths_with_fd = strengths

    tw = twiss
    betx = tw.betx
    bety = tw.bety
    mux = tw.mux
    muy = tw.muy
    qx = tw.qx
    qy = tw.qy
    s = tw.s

    out_data = {}

    if isinstance(rdt, str):
        rdt = [rdt]

    for rr in rdt:
        p,q,r,t = _parse_rdt_key(rr)

        n = p + q + r + t

        bnl = strengths_with_fd[f'k{n-1}l']
        anl = strengths_with_fd[f'k{n-1}sl']

        factorial_prod = (factorial(p) * factorial(q)
                        * factorial(r) * factorial(t))

        h_pqrt_l = -(
            (bnl * _omega(r + t) + 1j * anl * _omega(r + t + 1))
            / factorial_prod / 2 ** n
            * (1j ** (r + t))
            * betx ** ((p + q) / 2)
            * bety ** ((r + t) / 2)
        )

        integrand = h_pqrt_l * np.exp(1j * 2 * PI * ((p - q) * (-mux) + (r - t) * (-muy)))
        integrand_turn_m1 = h_pqrt_l * np.exp(1j * 2 * PI * ((p - q) * (-mux + qx)
                                                + (r - t) * (-muy + qy)))
        integrand_two_turns = np.concatenate((integrand_turn_m1, integrand))
        cumsum_integrand_two_turns = np.cumsum(integrand_two_turns)

        exp_obs = np.exp(1j * 2 * PI * ((p - q) * mux + (r - t) * muy))

        # RTD at all s
        f_pqrt_open = 0 * integrand
        for i in range(len(s)):
            integral = cumsum_integrand_two_turns[i + len(s)] - cumsum_integrand_two_turns[i]
            f_pqrt_open[i] = integral * exp_obs[i]

        denominator = 1 - np.exp(1j * 2 * PI * ((p - q) * qx + (r - t) * qy))
        f_pqrt = f_pqrt_open / denominator

        out_data[rr] = f_pqrt
        out_data[rr + '_open'] = f_pqrt_open
        out_data[rr + '_integrand'] = integrand
        out_data[rr + '_integrand_previous_turn'] = integrand_turn_m1

    out_data['name'] = twiss.name

    # Sort keys (fo visualization purposes)
    out_cols = {}
    out_cols['name'] = out_data['name']
    for rr in rdt:
        out_cols[rr] = out_data[rr]
    for rr in rdt:
        out_cols[rr + '_open'] = out_data[rr + '_open']
    for rr in rdt:
        out_cols[rr + '_integrand'] = out_data[rr + '_integrand']
    for rr in rdt:
        out_cols[rr + '_integrand_previous_turn'] = out_data[rr + '_integrand_previous_turn']

    out = xt.Table(data=out_cols)

    metadata = rdt_metadata(rdt, tw.qx, tw.qy)
    out._data.update(metadata)

    return out

def rdt_metadata(rdts: list[str], Qx: float, Qy: float) -> float:
    """
    Compute the frequency associated to a given RDT.

    Parameters
    ----------
    rdts : list of str
        RDT key like ``"f1020"``.
    Qx, Qy : float
        Tunes in the two planes.

    Returns
    -------
    freq : float
        Frequency associated to the RDT.
    """
    if isinstance(rdts, str):
        rdts = [rdts]
    out = {}
    for rdt_key in rdts:
        p, q, r, t = _parse_rdt_key(rdt_key)
        freq_x_expr = f'{1 - p + q} * Qx + {t - r} * Qy'
        freq_x = (1 - p + q) * Qx + (t - r) * Qy
        freq_y_expr = f'{q - p} * Qx + {1 - r + t} * Qy'
        freq_y = (q - p) * Qx + (1 - r + t) * Qy
        while freq_x < 0.0:
            freq_x += 1.0
        while freq_y < 0.0:
            freq_y += 1.0
        while freq_x >= 1.0:
            freq_x -= 1.0
        while freq_y >= 1.0:
            freq_y -= 1.0
        if p != 0:
            a_x_expr = f'Ix^{(p + q - 1)/2} * Iy^{(r + t)/2}'
        else:
            a_x_expr = '0'
        if r != 0:
            a_y_expr = f'Ix^{(p + q)/2} * Iy^{(r + t - 1)/2}'
        else:
            a_y_expr = '0'
        out[rdt_key + '_ampl_x_expr'] = a_x_expr
        out[rdt_key + '_freq_x_expr'] = freq_x_expr
        out[rdt_key + '_freq_x'] = float(freq_x)
        out[rdt_key + '_ampl_y_expr'] = a_y_expr
        out[rdt_key + '_freq_y_expr'] = freq_y_expr
        out[rdt_key + '_freq_y'] = float(freq_y)
    return out

def tracking_from_rdt(
    rdts: dict[str, complex],
    Ix: float | np.ndarray,
    Iy: float | np.ndarray,
    Qx: float | np.ndarray,
    Qy: float | np.ndarray,
    psi_x0: float | np.ndarray = 0.0,
    psi_y0: float | np.ndarray = 0.0,
    num_turns: int = 0,
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
    num_turns : int
        Number of turns to track.

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
    N = np.arange(int(num_turns), dtype=np.float64)

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

def _omega(idx):
    return 1 if idx % 2 == 0 else 0