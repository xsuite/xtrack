import numpy as np
from math import factorial, pi as PI
from feed_down import feed_down as feed_down_computation

import xtrack as xt

# TODO:
# - add feed down

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

def _omega(idx):
    return 1 if idx % 2 == 0 else 0