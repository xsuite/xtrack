import numpy as np
from math import factorial, pi as PI

# Based on https://arxiv.org/abs/1711.06589. See in particular: eq. 8 and A8.
# A. Franchi et al. "Analytic formulas for the rapid evaluation of the orbit
# response matrix and chromatic functions from lattice parameters in circular
# accelerators", 2017
# Reference implementation by OMC team:
# https://github.com/pylhc/optics_functions/blob/master/optics_functions/rdt.py


def compute_rdt_first_order_perturbation(rdt, twiss, strengths):

    tw = twiss
    betx = tw.betx
    bety = tw.bety
    mux = tw.mux
    muy = tw.muy
    qx = tw.qx
    qy = tw.qy
    s = tw.s

    p,q,r,t = map(int, rdt[1:])

    n = p + q + r + t

    bnl = strengths[f'k{n-1}l']
    anl = strengths[f'k{n-1}sl']

    def omega(idx):
        return 1 if idx % 2 == 0 else 0

    factorial_prod = (factorial(p) * factorial(q)
                    * factorial(r) * factorial(t))

    h_pqrt_l = -(
        (bnl * omega(r + t) + 1j * anl * omega(r + t + 1))
        / factorial_prod / 2 ** n
        * (1j ** (r + t))
        * betx ** ((p + q) / 2)
        * bety ** ((r + t) / 2)
    )

    denominator = 1 - np.exp(1j * 2 * PI * ((p - q) * qx + (r - t) * qy))

    integrand = h_pqrt_l * np.exp(1j * 2 * PI * ((p - q) * (-mux) + (r - t) * (-muy)))
    integrand_turn_m1 = h_pqrt_l * np.exp(1j * 2 * PI * ((p - q) * (-mux + qx)
                                            + (r - t) * (-muy + qy)))
    integrand_two_turns = np.concatenate((integrand_turn_m1, integrand))
    cumsum_integrand_two_turns = np.cumsum(integrand_two_turns)

    exp_obs = np.exp(1j * 2 * PI * ((p - q) * mux + (r - t) * muy))

    # RTD at all s
    f_pqrt = 0 * integrand
    for i in range(len(s)):

        integral = cumsum_integrand_two_turns[i + len(s)] - cumsum_integrand_two_turns[i]
        f_pqrt[i] = integral / denominator * exp_obs[i]

    return f_pqrt
