import xtrack as xt
import numpy as np
import math

# Based on https://arxiv.org/abs/1711.06589. See in particular: eq. 8 and A8.
# A. Franchi et al. "Analytic formulas for the rapid evaluation of the orbit
# response matrix and chromatic functions from lattice parameters in circular 
# accelerators", 2017
# Reference implementation by OMC team:
# https://github.com/pylhc/optics_functions/blob/master/optics_functions/rdt.py

env = xt.load('../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
rdt = 'f1020'

line = env.lhcb1

tw = line.twiss4d()
betx = tw.betx
bety = tw.bety
mux = tw.mux
muy = tw.muy
qx = tw.qx
qy = tw.qy
s = tw.s

tt = line.get_table(attr=True)

p,q,r,t = map(int, rdt[1:])

n = p + q + r + t

bnl = tt[f'k{n-1}l']
anl = tt[f'k{n-1}sl']

def omega(idx):
    return 1 if idx % 2 == 0 else 0

factorial_prod = (math.factorial(p) * math.factorial(q)
                * math.factorial(r) * math.factorial(t))

h_pqrt_l = -(
    (bnl * omega(r + t) + 1j * anl * omega(r + t + 1))
    / factorial_prod / 2 ** n
    * (1j ** (r + t))
    * betx ** ((p + q) / 2)
    * bety ** ((r + t) / 2)
)


denominator = 1 - np.exp(1j * 2 * math.pi * ((p - q) * qx + (r - t) * qy))

integrand = h_pqrt_l * np.exp(1j * 2 * math.pi * ((p - q) * (-mux) + (r - t) * (-muy)))
integrand_turn_m1 = h_pqrt_l * np.exp(1j * 2 * math.pi * ((p - q) * (-mux + qx)
                                          + (r - t) * (-muy + qy)))
integrand_two_turns = np.concatenate((integrand_turn_m1, integrand))
cumsum_integrand_two_turns = np.cumsum(integrand_two_turns)

# RTD at all s
f_pqrt = 0 * integrand
for i in range(len(s)):
    # delta_mux_i = 2 * math.pi * (-mux)
    # delta_muy_i = 2 * math.pi * (-muy)
    # delta_mux_add = 2 * math.pi * qx * np.ones_like(mux)
    # delta_muy_add = 2 * math.pi * qy * np.ones_like(muy)
    # delta_mux_add[:i] = 0
    # delta_muy_add[:i] = 0

    # delta_mux_i += delta_mux_add
    # delta_muy_i += delta_muy_add
    # integrand_i = h_pqrt_l * np.exp(1j * ((p - q) * delta_mux_i + (r - t) * delta_muy_i))

    # integrand_i = integrand_two_turns[i:i+len(s)]
    # integral = np.sum(integrand_i)

    integral = cumsum_integrand_two_turns[i + len(s)] - cumsum_integrand_two_turns[i]
    f_pqrt[i] = integral / denominator * np.exp(1j * 2 * math.pi * ((p - q) * (mux[i]) + (r - t) * (muy[i])))

# tw_ng = env.lhcb1.madng_twiss(rdts=[rdt])
