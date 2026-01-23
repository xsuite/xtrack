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

delta_mux = 2 * math.pi * (mux - mux[0])
delta_muy = 2 * math.pi * (muy - muy[0])
integrand = h_pqrt_l * np.exp(1j * ((p - q) * delta_mux + (r - t) * delta_muy))

denominator = 1 - np.exp(1j * 2 * math.pi * ((p - q) * qx + (r - t) * qy))

# Integral evaluated at s=0 over one turn
f_pqrt_s0 = np.sum(integrand) / denominator

delta_mux_second_turn = delta_mux + 2 * math.pi * qx
delta_muy_second_turn = delta_muy + 2 * math.pi * qy
integrand_second_turn = h_pqrt_l * np.exp(1j * ((p - q) * delta_mux_second_turn + (r - t) * delta_muy_second_turn))

integrand_two_turns = np.concatenate((integrand, integrand_second_turn))

integrand_two_turns_cumsum = np.cumsum(integrand_two_turns)

# nn = len(s)
# f_pqrt = np.zeros_like(s, dtype=complex)
# for i in range(nn):
#     f_pqrt[i] = (integrand_two_turns_cumsum[i + nn] - integrand_two_turns_cumsum[i]) / denominator

# RTD at all s
f_pqrt = 0 * integrand
for i in range(len(s)):
    delta_mux_i = 2 * math.pi * (mux[i]-mux)
    delta_muy_i = 2 * math.pi * (muy[i]-muy)
    delta_mux_add = 2 * math.pi * qx * np.ones_like(mux)
    delta_muy_add = 2 * math.pi * qy * np.ones_like(muy)
    delta_mux_add[delta_mux_i > 0] = 0
    delta_muy_add[delta_muy_i > 0] = 0

    delta_mux_i += delta_mux_add
    delta_muy_i += delta_muy_add
    integrand_i = h_pqrt_l * np.exp(1j * ((p - q) * delta_mux_i + (r - t) * delta_muy_i))

    delta_mux_source = -2 * math.pi * mux
    delta_muy_source = -2 * math.pi * muy
    delta_mux_obs = 2 * math.pi * mux[i]
    delta_muy_obs = 2 * math.pi * muy[i]

    delta_mux_turn = -2 * math.pi * qx * np.ones_like(mux)
    delta_muy_turn = -2 * math.pi * qy * np.ones_like(muy)

    delta_mux_turn[delta_mux_i > 0] = 0
    delta_muy_turn[delta_muy_i > 0] = 0

    # integrand_i_2 = h_pqrt_l * np.exp(1j * ((p - q) * (delta_mux_source + delta_mux_obs + delta_mux_turn)
    #                                + (r - t) * (delta_muy_source + delta_muy_obs + delta_muy_turn)))


    f_pqrt[i] = np.sum(integrand_i) / denominator

# tw_ng = env.lhcb1.madng_twiss(rdts=[rdt])
