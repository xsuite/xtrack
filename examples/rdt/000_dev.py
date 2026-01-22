import xtrack as xt
import numpy as np
import math

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

factorial_prod = math.factorial(p) * math.factorial(q) * math.factorial(r) * math.factorial(t)

h_pqrt_l = (
    (bnl * omega(r + t) + 1j * anl * omega(r + t + 1))
    / factorial_prod
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

# RTD at all s
f_pqrt = 0 * integrand
for i in range(len(s)):
    delta_mux_i = 2 * math.pi * (mux[i]-mux)
    delta_muy_i = 2 * math.pi * (muy[i]-muy)
    delta_mux_i[delta_mux_i < 0] += 2 * math.pi * qx
    delta_muy_i[delta_muy_i < 0] += 2 * math.pi * qy
    integrand_i = h_pqrt_l * np.exp(1j * ((p - q) * delta_mux_i + (r - t) * delta_muy_i))
    f_pqrt[i] = np.sum(integrand_i) / denominator

# tw_ng = env.lhcb1.madng_twiss(rdts=[rdt])
