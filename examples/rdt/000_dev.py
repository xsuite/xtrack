import xtrack as xt
import math

env = xt.load('../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
rdt = 'f2020'

line = env.lhcb1

tw = line.twiss4d()
betx = tw.betx
bety = tw.bety
mux = tw.mux
muy = tw.muy

tt = line.get_table(attr=True)

p,q,r,t = map(int, rdt[1:])

n = p + q + r + t

bnl = tt[f'k{n-1}l']
anl = tt[f'k{n-1}sl']

def omega(idx):
    return 1 if idx % 2 == 0 else 0

factorial_prod = math.factorial(p) * math.factorial(q) * math.factorial(r) * math.factorial(t)

h_pqrt = (
    (bnl * omega(r + t) + 1j * anl * omega(r + t + 1))
    / factorial_prod
    * (1j ** (r + t))
    * betx ** ((p + q) / 2)
    * bety ** ((r + t) / 2)
)


# tw_ng = env.lhcb1.madng_twiss(rdts=[rdt])
