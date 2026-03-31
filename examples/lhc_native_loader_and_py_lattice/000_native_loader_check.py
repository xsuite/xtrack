import xtrack as xt
import xobjects as xo
import xdeps as xd
from cpymad.madx import Madx
import numpy as np
import time

fpath = '../../test_data/lhc_2024/lhc.seq'
mode = 'direct' # 'direct' / 'dict' / 'copy'

with open(fpath, 'r') as fid:
    seq_text = fid.read()

assert ' at=' in seq_text
assert ',at=' not in seq_text
assert 'at =' not in seq_text
seq_text = seq_text.replace(' at=', 'at:=')

t1 = time.perf_counter()
env = xt.load(string=seq_text, format='madx', reverse_lines=['lhcb2'])
t2 = time.perf_counter()
print(f'Loading and parsing MAD-X file took {t2-t1:.2f} seconds')
env.lhcb1.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7000e9)
env.lhcb2.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7000e9)

env.vars.load('../../test_data/lhc_2024/injection_optics.madx')
