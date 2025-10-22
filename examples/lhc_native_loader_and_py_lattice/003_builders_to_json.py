import xtrack as xt
import xobjects as xo
import xdeps as xd
from cpymad.madx import Madx
import numpy as np

fpath = '../../test_data/lhc_2024/lhc.seq'
mode = 'direct' # 'direct' / 'dict' / 'copy'

with open(fpath, 'r') as fid:
    seq_text = fid.read()

assert ' at=' in seq_text
assert ',at=' not in seq_text
assert 'at =' not in seq_text
seq_text = seq_text.replace(' at=', 'at:=')

env = xt.load(string=seq_text, format='madx', reverse_lines=['lhcb2'])
env.lhcb1.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7000e9)
env.lhcb2.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7000e9)

env.vars.load('../../test_data/lhc_2024/injection_optics.madx')

# Keep only builders in the environment
for nn in ['lhcb1', 'lhcb2']:
    bb = env.lines[nn].builder
    del env.lines[nn]
    env.lines[nn] = bb

print('Remove drifts:')
# Remove all drifts
tt_elems = env.elements.get_table()
tt_drift = tt_elems.rows['drift_.*']
drift_names = tt_drift.name
for ii, dn in enumerate(drift_names):
    print(f'Removing drift {ii+1}/{len(drift_names)}', end='\r', flush=True)
    # I bypass the xdeps checks, I know there are no expressions in drifts
    del env._element_dict[dn]

print('To dict:')
ddd = env.to_dict()

env2 = xt.Environment.from_dict(ddd)

# dct = to_dict(builder)
# bb = from_dict(xt.Builder, dct, env)
