import numpy as np
import xtrack as xt

line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

tw = line.twiss(method='4d')

mng = line.to_madng(sequence_name='lhcb1')

mng.send('''
local damap in MAD
local lhc = MADX.lhcb1

-- list of octupolar RDTs
local rdts = {"f4000", "f3100", "f2020", "f1120"}

-- create phase-space damap at 4th order
local X0 = damap {nv=6, mo=4}

-- twiss with RDTs
local mtbl = twiss {sequence=lhc, X0=X0, trkrdt=rdts, info=2, saverdt=true}

-- send columns to Python
py:send({mtbl.s, mtbl.beta11, mtbl.f4000, mtbl.f3100, mtbl.f2020, mtbl.f1120})

''')


s, beta11, f4000, f3100, f2020, f1120 = mng.recv()

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)

plt.plot(s, np.abs(f4000), label='f4000')
plt.plot(s, np.abs(f2020), label='f2020')
plt.plot(s, np.abs(f1120), label='f1120')
plt.plot(s, np.abs(f3100), label='f3100')
plt.xlabel('s [m]')
plt.ylabel(r'|f_{jklm}|')
plt.legend()

plt.show()