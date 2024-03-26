import numpy as np
import xtrack as xt

line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

tw = line.twiss(method='4d')

mng = line.to_madng(sequence_name='lhcb1')

mng["mytwtable", 'mytwflow'] = mng.twiss(
    sequence=mng.lhcb1, method=4, mapdef=2, implicit=True, nslice=3, save="'atbody'")


mng.send(r'''
local damap, vector in MAD
local lhc = MADX.lhcb1

-- list of octupolar RDTs
local rdts = {"f4000", "f3100", "f2020", "f1120"}

-- create phase-space damap at 4th order
local X0 = damap {nv=6, mo=4}

-- twiss with RDTs
local mtbl = track {sequence=lhc, X0=X0, info=2, observe=0, savemap=true}

mtbl:addcol("symperror", \ri -> mtbl.__map[ri]:get1():symperr())

local symp_err_vect = vector(#mtbl):fill(mtbl.symperror)

symp_err_vect:print("symperror")

-- send columns to Python
py:send({mtbl.s, symp_err_vect})

''')

s, symperror = mng.recv()

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
ax1 = plt.subplot(2, 1, 1)
plt.plot(tw.s, tw.betx)

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(s, symperror)

plt.show()
