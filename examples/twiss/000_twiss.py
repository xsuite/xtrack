import pickle
import json
import pathlib
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp

fname_line_particles = './temp_precise_lattice/xtline.json'

context = xo.ContextCpu()

with open(fname_line_particles, 'r') as fid:
    input_data = json.load(fid)

line = xt.Line.from_dict(input_data['line'])
part0 = xp.Particles(_context=context, **input_data['particle'])

print('Build tracker...')
tracker = xt.Tracker(_context=context, line=line)

tw = tracker.twiss(particle_ref=part0)

import matplotlib.pyplot as plt

plt.close('all')

fig1 = plt.figure(1)
spbet = plt.subplot(3,1,1)
spco = plt.subplot(3,1,2, sharex=spbet)
spdisp = plt.subplot(3,1,3, sharex=spbet)

spbet.plot(tw['s'], tw['betx'])
spbet.plot(tw['s'], tw['bety'])

spco.plot(tw['s'], tw['x'])
spco.plot(tw['s'], tw['y'])

spdisp.plot(tw['s'], tw['dx'])
spdisp.plot(tw['s'], tw['dy'])

plt.show()
