import numpy as np
from scipy.constants import c as clight
from scipy.constants import hbar
from scipy.constants import epsilon_0

from cpymad.madx import Madx
import xtrack as xt

mad = Madx()
mad.call('../../test_data/sps_thick/sps.seq')

# higher energy
mad.input('beam, particle=electron, pc=45;')
v_mv = 250

mad.call('../../test_data/sps_thick/lhc_q20.str')

mad.use(sequence='sps')

mad.input('twiss, table=tw4d;')
twm4d = mad.table.tw4d

n_cav = 6

mad.sequence.sps.elements['actcse.31632'].volt = v_mv * 10 / n_cav   # To stay in the linear region
mad.sequence.sps.elements['actcse.31632'].freq = 10
mad.sequence.sps.elements['actcse.31632'].lag = 0.5

mad.input('twiss, table=tw6d;')
twm6d = mad.table.tw6d

mad.sequence.sps.beam.radiate = True
mad.emit()

line = xt.Line.from_madx_sequence(mad.sequence.sps, allow_thick=True,
                                  deferred_expressions=True)
line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV,
                                    q0=-1, gamma0=mad.sequence.sps.beam.gamma)
line.cycle('bpv.11706', inplace=True)

# Create thin cavities with same properties as actcse.31632
env = line.env
env.new('cav1', 'actcse.31632', length=0)
env.new('cav2', 'actcse.31632', length=0)
env.new('cav3', 'actcse.31632', length=0)
env.new('cav4', 'actcse.31632', length=0)
env.new('cav5', 'actcse.31632', length=0)
env.new('cav6', 'actcse.31632', length=0)

line.insert([
    env.place('cav1', at='bpv.11706'),
    env.place('cav2', at='bpv.21508'),
    env.place('cav3', at='bpv.31508'),
    env.place('cav4', at='bpv.41508'),
    env.place('cav5', at='bpv.51508'),
    env.place('cav6', at='bpv.61508'),
])

tt = line.get_table()

tw_thick = line.twiss()

Strategy = xt.slicing.Strategy
Teapot = xt.slicing.Teapot

line.discard_tracker()
slicing_strategies = [
    Strategy(slicing=Teapot(1)),  # Default
    Strategy(slicing=Teapot(2), element_type=xt.Bend),
    Strategy(slicing=Teapot(8), element_type=xt.Quadrupole),
    Strategy(slicing=None, element_type=xt.Cavity),
]

line.slice_thick_elements(slicing_strategies)
line.build_tracker()

tw = line.twiss()

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()
p = line.build_particles(x=35e-3, y=35e-3)
line.track(p, num_turns=1000, turn_by_turn_monitor=True)
mon_mean = line.record_last_track

line.configure_radiation(model='quantum')
p = line.build_particles(x=35e-3, y=35e-3)
line.track(p, num_turns=1000, turn_by_turn_monitor=True)
mon_quantum = line.record_last_track

import matplotlib.pyplot as plt
plt.close('all')
f1 = plt.figure(1)
spx = plt.subplot(2,1,1)
plt.plot(mon_mean.x.T)
plt.ylabel('x [m]')
spy = plt.subplot(2,1,2, sharex=spx)
plt.plot(mon_mean.y.T)
plt.ylabel('y [m]')
plt.xlabel('Turn')

f2 = plt.figure(2)
spqx = plt.subplot(2,1,1)
plt.plot(mon_quantum.x.T)
plt.ylabel('x [m]')
spqy = plt.subplot(2,1,2, sharex=spqx)
plt.plot(mon_quantum.y.T)
plt.ylabel('y [m]')
plt.xlabel('Turn')


plt.show()