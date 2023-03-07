# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import time
import numpy as np

from cpymad.madx import Madx

import xtrack as xt
import xpart as xp
import xobjects as xo

# Import thick sequence
mad = Madx()

# CLIC-DR
mad.call('../../test_data/clic_dr/sequence.madx')

# # ELETTRA
# np.sqrt(met[met.loc[:, 'parameter']=='emittance']['mode3'][0]*tw['betz0'])# ELETTRA
# mad.globals.update({'ON_SEXT': 1, 'ON_OCT': 1, 'ON_RF': 1, 'NRJ_GeV': 2.4,
#                    'SAVE_FIGS': False, 'SAVE_TWISS': False})
# mad.call("../../../elettra/elettra2_v15_VADER_2.3T.madx")
# mad.call("../../../elettra/optics_elettra2_v15_VADER_2.3T.madx")

mad.use('ring')

# Twiss
twthick = mad.twiss().dframe()

# Emit
mad.sequence.ring.beam.radiate = True
mad.emit()
mad_emit_table = mad.table.emit.dframe()
mad_emit_summ = mad.table.emitsumm.dframe()

# Makethin
mad.input(f'''
select, flag=MAKETHIN, SLICE=4, thick=false;
select, flag=MAKETHIN, pattern=wig, slice=1;
MAKETHIN, SEQUENCE=ring, MAKEDIPEDGE=true;
use, sequence=RING;
''')
mad.use('ring')
mad.twiss()

# Build xtrack line
print('Build xtrack line...')
line = xt.Line.from_madx_sequence(mad.sequence['RING'])
line.particle_ref = xp.Particles(
        mass0=xp.ELECTRON_MASS_EV,
        q0=-1,
        gamma0=mad.sequence.ring.beam.gamma)

context = xo.ContextCpu()

# Build tracker
print('Build tracker ...')
line.build_tracker(_context=context)
line.matrix_stability_tol = 1e-2

line.configure_radiation(model='mean')

# Twiss
print('Checks with twiss...')
tw = line.twiss(eneloss_and_damping=True)

# Checks
met = mad_emit_table
assert np.isclose(tw['eneloss_turn'], mad_emit_summ.u0[0]*1e9,
                  rtol=3e-3, atol=0)
assert np.isclose(tw['damping_constants_s'][0],
    met[met.loc[:, 'parameter']=='damping_constant']['mode1'][0],
    rtol=3e-3, atol=0
    )
assert np.isclose(tw['damping_constants_s'][1],
    met[met.loc[:, 'parameter']=='damping_constant']['mode2'][0],
    rtol=1e-3, atol=0
    )
assert np.isclose(tw['damping_constants_s'][2],
    met[met.loc[:, 'parameter']=='damping_constant']['mode3'][0],
    rtol=3e-3, atol=0
    )

assert np.isclose(tw['partition_numbers'][0],
    met[met.loc[:, 'parameter']=='damping_partion']['mode1'][0],
    rtol=3e-3, atol=0
    )
assert np.isclose(tw['partition_numbers'][1],
    met[met.loc[:, 'parameter']=='damping_partion']['mode2'][0],
    rtol=1e-3, atol=0
    )
assert np.isclose(tw['partition_numbers'][2],
    met[met.loc[:, 'parameter']=='damping_partion']['mode3'][0],
    rtol=3e-3, atol=0
    )

part_co = tw['particle_on_co']
particles = line.build_particles(
    x_norm=[500., 0, 0], y_norm=[0, 0.0001, 0], zeta=part_co.zeta[0],
    delta=np.array([0,0,1e-2]) + part_co.delta[0],
    nemitt_x=1e-9, nemitt_y=1e-9)

line.configure_radiation(model='quantum')

print('Track 3 particles ...')
num_turns = 5000
t1 = time.time()
line.track(particles, num_turns=num_turns, turn_by_turn_monitor=True)
t2 = time.time()
print(f'Track time: {(t2-t1)/num_turns:.2e} s/turn')
mon = line.record_last_track

import matplotlib.pyplot as plt
plt.close('all')
fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312, sharex=ax1)
ax3 = fig.add_subplot(313, sharex=ax1)

ax1.plot(mon.x[0, :].T)
ax2.plot(mon.y[1, :].T)
ax3.plot(mon.delta[2, :].T)
i_turn = np.arange(num_turns)
ax1.plot(part_co.x[0]
    +(mon.x[0,0]-part_co.x[0])*np.exp(-i_turn*tw['damping_constants_turns'][0]))
ax2.plot(part_co.y[0]
    +(mon.y[1,0]-part_co.y[0])*np.exp(-i_turn*tw['damping_constants_turns'][1]))
ax3.plot(part_co.delta[0]
    +(mon.delta[2,0]-part_co.delta[0])*np.exp(-i_turn*tw['damping_constants_turns'][2]))

plt.show()

# Switch radiation
line.configure_radiation(model='mean')
par_for_emit = line.build_particles(x_norm=50*[0],
                zeta=part_co.zeta[0], delta=part_co.delta[0])
line.configure_radiation(model='quantum')

num_turns=1500
print('Track 50 particles...')
t1 = time.time()
line.track(par_for_emit, num_turns=num_turns, turn_by_turn_monitor=True)
t2 = time.time()
print(f'Track time: {(t2-t1)/num_turns:.2e} s/turn')
mon = line.record_last_track

assert np.isclose(np.std(mon.zeta[:, 750:]),
    np.sqrt(met[met.loc[:, 'parameter']=='emittance']['mode3'][0]*tw['betz0']),
    rtol=0.2, atol=0
    )

assert np.isclose(np.std(mon.x[:, 750:]),
    np.sqrt(met[met.loc[:, 'parameter']=='emittance']['mode1'][0]*tw['betx'][0]),
    rtol=0.2, atol=0
    )

assert np.all(mon.y[:] < 1e-15)
