import numpy as np

import xtrack as xt
import xobjects as xo

# mad = Madx()
# mad.call('../../test_data/sps_thick/sps.seq')

# # mad.input('beam, particle=proton, pc=26;')
# # mad.input('beam, particle=electron, pc=20;')

# # # realistic
# # mad.input('beam, particle=electron, pc=20;')
# # 
# # num_turns = 8000

# # higher energy
# mad.input('beam, particle=electron, pc=50;')
# v_mv = 250

# mad.call('../../test_data/sps_thick/lhc_q20.str')

# mad.use(sequence='sps')

# mad.input('twiss, table=tw4d;')
# twm4d = mad.table.tw4d



# mad.sequence.sps.elements['actcse.31632'].volt = v_mv * 10 / n_cav   # To stay in the linear region
# mad.sequence.sps.elements['actcse.31632'].freq = 3
# mad.sequence.sps.elements['actcse.31632'].lag = 0.5


# mad.input('twiss, table=tw6d;')
# twm6d = mad.table.tw6d

# mad.sequence.sps.beam.radiate = True
# mad.emit()

# line = xt.Line.from_madx_sequence(mad.sequence.sps, allow_thick=True,
#                                   deferred_expressions=True)
env = xt.load(['../../test_data/sps_thick/sps.seq',
               '../../test_data/sps_thick/lhc_q20.str'
               ])
line = env['sps']

line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV,
                                    q0=-1, p0c=50e9)
line.cycle('bpv.11706', inplace=True)

v_mv = 250e6
n_cav = 6
env['actcse.31632'].voltage = v_mv * 10 / n_cav   # To stay in the linear region
env['actcse.31632'].frequency = 3e6
env['actcse.31632'].phase = np.pi

env = line.env
# Create thin cavities with same properties as actcse.31632
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

line.configure_radiation(model='mean')
tw1 = line.twiss(radiation_analysis=True)

# Make one cavity decelerating
env['cav4'].phase = env['cav4'].phase - np.pi
tw2 = line.twiss(radiation_analysis=True)

xo.assert_allclose(tw1.energy_loss, tw2.energy_loss, rtol=0.01)
xo.assert_allclose(tw1.partition_numbers, tw2.partition_numbers)