import numpy as np
from scipy.constants import c as clight
from scipy.constants import hbar
from scipy.constants import epsilon_0

from cpymad.madx import Madx
import xtrack as xt
import xobjects as xo

mad = Madx()
mad.call('../../test_data/sps_thick/sps.seq')

# mad.input('beam, particle=proton, pc=26;')
# mad.input('beam, particle=electron, pc=20;')

# # realistic
# mad.input('beam, particle=electron, pc=20;')
# v_mv = 25
# num_turns = 8000

# higher energy
mad.input('beam, particle=electron, pc=50;')
v_mv = 250
num_turns = 600

mad.call('../../test_data/sps_thick/lhc_q20.str')

mad.use(sequence='sps')

mad.input('twiss, table=tw4d;')
twm4d = mad.table.tw4d

n_cav = 6

mad.sequence.sps.elements['actcse.31632'].volt = v_mv * 10 / n_cav   # To stay in the linear region
mad.sequence.sps.elements['actcse.31632'].freq = 3
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

tt = line.get_table()

s_start_wig = tt['s', 'actcsg.31780']

line.discard_tracker()
line.insert([
    env.new('wig1', xt.Bend, length=0.5, rot_s_rad=np.pi/2, at=s_start_wig+0.1+0*0.5),
    env.new('wig2', xt.Bend, length=0.5, rot_s_rad=np.pi/2, at=s_start_wig+0.1+1*0.5),
    env.new('wig3', xt.Bend, length=0.5, rot_s_rad=np.pi/2, at=s_start_wig+0.1+2*0.5),
    env.new('wig4', xt.Bend, length=0.5, rot_s_rad=np.pi/2, at=s_start_wig+0.1+3*0.5),
    env.new('wig5', xt.Bend, length=0.5, rot_s_rad=np.pi/2, at=s_start_wig+0.1+4*0.5),
    env.new('wig6', xt.Bend, length=0.5, rot_s_rad=np.pi/2, at=s_start_wig+0.1+5*0.5),
    env.new('wig7', xt.Bend, length=0.5, rot_s_rad=np.pi/2, at=s_start_wig+0.1+6*0.5),
    env.new('wig8', xt.Bend, length=0.5, rot_s_rad=np.pi/2, at=s_start_wig+0.1+7*0.5),
])

line['k_wig'] = 0
line['wig1'].k0 = 'k_wig'
line['wig2'].k0 = '-k_wig'
line['wig3'].k0 = '-k_wig'
line['wig4'].k0 = 'k_wig'
line['wig5'].k0 = '-k_wig'
line['wig6'].k0 = 'k_wig'
line['wig7'].k0 = 'k_wig'
line['wig8'].k0 = '-k_wig'

line['k_wig'] = 1e-3
tw = line.twiss4d()

# Remove edge effects
# for nn in tt.rows[tt.element_type=='DipoleEdge'].name:
#     line[nn].k = 0

tw_thick = line.twiss()

Strategy = xt.slicing.Strategy
Teapot = xt.slicing.Teapot

line.discard_tracker()
slicing_strategies = [
    Strategy(slicing=Teapot(1)),  # Default
    Strategy(slicing=Teapot(2), element_type=xt.Bend),
    Strategy(slicing=Teapot(8), element_type=xt.Quadrupole),
    Strategy(slicing=Teapot(20), name='wig.*'),
]

line.slice_thick_elements(slicing_strategies)
line.build_tracker()

line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu())
line.configure_radiation(model=None)

line.configure_radiation(model='mean')
twr = line.twiss(radiation_analysis=True)
