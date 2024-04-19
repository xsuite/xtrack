from cpymad.madx import Madx
import xtrack as xt

mad = Madx()
mad.call('lep.seq9')
mad.call('lep.opt9')
mad.input('beam, particle=positron, energy=100, radiate=true;')
mad.use(sequence='lep')
mad.input('vrfc:=2.; vrfsc:=50.; vrfscn:=50.; ! LEP2  rf on')

twm = mad.twiss()

line = xt.Line.from_madx_sequence(mad.sequence.lep, deferred_expressions=True)
line.particle_ref = xt.Particles(
    mass0=xt.ELECTRON_MASS_EV, gamma0=mad.sequence.lep.beam.gamma)
line.build_tracker()
sv = line.survey()
tw = line.twiss()

# import xplt
# xplt.FloorPlot(sv, line, element_width=100)

# Slice
line.discard_tracker()
line.slice_thick_elements(
    slicing_strategies=[
        # Slicing with thin elements
        xt.Strategy(slicing=xt.Teapot(1)),
        xt.Strategy(slicing=xt.Uniform(2), element_type=xt.Bend),
        xt.Strategy(slicing=xt.Teapot(10), element_type=xt.Quadrupole),
    ])
line.build_tracker()
line.configure_radiation('mean')

tw_rad = line.twiss(eneloss_and_damping=True)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
sp1 = plt.subplot(3, 1, 1)
plt.plot(tw_rad.s, tw_rad.betx, label='betx')
plt.plot(tw_rad.s, tw_rad.bety, label='bety')
plt.ylabel(r'$\beta_{x,y}$ [m]')

sp2 = plt.subplot(3, 1, 2, sharex=sp1)
plt.plot(tw_rad.s, tw_rad.x, label='x')
plt.ylabel('x [m]')

sp3 = plt.subplot(3, 1, 3, sharex=sp1)
plt.plot(tw_rad.s, tw_rad.delta, label='delta')
plt.ylabel(r'$\delta$')
plt.xlabel('s [m]')

plt.show()