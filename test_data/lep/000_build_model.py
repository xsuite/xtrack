import xtrack as xt
from cpymad.madx import Madx

mad = Madx()
mad.call('lep98_cv20.madx')
mad.beam()
mad.use('lep')

line = xt.Line.from_madx_sequence(mad.sequence.lep, deferred_expressions=True)
line.vars.load('n6060pol70v5.str')
line.particle_ref = xt.Particles(
    mass0=xt.ELECTRON_MASS_EV, q0=1, energy0=45.6e9,
    anomalous_magnetic_moment=0.00115965218128
)

line.to_json('lep.json')
