import xtrack as xt

env = xt.load_madx_lattice('../../test_data/leir/leir.seq')
env.vars.load_madx('../../test_data/leir/leir_inj_nominal.str')

line = env.leir
line.configure_bend_model(edge='full', core='adaptive')
line.particle_ref = xt.Particles(kinetic_energy0=14e9)

from cpymad.madx import Madx
mad = Madx()

mad.call('../../test_data/leir/leir.seq')
mad.call('../../test_data/leir/leir_inj_nominal.str')
mad.beam()
mad.use(sequence='leir')

lref = xt.Line.from_madx_sequence(mad.sequence.leir)
lref.configure_bend_model(edge='full', core='adaptive')
lref.particle_ref = xt.Particles(kinetic_energy0=14e9)

tw_ref = lref.twiss4d(betx=1, bety=1)
tw = line.twiss4d(betx=1, bety=1)

mad.sequence.leir.expanded_elements['er.sol21'].cmdpar['ks'].expr # this is None ?!