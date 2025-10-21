import xtrack as xt
import xobjects as xo

env = xt.load('../../test_data/lhc_2024/lhc.seq',
               reverse_lines=['lhcb2'],
)
env.vars.load('../../test_data/lhc_2024/injection_optics.madx')

# Some checks based on direct inpection of MAD-X file
xo.assert_allclose(env['ip8ofs.b2'],  -154, atol=1e-12)
assert str(env.ref['aip2']._expr) == "f['atan'](((vars['sep_arc'] / 2.0) / vars['dsep2']))"

assert env['tanb'].prototype == 'collimator'
assert env['collimator'].prototype is None
assert isinstance(env['collimator'], xt.Drift)
assert str(env.ref['tanb'].length._expr) == "vars['l.tanb']"

assert env['mcbch'].prototype == 'hcorrector'
assert env['hcorrector'].prototype == 'hkicker'
assert env['hkicker'].prototype is None
assert isinstance(env['mcbch'], xt.Multipole)
assert isinstance(env['hcorrector'], xt.Multipole)
assert isinstance(env['hkicker'], xt.Multipole)
assert env['mcbch'].isthick
assert str(env.ref['mcbch'].length._expr) == "vars['l.mcbch']"
assert str(env.ref['mcbch'].extra['calib']._expr) == "(vars['kmax_mcbch'] / vars['imax_mcbch'])"
assert type(env['mcbch']).__name__ == 'View'
assert type(env['mcbch'].knl).__name__ == 'View'
assert type(env['mcbch'].extra).__name__ == 'dict'


assert env['bctfr'].prototype == 'instrument'
assert env['instrument'].prototype is None
assert isinstance(env['bctfr'], xt.Drift)
assert isinstance(env['instrument'], xt.Drift)

assert env['bpmwt'].prototype == 'monitor'
assert env['monitor'].prototype is None
assert isinstance(env['bpmwt'], xt.Drift)
assert isinstance(env['monitor'], xt.Drift)

assert env['dfbaj'].prototype == 'placeholder'
assert env['placeholder'].prototype is None
assert isinstance(env['dfbaj'], xt.Drift)
assert isinstance(env['placeholder'], xt.Drift)

assert env['mcd_unplugged'].prototype == 'placeholder'
assert env['placeholder'].prototype is None
assert isinstance(env['mcd_unplugged'], xt.Drift)
assert isinstance(env['placeholder'], xt.Drift)

assert env['mqm'].prototype == 'quadrupole'
assert env['quadrupole'].prototype is None
assert isinstance(env['mqm'], xt.Quadrupole)
assert isinstance(env['quadrupole'], xt.Quadrupole)

prrrr
# from cpymad.madx import Madx
# madx = Madx()
# madx.call('../../test_data/lhc_2024/lhc.seq')
# madx.call('../../test_data/lhc_2024/injection_optics.madx')
# madx.beam()
# madx.use('lhcb1')
# twmad = madx.twiss()
# lmad = xt.Line.from_madx_sequence(madx.sequence.lhcb1)

env.lhcb1.particle_ref = xt.Particles(p0c=7e12)
env.lhcb2.particle_ref = xt.Particles(p0c=7e12)

env.lhcb1.twiss4d().plot()
env.lhcb2.twiss4d(reverse=True).plot()


# Check builder
env.lhcb2.builder.name = None # Not to overwrite the line
env.lhcb1.builder.name = None # Not to overwrite the line
lb1 = env.lhcb1.builder.build()
lb2 = env.lhcb2.builder.build()
lb1.particle_ref = xt.Particles(p0c=7e12)
lb2.particle_ref = xt.Particles(p0c=7e12)
tb1 = lb1.twiss4d()
tb2 = lb2.twiss4d(reverse=True)
