import xtrack as xt
import xobjects as xo
import xdeps as xd

fpath = '../../test_data/lhc_2024/lhc.seq'

with open(fpath, 'r') as fid:
    seq_text = fid.read()

assert ' at=' in seq_text
assert ',at=' not in seq_text
assert 'at =' not in seq_text
seq_text = seq_text.replace(' at=', 'at:=')


env = xt.load(string=seq_text, format='madx', reverse_lines=['lhcb2'])

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
assert str(env.ref['bctfr'].length._expr) == "vars['l.bctfr']"

assert env['bpmwt'].prototype == 'monitor'
assert env['monitor'].prototype is None
assert isinstance(env['bpmwt'], xt.Drift)
assert isinstance(env['monitor'], xt.Drift)
assert str(env.ref['bpmwt'].length._expr) == "vars['l.bpmwt']"

assert env['dfbaj'].prototype == 'placeholder'
assert env['placeholder'].prototype is None
assert isinstance(env['dfbaj'], xt.Drift)
assert isinstance(env['placeholder'], xt.Drift)
assert str(env.ref['dfbaj'].length._expr) == "vars['l.dfbaj']"

assert env['mcd_unplugged'].prototype == 'placeholder'
assert env['placeholder'].prototype is None
assert isinstance(env['mcd_unplugged'], xt.Drift)
assert isinstance(env['placeholder'], xt.Drift)
assert env.ref['mcd_unplugged'].length._expr is None # The MAD-X file sets lrad not l

assert env['mqm'].prototype == 'quadrupole'
assert env['quadrupole'].prototype is None
assert isinstance(env['mqm'], xt.Quadrupole)
assert isinstance(env['quadrupole'], xt.Quadrupole)
assert str(env.ref['mqm'].length._expr) == "vars['l.mqm']"
assert str(env.ref['mqm'].extra['calib']._expr) == "(vars['kmax_mqm'] / vars['imax_mqm'])"
assert type(env['mqm']).__name__ == 'View'
assert type(env['mqm'].knl).__name__ == 'View'
assert type(env['mqm'].extra).__name__ == 'dict'

assert env['mbrs'].prototype == 'rbend'
assert env['rbend'].prototype is None
assert isinstance(env['mbrs'], xt.RBend)
assert isinstance(env['rbend'], xt.RBend)
assert env.ref['mbrs'].length._expr is None
assert str(env.ref['mbrs'].length_straight._expr) == "vars['l.mbrs']"
assert str(env.ref['mbrs'].extra['calib']._expr) == "(vars['kmax_mbrs_4.5k'] / vars['imax_mbrs_4.5k'])"
assert type(env['mbrs']).__name__ == 'View'
assert type(env['mbrs'].knl).__name__ == 'View'
assert type(env['mbrs'].extra).__name__ == 'dict'

assert env['mb'].prototype == 'sbend'
assert env['sbend'].prototype is None
assert isinstance(env['mb'], xt.Bend)
assert isinstance(env['sbend'], xt.Bend)
assert str(env.ref['mb'].length._expr) == "vars['l.mb']"
assert str(env.ref['mb'].extra['calib']._expr) == "(vars['kmax_mb'] / vars['imax_mb'])"
assert type(env['mb']).__name__ == 'View'
assert type(env['mb'].knl).__name__ == 'View'
assert type(env['mb'].extra).__name__ == 'dict'

assert env['adtkv'].prototype == 'tkicker'
assert env['tkicker'].prototype is None
assert isinstance(env['adtkv'], xt.Multipole)
assert env['adtkv'].isthick
assert not env['tkicker'].isthick
assert str(env.ref['adtkv'].length._expr) == "vars['l.adtkv']"
assert type(env['adtkv']).__name__ == 'View'
assert type(env['adtkv'].knl).__name__ == 'View'
assert type(env['adtkv'].extra).__name__ == 'dict'

assert env['mcbv'].prototype == 'vcorrector'
assert env['vcorrector'].prototype == 'vkicker'
assert env['vkicker'].prototype is None
assert isinstance(env['mcbv'], xt.Multipole)
assert isinstance(env['vcorrector'], xt.Multipole)
assert isinstance(env['vkicker'], xt.Multipole)
assert env['mcbv'].isthick
assert str(env.ref['mcbv'].length._expr) == "vars['l.mcbv']"
assert str(env.ref['mcbv'].extra['calib']._expr) == "(vars['kmax_mcbv'] / vars['imax_mcbv'])"

assert env['acsca'].prototype == 'rfcavity'
assert env['rfcavity'].prototype is None
assert isinstance(env['acsca'], xt.Cavity)
assert isinstance(env['rfcavity'], xt.Cavity)
assert str(env.ref['acsca'].length._expr) == "vars['l.acsca']"
assert type(env['acsca']).__name__ == 'View'
assert type(env['acsca'].extra).__name__ == 'dict'

assert env['mbas2'].prototype == 'solenoid'
assert env['solenoid'].prototype is None
assert isinstance(env['mbas2'], xt.UniformSolenoid)
assert isinstance(env['solenoid'], xt.UniformSolenoid)
assert str(env.ref['mbas2'].length._expr) == "vars['l.mbas2']"
assert type(env['mbas2']).__name__ == 'View'
assert type(env['mbas2'].extra).__name__ == 'dict'

# Check some B1 elements

assert env['mqxa.1r1/lhcb1'].prototype == 'mqxa'
assert env['mqxa'].prototype == 'quadrupole'
# inherited from mqxa
assert str(env.ref['mqxa'].length._expr) == "vars['l.mqxa']"
assert env.ref['mqxa.1r1/lhcb1'].extra['calib']._expr == "(vars['kmax_mqxa'] / vars['imax_mqxa'])" # inherited from mqxa
# set after element definition in MAD-X
assert str(env.ref['mqxa.1r1/lhcb1'].k1._expr) == "(vars['kqx.r1'] + vars['ktqx1.r1'])"
assert env.ref['mqxa.1r1/lhcb1'].extra['polarity']._value == 1.
assert env.ref['mqxa.1r1/lhcb1'].extra['polarity']._expr is None
for kk in ['kmax', 'kmin', 'calib', 'mech_sep', 'slot_id', 'assembly_id', 'polarity']:
    assert kk in env['mqxa.1r1/lhcb1'].extra

assert env['mcssx.3r1/lhcb1'].prototype == 'mcssx'
assert env['mcssx'].prototype == 'multipole'
assert env['multipole'].prototype is None
# inherited from mcssx
assert str(env.ref['mcssx'].length._expr) == "vars['l.mcssx']"
assert env.ref['mcssx.3r1/lhcb1'].extra['calib']._expr == "(vars['kmax_mcssx'] / vars['imax_mcssx'])" # inherited from mcssx
# set after element definition in MAD-X
assert str(env.ref['mcssx.3r1/lhcb1'].ksl[2]._expr) == "(vars['kcssx3.r1'] * vars['l.mcssx'])"
assert env.ref['mcssx.3r1/lhcb1'].extra['polarity']._value == -1.
assert env.ref['mcssx.3r1/lhcb1'].extra['polarity']._expr is None
for kk in ['kmax', 'kmin', 'calib', 'mech_sep', 'slot_id', 'assembly_id', 'polarity']:
    assert kk in env['mcssx.3r1/lhcb1'].extra

assert env['mb.a8r1.b1'].prototype == 'mb'
# inherited from mb
assert str(env.ref['mb'].length._expr) == "vars['l.mb']"
assert env.ref['mb.a8r1.b1'].extra['calib']._expr == "(vars['kmax_mb'] / vars['imax_mb'])" # inherited from mb
# set after element definition in MAD-X and reversed
assert str(env.ref['mb.a8r1.b1'].angle._expr) == "vars['ab.a12']"
assert str(env.ref['mb.a8r1.b1'].k0._expr) == "vars['kb.a12']"
assert env.ref['mb.a8r1.b1'].extra['polarity']._value == 1.
assert env.ref['mb.a8r1.b1'].extra['polarity']._expr is None
for kk in ['kmax', 'kmin', 'calib', 'mech_sep', 'slot_id', 'assembly_id', 'polarity']:
    assert kk in env['mb.a8r1.b1'].extra

# Check some B2 elements

assert env['mqxa.1r1/lhcb2'].prototype == 'mqxa'
assert env['mqxa'].prototype == 'quadrupole'
# inherited from mqxa
assert str(env.ref['mqxa'].length._expr) == "vars['l.mqxa']"
assert env.ref['mqxa.1r1/lhcb2'].extra['calib']._expr == "(vars['kmax_mqxa'] / vars['imax_mqxa'])" # inherited from mqxa
# set after element definition in MAD-X and reversed by the loader
assert str(env.ref['mqxa.1r1/lhcb2'].k1._expr) == "(-(vars['kqx.r1'] + vars['ktqx1.r1']))"
assert env.ref['mqxa.1r1/lhcb2'].extra['polarity']._value == 1.
assert env.ref['mqxa.1r1/lhcb2'].extra['polarity']._expr is None
for kk in ['kmax', 'kmin', 'calib', 'mech_sep', 'slot_id', 'assembly_id', 'polarity']:
    assert kk in env['mqxa.1r1/lhcb2'].extra

assert env['mcssx.3r1/lhcb2'].prototype == 'mcssx'
assert env['mcssx'].prototype == 'multipole'
assert env['multipole'].prototype is None
# inherited from mcssx
assert str(env.ref['mcssx'].length._expr) == "vars['l.mcssx']"
assert env.ref['mcssx.3r1/lhcb2'].extra['calib']._expr == "(vars['kmax_mcssx'] / vars['imax_mcssx'])" # inherited from mcssx
# set after element definition in MAD-X and reversed
assert str(env.ref['mcssx.3r1/lhcb2'].ksl[2]._expr) == "(-(vars['kcssx3.r1'] * vars['l.mcssx']))"
assert env.ref['mcssx.3r1/lhcb2'].extra['polarity']._value == -1.
assert env.ref['mcssx.3r1/lhcb2'].extra['polarity']._expr is None
for kk in ['kmax', 'kmin', 'calib', 'mech_sep', 'slot_id', 'assembly_id', 'polarity']:
    assert kk in env['mcssx.3r1/lhcb2'].extra

assert env['mb.a8r1.b2'].prototype == 'mb'
# inherited from mb
assert str(env.ref['mb'].length._expr) == "vars['l.mb']"
assert env.ref['mb.a8r1.b2'].extra['calib']._expr == "(vars['kmax_mb'] / vars['imax_mb'])" # inherited from mb
# set after element definition in MAD-X and reversed
assert str(env.ref['mb.a8r1.b2'].angle._expr) == "(-vars['ab.a12'])"
assert str(env.ref['mb.a8r1.b2'].k0._expr) == "(-vars['kb.a12'])"
assert env.ref['mb.a8r1.b2'].extra['polarity']._value == 1.
assert env.ref['mb.a8r1.b2'].extra['polarity']._expr is None
for kk in ['kmax', 'kmin', 'calib', 'mech_sep', 'slot_id', 'assembly_id', 'polarity']:
    assert kk in env['mb.a8r1.b2'].extra

# Check builder
assert not env.lhcb1.builder.mirror
assert env.lhcb2.builder.mirror

assert env.lhcb1.builder.components[1000].name == 'mco.b14r2.b1'
assert xd.refs.is_ref(env.lhcb1.builder.components[1000].at)
assert str(env.lhcb1.builder.components[1000].at) == "(578.4137 + ((138.0 - vars['ip2ofs.b1']) * vars['ds']))"
assert env.lhcb1.builder.components[1000].from_ == 'ip2'


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
env.lhcb2.twiss4d(reverse=True)


# Check builder
env.lhcb2.builder.name = None # Not to overwrite the line
env.lhcb1.builder.name = None # Not to overwrite the line
lb1 = env.lhcb1.builder.build()
lb2 = env.lhcb2.builder.build()
lb1.particle_ref = xt.Particles(p0c=7e12)
lb2.particle_ref = xt.Particles(p0c=7e12)
tb1 = lb1.twiss4d()
tb2 = lb2.twiss4d(reverse=True)

