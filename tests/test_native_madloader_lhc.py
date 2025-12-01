import xtrack as xt
import xobjects as xo
import xdeps as xd
from cpymad.madx import Madx
import numpy as np
import pathlib
import pytest

from xtrack._temp.python_lattice_writer import lattice_py_generation as lpg

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()

@pytest.fixture(scope='module')
def lines_ref():
    fpath = test_data_folder / 'lhc_2024/lhc.seq'

    settings = {}
    settings['vrf400'] = 16  # Check voltage expressions
    settings['lagrf400.b1'] = 0.5 + 0.02  # Check lag expressions
    settings['lagrf400.b2'] = 0.02  # Check lag expressions
    settings['on_x1'] = 100  # Check kicker expressions
    settings['on_sep2h'] = 2  # Check kicker expressions
    settings['on_x5'] = 123  # Check kicker expressions
    settings['dqx.b2'] = 3e-3  # Check quad expressions
    settings['dqx.b1'] = 2e-3  # Check quad expressions
    settings['dqpx.b1'] = 2.  # Check sext expressions
    settings['dqpx.b2'] = 3.  # Check sext expressions
    settings['kqs.l3b2'] = 1e-4  # Check skew expressions
    settings['kss.a45b2'] = 1e-4  # Check skew sext expressions
    settings['kof.a34b2'] = 3  # Check oct expressions
    settings['on_sol_atlas'] = 1  # Check solenoid expressions
    settings['kcd.a34b2'] = 1e-4  # Check decapole expressions
    settings['kcd.a34b1'] = 1e-4  # Check decapole expressions
    settings['kctx3.l1'] = 1e-5  # Check thin dodecapole expressions

    mad = Madx()
    mad.call(str(fpath))
    mad.input('beam, sequence=lhcb1, particle=proton, energy=7000;')
    mad.use('lhcb1')
    mad.input('beam, sequence=lhcb2, particle=proton, energy=7000, bv=-1;')
    mad.use('lhcb2')
    mad.call(str(test_data_folder / 'lhc_2024/injection_optics.madx'))

    for kk, vv in settings.items():
        mad.globals[kk] = vv

    line1_ref = xt.Line.from_madx_sequence(
            sequence=mad.sequence.lhcb1,
            allow_thick=True,
            deferred_expressions=True,
            replace_in_expr={'bv_aux': 'bvaux_b2'},
    )
    line1_ref.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7000e9,
                                          _context=xo.ContextCpu())
    line1_ref.build_tracker(_context=xo.ContextCpu())

    line2_ref = xt.Line.from_madx_sequence(
            sequence=mad.sequence.lhcb2,
            allow_thick=True,
            deferred_expressions=True,
            replace_in_expr={'bv_aux': 'bvaux_b2'},
        )
    line2_ref.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7000e9,
                                          _context=xo.ContextCpu())
    line2_ref.build_tracker(_context=xo.ContextCpu())

    return settings, line1_ref, line2_ref

@pytest.mark.parametrize("line_mode", ['normal', 'compose'])
@pytest.mark.parametrize("data_mode", ['direct', 'json', 'copy', 'py'])
def test_native_loader_lhc(line_mode, data_mode, tmpdir, lines_ref):

    if line_mode == 'normal' and data_mode == 'py':
        pytest.skip("Export only compose mode to py lattice")

    fpath = test_data_folder / 'lhc_2024/lhc.seq'

    with open(fpath, 'r') as fid:
        seq_text = fid.read()

    assert ' at=' in seq_text
    assert ',at=' not in seq_text
    assert 'at =' not in seq_text
    seq_text = seq_text.replace(' at=', 'at:=')

    env = xt.load(string=seq_text, format='madx', reverse_lines=['lhcb2'])
    env.lhcb1.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7000e9)
    env.lhcb2.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7000e9)

    env.vars.load(test_data_folder / 'lhc_2024/injection_optics.madx')

    if line_mode == 'compose':
        env.lhcb1.regenerate_from_composer()
        env.lhcb2.regenerate_from_composer()
        for nn in list(env.elements.keys()):
            if nn.startswith('drift_'):
                del env._element_dict[nn]
        assert env.lhcb1.mode == 'compose'
        assert env.lhcb2.mode == 'compose'
        assert env.lhcb1.composer.mirror is False
        assert env.lhcb2.composer.mirror is True
        assert env.lhcb1.element_names == '__COMPOSE__'
        assert env.lhcb2.element_names == '__COMPOSE__'

    if data_mode == 'json':
        env_orig = env
        env.to_json(tmpdir / f'lhc_{line_mode}.json')
        env = xt.load(tmpdir / f'lhc_{line_mode}.json', format='json')
    elif data_mode == 'copy':
        env = env.copy()
    elif data_mode == 'py':
        # Force k0_from_h to False (they are all provided)
        for nn in list(env.elements.keys()):
            if hasattr(env.elements[nn], 'k0_from_h'):
                env.elements[nn].k0_from_h = False
        lpg.write_py_lattice_file(env,
                                  output_fname=tmpdir / f'lhc_{line_mode}.py')
        env = xt.Environment()
        env.call(tmpdir / f'lhc_{line_mode}.py')
        env.lhcb1.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7000e9)
        env.lhcb2.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7000e9)
    else:
        assert data_mode == 'direct'

    # Some checks based on direct inspection of MAD-X file
    xo.assert_allclose(env['ip8ofs.b2'],  -154, atol=1e-12)
    assert (str(env.ref['aip2']._expr) == "f['atan'](((vars['sep_arc'] / 2.0) / vars['dsep2']))"
        or str(env.ref['aip2']._expr) == "f.atan(((vars['sep_arc'] / 2.0) / vars['dsep2']))")

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
    assert not hasattr(env['adtkv'], 'extra')

    assert env['mcbv'].prototype == 'vcorrector'
    assert env['vcorrector'].prototype == 'vkicker'
    assert env['vkicker'].prototype is None
    assert isinstance(env['mcbv'], xt.Multipole)
    assert isinstance(env['vcorrector'], xt.Multipole)
    assert isinstance(env['vkicker'], xt.Multipole)
    assert env['mcbv'].isthick
    assert str(env.ref['mcbv'].length._expr) == "vars['l.mcbv']"
    assert str(env.ref['mcbv'].extra['calib']._expr) == "(vars['kmax_mcbv'] / vars['imax_mcbv'])"
    assert type(env['mcbv'].extra).__name__ == 'dict'

    assert env['acsca'].prototype == 'rfcavity'
    assert env['rfcavity'].prototype is None
    assert isinstance(env['acsca'], xt.Cavity)
    assert isinstance(env['rfcavity'], xt.Cavity)
    assert str(env.ref['acsca'].length._expr) == "vars['l.acsca']"
    assert type(env['acsca']).__name__ == 'View'
    assert not hasattr(env['acsca'], 'extra')

    assert env['mbas2'].prototype == 'solenoid'
    assert env['solenoid'].prototype is None
    assert isinstance(env['mbas2'], xt.UniformSolenoid)
    assert isinstance(env['solenoid'], xt.UniformSolenoid)
    assert str(env.ref['mbas2'].length._expr) == "vars['l.mbas2']"
    assert type(env['mbas2']).__name__ == 'View'
    assert not hasattr(env['mbas2'], 'extra')

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
    assert type(env['mqxa.1r1/lhcb1']).__name__ == 'View'
    assert type(env['mqxa.1r1/lhcb1'].knl).__name__ == 'View'
    assert type(env['mqxa.1r1/lhcb1'].extra).__name__ == 'dict'

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
    assert type(env['mcssx.3r1/lhcb1']).__name__ == 'View'
    assert type(env['mcssx.3r1/lhcb1'].knl).__name__ == 'View'
    assert type(env['mcssx.3r1/lhcb1'].extra).__name__ == 'dict'

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
    assert type(env['mb.a8r1.b1']).__name__ == 'View'
    assert type(env['mb.a8r1.b1'].knl).__name__ == 'View'
    assert type(env['mb.a8r1.b1'].extra).__name__ == 'dict'

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
    assert type(env['mqxa.1r1/lhcb2']).__name__ == 'View'
    assert type(env['mqxa.1r1/lhcb2'].knl).__name__ == 'View'
    assert type(env['mqxa.1r1/lhcb2'].extra).__name__ == 'dict'

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
    assert type(env['mcssx.3r1/lhcb2']).__name__ == 'View'
    assert type(env['mcssx.3r1/lhcb2'].knl).__name__ == 'View'
    assert type(env['mcssx.3r1/lhcb2'].extra).__name__ == 'dict'

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
    assert type(env['mb.a8r1.b2']).__name__ == 'View'
    assert type(env['mb.a8r1.b2'].knl).__name__ == 'View'
    assert type(env['mb.a8r1.b2'].extra).__name__ == 'dict'

    # Check composer
    if data_mode == 'direct': # other cases not yet implemented
        assert not env.lhcb1.composer.mirror
        assert env.lhcb2.composer.mirror

        assert env.lhcb1.composer.components[1000].name == 'mco.b14r2.b1'
        assert xd.refs.is_ref(env.lhcb1.composer.components[1000].at)
        assert str(env.lhcb1.composer.components[1000].at) == "(578.4137 + ((138.0 - vars['ip2ofs.b1']) * vars['ds']))"
        assert env.lhcb1.composer.components[1000].from_ == 'ip2'

        assert env.lhcb2.composer.components[1000].name == 'mcbv.14r2.b2'
        assert xd.refs.is_ref(env.lhcb2.composer.components[1000].at)
        assert str(env.lhcb2.composer.components[1000].at) == "(599.4527 + ((-137.0 - vars['ip2ofs.b2']) * vars['ds']))"
        assert env.lhcb2.composer.components[1000].from_ == 'ip2'

    # Check against cpymad line
    settings, line1_ref, line2_ref = lines_ref

    for kk in settings:
        assert len(env.ref[kk]._find_dependant_targets())>1

    for kk, vv in settings.items():
        env[kk] = vv

    for lref, ltest, beam in [(line1_ref, env.lhcb1, 1), (line2_ref, env.lhcb2, 2)]:

        tt_ref = lref.get_table()
        tt_test = ltest.get_table()

        tt_ref_nodr = tt_ref.rows[tt_ref.element_type != 'Drift']
        tt_test_nodr = tt_test.rows[tt_test.element_type != 'Drift']

        # Check s
        lref_names = list(tt_ref_nodr.name)
        ltest_names = list(tt_test_nodr.name)

        for nn in ['lhcb1$start', 'lhcb1$end', 'lhcb2$start', 'lhcb2$end']:
            if nn in lref_names:
                lref_names.remove(nn)

        assert lref_names == [
            nn[:-len(f'/lhcb{beam}')] if nn.endswith(f'/lhcb{beam}') else nn for nn in ltest_names]

        xo.assert_allclose(
            tt_ref_nodr.rows[lref_names].s_center, tt_test_nodr.rows[ltest_names].s_center,
            rtol=0, atol=5e-9)

        for nn in ltest_names:
            print(f'Checking: {nn}                     ', end='\r', flush=True)
            if nn == '_end_point':
                continue
            nn_straight = nn[:-len(f'/lhcb{beam}')] if nn.endswith(f'/lhcb{beam}') else nn
            eref = lref[nn_straight]
            etest = ltest[nn]
            dref = eref.to_dict()
            dtest = etest.to_dict()
            is_rbend = isinstance(etest, xt.RBend)

            for kk in dref.keys():

                if kk == 'prototype':
                    continue  # prototype is always None from cpymad

                if kk in ('__class__', 'model', 'side'):
                    assert dref[kk] == dtest[kk]
                    continue

                if kk == '_isthick' and eref.length == 0:
                    continue  # Skip the check for zero-length elements

                if kk in {
                    'order',  # Always assumed to be 5, not always the same
                    'frequency',  # If not specified, depends on the beam,
                                    # so for now we ignore it
                }:
                    continue

                if kk in {'knl', 'ksl'}:
                    maxlen = max(len(dref[kk]), len(dtest[kk]))
                    lhs = np.pad(dref[kk], (0, maxlen - len(dref[kk])), mode='constant')
                    rhs = np.pad(dtest[kk], (0, maxlen - len(dtest[kk])), mode='constant')
                    xo.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-16)
                    continue

                if is_rbend and kk in ('length', 'length_straight'):
                    xo.assert_allclose(dref[kk], dtest[kk], rtol=1e-7, atol=1e-6)
                    continue

                if is_rbend and kk in ('h', 'k0'):
                    xo.assert_allclose(dref[kk], dtest[kk], rtol=1e-7, atol=5e-10)
                    continue

                xo.assert_allclose(dref[kk], dtest[kk], rtol=1e-10, atol=1e-16)

        twref = lref.twiss4d()
        twtest = ltest.twiss4d()

        xo.assert_allclose(twref.rows['ip.*'].betx, twtest.rows['ip.*'].betx, rtol=1e-6,
                        atol=0)
        xo.assert_allclose(twref.rows['ip.*'].bety, twtest.rows['ip.*'].bety, rtol=1e-6,
                        atol=0)
        xo.assert_allclose(twref.rows['ip.*'].dx, twtest.rows['ip.*'].dx, rtol=0,
                        atol=1e-6)
        xo.assert_allclose(twref.rows['ip.*'].dy, twtest.rows['ip.*'].dy, rtol=1e-6,
                        atol=1e-6)
        xo.assert_allclose(twref.rows['ip.*'].ax_chrom, twtest.rows['ip.*'].ax_chrom,
                        rtol=1e-4, atol=1e-5)
        xo.assert_allclose(twref.rows['ip.*'].ay_chrom, twtest.rows['ip.*'].ay_chrom,
                        rtol=1e-4, atol=1e-5)

def test_load_multipoles_long_knl_ksl():

    mad_src = '''
        m1: multipole, knl={0, 0.01, 0.0, 0,0, 0.1, 0.3, 0.7};

        seq: sequence, l=10.0;
        m1a: m1, at=2.0;
        endsequence;
    '''

    env = xt.load(string=mad_src, format='madx')
    xo.assert_allclose(env.elements['m1'].knl,
                    [0, 0.01, 0.0, 0, 0, 0.1, 0.3, 0.7], rtol=0, atol=1e-12)


    mad = Madx()
    mad.input(mad_src)
    mad.beam()
    mad.use('seq')
    line = xt.Line.from_madx_sequence(mad.sequence.seq)
    xo.assert_allclose(line['m1a'].knl,
                    [0, 0.01, 0.0, 0, 0, 0.1, 0.3, 0.7], rtol=0, atol=1e-12)
