import pathlib

import numpy as np

import xtrack as xt
import xpart as xp
import xdeps as xd

from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()

@for_all_test_contexts
def test_var_cache(test_context):
    # Load the collider
    collider = xt.Multiline.from_json(test_data_folder /
                    'hllhc15_collider/collider_00_from_mad.json')
    collider.build_trackers(_context=test_context)

    collider.lhcb1.twiss_default['method'] = '4d'
    collider.lhcb2.twiss_default['method'] = '4d'
    collider.lhcb2.twiss_default['reverse'] = True

    collider.vars['on_x2'] = 123

    assert 'on_x5' in collider.vars
    assert 'on_x9' not in collider.vars

    assert xd.refs._isref(collider.vars['on_x5'])
    try:
        collider.vars['on_x9']
    except KeyError:
        pass
    else:
        raise ValueError('Should have raised KeyError')

    collider.vars.cache_active = True

    assert 'on_x5' in collider.vars
    assert 'on_x9' not in collider.vars

    assert isinstance(collider.vars['on_x5'], xt.line.VarSetter)
    try:
        collider.vars['on_x9']
    except KeyError:
        pass
    else:
        raise ValueError('Should have raised KeyError')

    assert isinstance(collider.vars['on_x1'], xt.line.VarSetter)
    collider.vars['on_x1'] = 11
    collider.vars['on_x5'] = 55

    assert collider.vars['on_x1']._value == 11
    assert collider.vars['on_x5']._value == 55

    assert np.isclose(
        collider['lhcb1'].twiss()['px', 'ip1'], 11e-6, atol=1e-9, rtol=0)
    assert np.isclose(
        collider['lhcb1'].twiss()['py', 'ip5'], 55e-6, atol=1e-9, rtol=0)

    collider.vars['on_x1'] = 234
    collider.vars['on_x5'] = 123

    assert collider.vars['on_x1']._value == 234
    assert collider.vars['on_x5']._value == 123

    assert np.isclose(
        collider['lhcb1'].twiss()['px', 'ip1'], 234e-6, atol=1e-9, rtol=0)
    assert np.isclose(
        collider['lhcb1'].twiss()['py', 'ip5'], 123e-6, atol=1e-9, rtol=0)

    collider.vars['on_x2'] = 234

    assert np.isclose(
        collider['lhcb1'].twiss()['py', 'ip2'], 234e-6, atol=1e-9, rtol=0)

    collider.vars.cache_active = False
    assert isinstance(collider.vars['on_x1'], xd.refs.MutableRef)

    # Same check on single line within collider

    # Load the line
    collider = xt.Multiline.from_json(test_data_folder /
                    'hllhc15_collider/collider_00_from_mad.json')
    collider.build_trackers(_context=test_context)

    collider.lhcb1.twiss_default['method'] = '4d'
    collider.lhcb2.twiss_default['method'] = '4d'
    collider.lhcb2.twiss_default['reverse'] = True

    line = collider['lhcb1']

    line.vars['on_x2'] = 123

    assert 'on_x5' in line.vars
    assert 'on_x9' not in line.vars

    assert xd.refs._isref(line.vars['on_x5'])
    try:
        line.vars['on_x9']
    except KeyError:
        pass
    else:
        raise ValueError('Should have raised KeyError')

    line.vars.cache_active = True

    assert 'on_x5' in line.vars
    assert 'on_x9' not in line.vars

    assert isinstance(line.vars['on_x5'], xt.line.VarSetter)
    try:
        line.vars['on_x9']
    except KeyError:
        pass
    else:
        raise ValueError('Should have raised KeyError')

    line.vars['on_x1'] = 11
    line.vars['on_x5'] = 55

    assert line.vars['on_x1']._value == 11
    assert line.vars['on_x5']._value == 55

    assert np.isclose(
        line.twiss()['px', 'ip1'], 11e-6, atol=1e-9, rtol=0)
    assert np.isclose(
        line.twiss()['py', 'ip5'], 55e-6, atol=1e-9, rtol=0)

    line.vars['on_x1'] = 234
    line.vars['on_x5'] = 123

    assert line.vars['on_x1']._value == 234
    assert line.vars['on_x5']._value == 123

    assert np.isclose(
        line.twiss()['px', 'ip1'], 234e-6, atol=1e-9, rtol=0)
    assert np.isclose(
        line.twiss()['py', 'ip5'], 123e-6, atol=1e-9, rtol=0)

    line.vars['on_x2'] = 234

    assert np.isclose(
        line.twiss()['py', 'ip2'], 234e-6, atol=1e-9, rtol=0)

    line.vars.cache_active = False
    assert isinstance(line.vars['on_x1'], xd.refs.MutableRef)


    # Checks on isolated line

    line = xt.Line.from_json(test_data_folder /
                'hllhc15_noerrors_nobb/line_w_knobs_and_particle.json')
    line.particle_ref = xp.Particles(p0c=7e12, mass=xp.PROTON_MASS_EV)
    line.twiss_default['method'] = '4d'
    line.build_tracker(_context=test_context)

    line.vars['on_x2'] = 123

    assert xd.refs._isref(line.vars['on_x5'])
    try:
        line.vars['on_x9']
    except KeyError:
        pass
    else:
        raise ValueError('Should have raised KeyError')

    line.vars.cache_active = True

    assert 'on_x5' in line.vars
    assert 'on_x9' not in line.vars

    assert isinstance(line.vars['on_x5'], xt.line.VarSetter)
    try:
        line.vars['on_x9']
    except KeyError:
        pass
    else:
        raise ValueError('Should have raised KeyError')

    line.vars['on_x1'] = 11
    line.vars['on_x5'] = 55

    assert line.vars['on_x1']._value == 11
    assert line.vars['on_x5']._value == 55

    assert np.isclose(
        line.twiss()['px', 'ip1'], 11e-6, atol=1e-9, rtol=0)
    assert np.isclose(
        line.twiss()['py', 'ip5'], 55e-6, atol=1e-9, rtol=0)

    line.vars['on_x1'] = 234
    line.vars['on_x5'] = 123

    assert line.vars['on_x1']._value == 234
    assert line.vars['on_x5']._value == 123

    assert np.isclose(
        line.twiss()['px', 'ip1'], 234e-6, atol=1e-9, rtol=0)
    assert np.isclose(
        line.twiss()['py', 'ip5'], 123e-6, atol=1e-9, rtol=0)

    line.vars['on_x2'] = 234

    assert np.isclose(
        line.twiss()['py', 'ip2'], 234e-6, atol=1e-9, rtol=0)

    line.vars.cache_active = False
    assert isinstance(line.vars['on_x1'], xd.refs.MutableRef)