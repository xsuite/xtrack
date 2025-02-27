import xtrack as xt
import xobjects as xo
import numpy as np
import pathlib

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()

def test_legacy_multiline_to_env_conversion():
    env = xt.Multiline.from_json(
        test_data_folder / 'hllhc15_thick/hllhc15_collider_thick_legacy_multiline.json')
    assert isinstance(env, xt.Environment)
    env.lhcb2.twiss_default['reverse'] = True

    assert env.lhcb1.element_dict is env.element_dict
    assert env.lhcb2.element_dict is env.element_dict

    assert 'drift_1' in env.lhcb1.element_names
    assert 'drift_1' not in env.lhcb2.element_names

    assert 'ip1' in env.lhcb1.element_names
    assert 'ip1' in env.lhcb2.element_names
    assert 'ip1/b2' not in env.lhcb2.element_names

    assert 'mqxfa.a1r1/lhcb1' in env.lhcb1.element_names
    assert 'mqxfa.a1r1/lhcb2' in env.lhcb2.element_names
    assert 'mqxfa.a1r1'  not in env.lhcb1.element_names
    assert 'mqxfa.a1r1'  not in env.lhcb2.element_names

    assert 'taxs.1l5/lhcb1' in env.lhcb1.element_names
    assert 'taxs.1l5/lhcb2' in env.lhcb2.element_names
    assert 'taxs.1l5'  not in env.lhcb1.element_names
    assert 'taxs.1l5'  not in env.lhcb2.element_names

    assert 'mq.12l6.b1' in env.lhcb1.element_names
    assert 'mq.12l6.b2' in env.lhcb2.element_names
    assert 'mq.12l6.b2/lhcb2' not in env.lhcb2.element_names

    assert 'drift_mcbyh.a4r1.b1..2' in env.lhcb1.element_names
    assert 'drift_mcbrdv.4l1.b2..2' in env.lhcb2.element_names

    twb1 = env.lhcb1.twiss4d()
    twb2 = env.lhcb2.twiss4d()

    assert np.all(twb1.rows['ip.*'].name ==
        np.array(['ip1', 'ip2', 'ip3', 'ip4', 'ip5', 'ip6', 'ip7', 'ip8', 'ip1.l1']))
    assert np.all(twb2.rows['ip.*'].name ==
        np.array(['ip1', 'ip2', 'ip3', 'ip4', 'ip5', 'ip6', 'ip7', 'ip8', 'ip1.l1']))

    xo.assert_allclose(twb1.qx, 62.31, rtol=0, atol=1e-5)
    xo.assert_allclose(twb1.qy, 60.32, rtol=0, atol=1e-5)
    xo.assert_allclose(twb2.qx, 62.31, rtol=0, atol=1e-5)
    xo.assert_allclose(twb2.qy, 60.32, rtol=0, atol=1e-5)

    env['on_disp'] = 0
    env['on_x1'] = 10
    env['on_x5'] = 12

    twb1cross = env.lhcb1.twiss4d()
    twb2cross = env.lhcb2.twiss4d()

    xo.assert_allclose(twb1cross['px', 'ip1'], +10e-6, rtol=0, atol=1e-9)
    xo.assert_allclose(twb2cross['px', 'ip1'], -10e-6, rtol=0, atol=1e-9)
    xo.assert_allclose(twb1cross['py', 'ip5'], +12e-6, rtol=0, atol=1e-9)
    xo.assert_allclose(twb2cross['py', 'ip5'], -12e-6, rtol=0, atol=1e-9)

    # Check combined twiss
    twcross = env.twiss(method='4d')
    xo.assert_allclose(twcross.lhcb1['px', 'ip1'], +10e-6, rtol=0, atol=1e-9)
    xo.assert_allclose(twcross.lhcb2['px', 'ip1'], -10e-6, rtol=0, atol=1e-9)
    xo.assert_allclose(twcross.lhcb1['py', 'ip5'], +12e-6, rtol=0, atol=1e-9)
    xo.assert_allclose(twcross.lhcb2['py', 'ip5'], -12e-6, rtol=0, atol=1e-9)

    env.discard_trackers()
    assert env.lhcb1.tracker is None
    assert env.lhcb2.tracker is None

    mycontext = xo.ContextCpu()
    env.build_trackers(_context=mycontext)
    assert env.lhcb1.tracker is not None
    assert env.lhcb2.tracker is not None
    assert env.lhcb1.tracker._context is mycontext
    assert env.lhcb2.tracker._context is mycontext
