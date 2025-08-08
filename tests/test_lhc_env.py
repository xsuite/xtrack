import xtrack as xt
import xpart as xp
import xobjects as xo
import pathlib
import numpy as np

from cpymad.madx import Madx

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

def test_lhc_environment():

    mad1=Madx()
    mad1.call(str(test_data_folder / 'hllhc15_thick/lhc.seq'))
    mad1.call(str(test_data_folder / 'hllhc15_thick/hllhc_sequence.madx'))
    mad1.input('beam, sequence=lhcb1, particle=proton, energy=7000;')
    mad1.use('lhcb1')
    mad1.call(str(test_data_folder / "hllhc15_thick/opt_round_150_1500.madx"))
    mad1.twiss()

    mad4=Madx()
    mad4.input('mylhcbeam=4')
    mad4.call(str(test_data_folder / 'hllhc15_thick/lhcb4.seq'))
    mad4.call(str(test_data_folder / 'hllhc15_thick/hllhc_sequence.madx'))
    mad4.input('beam, sequence=lhcb2, particle=proton, energy=7000;')
    mad4.use('lhcb2')
    mad4.call(str(test_data_folder / 'hllhc15_thick/opt_round_150_1500.madx'))
    mad4.twiss()


    line1=xt.Line.from_madx_sequence(mad1.sequence.lhcb1,
                                    allow_thick=True,
                                    deferred_expressions=True,
                                    replace_in_expr={'bv_aux':'bvaux_b1'})
    line1.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, p0c=7000e9)

    line4=xt.Line.from_madx_sequence(mad4.sequence.lhcb2,
                                    allow_thick=True,
                                    deferred_expressions=True,
                                    replace_in_expr={'bv_aux':'bvaux_b2'})
    line4.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, p0c=7000e9)
    line4.twiss_default['reverse'] = True

    env = xt.Environment(lines={'lhcb1': line1, 'lhcb2': line4})
    assert env.lhcb2.twiss_default['reverse'] == True

    env.to_json('lhc.json')

    env = xt.load('lhc.json')

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

    assert 'mq.12l6.b1' in env.lhcb1.element_names
    assert 'mq.12l6.b2' in env.lhcb2.element_names
    assert 'mq.12l6.b2/lhcb2' not in env.lhcb2.element_names

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