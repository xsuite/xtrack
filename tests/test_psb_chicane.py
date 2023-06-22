import pathlib

import numpy as np
from cpymad.madx import Madx

import xtrack as xt
import xpart as xp

from xobjects.test_helpers import for_all_test_contexts

from cpymad.madx import Madx

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

@for_all_test_contexts
def test_psb_chicane(test_context):
    mad = Madx()

    # Load mad model and apply element shifts
    mad.input(f'''
    call, file = '{str(test_data_folder)}/psb_chicane/psb.seq';
    call, file = '{str(test_data_folder)}/psb_chicane/psb_fb_lhc.str';

    beam, particle=PROTON, pc=0.5708301551893517;
    use, sequence=psb1;

    select,flag=error,clear;
    select,flag=error,pattern=bi1.bsw1l1.1*;
    ealign, dx=-0.0057;

    select,flag=error,clear;
    select,flag=error,pattern=bi1.bsw1l1.2*;
    select,flag=error,pattern=bi1.bsw1l1.3*;
    select,flag=error,pattern=bi1.bsw1l1.4*;
    ealign, dx=-0.0442;

    twiss;
    ''')

    line = xt.Line.from_madx_sequence(mad.sequence.psb1,
                                allow_thick=True,
                                enable_align_errors=True,
                                deferred_expressions=True)
    line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV,
                                gamma0=mad.sequence.psb1.beam.gamma)
    line.twiss_default['method'] = '4d'
    line.configure_bend_method('full')

    # Build chicane knob (k0)
    line.vars['bsw_k0l'] = 0
    line.vars['k0bi1bsw1l11'] = (line.vars['bsw_k0l'] / line['bi1.bsw1l1.1'].length)
    line.vars['k0bi1bsw1l12'] = (-line.vars['bsw_k0l'] / line['bi1.bsw1l1.2'].length)
    line.vars['k0bi1bsw1l13'] = (-line.vars['bsw_k0l'] / line['bi1.bsw1l1.3'].length)
    line.vars['k0bi1bsw1l14'] = (line.vars['bsw_k0l'] / line['bi1.bsw1l1.4'].length)


    # Build knob to model eddy currents (k2)
    line.vars['bsw_k2l'] = 0
    line.element_refs['bi1.bsw1l1.1'].knl[2] = line.vars['bsw_k2l']
    line.element_refs['bi1.bsw1l1.2'].knl[2] = -line.vars['bsw_k2l']
    line.element_refs['bi1.bsw1l1.3'].knl[2] = -line.vars['bsw_k2l']
    line.element_refs['bi1.bsw1l1.4'].knl[2] = line.vars['bsw_k2l']

    # Test to_dict roundtrip
    line = xt.Line.from_dict(line.to_dict())
    line.build_tracker(_context=test_context)

    bsw_k2l_ref = -9.7429e-2 # Maximum ramp rate
    bsw_k0l_ref = 6.6e-2 # Full bump amplitude

    line.vars['bsw_k2l'] = bsw_k2l_ref
    line.vars['bsw_k0l'] = bsw_k0l_ref
    assert np.isclose(line['bi1.bsw1l1.1'].knl[2], bsw_k2l_ref, rtol=0, atol=1e-10)
    assert np.isclose(line['bi1.bsw1l1.2'].knl[2], -bsw_k2l_ref, rtol=0, atol=1e-10)
    assert np.isclose(line['bi1.bsw1l1.3'].knl[2], -bsw_k2l_ref, rtol=0, atol=1e-10)
    assert np.isclose(line['bi1.bsw1l1.4'].knl[2], bsw_k2l_ref, rtol=0, atol=1e-10)
    assert np.isclose(line['bi1.bsw1l1.1'].k0, bsw_k0l_ref / line['bi1.bsw1l1.1'].length, rtol=0, atol=1e-10)
    assert np.isclose(line['bi1.bsw1l1.2'].k0, -bsw_k0l_ref / line['bi1.bsw1l1.2'].length, rtol=0, atol=1e-10)
    assert np.isclose(line['bi1.bsw1l1.3'].k0, -bsw_k0l_ref / line['bi1.bsw1l1.3'].length, rtol=0, atol=1e-10)
    assert np.isclose(line['bi1.bsw1l1.4'].k0, bsw_k0l_ref / line['bi1.bsw1l1.4'].length, rtol=0, atol=1e-10)

    tw = line.twiss()
    assert np.isclose(tw['x', 'bi1.tstr1l1'], -0.0457367, rtol=0, atol=1e-5)
    assert np.isclose(tw['y', 'bi1.tstr1l1'], 0.0000000, rtol=0, atol=1e-5)
    assert np.isclose(tw['betx', 'bi1.tstr1l1'], 5.20006, rtol=0, atol=1e-4)
    assert np.isclose(tw['bety', 'bi1.tstr1l1'], 6.91701, rtol=0, atol=1e-4)
    assert np.isclose(tw.qy, 4.474490031799888, rtol=0, atol=1e-6) # verify that it does not change from one version to the other
    assert np.isclose(tw.qx, 4.396711590204319, rtol=0, atol=1e-6)
    assert np.isclose(tw.dqy, -8.636405235646905, rtol=0, atol=1e-4)
    assert np.isclose(tw.dqx, -3.560656125021211, rtol=0, atol=1e-4)

    line.vars['bsw_k2l'] = bsw_k2l_ref / 3
    assert np.isclose(line['bi1.bsw1l1.1'].knl[2], bsw_k2l_ref / 3, rtol=0, atol=1e-10)
    assert np.isclose(line['bi1.bsw1l1.2'].knl[2], -bsw_k2l_ref / 3, rtol=0, atol=1e-10)
    assert np.isclose(line['bi1.bsw1l1.3'].knl[2], -bsw_k2l_ref / 3, rtol=0, atol=1e-10)
    assert np.isclose(line['bi1.bsw1l1.4'].knl[2], bsw_k2l_ref / 3, rtol=0, atol=1e-10)
    assert np.isclose(line['bi1.bsw1l1.1'].k0, bsw_k0l_ref / line['bi1.bsw1l1.1'].length, rtol=0, atol=1e-10)
    assert np.isclose(line['bi1.bsw1l1.2'].k0, -bsw_k0l_ref / line['bi1.bsw1l1.2'].length, rtol=0, atol=1e-10)
    assert np.isclose(line['bi1.bsw1l1.3'].k0, -bsw_k0l_ref / line['bi1.bsw1l1.3'].length, rtol=0, atol=1e-10)
    assert np.isclose(line['bi1.bsw1l1.4'].k0, bsw_k0l_ref / line['bi1.bsw1l1.4'].length, rtol=0, atol=1e-10)

    tw = line.twiss()
    assert np.isclose(tw['x', 'bi1.tstr1l1'], -0.04588556, rtol=0, atol=1e-5)
    assert np.isclose(tw['y', 'bi1.tstr1l1'], 0.0000000, rtol=0, atol=1e-5)
    assert np.isclose(tw['betx', 'bi1.tstr1l1'], 5.263928, rtol=0, atol=1e-4)
    assert np.isclose(tw['bety', 'bi1.tstr1l1'], 6.322020, rtol=0, atol=1e-4)
    assert np.isclose(tw.qy, 4.471798396829118, rtol=0, atol=1e-6)
    assert np.isclose(tw.qx, 4.398925843617764, rtol=0, atol=1e-6)
    assert np.isclose(tw.dqy, -8.20730683661175, rtol=0, atol=1e-4)
    assert np.isclose(tw.dqx, -3.5636345521616875, rtol=0, atol=1e-4)

    # Switch off bsws
    line.vars['bsw_k0l'] = 0
    line.vars['bsw_k2l'] = 0
    assert np.isclose(line['bi1.bsw1l1.1'].knl[2], 0, rtol=0, atol=1e-10)
    assert np.isclose(line['bi1.bsw1l1.2'].knl[2], 0, rtol=0, atol=1e-10)
    assert np.isclose(line['bi1.bsw1l1.3'].knl[2], 0, rtol=0, atol=1e-10)
    assert np.isclose(line['bi1.bsw1l1.4'].knl[2], 0, rtol=0, atol=1e-10)
    assert np.isclose(line['bi1.bsw1l1.1'].k0, 0, rtol=0, atol=1e-10)
    assert np.isclose(line['bi1.bsw1l1.2'].k0, 0, rtol=0, atol=1e-10)
    assert np.isclose(line['bi1.bsw1l1.3'].k0, 0, rtol=0, atol=1e-10)
    assert np.isclose(line['bi1.bsw1l1.4'].k0, 0, rtol=0, atol=1e-10)

    tw = line.twiss()
    assert np.isclose(tw['x', 'bi1.tstr1l1'], 0, rtol=0, atol=1e-5)
    assert np.isclose(tw['y', 'bi1.tstr1l1'], 0, rtol=0, atol=1e-5)
    assert np.isclose(tw['betx', 'bi1.tstr1l1'], 5.2996347, rtol=0, atol=1e-4)
    assert np.isclose(tw['bety', 'bi1.tstr1l1'], 3.838857, rtol=0, atol=1e-4)
    assert np.isclose(tw.qy, 4.45, rtol=0, atol=1e-6)
    assert np.isclose(tw.qx, 4.4, rtol=0, atol=1e-6)
    assert np.isclose(tw.dqy, -7.149781341846406, rtol=0, atol=1e-4)
    assert np.isclose(tw.dqx, -3.5655757511587893, rtol=0, atol=1e-4)
