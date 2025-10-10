import pathlib

import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()

@for_all_test_contexts
def test_cavity_absolute_time(test_context):
    line = xt.load(test_data_folder /
                            'hllhc15_thick/lhc_thick_with_knobs.json')
    line.build_tracker(_context=test_context)
    line.vv['vrf400'] = 16

    for vv in line.vars.get_table().rows[
        'on_x.*|on_sep.*|on_crab.*|on_alice|on_lhcb|corr_.*'].name:
        line.vars[vv] = 0

    tw = line.twiss()

    df_hz = -50
    h_rf = 35640
    f_rev = 1/tw.T_rev0
    df_rev = df_hz / h_rf
    eta = tw.slip_factor

    f_rf0 = 1/tw.T_rev0 * h_rf

    f_rf = f_rf0 + df_hz
    line.vars['f_rf'] = f_rf
    tt = line.get_table()
    for nn in tt.rows[tt.element_type=='Cavity'].name:
        line.element_refs[nn].absolute_time = 1 # Need property
        line.element_refs[nn].frequency = line.vars['f_rf']

    tw1 = line.twiss(search_for_t_rev=True)

    f_rev_expected = f_rf / h_rf

    xo.assert_allclose(f_rev_expected, 1/tw1.T_rev, atol=1e-5, rtol=0)
    xo.assert_allclose(tw1.delta, tw1.delta[0], atol=1e-5, rtol=0) # Check that it is flat
    delta_expected = -df_rev / f_rev / eta
    xo.assert_allclose(tw1.delta, delta_expected, atol=2e-6, rtol=0)
    tw_off_mom = line.twiss(method='4d', delta0=tw1.delta[0])
    xo.assert_allclose(tw1.x, tw_off_mom.x, atol=1e-5, rtol=0)