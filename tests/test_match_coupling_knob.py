import pathlib

import numpy as np

import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

@for_all_test_contexts
def test_match_coupling_knob(test_context):
    line = xt.load(test_data_folder /
        'hllhc14_no_errors_with_coupling_knobs/line_b1.json')
    line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, energy0=7e12)

    line.build_tracker(_context=test_context)

    line.cycle('ip1', inplace=True)
    line.twiss_default['method'] = '4d'

    # Flat machine and skew quadrupoles off
    for nn in line.vars.get_table().rows['on_.*|acb.*|cm.*'].name:
        line.vars[nn] = 0

    # see Eq. 47 in https://cds.cern.ch/record/522049/files/lhc-project-report-501.pdf
    class ActionCmin(xt.Action):
        def __init__(self, line):
            self.line = line
        def run(self):
            tw = self.line.twiss()
            tt = self.line.get_table(attr=True)
            k1sl = tt['k1sl']
            c_min = 1 / (2*np.pi) * np.sum(k1sl * np.sqrt(tw.betx * tw.bety)
                                    * np.exp(1j * 2 * np.pi * (tw.mux - tw.muy)))
            return {'c_min_re': c_min.real, 'c_min_im': c_min.imag}

    act_cmin = ActionCmin(line)

    line.vars['kqs.a34b1'] = 0
    line.vars['kqs.a78b1'] = 0
    line.vars['kqs.l4b1'] = line.vars['kqs.a34b1']
    line.vars['kqs.l8b1'] = line.vars['kqs.a78b1']
    line.vars['kqs.r3b1'] = line.vars['kqs.a34b1']
    line.vars['kqs.r7b1'] = line.vars['kqs.a78b1']
    vary=xt.VaryList(['kqs.a23b1', 'kqs.a34b1', 'kqs.a67b1', 'kqs.a78b1'], step=5e-5)

    c_min_match = 1e-4

    opt_re = line.match_knob(knob_name='c_minus_re.b1',
        knob_value_start=0, knob_value_end=c_min_match,
        run=False,
        vary=vary,
        targets=[
            act_cmin.target('c_min_re', value=c_min_match, tol=1e-8),
            act_cmin.target('c_min_im', value=0, tol=1e-8),
        ])
    opt_re.solve()
    opt_re.generate_knob()

    opt_im = line.match_knob(knob_name='c_minus_im.b1',
        knob_value_start=0, knob_value_end=c_min_match,
        run=False,
        vary=vary,
        targets=[
            act_cmin.target('c_min_re', value=0, tol=1e-8),
            act_cmin.target('c_min_im', value=c_min_match, tol=1e-8),
        ])
    opt_im.solve()
    opt_im.generate_knob()

    line.vars['c_minus_re.b1'] = 5e-3
    line.vars['c_minus_im.b1'] = 0
    tt_re = line.get_table(attr=True)

    line.vars['c_minus_re.b1'] = 0
    line.vars['c_minus_im.b1'] = 5e-3
    tt_im = line.get_table(attr=True)

    # Check orthogonality
    line.vars['c_minus_re.b1'] = 1e-3
    line.vars['c_minus_im.b1'] = 1e-3
    xo.assert_allclose(line.twiss().c_minus/np.sqrt(2), 1e-3, rtol=0, atol=1.5e-5)

    # Compare against "legacy" knobs
    line_legacy = xt.load(test_data_folder /
        'hllhc14_no_errors_with_coupling_knobs/line_b1.json')
    line_legacy.cycle('ip1', inplace=True)

    # Flat machine and skew quadrupoles off
    for nn in line_legacy.vars.get_table().rows['on_.*|acb.*|cm.*'].name:
        line_legacy.vars[nn] = 0

    tw_leg = line_legacy.twiss()

    line_legacy.vars['cmrskew'] = 5e-3
    line_legacy.vars['cmiskew'] = 0
    tt_re_leg = line_legacy.get_table(attr=True)

    line_legacy.vars['cmrskew'] = 0
    line_legacy.vars['cmiskew'] = 5e-3
    tt_im_leg = line_legacy.get_table(attr=True)

    # Withing 12% of the maximum value
    xo.assert_allclose(tt_re.k1sl, tt_re_leg.k1sl, rtol=0,
                    atol=0.12 * np.max(tt_re_leg.k1sl))
    xo.assert_allclose(tt_im.k1sl, tt_im_leg.k1sl, rtol=0,
                    atol=0.12 * np.max(tt_im_leg.k1sl))

