import pathlib

import numpy as np
import pytest
from cpymad.madx import Madx

import xobjects as xo
import xtrack as xt

test_data_folder = pathlib.Path(
            __file__).parent.joinpath('../test_data').absolute()

@pytest.mark.parametrize('case', ['thin', 'thick'])
def test_mad_writer(case):

    if case == 'thick':
        line = xt.load(
            test_data_folder / 'hllhc15_thick/lhc_thick_with_knobs.json')
    else:
        line = xt.load(
            test_data_folder / 'hllhc15_collider/collider_00_from_mad.json').lhcb1
        # Rotations not supported in thin
        for nn in list(line.element_names):
            if (nn.startswith('mb') and
            (nn.endswith('_tilt_entry') or nn.endswith('_tilt_exit'))):
                line.element_names.remove(nn)

    line.build_tracker()
    mad_seq = line.to_madx_sequence(sequence_name='myseq')

    mad = Madx(stdout=True)
    mad.options.rbarc = False
    mad.input(mad_seq)
    mad.beam(particle='proton', energy=7000e9)
    mad.use('myseq')

    line2 = xt.Line.from_madx_sequence(mad.sequence.myseq, deferred_expressions=True)
    line2.particle_ref = line.particle_ref

    for ll in [line, line2]:
        ll.vars['vrf400'] = 16 # Check voltage expressions
        ll.vars['lagrf400.b1'] = 0.52 # Check lag expressions
        ll.vars['on_x1'] = 100 # Check kicker expressions
        ll.vars['on_sep2'] = 2 # Check kicker expressions
        ll.vars['on_x5'] = 123 # Check kicker expressions
        ll.vv['kqtf.b1'] += 1e-5 # Check quad expressions
        ll.vv['ksf.b1'] += 1e-3  # Check sext expressions
        ll.vv['kqs.l4b1'] += 1e-4 # Check skew expressions
        ll.vv['kof.a34b1'] = 3 # Check oct expressions
        ll.vars['on_crab1'] = -190 # Check cavity expressions
        ll.vars['on_crab5'] = -130 # Check cavity expressions

    tw = line.twiss()
    tw2 = line2.twiss()

    assert np.all(tw2.rows['ip.*'].name == tw.rows['ip.*'].name)

    xo.assert_allclose(tw2.rows['ip.*'].s, tw.rows['ip.*'].s, rtol=0, atol=2e-5)
    xo.assert_allclose(tw2.rows['ip.*'].x, tw.rows['ip.*'].x, rtol=0, atol=1e-9)
    xo.assert_allclose(tw2.rows['ip.*'].y, tw.rows['ip.*'].y, rtol=0, atol=1e-9)
    xo.assert_allclose(tw2.rows['ip.*'].px, tw.rows['ip.*'].px, rtol=0, atol=1e-9)
    xo.assert_allclose(tw2.rows['ip.*'].py, tw.rows['ip.*'].py, rtol=0, atol=1e-9)
    xo.assert_allclose(tw2.rows['ip.*'].mux, tw.rows['ip.*'].mux, rtol=0, atol=1e-8)
    xo.assert_allclose(tw2.rows['ip.*'].muy, tw.rows['ip.*'].muy, rtol=0, atol=1e-8)
    xo.assert_allclose(tw2.rows['ip.*'].betx, tw.rows['ip.*'].betx, rtol=1e-7, atol=0)
    xo.assert_allclose(tw2.rows['ip.*'].bety, tw.rows['ip.*'].bety, rtol=1e-7, atol=0)
    xo.assert_allclose(tw2.rows['ip.*'].ax_chrom, tw.rows['ip.*'].ax_chrom, rtol=1e-5, atol=0)
    xo.assert_allclose(tw2.rows['ip.*'].dx_zeta, tw.rows['ip.*'].dx_zeta, rtol=2e-4, atol=1e-6)
    xo.assert_allclose(tw2.rows['ip.*'].dy_zeta, tw.rows['ip.*'].dy_zeta, rtol=2e-4, atol=1e-6)

    xo.assert_allclose(tw2.qs, tw.qs, rtol=1e-4, atol=0)
    xo.assert_allclose(tw2.ddqx, tw.ddqx, rtol=1e-3, atol=0)
