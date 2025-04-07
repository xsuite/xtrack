import pathlib

import numpy as np
import pytest
from cpymad.madx import Madx

import xobjects as xo
import xpart as xp
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts
from xtrack.slicing import Strategy, Teapot

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

@pytest.mark.parametrize('deferred_expressions', [True, False])
@for_all_test_contexts
def test_sps_thick(test_context, deferred_expressions):

    mad = Madx(stdout=False)
    mad.call(str(test_data_folder) + '/sps_thick/sps.seq')
    mad.input('beam, particle=proton, pc=26;')
    mad.call(str(test_data_folder) + '/sps_thick/lhc_q20.str')
    mad.use(sequence='sps')
    twmad = mad.twiss()

    line = xt.Line.from_madx_sequence(
        sequence=mad.sequence.sps,
        deferred_expressions=deferred_expressions,
        allow_thick=True)
    line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV,
                                    gamma0=mad.sequence.sps.beam.gamma)
    line.build_tracker(_context=test_context)
    line.twiss_default['method'] = '4d'

    # Check a bend
    assert line.element_names[26] == 'mbb.10150'

    assert isinstance(line['mbb.10150'], xt.RBend)

    assert line['mbb.10150'].edge_entry_model == 'linear'
    assert line['mbb.10150'].edge_exit_model == 'linear'
    assert line['mbb.10150'].model == 'adaptive'

    ang = line['mbb.10150'].k0 * line['mbb.10150'].length
    xo.assert_allclose(line['mbb.10150'].edge_entry_angle, 0, atol=1e-11, rtol=0)
    xo.assert_allclose(line['mbb.10150'].edge_exit_angle, 0, atol=1e-11, rtol=0)

    tw = line.twiss()
    xo.assert_allclose(twmad.s[-1], tw.s[-1], atol=1e-9, rtol=0)
    xo.assert_allclose(twmad.summary.q1, tw.qx, rtol=0, atol=1e-7)
    xo.assert_allclose(twmad.summary.q2, tw.qy, rtol=0, atol=1e-7)
    xo.assert_allclose(twmad.summary.dq1, tw.dqx, rtol=0, atol=0.2)
    xo.assert_allclose(twmad.summary.dq2, tw.dqy, rtol=0, atol=0.2)

    line.configure_bend_model(edge='full', core='full')

    tw = line.twiss()

    assert line['mbb.10150'].edge_entry_model == 'full'
    assert line['mbb.10150'].edge_exit_model == 'full'
    assert line['mbb.10150'].model == 'full'

    xo.assert_allclose(twmad.s[-1], tw.s[-1], atol=1e-9, rtol=0)
    xo.assert_allclose(twmad.summary.q1, tw.qx, rtol=0, atol=1e-7)
    xo.assert_allclose(twmad.summary.q2, tw.qy, rtol=0, atol=1e-7)
    xo.assert_allclose(twmad.summary.dq1, tw.dqx, rtol=0, atol=0.01)
    xo.assert_allclose(twmad.summary.dq2, tw.dqy, rtol=0, atol=0.01)

    line.configure_bend_model(core='expanded')

    tw = line.twiss()

    assert line['mbb.10150'].edge_entry_model == 'full'
    assert line['mbb.10150'].edge_exit_model == 'full'
    assert line['mbb.10150'].model == 'mat-kick-mat'

    xo.assert_allclose(twmad.s[-1], tw.s[-1], atol=1e-9, rtol=0)
    xo.assert_allclose(twmad.summary.q1, tw.qx, rtol=0, atol=1e-7)
    xo.assert_allclose(twmad.summary.q2, tw.qy, rtol=0, atol=1e-7)
    xo.assert_allclose(twmad.summary.dq1, tw.dqx, rtol=0, atol=0.2)
    xo.assert_allclose(twmad.summary.dq2, tw.dqy, rtol=0, atol=0.2)

    line.configure_bend_model(edge='linear')

    assert line['mbb.10150'].edge_entry_model == 'linear'
    assert line['mbb.10150'].edge_exit_model == 'linear'
    assert line['mbb.10150'].model == 'mat-kick-mat'

    line.configure_bend_model(core='full')
    line.configure_bend_model(edge='full')

    assert line['mbb.10150'].edge_entry_model == 'full'
    assert line['mbb.10150'].edge_exit_model == 'full'
    assert line['mbb.10150'].model == 'full'

    # Test from_dict/to_dict roundtrip
    dct = line.to_dict()
    line = xt.Line.from_dict(dct)

    assert line['mbb.10150'].edge_entry_model == 'full'
    assert line['mbb.10150'].edge_exit_model == 'full'
    assert line['mbb.10150'].model == 'full'

    line.discard_tracker()
    slicing_strategies = [
        Strategy(slicing=Teapot(1)),  # Default
        Strategy(slicing=Teapot(2), element_type=xt.Bend),
        Strategy(slicing=Teapot(2), element_type=xt.RBend),
        Strategy(slicing=Teapot(8), element_type=xt.Quadrupole),
    ]

    line.slice_thick_elements(slicing_strategies)
    line.build_tracker(_context=test_context)

    # Check a bend
    assert line.element_names[112] == 'mbb.10150_entry'
    assert line.element_names[113] == 'mbb.10150..entry_map'
    assert line.element_names[114] == 'drift_mbb.10150..0'
    assert line.element_names[115] == 'mbb.10150..0'
    assert line.element_names[116] == 'drift_mbb.10150..1'
    assert line.element_names[117] == 'mbb.10150..1'
    assert line.element_names[118] == 'drift_mbb.10150..2'
    assert line.element_names[119] == 'mbb.10150..exit_map'
    assert line.element_names[120] == 'mbb.10150_exit'

    assert isinstance(line['mbb.10150_entry'], xt.Marker)
    assert isinstance(line['mbb.10150..entry_map'], xt.ThinSliceRBendEntry)
    assert isinstance(line['drift_mbb.10150..0'], xt.DriftSliceRBend)
    assert isinstance(line['mbb.10150..0'], xt.ThinSliceRBend)
    assert isinstance(line['drift_mbb.10150..1'], xt.DriftSliceRBend)
    assert isinstance(line['mbb.10150..1'], xt.ThinSliceRBend)
    assert isinstance(line['drift_mbb.10150..2'], xt.DriftSliceRBend)
    assert isinstance(line['mbb.10150..exit_map'], xt.ThinSliceRBendExit)
    assert isinstance(line['mbb.10150_exit'], xt.Marker)

    # Check a quadrupole
    assert line.element_names[156] == 'qf.10210_entry'
    assert line.element_names[157] == 'qf.10210..entry_map'
    assert line.element_names[158] == 'drift_qf.10210..0'
    assert line.element_names[159] == 'qf.10210..0'
    assert line.element_names[160] == 'drift_qf.10210..1'
    assert line.element_names[161] == 'qf.10210..1'
    assert line.element_names[162] == 'drift_qf.10210..2'
    assert line.element_names[163] == 'qf.10210..2'
    assert line.element_names[164] == 'drift_qf.10210..3'
    assert line.element_names[165] == 'qf.10210..3'
    assert line.element_names[166] == 'drift_qf.10210..4'
    assert line.element_names[167] == 'qf.10210..4'
    assert line.element_names[168] == 'drift_qf.10210..5'
    assert line.element_names[169] == 'qf.10210..5'
    assert line.element_names[170] == 'drift_qf.10210..6'
    assert line.element_names[171] == 'qf.10210..6'
    assert line.element_names[172] == 'drift_qf.10210..7'
    assert line.element_names[173] == 'qf.10210..7'
    assert line.element_names[174] == 'drift_qf.10210..8'
    assert line.element_names[175] == 'qf.10210..exit_map'
    assert line.element_names[176] == 'qf.10210_exit'

    assert isinstance(line['qf.10210..7'], xt.ThinSliceQuadrupole)

    assert line['mbb.10150..entry_map']._parent.model == 'full'
    assert line['mbb.10150..exit_map']._parent.model == 'full'

    line.configure_bend_model(edge='linear')

    assert line['mbb.10150..entry_map']._parent.edge_entry_model == 'linear'
    assert line['mbb.10150..exit_map']._parent.edge_exit_model == 'linear'

    line.configure_bend_model(edge='full')

    assert line['mbb.10150..entry_map']._parent.edge_entry_model == 'full'
    assert line['mbb.10150..exit_map']._parent.edge_exit_model == 'full'

    tw_edge_full = line.twiss()

    xo.assert_allclose(twmad.s[-1], tw_edge_full.s[-1], atol=1e-9, rtol=0)
    xo.assert_allclose(twmad.summary.q1, tw_edge_full.qx, rtol=0, atol=0.5e-3)
    xo.assert_allclose(twmad.summary.q2, tw_edge_full.qy, rtol=0, atol=0.5e-3)
    xo.assert_allclose(twmad.summary.dq1, tw_edge_full.dqx, rtol=0, atol=0.2)
    xo.assert_allclose(twmad.summary.dq2, tw_edge_full.dqy, rtol=0, atol=0.2)

    line.configure_bend_model(edge='linear')
    tw_edge_linear = line.twiss()
    xo.assert_allclose(twmad.s[-1], tw_edge_linear.s[-1], atol=1e-9, rtol=0)
    xo.assert_allclose(twmad.summary.q1, tw_edge_linear.qx, rtol=0, atol=0.5e-3)
    xo.assert_allclose(twmad.summary.q2, tw_edge_linear.qy, rtol=0, atol=0.5e-3)
    xo.assert_allclose(twmad.summary.dq1, tw_edge_linear.dqx, rtol=0, atol=0.2)
    xo.assert_allclose(twmad.summary.dq2, tw_edge_linear.dqy, rtol=0, atol=0.2)

    tw_backwards = line.twiss(start=line.element_names[0],
                end=line.element_names[-1],
                init=tw_edge_linear.get_twiss_init(line.element_names[-1]),
                compute_chromatic_properties=True)

    assert_allclose = np.testing.assert_allclose

    assert_allclose(tw_backwards.s, tw_edge_linear.s, rtol=0, atol=1e-10)
    assert_allclose(tw_backwards.x, tw_edge_linear.x, rtol=0, atol=1e-10)
    assert_allclose(tw_backwards.px, tw_edge_linear.px, rtol=0, atol=1e-10)
    assert_allclose(tw_backwards.y, tw_edge_linear.y, rtol=0, atol=1e-10)
    assert_allclose(tw_backwards.py, tw_edge_linear.py, rtol=0, atol=1e-10)
    assert_allclose(tw_backwards.zeta, tw_edge_linear.zeta, rtol=0, atol=1e-10)
    assert_allclose(tw_backwards.delta, tw_edge_linear.delta, rtol=0, atol=1e-10)
    assert_allclose(tw_backwards.betx, tw_edge_linear.betx, rtol=5e-9, atol=1e-10)
    assert_allclose(tw_backwards.bety, tw_edge_linear.bety, rtol=5e-9, atol=1e-10)
    assert_allclose(tw_backwards.ax_chrom, tw_edge_linear.ax_chrom, rtol=0, atol=1e-5)
    assert_allclose(tw_backwards.ay_chrom, tw_edge_linear.ay_chrom, rtol=0, atol=1e-5)
    assert_allclose(tw_backwards.bx_chrom, tw_edge_linear.bx_chrom, rtol=0, atol=1e-5)
    assert_allclose(tw_backwards.by_chrom, tw_edge_linear.by_chrom, rtol=0, atol=1e-5)
    assert_allclose(tw_backwards.dx, tw_edge_linear.dx, rtol=0, atol=1e-8)
    assert_allclose(tw_backwards.dy, tw_edge_linear.dy, rtol=0, atol=1e-8)
    assert_allclose(tw_backwards.mux[-1] - tw_edge_linear.mux[0], tw_edge_linear.qx,
                    rtol=0, atol=1e-10)
    assert_allclose(tw_backwards.muy[-1] - tw_edge_linear.muy[0], tw_edge_linear.qy,
                    rtol=0, atol=1e-10)
