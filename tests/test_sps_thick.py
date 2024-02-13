import pathlib

import numpy as np
import pytest

from cpymad.madx import Madx

import xpart as xp
import xtrack as xt
from xtrack.slicing import Strategy, Teapot
from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

@pytest.mark.parametrize('deferred_expressions', [True, False])
@for_all_test_contexts
def test_sps_thick(test_context, deferred_expressions):

    mad = Madx()
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
    assert line.element_names[58] == 'mbb.10150_entry'
    assert line.element_names[59] == 'mbb.10150_den'
    assert line.element_names[60] == 'mbb.10150'
    assert line.element_names[61] == 'mbb.10150_dex'
    assert line.element_names[62] == 'mbb.10150_exit'

    assert isinstance(line['mbb.10150_entry'], xt.Marker)
    assert isinstance(line['mbb.10150_den'], xt.DipoleEdge)
    assert isinstance(line['mbb.10150'], xt.Bend)
    assert isinstance(line['mbb.10150_den'], xt.DipoleEdge)
    assert isinstance(line['mbb.10150_exit'], xt.Marker)

    assert line['mbb.10150_den'].model == 'linear'
    assert line['mbb.10150_den'].side == 'entry'
    assert line['mbb.10150_dex'].model == 'linear'
    assert line['mbb.10150_dex'].side == 'exit'
    assert line['mbb.10150'].model == 'adaptive'

    ang = line['mbb.10150'].k0 * line['mbb.10150'].length
    assert np.isclose(line['mbb.10150_den'].e1, ang / 2, atol=1e-11, rtol=0)
    assert np.isclose(line['mbb.10150_dex'].e1, ang / 2, atol=1e-11, rtol=0)

    tw = line.twiss()
    assert np.isclose(twmad.s[-1], tw.s[-1], atol=1e-9, rtol=0)
    assert np.isclose(twmad.summary.q1, tw.qx, rtol=0, atol=1e-7)
    assert np.isclose(twmad.summary.q2, tw.qy, rtol=0, atol=1e-7)
    assert np.isclose(twmad.summary.dq1, tw.dqx, rtol=0, atol=0.2)
    assert np.isclose(twmad.summary.dq2, tw.dqy, rtol=0, atol=0.2)

    line.configure_bend_model(edge='full', core='full')

    tw = line.twiss()

    assert line['mbb.10150_den'].model == 'full'
    assert line['mbb.10150_den'].side == 'entry'
    assert line['mbb.10150_dex'].model == 'full'
    assert line['mbb.10150_dex'].side == 'exit'
    assert line['mbb.10150'].model == 'full'

    assert np.isclose(twmad.s[-1], tw.s[-1], atol=1e-9, rtol=0)
    assert np.isclose(twmad.summary.q1, tw.qx, rtol=0, atol=1e-7)
    assert np.isclose(twmad.summary.q2, tw.qy, rtol=0, atol=1e-7)
    assert np.isclose(twmad.summary.dq1, tw.dqx, rtol=0, atol=0.01)
    assert np.isclose(twmad.summary.dq2, tw.dqy, rtol=0, atol=0.01)

    line.configure_bend_model(core='expanded')

    tw = line.twiss()

    assert line['mbb.10150_den'].model == 'full'
    assert line['mbb.10150_den'].side == 'entry'
    assert line['mbb.10150_dex'].model == 'full'
    assert line['mbb.10150_dex'].side == 'exit'
    assert line['mbb.10150'].model == 'expanded'

    assert np.isclose(twmad.s[-1], tw.s[-1], atol=1e-11, rtol=0)
    assert np.isclose(twmad.summary.q1, tw.qx, rtol=0, atol=1e-7)
    assert np.isclose(twmad.summary.q2, tw.qy, rtol=0, atol=1e-7)
    assert np.isclose(twmad.summary.dq1, tw.dqx, rtol=0, atol=0.2)
    assert np.isclose(twmad.summary.dq2, tw.dqy, rtol=0, atol=0.2)

    line.configure_bend_model(edge='linear')

    assert line['mbb.10150_den'].model == 'linear'
    assert line['mbb.10150_den'].side == 'entry'
    assert line['mbb.10150_dex'].model == 'linear'
    assert line['mbb.10150_dex'].side == 'exit'
    assert line['mbb.10150'].model == 'expanded'

    assert line['mbb.10150_den'].model == 'linear'
    assert line['mbb.10150_den'].side == 'entry'
    assert line['mbb.10150_dex'].model == 'linear'
    assert line['mbb.10150_dex'].side == 'exit'
    assert line['mbb.10150'].model == 'expanded'

    line.configure_bend_model(core='full')
    line.configure_bend_model(edge='full')

    assert line['mbb.10150_den'].model == 'full'
    assert line['mbb.10150_den'].side == 'entry'
    assert line['mbb.10150_dex'].model == 'full'
    assert line['mbb.10150_dex'].side == 'exit'
    assert line['mbb.10150'].model == 'full'

    # Test from_dict/to_dict roundtrip
    dct = line.to_dict()
    line = xt.Line.from_dict(dct)

    assert line['mbb.10150_den'].model == 'full'
    assert line['mbb.10150_den'].side == 'entry'
    assert line['mbb.10150_dex'].model == 'full'
    assert line['mbb.10150_dex'].side == 'exit'
    assert line['mbb.10150'].model == 'full'

    line.discard_tracker()
    slicing_strategies = [
        Strategy(slicing=Teapot(1)),  # Default
        Strategy(slicing=Teapot(2), element_type=xt.Bend),
        Strategy(slicing=Teapot(8), element_type=xt.Quadrupole),
    ]

    line.slice_thick_elements(slicing_strategies)
    line.build_tracker(_context=test_context)

    # Check a bend
    assert line.element_names[112] == 'mbb.10150_entry'
    assert line.element_names[113] == 'mbb.10150_den'
    assert line.element_names[114] == 'drift_mbb.10150..0'
    assert line.element_names[115] == 'mbb.10150..0'
    assert line.element_names[116] == 'drift_mbb.10150..1'
    assert line.element_names[117] == 'mbb.10150..1'
    assert line.element_names[118] == 'drift_mbb.10150..2'
    assert line.element_names[119] == 'mbb.10150_dex'
    assert line.element_names[120] == 'mbb.10150_exit'

    assert isinstance(line['mbb.10150_entry'], xt.Marker)
    assert isinstance(line['mbb.10150_den'], xt.DipoleEdge)
    assert isinstance(line['drift_mbb.10150..0'], xt.Drift)
    assert isinstance(line['mbb.10150..0'], xt.Multipole)
    assert isinstance(line['drift_mbb.10150..1'], xt.Drift)
    assert isinstance(line['mbb.10150..1'], xt.Multipole)
    assert isinstance(line['drift_mbb.10150..2'], xt.Drift)
    assert isinstance(line['mbb.10150_dex'], xt.DipoleEdge)
    assert isinstance(line['mbb.10150_exit'], xt.Marker)

    # Check a quadrupole
    assert line.element_names[158] == 'qf.10210_entry'
    assert line.element_names[159] == 'drift_qf.10210..0'
    assert line.element_names[160] == 'qf.10210..0'
    assert line.element_names[161] == 'drift_qf.10210..1'
    assert line.element_names[162] == 'qf.10210..1'
    assert line.element_names[163] == 'drift_qf.10210..2'
    assert line.element_names[164] == 'qf.10210..2'
    assert line.element_names[165] == 'drift_qf.10210..3'
    assert line.element_names[166] == 'qf.10210..3'
    assert line.element_names[167] == 'drift_qf.10210..4'
    assert line.element_names[168] == 'qf.10210..4'
    assert line.element_names[169] == 'drift_qf.10210..5'
    assert line.element_names[170] == 'qf.10210..5'
    assert line.element_names[171] == 'drift_qf.10210..6'
    assert line.element_names[172] == 'qf.10210..6'
    assert line.element_names[173] == 'drift_qf.10210..7'
    assert line.element_names[174] == 'qf.10210..7'
    assert line.element_names[175] == 'drift_qf.10210..8'
    assert line.element_names[176] == 'qf.10210_exit'

    assert line['mbb.10150_den'].model == 'full'
    assert line['mbb.10150_den'].side == 'entry'
    assert line['mbb.10150_dex'].model == 'full'
    assert line['mbb.10150_dex'].side == 'exit'

    line.configure_bend_model(edge='linear')

    assert line['mbb.10150_den'].model == 'linear'
    assert line['mbb.10150_den'].side == 'entry'
    assert line['mbb.10150_dex'].model == 'linear'
    assert line['mbb.10150_dex'].side == 'exit'

    line.configure_bend_model(edge='full')

    assert line['mbb.10150_den'].model == 'full'
    assert line['mbb.10150_den'].side == 'entry'
    assert line['mbb.10150_dex'].model == 'full'
    assert line['mbb.10150_dex'].side == 'exit'

    tw = line.twiss()

    assert np.isclose(twmad.s[-1], tw.s[-1], atol=1e-9, rtol=0)
    assert np.isclose(twmad.summary.q1, tw.qx, rtol=0, atol=0.5e-3)
    assert np.isclose(twmad.summary.q2, tw.qy, rtol=0, atol=0.5e-3)
    assert np.isclose(twmad.summary.dq1, tw.dqx, rtol=0, atol=0.2)
    assert np.isclose(twmad.summary.dq2, tw.dqy, rtol=0, atol=0.2)
