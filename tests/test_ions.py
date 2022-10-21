import pathlib
import numpy as np

import xtrack as xt
import xpart as xp
import xobjects as xo
from cpymad.madx import Madx

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()


def test_ions():

    mad = Madx()
    mad.call(str(test_data_folder.joinpath(
        'sps_ions/SPS_2021_Pb_ions_thin_test.seq')))
    mad.use('sps')
    mad.twiss()

    twmad_4d = mad.table.twiss.dframe()
    summad_4d = mad.table.summ.dframe()

    V_RF = 1.7e6 # V (control room definition, energy gain per charge)

    # I switch on one cavity
    charge = mad.sequence.sps.beam.charge
    mad.sequence.sps.elements['actcse.31632'].volt = V_RF/1e6 * charge
    mad.sequence.sps.elements['actcse.31632'].lag = 0
    mad.sequence.sps.elements['actcse.31632'].freq = 200.

    twmad_6d = mad.table.twiss.dframe()
    summad_6d = mad.table.summ.dframe()

    mad.emit()
    qs_mad = mad.table.emitsumm.qs[0]

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")
        # Make xsuite line and tracker
        line = xt.Line.from_madx_sequence(mad.sequence.sps, deferred_expressions=True)
        line.particle_ref = xp.Particles(mass0=mad.sequence.sps.beam.mass*1e9,
                                        q0=mad.sequence.sps.beam.charge,
                                        gamma0=mad.sequence.sps.beam.gamma)

        assert np.isclose(line['actcse.31632'].voltage, V_RF, atol=1e-10)

        tracker = line.build_tracker()

        tw = tracker.twiss()

        assert np.isclose(tw.qs, qs_mad, atol=1e-6)
        assert np.isclose(tw.qx, summad_4d.q1, atol=1e-5)
        assert np.isclose(tw.qy, summad_4d.q2, atol=1e-5)
        assert np.isclose(tw.dqx, summad_6d.dq1, atol=0.5)
        assert np.isclose(tw.dqy, summad_6d.dq2, atol=0.5)