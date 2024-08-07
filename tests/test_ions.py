import pathlib

from cpymad.madx import Madx

import xobjects as xo
import xpart as xp
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()


@for_all_test_contexts
def test_ions(test_context):

    mad = Madx(stdout=False)
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

    print(f"Test {test_context.__class__}")
    # Make xsuite line and line
    line = xt.Line.from_madx_sequence(mad.sequence.sps, deferred_expressions=True)
    line.particle_ref = xp.Particles(mass0=mad.sequence.sps.beam.mass*1e9,
                                    q0=mad.sequence.sps.beam.charge,
                                    gamma0=mad.sequence.sps.beam.gamma)

    xo.assert_allclose(line['actcse.31632'].voltage, V_RF, atol=1e-10)

    line.build_tracker()

    tw = line.twiss()

    xo.assert_allclose(tw.qs, qs_mad, rtol=0, atol=1e-6)
    xo.assert_allclose(tw.qx, float(summad_4d.q1.iloc[0]), rtol=0, atol=1e-5)
    xo.assert_allclose(tw.qy, float(summad_4d.q2.iloc[0]), rtol=0, atol=1e-5)
    xo.assert_allclose(tw.dqx - float(summad_6d.dq1.iloc[0]), 0, rtol=0, atol=0.5)
    xo.assert_allclose(tw.dqy - float(summad_6d.dq2.iloc[0]), 0, rtol=0, atol=0.5)
