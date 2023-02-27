import pathlib

import numpy as np
import xpart as xp
from cpymad.madx import Madx
from xobjects.test_helpers import for_all_test_contexts

import xtrack as xt

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()


@for_all_test_contexts
def test_ions(test_context):

    mad = Madx()
    mad.call(str(test_data_folder.joinpath(
        'ps_ion/PS_2022_Pb_ions_thin_matched.seq')))
    mad.use('ps')
    mad.twiss()

    twmad_4d = mad.table.twiss.dframe()
    summad_4d = mad.table.summ.dframe()
    
    # RF values as taken from LSA for Pb ions of PS
    # In the ps_ss.seq, the 10 MHz cavities are PR_ACC10 - there are 12 of them in the straight sections
    harmonic_nb = 16
    V_RF = 38.0958e3  # V (control room definition, energy gain per charge). 
    nn = 'pa.c10.11'  # for now use the first of the 10 MHz RF cavities 
    
    # We switch on one cavity 
    charge = mad.sequence.ps.beam.charge
    mad.sequence.ps.elements[nn].lag = 0 # 0 as we are below transition
    mad.sequence.ps.elements[nn].volt = V_RF/1e6*charge # different convention between madx and xsuite
    mad.sequence.ps.elements[nn].freq = mad.sequence['ps'].beam.freq0*harmonic_nb
    
    twmad_6d = mad.table.twiss.dframe()
    summad_6d = mad.table.summ.dframe()
    
    mad.emit()
    qs_mad = mad.table.emitsumm.qs[0]

    print(f"Test {test_context.__class__}")
    # Make xsuite line and tracker
    line = xt.Line.from_madx_sequence(mad.sequence.ps)
    line.particle_ref = xp.Particles(mass0=mad.sequence.ps.beam.mass*1e9,
                                     q0=mad.sequence.ps.beam.charge,
                                     gamma0=mad.sequence.ps.beam.gamma)
    
    # Activate the RF cavity in Xsuite
    line[nn].lag = 0  
    line[nn].voltage =  V_RF # In Xsuite for ions, do not multiply by charge as in MADX
    line[nn].frequency = mad.sequence['ps'].beam.freq0*1e6*harmonic_nb
    
    # Build tracker and perform Twiss
    tracker = line.build_tracker()
    tw = tracker.twiss()

    # Relativistic beta factor needed when comparing MAD-X and Xsuite
    beta0 = mad.sequence['ps'].beam.beta 

    assert np.isclose(tw.qs, qs_mad, atol=1e-6)
    assert np.isclose(tw.qx, summad_4d.q1, atol=1e-5)
    assert np.isclose(tw.qy, summad_4d.q2, atol=1e-5)
    assert np.isclose(tw.dqx, summad_6d.dq1*beta0, atol=0.5)
    assert np.isclose(tw.dqy, summad_6d.dq2*beta0, atol=0.5)