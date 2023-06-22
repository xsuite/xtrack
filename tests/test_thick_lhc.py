import pathlib

from cpymad.madx import Madx
import xtrack as xt
import xpart as xp
import xdeps as xd

from xobjects.test_helpers import for_all_test_contexts

import numpy as np

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

@for_all_test_contexts
def test_madloader_thick(test_context):
    mad = Madx()

    mad.input(f"""
    call,file="{str(test_data_folder)}/hllhc15_thick/lhc.seq";
    call,file="{str(test_data_folder)}/hllhc15_thick/hllhc_sequence.madx";
    seqedit,sequence=lhcb1;flatten;cycle,start=IP7;flatten;endedit;
    seqedit,sequence=lhcb2;flatten;cycle,start=IP7;flatten;endedit;
    beam, sequence=lhcb1, particle=proton, pc=7000;
    call,file="{str(test_data_folder)}/hllhc15_thick/opt_round_150_1500.madx";
    """)

    mad.use(sequence="lhcb1")
    seq = mad.sequence.lhcb1
    mad.twiss()

    line = xt.Line.from_madx_sequence(mad.sequence.lhcb1,
                allow_thick=True, deferred_expressions=True)
    line.particle_ref = xp.Particles(mass0=seq.beam.mass*1e9, gamma0=seq.beam.gamma)
    line.twiss_default['method'] = '4d'
    line.twiss_default['matrix_stability_tol'] = 100


    line.build_tracker(_context=test_context)

    # To have the very same model
    for ee in seq.elements:
        if hasattr(ee, 'KILL_ENT_FRINGE'):
            ee.KILL_ENT_FRINGE = True
        if hasattr(ee, 'KILL_EXI_FRINGE'):
            ee.KILL_EXI_FRINGE = True

    for ee in line.elements:
        if isinstance(ee, xt.DipoleEdge):
            ee.r21 = 0
            ee.r43 = 0


    tw0 = line.twiss()
    twmad = mad.twiss(table='twiss')
    tmad = xd.Table(twmad)

    print(f'dqx xsuite:      {tw0.dqx}')
    print(f'dqx mad nochrom: {twmad.summary.dq1}')
    print(f'dqy xsuite:      {tw0.dqy}')
    print(f'dqy mad nochrom: {twmad.summary.dq2}')

    assert np.isclose(tw0.dqx, twmad.summary.dq1, atol=0.2, rtol=0)
    assert np.isclose(tw0.dqy, twmad.summary.dq2, atol=0.2, rtol=0)
    assert np.isclose(tw0.qx, twmad.summary.q1, atol=1e-6, rtol=0)
    assert np.isclose(tw0.qy, twmad.summary.q2, atol=1e-6, rtol=0)
    assert np.isclose(tw0['betx', 'ip1'], tmad['betx', 'ip1:1'], atol=0, rtol=1e-5)
    assert np.isclose(tw0['bety', 'ip1'], tmad['bety', 'ip1:1'], atol=0, rtol=1e-5)
    assert np.isclose(tw0['betx', 'ip5'], tmad['betx', 'ip5:1'], atol=0, rtol=1e-5)
    assert np.isclose(tw0['bety', 'ip5'], tmad['bety', 'ip5:1'], atol=0, rtol=1e-5)