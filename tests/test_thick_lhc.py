import pathlib

from cpymad.madx import Madx

import xdeps as xd
import xobjects as xo
import xpart as xp
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts
from xtrack.slicing import Teapot, Strategy

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

@for_all_test_contexts
def test_madloader_lhc_thick(test_context):
    mad = Madx(stdout=False)

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

    line.configure_bend_model(edge='suppressed')

    tw0 = line.twiss()
    twmad = mad.twiss(table='twiss')
    tmad = xd.Table(twmad)

    print(f'dqx xsuite:      {tw0.dqx}')
    print(f'dqx mad nochrom: {twmad.summary.dq1}')
    print(f'dqy xsuite:      {tw0.dqy}')
    print(f'dqy mad nochrom: {twmad.summary.dq2}')

    xo.assert_allclose(tw0.dqx, twmad.summary.dq1, atol=0.2, rtol=0)
    xo.assert_allclose(tw0.dqy, twmad.summary.dq2, atol=0.2, rtol=0)
    xo.assert_allclose(tw0.qx, twmad.summary.q1, atol=1e-6, rtol=0)
    xo.assert_allclose(tw0.qy, twmad.summary.q2, atol=1e-6, rtol=0)
    xo.assert_allclose(tw0['betx', 'ip1'], tmad['betx', 'ip1:1'], atol=0, rtol=1e-5)
    xo.assert_allclose(tw0['bety', 'ip1'], tmad['bety', 'ip1:1'], atol=0, rtol=1e-5)
    xo.assert_allclose(tw0['betx', 'ip5'], tmad['betx', 'ip5:1'], atol=0, rtol=1e-5)
    xo.assert_allclose(tw0['bety', 'ip5'], tmad['bety', 'ip5:1'], atol=0, rtol=1e-5)


@for_all_test_contexts
def test_slicing_lhc_thick(test_context):
    line = xt.load(test_data_folder /
            'hllhc15_thick/lhc_thick_with_knobs.json')
    line.twiss_default['method'] = '4d'
    line.build_tracker(_context=test_context)

    line_thick = line.copy()
    line_thick.build_tracker(_context=test_context)

    slicing_strategies = [
        Strategy(slicing=Teapot(1)),  # Default catch-all as in MAD-X
        Strategy(slicing=None, element_type=xt.UniformSolenoid),
        Strategy(slicing=Teapot(4), element_type=xt.Bend),
        Strategy(slicing=Teapot(20), element_type=xt.Quadrupole),
        Strategy(slicing=Teapot(2), name=r'^mb\..*'),
        Strategy(slicing=Teapot(5), name=r'^mq\..*'),
        Strategy(slicing=Teapot(2), name=r'^mqt.*'),
        Strategy(slicing=Teapot(60), name=r'^mqx.*'),
    ]

    line.discard_tracker()
    line.slice_thick_elements(slicing_strategies=slicing_strategies)
    line.build_tracker(_context=test_context)

    tw = line.twiss()
    tw_thick = line_thick.twiss()

    beta_beat_x_at_ips = [tw['betx', f'ip{nn}'] / tw_thick['betx', f'ip{nn}'] - 1
                            for nn in range(1, 9)]
    beta_beat_y_at_ips = [tw['bety', f'ip{nn}'] / tw_thick['bety', f'ip{nn}'] - 1
                            for nn in range(1, 9)]

    xo.assert_allclose(beta_beat_x_at_ips, 0, atol=3e-3)
    xo.assert_allclose(beta_beat_y_at_ips, 0, atol=3e-3)

    # Checks on orbit knobs
    xo.assert_allclose(tw_thick['px', 'ip1'], 0, rtol=0, atol=1e-7)
    xo.assert_allclose(tw_thick['py', 'ip1'], 0, rtol=0, atol=1e-7)
    xo.assert_allclose(tw_thick['px', 'ip5'], 0, rtol=0, atol=1e-7)
    xo.assert_allclose(tw_thick['py', 'ip5'], 0, rtol=0, atol=1e-7)
    xo.assert_allclose(tw['px', 'ip1'], 0, rtol=0, atol=1e-7)
    xo.assert_allclose(tw['py', 'ip1'], 0, rtol=0, atol=1e-7)
    xo.assert_allclose(tw['px', 'ip5'], 0, rtol=0, atol=1e-7)
    xo.assert_allclose(tw['py', 'ip5'], 0, rtol=0, atol=1e-7)

    line.vars['on_x1'] = 50
    line.vars['on_x5'] = 60
    line_thick.vars['on_x1'] = 50
    line_thick.vars['on_x5'] = 60

    tw = line.twiss()
    tw_thick = line_thick.twiss()

    xo.assert_allclose(tw_thick['px', 'ip1'], 50e-6, rtol=0, atol=5e-7)
    xo.assert_allclose(tw_thick['py', 'ip1'], 0, rtol=0, atol=5e-7)
    xo.assert_allclose(tw_thick['px', 'ip5'], 0, rtol=0, atol=5e-7)
    xo.assert_allclose(tw_thick['py', 'ip5'], 60e-6, rtol=0, atol=5e-7)
    xo.assert_allclose(tw['px', 'ip1'], 50e-6, rtol=0, atol=5e-7)
    xo.assert_allclose(tw['py', 'ip1'], 0, rtol=0, atol=5e-7)
    xo.assert_allclose(tw['px', 'ip5'], 0, rtol=0, atol=5e-7)
    xo.assert_allclose(tw['py', 'ip5'], 60e-6, rtol=0, atol=5e-7)

    line_thick.match(
        vary=[
            xt.Vary('kqtf.b1', step=1e-8),
            xt.Vary('kqtd.b1', step=1e-8),
            xt.Vary('ksf.b1', step=1e-8),
            xt.Vary('ksd.b1', step=1e-8),
        ],
        targets = [
            xt.Target('qx', 62.27, tol=1e-4),
            xt.Target('qy', 60.29, tol=1e-4),
            xt.Target('dqx', 10.0, tol=0.05),
            xt.Target('dqy', 12.0, tol=0.05)])

    line.match(
        vary=[
            xt.Vary('kqtf.b1', step=1e-8),
            xt.Vary('kqtd.b1', step=1e-8),
            xt.Vary('ksf.b1', step=1e-8),
            xt.Vary('ksd.b1', step=1e-8),
        ],
        targets = [
            xt.Target('qx', 62.27, tol=1e-4),
            xt.Target('qy', 60.29, tol=1e-4),
            xt.Target('dqx', 10.0, tol=0.05),
            xt.Target('dqy', 12.0, tol=0.05)])

    tw = line.twiss()
    tw_thick = line_thick.twiss()

    xo.assert_allclose(tw_thick.qx, 62.27, rtol=0, atol=1e-4)
    xo.assert_allclose(tw_thick.qy, 60.29, rtol=0, atol=1e-4)
    xo.assert_allclose(tw_thick.dqx, 10.0, rtol=0, atol=0.05)
    xo.assert_allclose(tw_thick.dqy, 12.0, rtol=0, atol=0.05)
    xo.assert_allclose(tw.qx, 62.27, rtol=0, atol=1e-4)
    xo.assert_allclose(tw.qy, 60.29, rtol=0, atol=1e-4)
    xo.assert_allclose(tw.dqx, 10.0, rtol=0, atol=0.05)
    xo.assert_allclose(tw.dqy, 12.0, rtol=0, atol=0.05)

    xo.assert_allclose(line.vars['kqtf.b1']._value, line_thick.vars['kqtf.b1']._value,
                        rtol=0.03, atol=0)
    xo.assert_allclose(line.vars['kqtd.b1']._value, line_thick.vars['kqtd.b1']._value,
                        rtol=0.03, atol=0)
    xo.assert_allclose(line.vars['ksf.b1']._value, line_thick.vars['ksf.b1']._value,
                        rtol=0.03, atol=0)
    xo.assert_allclose(line.vars['ksd.b1']._value, line_thick.vars['ksd.b1']._value,
                        rtol=0.03, atol=0)
