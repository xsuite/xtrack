import pytest
import xtrack as xt
import xobjects as xo
import pathlib
import numpy as np
from xtrack._temp import lhc_match as lm

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()

def test_madng_twiss():
    rdts = ["f4000", "f3100", "f2020", "f1120"]

    line = xt.load(test_data_folder /
                            'hllhc15_thick/lhc_thick_with_knobs.json')

    line['on_disp'] = 0
    line['test_dk1'] = 0
    line['mb.b32l8.b1'].knl[1] = 'test_dk1'

    tw = line.madng_twiss(normal_form=False)

    opt = line.match(
        solve=False,
        vary=[
            xt.Vary('on_x1', step=1e-3),
        ],
        targets=(
            tw.target('px_ng', 50e-6, at='ip1'),
            tw.target('py_ng', 0, at='ip1'),
        ),
    )
    opt.step(3)
    tw_after = line.madng_twiss(normal_form=False)
    xo.assert_allclose(tw_after['px_ng', 'ip1'], 50e-6, rtol=5e-3, atol=0)

    line['on_x1'] = 1.
    xo.assert_allclose(line.madng_twiss()['px_ng', 'ip1'], 1e-6, rtol=5e-3, atol=0)

    line['on_x1'] = -2.
    xo.assert_allclose(line.madng_twiss()['px_ng', 'ip1'], -2e-6, rtol=5e-3, atol=0)

    # We just check that rdts are there
    tw_rdt = line.madng_twiss(rdts=rdts)
    assert np.abs(tw_rdt.f4000).max() > 0
    assert np.abs(tw_rdt.f3100).max() > 0
    assert np.abs(tw_rdt.f2020).max() > 0
    assert np.abs(tw_rdt.f1120).max() > 0

def test_madng_interface_with_multipole_errors_and_misalignments():
    line = xt.load(test_data_folder /
                            'hllhc15_thick/lhc_thick_with_knobs.json')

    tt = line.get_table()
    tt_quads = tt.rows[tt.element_type=='Quadrupole']

    # Introduce misalignments on all quadrupoles
    tt = line.get_table()
    tt_quad = tt.rows[r'mq\..*']
    rgen = np.random.RandomState(1) # fix seed for random number generator
                                    # (to have reproducible results)
    shift_x = rgen.randn(len(tt_quad)) * 0.01e-3 # 0.01 mm rms shift on all quads
    shift_y = rgen.randn(len(tt_quad)) * 0.01e-3 # 0.01 mm rms shift on all quads
    rot_s = rgen.randn(len(tt_quad)) * 1e-3 # 1 mrad rms rotation on all quads
    k2l = rgen.rand(len(tt_quad)) * 1e-3

    line['on_error'] = 1.
    for nn_quad, sx, sy, rr, kkk in zip(tt_quad.name, shift_x, shift_y, rot_s, k2l):
        line[nn_quad].shift_x = sx * line.ref['on_error']
        line[nn_quad].shift_y = sy * line.ref['on_error']
        line[nn_quad].rot_s_rad = rr * line.ref['on_error']
        line[nn_quad].knl[2] = kkk * line.ref['on_error']
    tw = line.madng_twiss(coupling_edw_teng=True, compute_chromatic_properties=True)

    xo.assert_allclose(tw.x, tw.x_ng, atol=5e-4*tw.x.std(), rtol=0)
    xo.assert_allclose(tw.y, tw.y_ng, atol=5e-4*tw.y.std(), rtol=0)
    xo.assert_allclose(tw.betx2, tw.beta12_ng, atol=0, rtol=2e-3)
    xo.assert_allclose(tw.bety1, tw.beta21_ng, atol=0, rtol=2e-3)
    xo.assert_allclose(tw.wx_chrom, tw.wx_ng, atol=5e-3*tw.wx_chrom.max(), rtol=0)
    xo.assert_allclose(tw.wy_chrom, tw.wy_ng, atol=5e-3*tw.wy_chrom.max(), rtol=0)
    xo.assert_allclose(tw.ax_chrom, tw.ax_ng, atol=5e-3*tw.wx_chrom.max(), rtol=0)
    xo.assert_allclose(tw.ay_chrom, tw.ay_ng, atol=5e-3*tw.wy_chrom.max(), rtol=0)
    xo.assert_allclose(tw.bx_chrom, tw.bx_ng, atol=5e-3*tw.wx_chrom.max(), rtol=0)
    xo.assert_allclose(tw.by_chrom, tw.by_ng, atol=5e-3*tw.wy_chrom.max(), rtol=0)

    line['on_error'] = 0
    tw = line.madng_twiss(coupling_edw_teng=True, compute_chromatic_properties=True)
    xo.assert_allclose(tw.x, 0, atol=1e-10, rtol=0)
    xo.assert_allclose(tw.y, 0, atol=1e-10, rtol=0)
    xo.assert_allclose(tw.betx2, 0, atol=1e-10, rtol=0)
    xo.assert_allclose(tw.bety1, 0, atol=1e-10, rtol=0)
    xo.assert_allclose(tw.x, tw.x_ng, atol=1e-9, rtol=0)
    xo.assert_allclose(tw.y, tw.y_ng, atol=1e-9, rtol=0)
    xo.assert_allclose(tw.betx2, tw.beta12_ng, atol=1e-10, rtol=0)
    xo.assert_allclose(tw.bety1, tw.beta21_ng, atol=1e-19, rtol=0)
    xo.assert_allclose(tw.wx_chrom, tw.wx_ng, atol=5e-3*tw.wx_chrom.max(), rtol=0)
    xo.assert_allclose(tw.wy_chrom, tw.wy_ng, atol=5e-3*tw.wy_chrom.max(), rtol=0)
    xo.assert_allclose(tw.ax_chrom, tw.ax_ng, atol=5e-3*tw.wx_chrom.max(), rtol=0)
    xo.assert_allclose(tw.ay_chrom, tw.ay_ng, atol=5e-3*tw.wy_chrom.max(), rtol=0)
    xo.assert_allclose(tw.bx_chrom, tw.bx_ng, atol=5e-3*tw.wx_chrom.max(), rtol=0)
    xo.assert_allclose(tw.by_chrom, tw.by_ng, atol=5e-3*tw.wy_chrom.max(), rtol=0)

def test_madng_survey():
    line = xt.load(test_data_folder /
                            'hllhc15_thick/lhc_thick_with_knobs.json')
    survey = line.madng_survey()

    # Check that columns exist:
    SURVEY_COLS = ['X', 'Y', 'Z', 'theta', 'phi', 'psi', 'name', 's', 'length', 'slc',
                   'turn', 'tdir', 'eidx', 'ename', 'element_type', 'angle', 'tilt']
    assert isinstance(survey, xt.survey.SurveyTable)
    assert len(survey.cols) == len(SURVEY_COLS)
    for col in SURVEY_COLS:
        assert col in survey.cols, f"Column '{col}' not found in survey"

    # Compare MAD-NG survey with Xsuite survey
    xsurvey = line.survey()

    assert len(survey) == len(xsurvey), "Length of MAD-NG survey and Xsuite survey do not match"

    xo.assert_allclose(survey.X, xsurvey.X, atol=1e-5, rtol=0)
    xo.assert_allclose(survey.Y, xsurvey.Y, atol=1e-5, rtol=0)
    xo.assert_allclose(survey.Z, xsurvey.Z, atol=1e-5, rtol=0)
    xo.assert_allclose(survey.theta, xsurvey.theta, atol=1e-5, rtol=0)
    xo.assert_allclose(survey.phi, xsurvey.phi, atol=1e-5, rtol=0)
    xo.assert_allclose(survey.psi, xsurvey.psi, atol=1e-5, rtol=0)
    xo.assert_allclose(survey.s, xsurvey.s, atol=1e-5, rtol=0)
    xo.assert_allclose(survey.angle, xsurvey.angle, atol=1e-5, rtol=0)
    xo.assert_allclose(survey.tilt, xsurvey.rot_s_rad, atol=1e-5, rtol=0)
    # Length doesn't work because of multipoles.
    #xo.assert_allclose(survey.length, xsurvey.length, atol=1e-5, rtol=0)

def test_madng_conversion_drift_slice():

    env = xt.Environment()
    env.particle_ref = xt.Particles(p0c=1e9)

    line = env.new_line(length=10, components=[
        env.new('drift_1', xt.Drift, length=3.5, anchor='start', at=0),
        env.new('q1', xt.Quadrupole, length=1, k1=0.3, at=4),
        env.new('q2', xt.Quadrupole, length=1, k1=-0.3, at=6)
    ])

    line.insert('m', xt.Marker(), at=2)

    tt = line.get_table()
    # is:
    # Table: 8 rows, 11 cols
    # name                   s element_type isthick isreplica parent_name ...
    # drift_1..0             0 DriftSlice      True     False drift_1
    # m                      2 Marker         False     False None
    # drift_1..1             2 DriftSlice      True     False drift_1
    # q1                   3.5 Quadrupole      True     False None
    # drift_2              4.5 Drift           True     False None
    # q2                   5.5 Quadrupole      True     False None
    # drift_3              6.5 Drift           True     False None
    # _end_point            10                False     False None

    assert np.all(tt.name == [
        'drift_1..0', 'm', 'drift_1..1', 'q1', '||drift_1', 'q2',
       '||drift_2', '_end_point'])
    xo.assert_allclose(tt.s, [0, 2, 2, 3.5, 4.5, 5.5, 6.5, 10], atol=1e-10)
    assert np.all(tt.element_type == [
        'DriftSlice', 'Marker', 'DriftSlice', 'Quadrupole', 'Drift', 'Quadrupole', 'Drift', ''])

    tw = line.twiss4d()
    tw_ng = line.madng_twiss(normal_form=False)

    xo.assert_allclose(tw_ng.beta11_ng, tw.betx, rtol=1e-8)
    xo.assert_allclose(tw_ng.beta22_ng, tw.bety, rtol=1e-8)

def test_madng_interface_with_slicing():
    line = xt.load(test_data_folder /
                            'hllhc15_thick/lhc_thick_with_knobs.json')

    # Cut line at every 1 up to s = 1000
    line.cut_at_s(np.arange(1000))

    tw_xs = line.twiss4d()
    tw = line.madng_twiss(coupling_edw_teng=True, compute_chromatic_properties=True)

    assert len(tw) == len(tw_xs)

    xo.assert_allclose(tw.x, 0, atol=1e-10, rtol=0)
    xo.assert_allclose(tw.y, 0, atol=1e-10, rtol=0)
    xo.assert_allclose(tw.betx2, 0, atol=1e-10, rtol=0)
    xo.assert_allclose(tw.bety1, 0, atol=1e-10, rtol=0)
    xo.assert_allclose(tw.x, tw.x_ng, atol=1e-8, rtol=0)
    xo.assert_allclose(tw.y, tw.y_ng, atol=1e-10, rtol=0)
    xo.assert_allclose(tw.betx2, tw.beta12_ng, atol=1e-10, rtol=0)
    xo.assert_allclose(tw.bety1, tw.beta21_ng, atol=1e-19, rtol=0)
    xo.assert_allclose(tw.wx_chrom, tw.wx_ng, atol=5e-3*tw.wx_chrom.max(), rtol=0)
    xo.assert_allclose(tw.wy_chrom, tw.wy_ng, atol=5e-3*tw.wy_chrom.max(), rtol=0)
    xo.assert_allclose(tw.ax_chrom, tw.ax_ng, atol=5e-3*tw.wx_chrom.max(), rtol=0)
    xo.assert_allclose(tw.ay_chrom, tw.ay_ng, atol=5e-3*tw.wy_chrom.max(), rtol=0)
    xo.assert_allclose(tw.bx_chrom, tw.bx_ng, atol=5e-3*tw.wx_chrom.max(), rtol=0)
    xo.assert_allclose(tw.by_chrom, tw.by_ng, atol=5e-3*tw.wy_chrom.max(), rtol=0)

def test_madng_twiss_with_initial_conditions():
    line = xt.load(test_data_folder /
                            'hllhc15_thick/lhc_thick_with_knobs.json')
    #pytest.set_trace()
    tw_xs = line.twiss(betx=120, bety=150, alfx=5, alfy=5, dx=1e-4)
    tw = line.madng_twiss(beta11=120, beta22=150, alfa11=5, alfa22=5, dx=1e-4)

    assert len(tw) == len(tw_xs)
    assert len(tw.betx) == len(tw.beta11_ng)

    xo.assert_allclose(tw.betx, tw.beta11_ng, rtol=1e-6, atol=1e-6)
    xo.assert_allclose(tw.bety, tw.beta22_ng, rtol=1e-6, atol=1e-6)
    xo.assert_allclose(tw.alfx, tw.alfa11_ng, rtol=1e-6, atol=1e-6)
    xo.assert_allclose(tw.alfy, tw.alfa22_ng, rtol=1e-6, atol=1e-6)
    xo.assert_allclose(tw.dx, tw.dx_ng, rtol=1e-8, atol=1e-6)
    xo.assert_allclose(tw.dy, tw.dy_ng, rtol=1e-8, atol=1e-6)
    xo.assert_allclose(tw.dpx, tw.dpx_ng, rtol=1e-8, atol=1e-6)
    xo.assert_allclose(tw.dpy, tw.dpy_ng, rtol=1e-8, atol=1e-6)
    xo.assert_allclose(tw.x, tw.x_ng, rtol=1e-8, atol=1e-6)
    xo.assert_allclose(tw.y, tw.y_ng, rtol=1e-8, atol=1e-6)

    tw2_xs = line.twiss(start='s.ds.l8.b1', end='ip1', betx=100, bety=34, dx=1e-5)
    tw2_xsng = line.madng_twiss(start='s.ds.l8.b1', end='ip1', beta11=100, beta22=34, dx=1e-5, xsuite_tw=False)

    assert len(tw2_xs.betx) == len(tw2_xsng.beta11_ng)
    xo.assert_allclose(tw2_xs.betx, tw2_xsng.beta11_ng, rtol=1e-8, atol=1e-6)
    xo.assert_allclose(tw2_xs.bety, tw2_xsng.beta22_ng, rtol=1e-8, atol=1e-6)
    xo.assert_allclose(tw2_xs.alfx, tw2_xsng.alfa11_ng, rtol=1e-8, atol=1e-6)
    xo.assert_allclose(tw2_xs.alfy, tw2_xsng.alfa22_ng, rtol=1e-8, atol=1e-6)
    xo.assert_allclose(tw2_xs.dx, tw2_xsng.dx_ng, rtol=1e-8, atol=1e-6)
    xo.assert_allclose(tw2_xs.dy, tw2_xsng.dy_ng, rtol=1e-8, atol=1e-6)
    xo.assert_allclose(tw2_xs.dpx, tw2_xsng.dpx_ng, rtol=1e-8, atol=1e-6)
    xo.assert_allclose(tw2_xs.dpy, tw2_xsng.dpy_ng, rtol=1e-8, atol=1e-6)
    xo.assert_allclose(tw2_xs.x, tw2_xsng.x_ng, rtol=1e-8, atol=1e-8)
    xo.assert_allclose(tw2_xs.y, tw2_xsng.y_ng, rtol=1e-8, atol=1e-8)

    tw3_xs = line.twiss(start='ip8', end='ip2', betx=1.5, bety=1.5)
    tw3_xsng = line.madng_twiss(start='ip8', end='ip2', beta11=1.5, beta22=1.5, xsuite_tw=True)

    assert len(tw3_xs.betx) == len(tw3_xsng.beta11_ng)
    xo.assert_allclose(tw3_xs.betx, tw3_xsng.beta11_ng, rtol=1e-8, atol=1e-6)
    xo.assert_allclose(tw3_xs.bety, tw3_xsng.beta22_ng, rtol=1e-8, atol=1e-6)
    xo.assert_allclose(tw3_xs.alfx, tw3_xsng.alfa11_ng, rtol=1e-8, atol=1e-6)
    xo.assert_allclose(tw3_xs.alfy, tw3_xsng.alfa22_ng, rtol=1e-8, atol=1e-6)
    xo.assert_allclose(tw3_xs.dx, tw3_xsng.dx_ng, rtol=1e-8, atol=1e-6)
    xo.assert_allclose(tw3_xs.dy, tw3_xsng.dy_ng, rtol=1e-8, atol=1e-6)
    xo.assert_allclose(tw3_xs.dpx, tw3_xsng.dpx_ng, rtol=1e-8, atol=1e-6)
    xo.assert_allclose(tw3_xs.dpy, tw3_xsng.dpy_ng, rtol=1e-8, atol=1e-6)
    xo.assert_allclose(tw3_xs.x, tw3_xsng.x_ng, rtol=1e-8, atol=1e-8)
    xo.assert_allclose(tw3_xs.y, tw3_xsng.y_ng, rtol=1e-8, atol=1e-8)

    xo.assert_allclose(tw3_xsng.betx, tw3_xsng.beta11_ng, rtol=1e-8, atol=1e-6)
    xo.assert_allclose(tw3_xsng.bety, tw3_xsng.beta22_ng, rtol=1e-8, atol=1e-6)
    xo.assert_allclose(tw3_xsng.alfx, tw3_xsng.alfa11_ng, rtol=1e-8, atol=1e-6)
    xo.assert_allclose(tw3_xsng.alfy, tw3_xsng.alfa22_ng, rtol=1e-8, atol=1e-6)

    tw4_xs = line.twiss(start='ip3', end='ip4', betx=121.5668, bety=218.58374, alfx=2.295, alfy=-2.6429, dx=-0.51)
    tw4_xsng = line.madng_twiss(start='ip3', end='ip4', beta11=121.5668, beta22=218.58374, alfa11=2.295,
                                alfa22=-2.6429, dx=-0.51, xsuite_tw=False)

    assert len(tw4_xs.betx) == len(tw4_xsng.beta11_ng)
    xo.assert_allclose(tw4_xs.betx, tw4_xsng.beta11_ng, rtol=1e-6, atol=1e-5)
    xo.assert_allclose(tw4_xs.bety, tw4_xsng.beta22_ng, rtol=1e-6, atol=1e-5)
    xo.assert_allclose(tw4_xs.alfx, tw4_xsng.alfa11_ng, rtol=1e-6, atol=1e-5)
    xo.assert_allclose(tw4_xs.alfy, tw4_xsng.alfa22_ng, rtol=1e-6, atol=1e-5)
    xo.assert_allclose(tw4_xs.dx, tw4_xsng.dx_ng, rtol=1e-7, atol=1e-8)
    xo.assert_allclose(tw4_xs.dy, tw4_xsng.dy_ng, rtol=1e-7, atol=1e-8)
    xo.assert_allclose(tw4_xs.x, tw4_xsng.x_ng, rtol=1e-8, atol=1e-10)
    xo.assert_allclose(tw4_xs.y, tw4_xsng.y_ng, rtol=1e-8, atol=1e-10)
    xo.assert_allclose(tw4_xs.px, tw4_xsng.px_ng, rtol=1e-8, atol=1e-10)
    xo.assert_allclose(tw4_xs.py, tw4_xsng.py_ng, rtol=1e-8, atol=1e-10)
    xo.assert_allclose(tw4_xs.mux, tw4_xsng.mu1_ng, rtol=1e-8, atol=1e-5)
    xo.assert_allclose(tw4_xs.muy, tw4_xsng.mu2_ng, rtol=1e-8, atol=1e-5)

def test_madng_slices():
    line = xt.load(test_data_folder /
                            'hllhc15_thick/lhc_thick_with_knobs.json')
    tw = line.twiss4d()

    twng = line.madng_twiss(compute_chromatic_properties=True)

    line.cut_at_s(np.linspace(0, line.get_length(), 5000))
    tw_sliced = line.twiss4d()
    twng_sliced = line.madng_twiss(compute_chromatic_properties=True)
    tt_sliced = line.get_table()

    assert np.all(np.array(sorted(list(set(tt_sliced.element_type)))) ==
        ['',
        'Cavity',
        'Drift',
        'DriftSlice',
        'LimitRectEllipse',
        'Marker',
        'Multipole',
        'Octupole',
        'Quadrupole',
        'RBend',
        'Sextupole',
        'ThickSliceBend',
        'ThickSliceCavity',
        'ThickSliceMultipole',
        'ThickSliceOctupole',
        'ThickSliceQuadrupole',
        'ThickSliceRBend',
        'ThickSliceSextupole',
        'ThickSliceUniformSolenoid',
        'ThinSliceBendEntry',
        'ThinSliceBendExit',
        'ThinSliceOctupoleEntry',
        'ThinSliceOctupoleExit',
        'ThinSliceQuadrupoleEntry',
        'ThinSliceQuadrupoleExit',
        'ThinSliceRBendEntry',
        'ThinSliceRBendExit',
        'ThinSliceSextupoleEntry',
        'ThinSliceSextupoleExit',
        'ThinSliceUniformSolenoidEntry',
        'ThinSliceUniformSolenoidExit',
        'UniformSolenoid'])

    twng_ip = twng.rows['ip.*']
    twng_ip_sliced = twng_sliced.rows['ip.*']
    xo.assert_allclose(twng_ip.s, twng_ip_sliced.s, rtol=1e-8)
    xo.assert_allclose(twng_ip.beta11_ng, twng_ip_sliced.beta11_ng, rtol=1e-3)
    xo.assert_allclose(twng_ip.beta22_ng, twng_ip_sliced.beta22_ng, rtol=1e-3)
    xo.assert_allclose(twng_ip.wx_ng, twng_ip_sliced.wx_ng, rtol=1e-3)
    xo.assert_allclose(twng_ip.wy_ng, twng_ip_sliced.wy_ng, rtol=1e-3)
    xo.assert_allclose(twng_ip.dx_ng, twng_ip_sliced.dx_ng, atol=1e-6)
    xo.assert_allclose(twng_ip.dy_ng, twng_ip_sliced.dy_ng, atol=1e-6)


def test_madng_interface_amplitude_detuning_and_second_order_chrom():
    line = xt.load(test_data_folder /
                            'hllhc15_thick/lhc_thick_with_knobs.json')

    twng = line.madng_twiss()
    det = line.get_amplitude_detuning_coefficients(num_turns=512)

    xo.assert_allclose(twng.dqxdjx_nf_ng, det['det_xx'], rtol=7e-2)
    xo.assert_allclose(twng.dqydjy_nf_ng, det['det_yy'], rtol=7e-2)
    xo.assert_allclose(twng.dqxdjy_nf_ng, det['det_xy'], rtol=7e-2)
    xo.assert_allclose(twng.dqydjx_nf_ng, det['det_yx'], rtol=7e-2)

    tw = line.twiss4d()
    xo.assert_allclose(tw.ddqx, twng.d2q1_nf_ng, rtol=1e-2)
    xo.assert_allclose(tw.ddqy, twng.d2q2_nf_ng, rtol=1e-2)


def test_madng_match_optics():
    collider = xt.Environment.from_json(test_data_folder /
                    'hllhc15_thick/hllhc15_collider_thick.json')
    collider.vars.load_madx(test_data_folder /
                    'hllhc15_thick/opt_round_150_1500.madx')

    line = collider.lhcb1
    tw0 = line.madng_twiss()

    lm.set_var_limits_and_steps(collider)

    # Match with Xsuite Targets
    opt = line.match(
    solve=False,
    default_tol={None: 1e-8, 'betx': 1e-6, 'bety': 1e-6, 'alfx': 1e-6, 'alfy': 1e-6},
    start='s.ds.l8.b1', end='ip1',
    init=tw0, init_at=xt.START,
    vary=[
        # Only IR8 quadrupoles including DS
        xt.VaryList(['kq6.l8b1', 'kq7.l8b1', 'kq8.l8b1', 'kq9.l8b1', 'kq10.l8b1',
            'kqtl11.l8b1', 'kqt12.l8b1', 'kqt13.l8b1',
            'kq4.l8b1', 'kq5.l8b1', 'kq4.r8b1', 'kq5.r8b1',
            'kq6.r8b1', 'kq7.r8b1', 'kq8.r8b1', 'kq9.r8b1',
            'kq10.r8b1', 'kqtl11.r8b1', 'kqt12.r8b1', 'kqt13.r8b1'])],
    targets=[
        xt.TargetSet(at='ip8', tars=('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'), value=tw0, weight=1),
        xt.TargetSet(at='ip1', betx=0.15, bety=0.1, alfx=0, alfy=0, dx=0, dpx=0, weight=1),
        xt.TargetRelPhaseAdvance('mux', value = tw0['mux', 'ip1.l1'] - tw0['mux', 's.ds.l8.b1'], start='s.ds.l8.b1', end='ip1.l1', weight=1),
        xt.TargetRelPhaseAdvance('muy', value = tw0['muy', 'ip1.l1'] - tw0['muy', 's.ds.l8.b1'], start='s.ds.l8.b1', end='ip1.l1', weight=1),
    ],
    use_tpsa=True)

    opt.step(30)

    assert opt._err.call_counter < 20
    assert len(opt.log()) < 10

    tw = line.twiss(init=tw0, start='s.ds.l8.b1', end='ip1')

    xo.assert_allclose(tw['betx', 'ip1'], 0.15, atol=1e-6, rtol=0)
    xo.assert_allclose(tw['bety', 'ip1'], 0.1, atol=1e-6, rtol=0)
    xo.assert_allclose(tw['alfx', 'ip1'], 0., atol=1e-6, rtol=0)
    xo.assert_allclose(tw['alfy', 'ip1'], 0., atol=1e-6, rtol=0)
    xo.assert_allclose(tw['dx', 'ip1'], 0., atol=1e-6, rtol=0)
    xo.assert_allclose(tw['dy', 'ip1'], 0., atol=1e-6, rtol=0)

    xo.assert_allclose(tw['betx', 'ip8'], tw0['betx', 'ip8'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['bety', 'ip8'], tw0['bety', 'ip8'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['alfx', 'ip8'], tw0['alfx', 'ip8'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['alfy', 'ip8'], tw0['alfy', 'ip8'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['dx', 'ip8'], tw0['dx', 'ip8'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['dy', 'ip8'], tw0['dy', 'ip8'], atol=1e-6, rtol=0)

    xo.assert_allclose(tw['mux', 'ip1.l1'] - tw['mux', 's.ds.l8.b1'], tw0['mux', 'ip1.l1'] - tw0['mux', 's.ds.l8.b1'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['muy', 'ip1.l1'] - tw['muy', 's.ds.l8.b1'], tw0['muy', 'ip1.l1'] - tw0['muy', 's.ds.l8.b1'], atol=1e-6, rtol=0)

    opt.reload(0)
    opt.actions[0].cleanup()

    # Match with MAD-NG and Xsuite Targets mixed
    opt = line.match(
    solve=False,
    default_tol={None: 1e-8, 'betx': 1e-6, 'bety': 1e-6, 'alfx': 1e-6, 'alfy': 1e-6},
    start='s.ds.l8.b1', end='ip1',
    init=tw0, init_at=xt.START,
    vary=[
        # Only IR8 quadrupoles including DS
        xt.VaryList(['kq6.l8b1', 'kq7.l8b1', 'kq8.l8b1', 'kq9.l8b1', 'kq10.l8b1',
            'kqtl11.l8b1', 'kqt12.l8b1', 'kqt13.l8b1',
            'kq4.l8b1', 'kq5.l8b1', 'kq4.r8b1', 'kq5.r8b1',
            'kq6.r8b1', 'kq7.r8b1', 'kq8.r8b1', 'kq9.r8b1',
            'kq10.r8b1', 'kqtl11.r8b1', 'kqt12.r8b1', 'kqt13.r8b1'])],
    targets=[
        xt.TargetSet(at='ip8', tars=('beta11_ng', 'bety', 'alfa11_ng', 'alfy', 'dx_ng', 'dpx'), value=tw0, weight=1),
        xt.TargetSet(at='ip1', betx=0.15, beta22_ng=0.1, alfx=0, alfa22_ng=0, dx=0, dpx_ng=0, weight=1),
        xt.TargetRelPhaseAdvance('mux', value = tw0['mux', 'ip1.l1'] - tw0['mux', 's.ds.l8.b1'], start='s.ds.l8.b1', end='ip1.l1', weight=1),
        xt.TargetRelPhaseAdvance('mu2_ng', value = tw0['mu2_ng', 'ip1.l1'] - tw0['mu2_ng', 's.ds.l8.b1'], start='s.ds.l8.b1', end='ip1.l1', weight=1),
    ],
    use_tpsa=True)

    opt.step(30)

    assert opt._err.call_counter < 20
    assert len(opt.log()) < 10

    tw = line.twiss(init=tw0, start='s.ds.l8.b1', end='ip1')

    xo.assert_allclose(tw['betx', 'ip1'], 0.15, atol=1e-6, rtol=0)
    xo.assert_allclose(tw['bety', 'ip1'], 0.1, atol=1e-6, rtol=0)
    xo.assert_allclose(tw['alfx', 'ip1'], 0., atol=1e-6, rtol=0)
    xo.assert_allclose(tw['alfy', 'ip1'], 0., atol=1e-6, rtol=0)
    xo.assert_allclose(tw['dx', 'ip1'], 0., atol=1e-6, rtol=0)
    xo.assert_allclose(tw['dy', 'ip1'], 0., atol=1e-6, rtol=0)

    xo.assert_allclose(tw['betx', 'ip8'], tw0['betx', 'ip8'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['bety', 'ip8'], tw0['bety', 'ip8'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['alfx', 'ip8'], tw0['alfx', 'ip8'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['alfy', 'ip8'], tw0['alfy', 'ip8'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['dx', 'ip8'], tw0['dx', 'ip8'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['dy', 'ip8'], tw0['dy', 'ip8'], atol=1e-6, rtol=0)

    xo.assert_allclose(tw['mux', 'ip1.l1'] - tw['mux', 's.ds.l8.b1'], tw0['mux', 'ip1.l1'] - tw0['mux', 's.ds.l8.b1'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['muy', 'ip1.l1'] - tw['muy', 's.ds.l8.b1'], tw0['muy', 'ip1.l1'] - tw0['muy', 's.ds.l8.b1'], atol=1e-6, rtol=0)

    opt.reload(0)
    opt.actions[0].cleanup()

    # Match on full line without initial conditions
    opt = line.match(
    solve=False,
    default_tol={None: 1e-8, 'betx': 1e-6, 'bety': 1e-6, 'alfx': 1e-6, 'alfy': 1e-6},
    vary=[
        # Only IR8 quadrupoles including DS
        xt.VaryList(['kq6.l8b1', 'kq7.l8b1', 'kq8.l8b1', 'kq9.l8b1', 'kq10.l8b1',
            'kqtl11.l8b1', 'kqt12.l8b1', 'kqt13.l8b1',
            'kq4.l8b1', 'kq5.l8b1', 'kq4.r8b1', 'kq5.r8b1',
            'kq6.r8b1', 'kq7.r8b1', 'kq8.r8b1', 'kq9.r8b1',
            'kq10.r8b1', 'kqtl11.r8b1', 'kqt12.r8b1', 'kqt13.r8b1'])],
    targets=[
        xt.TargetSet(at='ip8', tars=('beta11_ng', 'beta22_ng', 'alfa11_ng', 'alfa22_ng', 'dx_ng', 'dpx_ng'), value=tw0, weight=1),
        xt.TargetSet(at='ip1.l1', beta11_ng=0.15, beta22_ng=0.1, alfa11_ng=0, alfa22_ng=0, dx_ng=0, dpx_ng=0, weight=1),
        xt.TargetRelPhaseAdvance('mu1_ng', value = tw0['mu1_ng', 'ip1.l1'] - tw0['mu1_ng', 's.ds.l8.b1'], start='s.ds.l8.b1', end='ip1.l1', weight=1),
        xt.TargetRelPhaseAdvance('mu2_ng', value = tw0['mu2_ng', 'ip1.l1'] - tw0['mu2_ng', 's.ds.l8.b1'], start='s.ds.l8.b1', end='ip1.l1', weight=1),
    ],
    use_tpsa=True)

    opt.step(30)

    assert opt._err.call_counter < 20
    assert len(opt.log()) < 10

    tw = line.twiss(init=tw0)

    xo.assert_allclose(tw['betx', 'ip1.l1'], 0.15, atol=1e-6, rtol=0)
    xo.assert_allclose(tw['bety', 'ip1.l1'], 0.1, atol=1e-6, rtol=0)
    xo.assert_allclose(tw['alfx', 'ip1.l1'], 0., atol=1e-6, rtol=0)
    xo.assert_allclose(tw['alfy', 'ip1.l1'], 0., atol=1e-6, rtol=0)
    xo.assert_allclose(tw['dx', 'ip1.l1'], 0., atol=1e-6, rtol=0)
    xo.assert_allclose(tw['dy', 'ip1.l1'], 0., atol=1e-6, rtol=0)

    xo.assert_allclose(tw['betx', 'ip8'], tw0['betx', 'ip8'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['bety', 'ip8'], tw0['bety', 'ip8'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['alfx', 'ip8'], tw0['alfx', 'ip8'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['alfy', 'ip8'], tw0['alfy', 'ip8'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['dx', 'ip8'], tw0['dx', 'ip8'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['dy', 'ip8'], tw0['dy', 'ip8'], atol=1e-6, rtol=0)

    xo.assert_allclose(tw['mux', 'ip1.l1'] - tw['mux', 's.ds.l8.b1'], tw0['mux', 'ip1.l1'] - tw0['mux', 's.ds.l8.b1'], atol=1e-6, rtol=0)
    xo.assert_allclose(tw['muy', 'ip1.l1'] - tw['muy', 's.ds.l8.b1'], tw0['muy', 'ip1.l1'] - tw0['muy', 's.ds.l8.b1'], atol=1e-6, rtol=0)


def test_madng_orbit_bump():
    env = xt.Environment()
    env.vars.default_to_zero = True
    line = env.new_line(length=10, components=[
        env.new('corr1', xt.Multipole, isthick=True,
                knl=['kick_h_1'], ksl=['kick_v_1'], length=0.1, at=1),
        env.new('corr2', xt.Multipole, isthick=True,
                knl=['kick_h_2'], ksl=['kick_v_2'], length=0.1, at=2),
        env.new('corr3', xt.Multipole, isthick=True,
                knl=['kick_h_3'], ksl=['kick_v_3'], length=0.1, at=8),
        env.new('corr4', xt.Multipole, isthick=True,
                knl=['kick_h_4'], ksl=['kick_v_4'], length=0.1, at=9),
        env.new('mid', xt.Marker, at=5),
        env.new('end', xt.Marker, at=10)
        ])
    line.set_particle_ref('proton', p0c=26e9)

    opt = line.match(
        solve=False,
        betx=1, bety=1,
        vary=xt.VaryList(['kick_h_1', 'kick_v_1',
                        'kick_h_2', 'kick_v_2',
                        'kick_h_3', 'kick_v_3',
                        'kick_h_4', 'kick_v_4']),
        targets=[
            xt.TargetSet(x=1e-3, y=-2e-3, px=0, py=0, at='mid'),
            xt.TargetSet(x=0, y=0, px=0, py=0, at='end'),
        ],
        use_tpsa=True
    )

    jac_ng = opt._err.get_jacobian(opt._err._get_x())

    jac_opt = np.array([[-40, 0, -30, 0, 0, 0, 0, 0],
                        [-100, 0, -100, 0, 0, 0, 0, 0],
                        [0, 40, 0, 30, 0, 0, 0, 0],
                        [0, 100, 0, 100, 0, 0, 0, 0],
                        [-90, 0, -80, 0, -20, 0, -10, 0],
                        [-100, 0, -100, 0, -100, 0, -100, 0],
                        [0, 90, 0, 80, 0, 20, 0, 10],
                        [0, 100, 0, 100, 0, 100, 0, 100]])

    xo.assert_allclose(jac_ng, jac_opt, rtol=1e-12, atol=1e-12)

    opt.solve()

    assert opt._err.call_counter < 7
