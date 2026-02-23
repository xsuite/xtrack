import pytest
import xtrack as xt
import xobjects as xo
import pathlib
import numpy as np

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
    tw = line.madng_twiss()

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
    tw = line.madng_twiss()
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
    tw = line.madng_twiss()

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

    twng = line.madng_twiss()

    line.cut_at_s(np.linspace(0, line.get_length(), 5000))
    tw_sliced = line.twiss4d()
    twng_sliced = line.madng_twiss()
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
