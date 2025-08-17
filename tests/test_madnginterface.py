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
    xo.assert_allclose(tw.x, tw.x_ng, atol=1e-10, rtol=0)
    xo.assert_allclose(tw.y, tw.y_ng, atol=1e-10, rtol=0)
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