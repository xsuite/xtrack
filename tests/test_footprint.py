import pathlib

import numpy as np
import pytest

import xobjects as xo
import xpart as xp
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(__file__).parent.joinpath('../test_data').absolute()

@for_all_test_contexts
@pytest.mark.parametrize('freeze_longitudinal', [True, False])
def test_footprint(test_context, freeze_longitudinal):

    if isinstance(test_context, xo.ContextPyopencl):
        pytest.skip('Pyopencl not yet supported for footprint')
        return

    nemitt_x = 1e-6
    nemitt_y = 1e-6

    line = xt.Line.from_json(test_data_folder /
                'hllhc15_noerrors_nobb/line_w_knobs_and_particle.json')
    line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, p0c=7e12)
    line.build_tracker(_context=test_context)

    if freeze_longitudinal:
        kwargs = {'freeze_longitudinal': True}
        for ee in line.elements:
            if isinstance(ee, xt.Cavity):
                ee.voltage = 0
    else:
        kwargs = {}

    line.vars['i_oct_b1'] = 0
    fp0 = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                             **kwargs)

    line.vars['i_oct_b1'] = 500
    fp1 = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                            n_r=11, n_theta=7, r_range=[0.05, 7],
                            theta_range=[0.01, np.pi/2-0.01],
                            keep_tracking_data=True,
                            **kwargs)

    assert hasattr(fp1, 'tracking_data')

    assert hasattr(fp1, 'theta_grid')
    assert hasattr(fp1, 'r_grid')

    assert len(fp1.r_grid) == 11
    assert len(fp1.theta_grid) == 7

    xo.assert_allclose(fp1.r_grid[0], 0.05, rtol=0, atol=1e-10)
    xo.assert_allclose(fp1.r_grid[-1], 7, rtol=0, atol=1e-10)

    xo.assert_allclose(fp1.theta_grid[0], 0.01, rtol=0, atol=1e-10)
    xo.assert_allclose(fp1.theta_grid[-1], np.pi/2 - 0.01, rtol=0, atol=1e-10)

    #i_theta = 0, i_r = 0
    xo.assert_allclose(fp1.x_norm_2d[0, 0], 0.05, rtol=0, atol=1e-3)
    xo.assert_allclose(fp1.y_norm_2d[0, 0], 0, rtol=0, atol=1e-3)
    xo.assert_allclose(fp1.qx[0, 0], 0.31, rtol=0, atol=5e-5)
    xo.assert_allclose(fp1.qy[0, 0], 0.32, rtol=0, atol=5e-5)

    #i_theta = 0, i_r = 10
    xo.assert_allclose(fp1.x_norm_2d[0, -1], 7, rtol=0, atol=1e-3)
    xo.assert_allclose(fp1.y_norm_2d[0, -1], 0.07, rtol=0, atol=1e-3)
    xo.assert_allclose(fp1.qx[0, -1], 0.3129, rtol=0, atol=2e-4)
    xo.assert_allclose(fp1.qy[0, -1], 0.3185, rtol=0, atol=2e-4)

    #i_theta = 6, i_r = 0
    xo.assert_allclose(fp1.x_norm_2d[-1, 0], 0, rtol=0, atol=1e-3)
    xo.assert_allclose(fp1.y_norm_2d[-1, 0], 0.05, rtol=0, atol=1e-3)
    xo.assert_allclose(fp1.qx[0, 0], 0.31, rtol=0, atol=5e-5)
    xo.assert_allclose(fp1.qy[0, 0], 0.32, rtol=0, atol=5e-5)

    #i_theta = 6, i_r = 10
    xo.assert_allclose(fp1.x_norm_2d[-1, -1], 0.07, rtol=0, atol=1e-3)
    xo.assert_allclose(fp1.y_norm_2d[-1, -1], 7, rtol=0, atol=1e-3)
    xo.assert_allclose(fp1.qx[-1, -1], 0.3085, rtol=0, atol=2e-4)
    xo.assert_allclose(fp1.qy[-1, -1], 0.3229, rtol=0, atol=2e-4)

    xo.assert_allclose(np.max(fp1.qx[:]) - np.min(fp1.qx[:]), 4.4e-3, rtol=0, atol=1e-4)
    xo.assert_allclose(np.max(fp1.qy[:]) - np.min(fp1.qy[:]), 4.4e-3, rtol=0, atol=1e-4)

    line.vars['i_oct_b1'] = 0
    fp0 = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                             **kwargs)

    assert hasattr(fp0, 'theta_grid')
    assert hasattr(fp0, 'r_grid')

    xo.assert_allclose(fp0.r_grid[0], 0.1, rtol=0, atol=1e-10)
    xo.assert_allclose(fp0.r_grid[-1], 6, rtol=0, atol=1e-10)

    xo.assert_allclose(fp0.theta_grid[0], 0.05, rtol=0, atol=1e-10)
    xo.assert_allclose(fp0.theta_grid[-1], np.pi/2-0.05, rtol=0, atol=1e-10)

    assert len(fp0.r_grid) == 10
    assert len(fp0.theta_grid) == 10

    #i_theta = 0, i_r = 0
    xo.assert_allclose(fp0.x_norm_2d[0, 0], 0.1, rtol=0, atol=1e-3)
    xo.assert_allclose(fp0.y_norm_2d[0, 0], 0.005, rtol=0, atol=1e-3)
    xo.assert_allclose(fp0.qx[0, 0], 0.31, rtol=0, atol=5e-5)
    xo.assert_allclose(fp0.qy[0, 0], 0.32, rtol=0, atol=5e-5)

    xo.assert_allclose(np.max(fp0.qx[:]) - np.min(fp0.qx[:]), 0.0003, rtol=0, atol=2e-5)
    xo.assert_allclose(np.max(fp0.qy[:]) - np.min(fp0.qy[:]), 0.0003, rtol=0, atol=2e-5)

    line.vars['i_oct_b1'] = 500
    fp1_jgrid = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                                x_norm_range=[0.01, 6], y_norm_range=[0.01, 6],
                                n_x_norm=9, n_y_norm=8,
                                mode='uniform_action_grid',
                                **kwargs)

    assert hasattr(fp1_jgrid,  'Jx_grid')
    assert hasattr(fp1_jgrid,  'Jy_grid')

    assert len(fp1_jgrid.Jx_grid) == 9
    assert len(fp1_jgrid.Jy_grid) == 8

    xo.assert_allclose(np.diff(fp1_jgrid.Jx_grid), np.diff(fp1_jgrid.Jx_grid)[0],
                    rtol=0, atol=1e-10)
    xo.assert_allclose(np.diff(fp1_jgrid.Jy_grid), np.diff(fp1_jgrid.Jy_grid)[0],
                        rtol=0, atol=1e-10)

    xo.assert_allclose(fp1_jgrid.x_norm_2d[0, 0], 0.01, rtol=0, atol=1e-10)
    xo.assert_allclose(fp1_jgrid.y_norm_2d[0, 0], 0.01, rtol=0, atol=1e-10)
    xo.assert_allclose(fp1_jgrid.qx[0, 0], 0.31, rtol=0, atol=5e-5)
    xo.assert_allclose(fp1_jgrid.qy[0, 0], 0.32, rtol=0, atol=5e-5)

    xo.assert_allclose(fp1_jgrid.x_norm_2d[0, -1], 6, rtol=0, atol=1e-10)
    xo.assert_allclose(fp1_jgrid.y_norm_2d[0, -1], 0.01, rtol=0, atol=1e-10)
    xo.assert_allclose(fp1_jgrid.qx[0, -1], 0.3121, rtol=0, atol=1e-4)
    xo.assert_allclose(fp1_jgrid.qy[0, -1], 0.3189, rtol=0, atol=1e-4)

    xo.assert_allclose(fp1_jgrid.x_norm_2d[-1, 0], 0.01, rtol=0, atol=1e-10)
    xo.assert_allclose(fp1_jgrid.y_norm_2d[-1, 0], 6, rtol=0, atol=1e-10)
    xo.assert_allclose(fp1_jgrid.qx[-1, 0], 0.3089, rtol=0, atol=1e-4)
    xo.assert_allclose(fp1_jgrid.qy[-1, 0], 0.3221, rtol=0, atol=1e-4)

    xo.assert_allclose(fp1_jgrid.x_norm_2d[-1, -1], 6, rtol=0, atol=1e-10)
    xo.assert_allclose(fp1_jgrid.y_norm_2d[-1, -1], 6, rtol=0, atol=1e-10)
    xo.assert_allclose(fp1_jgrid.qx[-1, -1], 0.3111, rtol=0, atol=1e-4)
    xo.assert_allclose(fp1_jgrid.qy[-1, -1], 0.3210, rtol=0, atol=1e-4)

    xo.assert_allclose(np.max(fp1_jgrid.qx[:]) - np.min(fp1_jgrid.qx[:]), 0.0032,
                        rtol=0, atol=1e-4)
    xo.assert_allclose(np.max(fp1_jgrid.qy[:]) - np.min(fp1_jgrid.qy[:]), 0.0032,
                        rtol=0, atol=1e-4)

    x_norm_range = [1, 6]
    y_norm_range = [1, 6]
    line.vars['i_oct_b1'] = 50000 # Particles are lost for such high octupole current
    fp50k = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                                x_norm_range=x_norm_range, y_norm_range=y_norm_range,
                                n_x_norm=9, n_y_norm=8,
                                mode='uniform_action_grid',
                                linear_rescale_on_knobs=xt.LinearRescale(
                                knob_name='i_oct_b1', v0=500, dv=100),
                                **kwargs)


    line.vars['i_oct_b1'] = 60000 # Particles are lost for such high octupole current
    fp60k = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                                x_norm_range=x_norm_range, y_norm_range=y_norm_range,
                                n_x_norm=9, n_y_norm=8,
                                mode='uniform_action_grid',
                                linear_rescale_on_knobs=xt.LinearRescale(
                                knob_name='i_oct_b1', v0=500, dv=100),
                                **kwargs)

    line.vars['i_oct_b1'] = 500
    fp500 = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                                x_norm_range=x_norm_range, y_norm_range=y_norm_range,
                                n_x_norm=9, n_y_norm=8,
                                mode='uniform_action_grid',
                                **kwargs)

    line.vars['i_oct_b1'] = 600
    fp600 = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                                x_norm_range=x_norm_range, y_norm_range=y_norm_range,
                                n_x_norm=9, n_y_norm=8,
                                mode='uniform_action_grid',
                                **kwargs)

    xo.assert_allclose((fp60k.qx - fp50k.qx)/(fp600.qx-fp500.qx), 100, rtol=0, atol=1e-2)
    xo.assert_allclose((fp60k.qy - fp50k.qy)/(fp600.qy-fp500.qy), 100, rtol=0, atol=1e-2)

@for_all_test_contexts
def test_footprint_delta0(test_context):

    if isinstance(test_context, xo.ContextPyopencl):
        pytest.skip('Pyopencl not yet supported for footprint')
        return

    nemitt_x = 1e-6
    nemitt_y = 1e-6

    line = xt.Line.from_json(test_data_folder /
                        'hllhc15_noerrors_nobb/line_w_knobs_and_particle.json')
    line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, p0c=7e12)
    line.build_tracker(_context=test_context)

    # Some octupoles and chromaticity to see the footprint moving
    line.vars['i_oct_b1'] = 500
    line.match(
        targets=[xt.TargetList(['dqx', 'dqy'], value=10, tol=0.01)],
        vary=[xt.VaryList(['ksf.b1', 'ksd.b1'], step=1e-3)])

    # Compute and plot footprint on momentum
    fp1 = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                            freeze_longitudinal=True)

    # Compute and plot footprint off momentum
    fp2 = line.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                            freeze_longitudinal=True, delta0=1e-4)

    xo.assert_allclose(line.record_last_track.delta[:], 1e-4, atol=1e-12, rtol=0)

    # It is 12, 9 instead of 10, 10 because of non-linear chromaticity
    xo.assert_allclose((np.max(fp2.qx) - np.max(fp1.qx))/1e-4, 12, atol=0.5, rtol=0)
    xo.assert_allclose((np.max(fp2.qy) - np.max(fp1.qy))/1e-4, 9, atol=0.5, rtol=0)

