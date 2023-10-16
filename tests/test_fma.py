import pathlib

import numpy as np
import pytest

import xtrack as xt
import xpart as xp
import xobjects as xo

from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(__file__).parent.joinpath('../test_data').absolute()

@for_all_test_contexts
@pytest.mark.parametrize('freeze_longitudinal', [True])
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
    fp0 = line.get_fma(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                             **kwargs)

    line.vars['i_oct_b1'] = 500
    fp1 = line.get_fma(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                            n_r=11, n_theta=7, r_range=[0.05, 7],
                            theta_range=[0.01, np.pi/2-0.01],
                            keep_tracking_data=True,
                            **kwargs)


    assert hasattr(fp1, 'tracking_data')

    assert hasattr(fp1, 'theta_grid')
    assert hasattr(fp1, 'r_grid')
    
    assert hasattr(fp1, 'qx1')
    assert hasattr(fp1, 'qx2')
    assert hasattr(fp1, 'qy1')
    assert hasattr(fp1, 'qy2')
    assert hasattr(fp1, 'diffusion')

    assert len(fp1.r_grid) == 11
    assert len(fp1.theta_grid) == 7

    assert np.isclose(fp1.r_grid[0], 0.05, rtol=0, atol=1e-10)
    assert np.isclose(fp1.r_grid[-1], 7, rtol=0, atol=1e-10)

    assert np.isclose(fp1.theta_grid[0], 0.01, rtol=0, atol=1e-10)
    assert np.isclose(fp1.theta_grid[-1], np.pi/2 - 0.01, rtol=0, atol=1e-10)

    #i_theta = 0, i_r = 0
    assert np.isclose(fp1.x_norm_2d[0, 0], 0.05, rtol=0, atol=1e-3)
    assert np.isclose(fp1.y_norm_2d[0, 0], 0, rtol=0, atol=1e-3)
    assert np.isclose(fp1.qx1[0, 0], 0.31, rtol=0, atol=5e-5)
    assert np.isclose(fp1.qy1[0, 0], 0.32, rtol=0, atol=5e-5)
    assert np.isclose(fp1.qx2[0, 0], 0.31, rtol=0, atol=5e-5)
    assert np.isclose(fp1.qy2[0, 0], 0.32, rtol=0, atol=5e-5)


