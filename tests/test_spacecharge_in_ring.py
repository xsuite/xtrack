# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
import pathlib

import numpy as np
import pytest

import xfields as xf
import xobjects as xo
import xpart as xp
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts, fix_random_seed


@for_all_test_contexts
@pytest.mark.parametrize(
    'mode',
    ['frozen', 'quasi-frozen', 'pic', 'pic_average_transverse'],
)
@fix_random_seed(746483)
def test_ring_with_spacecharge(test_context, mode):

    test_data_folder = pathlib.Path(
            __file__).parent.joinpath('../test_data').absolute()
    fname_line = test_data_folder.joinpath('sps_w_spacecharge/'
                      'line_no_spacecharge_and_particle.json')

    # Test settings (fast but inaccurate)
    bunch_intensity = 1e11/3  # Need short bunch to avoid bucket non-linearity
    sigma_z = 22.5e-2/3
    nemitt_x = 2.5e-6
    nemitt_y = 2.5e-6
    n_part = int(1e6/10)*10
    nz_grid = 100//20
    z_range = (-3*sigma_z/40, 3*sigma_z/40)

    num_spacecharge_interactions = 540

    ##############
    # Get a line #
    ##############

    with open(fname_line, 'r') as fid:
         input_data = json.load(fid)
    line0_no_sc = xt.Line.from_dict(input_data['line'])
    line0_no_sc.particle_ref=xp.Particles.from_dict(input_data['particle'])

    lprofile = xf.LongitudinalProfileQGaussian(
        number_of_particles=bunch_intensity,
        sigma_z=sigma_z,
        z0=0.,
        q_parameter=1.)

    ##################
    # Make particles #
    ##################
    line_temp = line0_no_sc.filter_elements(
        exclude_types_starting_with='SpaceCh')
    line_temp.build_tracker()
    import warnings
    warnings.filterwarnings('ignore')
    particle_probe = line_temp.build_particles(
                weight=0,  # pure probe particles
                zeta=0, delta=0,
                x_norm=2, px_norm=0,
                y_norm=2, py_norm=0,
                nemitt_x=nemitt_x, nemitt_y=nemitt_y)

    particles_gaussian = xp.generate_matched_gaussian_bunch(line=line_temp,
             num_particles= 2 * n_part, # will mark half of them as lost
             total_intensity_particles = 2 * bunch_intensity, # will mark half of them as lost
             nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z)

    particles_gaussian.state[1::2] = -222 # Mark half of them as lost

    particles0 = xp.Particles.merge([particle_probe, particles_gaussian])

    if isinstance(particles0._context, xo.ContextCpu):
        particles0.reorganize()

    warnings.filterwarnings('default')

    if (isinstance(test_context, xo.ContextPyopencl)
            and mode.startswith('pic')):
        # TODO With pyopencl the test gets to the end
        # but then hangs or crashes python
        pytest.skip('This test in broken on OpenCL. Known issue...')
        return

    # We need only particles at zeta close to the probe
    if mode == 'frozen':
        particles = particles0.filter(particles0.particle_id < 100)
    elif mode == 'quasi-frozen':
        particles = particles0.filter(particles0.particle_id < 1e5)
        particles.weight[:] = bunch_intensity / 1e5 * 2 # need to have the right peak density
    elif mode == 'pic' or mode == 'pic_average_transverse':
        particles = particles0.filter((particles0.zeta>z_range[0]*5)
                                      & (particles0.zeta<z_range[1]*5))
    else:
        raise ValueError('Invalid mode!')

    particles.move(_context=test_context)

    warnings.filterwarnings('ignore')
    line = line0_no_sc.copy()
    xf.install_spacecharge_frozen(
            line=line,
            longitudinal_profile=lprofile,
            nemitt_x=nemitt_x, nemitt_y=nemitt_y,
            sigma_z=sigma_z,
            num_spacecharge_interactions=num_spacecharge_interactions)
    warnings.filterwarnings('default')

    # Move to the right context
    line.build_tracker(_context=test_context)
    assert line._context is test_context
    buffer_for_check = line._buffer

    ##########################
    # Configure space-charge #
    ##########################

    if mode == 'frozen':
        pass # Already configured in line
    elif mode == 'quasi-frozen':
        xf.replace_spacecharge_with_quasi_frozen(
                                        line,
                                        update_mean_x_on_track=True,
                                        update_mean_y_on_track=True)
    elif mode == 'pic' or mode == 'pic_average_transverse':
        if mode == 'pic':
            solver = 'FFTSolver2p5D'
        elif mode == 'pic_average_transverse':
            solver = 'FFTSolver2p5DAveraged'
        pic_collection, all_pics = xf.replace_spacecharge_with_PIC(
            line=line,
            n_sigmas_range_pic_x=5,
            n_sigmas_range_pic_y=5,
            nx_grid=256, ny_grid=256, nz_grid=nz_grid,
            n_lims_x=7, n_lims_y=3,
            z_range=z_range,
            solver=solver)
    else:
        raise ValueError(f'Invalid mode: {mode}')

    # rebuild the tracker after editing
    line.build_tracker(_context=test_context)
    assert line._buffer is buffer_for_check

    if mode == 'pic_average_transverse' or mode == 'pic':
        assert isinstance(line[0], xf.SpaceCharge3D)
    else:
        assert isinstance(line[0], xf.SpaceChargeBiGaussian)

    if mode != 'frozen':
        assert line.iscollective
    else:
        assert not line.iscollective

    ###############################
    # Tune shift from single turn #
    ###############################

    line_no_sc = line.filter_elements(exclude_types_starting_with='SpaceCh')
    tw = line_no_sc.twiss(at_elements=[0])

    p_probe_before = particles.filter(
            particles.particle_id == 0).to_dict()

    assert line._buffer is buffer_for_check

    print('Start tracking...')
    line.track(particles)
    print('Done tracking.')

    p_probe_after = particles.filter(
            particles.particle_id == 0).to_dict()

    betx = tw['betx'][0]
    alfx = tw['alfx'][0]
    print(f'{alfx=} {betx=}')
    phasex_0 = np.angle(p_probe_before['x'] / np.sqrt(betx) -
                        1j*(p_probe_before['x'] * alfx / np.sqrt(betx) +
                        p_probe_before['px'] * np.sqrt(betx)))[0]
    phasex_1 = np.angle(p_probe_after['x'] / np.sqrt(betx) -
                       1j*(p_probe_after['x'] * alfx / np.sqrt(betx) +
                           p_probe_after['px'] * np.sqrt(betx)))[0]
    bety = tw['bety'][0]
    alfy = tw['alfy'][0]
    print(f'{alfy=} {bety=}')
    phasey_0 = np.angle(p_probe_before['y'] / np.sqrt(bety) -
                        1j*(p_probe_before['y'] * alfy / np.sqrt(bety) +
                        p_probe_before['py'] * np.sqrt(bety)))[0]
    phasey_1 = np.angle(p_probe_after['y'] / np.sqrt(bety) -
                        1j*(p_probe_after['y'] * alfy / np.sqrt(bety) +
                        p_probe_after['py'] * np.sqrt(bety)))[0]
    qx_probe = (phasex_1 - phasex_0)/(2*np.pi)
    qy_probe = (phasey_1 - phasey_0)/(2*np.pi)

    qx_target = 0.12424673159186882
    qy_target = 0.21993469870358598

    print(f'ex={(qx_probe - qx_target)/1e-3:.6f}e-3 '
          f'ey={(qy_probe - qy_target)/1e-3:.6f}e-3')

    xo.assert_allclose(qx_probe, qx_target, atol=5e-4, rtol=0)
    xo.assert_allclose(qy_probe, qy_target, atol=5e-4, rtol=0)

    if mode == 'pic_average_transverse':
        sc_test = all_pics[50]
        assert sc_test.fieldmap.solver.__class__.__name__ == 'FFTSolver2p5DAveraged'
        ctx2np = sc_test._context.nparray_from_context_array
        for dtest in [sc_test.fieldmap.dphi_dx, sc_test.fieldmap.dphi_dy]:
            # Check that the normalized electric field is the same
            dtest = ctx2np(dtest)
            xo.assert_allclose(
                dtest[:, :, 3] / np.max(dtest[:, :, 3]),
                dtest[:, :, 4] / np.max(dtest[:, :, 4]),
                atol=1e-10, rtol=1e-5)
    elif mode == 'pic':
        sc_test = all_pics[50]
        assert sc_test.fieldmap.solver.__class__.__name__ == 'FFTSolver2p5D'
