import pathlib
import json
import numpy as np

import xobjects as xo
import xpart as xp
import xtrack as xt
import xfields as xf

def test_ring_with_spacecharge():

    test_data_folder = pathlib.Path(
            __file__).parent.joinpath('../test_data').absolute()
    fname_line = test_data_folder.joinpath('sps_w_spacecharge/'
                      'line_no_spacecharge_and_particle.json')

    # Test settings (fast but inaccurate)
    bunch_intensity = 1e11/3 # Need short bunch to avoid bucket non-linearity
    sigma_z = 22.5e-2/3
    nemitt_x=2.5e-6
    nemitt_y=2.5e-6
    n_part=int(1e6/10)*10
    nz_grid = 100//20
    z_range = (-3*sigma_z/40, 3*sigma_z/40)

    num_spacecharge_interactions = 540
    tol_spacecharge_position = 1e-2

    ##############
    # Get a line #
    ##############

    with open(fname_line, 'r') as fid:
         input_data = json.load(fid)
    line0_no_sc = xt.Line.from_dict(input_data['line'])
    particle_ref=xp.Particles.from_dict(input_data['particle'])

    lprofile = xf.LongitudinalProfileQGaussian(
        number_of_particles=bunch_intensity,
        sigma_z=sigma_z,
        z0=0.,
        q_parameter=1.)


    ##################
    # Make particles #
    ##################
    tracker_temp=xt.Tracker(# I make a temp tracker to gen. particles only once 
            line=line0_no_sc.filter_elements(exclude_types_starting_with='SpaceCh'))
    import warnings
    warnings.filterwarnings('ignore')
    particle_probe = xp.build_particles(
                tracker=tracker_temp,
                particle_ref=particle_ref,
                weight=0, # pure probe particles
                zeta=0, delta=0,
                x_norm=2, px_norm=0,
                y_norm=2, py_norm=0,
                scale_with_transverse_norm_emitt=(nemitt_x, nemitt_y))

    particles_gaussian = xp.generate_matched_gaussian_bunch(
             num_particles=n_part, total_intensity_particles=bunch_intensity,
             nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
             particle_ref=particle_ref, tracker=tracker_temp)

    particles0 = xp.Particles.merge([particle_probe, particles_gaussian])
    warnings.filterwarnings('default')

    for context in xo.context.get_test_contexts():
        for mode in ['frozen', 'quasi-frozen', 'pic']:
            print('\n\n')
            print(f"Test {context.__class__}")
            print(f'mode = {mode}')
            print('\n\n')

            if isinstance(context, xo.ContextPyopencl) and mode == 'pic':
                # TODO With pyopencl the test gets to the end
                # but then hangs or crashes python
                print('Skipped! Known issue...')
                continue

            # We need only particles at zeta close to the probe
            if mode == 'frozen':
                particles = particles0.filter(particles0.particle_id < 100)
            elif mode == 'quasi-frozen':
                particles = particles0.filter(particles0.particle_id < 1e5)
            elif mode == 'pic':
                particles = particles0.filter((particles0.zeta>z_range[0]*5)
                                              & (particles0.zeta<z_range[1]*5))
            else:
                raise ValueError('Invalid mode!')

            particles = particles.copy(_context=context)

            warnings.filterwarnings('ignore')
            line = line0_no_sc.copy()
            xf.install_spacecharge_frozen(line=line,
                           particle_ref=particle_ref,
                           longitudinal_profile=lprofile,
                           nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                           sigma_z=sigma_z,
                           num_spacecharge_interactions=num_spacecharge_interactions,
                           tol_spacecharge_position=tol_spacecharge_position)
            warnings.filterwarnings('default')

            ##########################
            # Configure space-charge #
            ##########################

            if mode == 'frozen':
                pass # Already configured in line
            elif mode == 'quasi-frozen':
                xf.replace_spacecharge_with_quasi_frozen(
                                                line, _buffer=context.new_buffer(),
                                                update_mean_x_on_track=True,
                                                update_mean_y_on_track=True)
            elif mode == 'pic':
                pic_collection, all_pics = xf.replace_spacecharge_with_PIC(
                    _context=context, line=line,
                    n_sigmas_range_pic_x=5,
                    n_sigmas_range_pic_y=5,
                    nx_grid=256, ny_grid=256, nz_grid=nz_grid,
                    n_lims_x=7, n_lims_y=3,
                    z_range=z_range)
            else:
                raise ValueError(f'Invalid mode: {mode}')

            #################
            # Build Tracker #
            #################
            tracker = xt.Tracker(_context=context,
                                line=line)

            ###############################
            # Tune shift from single turn #
            ###############################

            tracker_no_sc = tracker.filter_elements(exclude_types_starting_with='SpaceCh')
            tw = tracker_no_sc.twiss(
                    particle_ref=particle_ref,  at_elements=[0])

            p_probe_before = particles.filter(
                    particles.particle_id == 0).to_dict()

            print('Start tracking...')
            tracker.track(particles)
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
            assert np.isclose(qx_probe, qx_target, atol=5e-4, rtol=0)
            assert np.isclose(qy_probe, qy_target, atol=5e-4, rtol=0)

