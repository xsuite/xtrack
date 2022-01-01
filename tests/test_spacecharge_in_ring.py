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
                      'line_with_spacecharge_and_particle.json')

    # Test settings (fast but inaccurate)
    bunch_intensity = 1e11/3 # Need short bunch to avoid bucket non-linearity
    sigma_z = 22.5e-2/3
    neps_x=2.5e-6
    neps_y=2.5e-6
    n_part=int(1e6/10)*10
    rf_voltage=3e6
    nz_grid = 100//20
    z_range = (-3*sigma_z/40, 3*sigma_z/40)

    ##############
    # Get a line #
    ##############

    with open(fname_line, 'r') as fid:
         input_data = json.load(fid)
    line0 = xt.Line.from_dict(input_data['line'])
    particle_ref=xp.Particles.from_dict(input_data['particle'])

    first_sc = line0.elements[1]
    sigma_x = first_sc.sigma_x
    sigma_y = first_sc.sigma_y

    ##################
    # Make particles #
    ##################
    import warnings
    warnings.filterwarnings('ignore')
    particles0 = xp.generate_matched_gaussian_bunch(
             num_particles=n_part, total_intensity_particles=bunch_intensity,
             nemitt_x=neps_x, nemitt_y=neps_y, sigma_z=sigma_z,
             particle_ref=particle_ref,
             tracker=xt.Tracker(# I make a temp tracker to gen. particles only once 
                 line=line0.filter_elements(exclude_types_starting_with='SpaceCh')))
    warnings.filterwarnings('default')

    # Add a probe at 1 sigma
    particles0.x[0] = 2*sigma_x
    particles0.y[0] = 2*sigma_y
    particles0.px[0] = 0.
    particles0.py[0] = 0.
    particles0.zeta[0] = 0.
    particles0.delta[0] = 0.

    ####################
    # Choose a context #
    ####################


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
                particles = particles0.filter(
                     (particles0.zeta>z_range[0]*5) & (particles0.zeta<z_range[1]*5))
            else:
                raise ValueError('Invalid mode!')

            particles = particles.copy(_context=context)

            line = xt.Line.from_dict(input_data['line'])

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

            qx_target = 0.13622046302275012
            qy_target = 0.23004568206474874
            print(f'ex={(qx_probe - qx_target)/1e-3:.6f}e-3 '
                  f'ey={(qy_probe - qy_target)/1e-3:.6f}e-3')
            assert np.isclose(qx_probe, qx_target, atol=5e-4, rtol=0)
            assert np.isclose(qy_probe, qy_target, atol=5e-4, rtol=0)

