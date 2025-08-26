import pathlib

import numpy as np
from scipy.constants import c as clight

import xtrack as xt
from xobjects.test_helpers import fix_random_seed

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()


@fix_random_seed(8837465)
def test_coasting():
    delta0 = 1e-2

    line = xt.load(test_data_folder /
                             'psb_injection/line_and_particle.json')

    # RF off!
    tt = line.get_table()
    ttcav = tt.rows[tt.element_type == 'Cavity']
    for nn in ttcav.name:
        line.element_refs[nn].voltage=0

    line.configure_bend_model(core='bend-kick-bend', edge='full')
    line.twiss_default['method'] = '4d'

    tw = line.twiss()
    twom = line.twiss(delta0=delta0)
    line.discard_tracker()

    # Install dummy collective elements
    s_sync = np.linspace(0, tw.circumference, 10)
    line.cut_at_s(s_sync)
    for ii, ss in enumerate(s_sync):
        nn = f'sync_here_{ii}'
        line.insert_element(element=xt.Marker(), name=nn, at_s=ss)
        line[nn].iscollective = True

    import xtrack.synctime as st
    st.install_sync_time_at_collective_elements(line)

    import xobjects as xo
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))

    beta1 = tw.beta0 / 0.9

    circumference = tw.circumference
    zeta_min0 = -circumference/2*tw.beta0/beta1
    zeta_max0 = circumference/2*tw.beta0/beta1

    num_particles = 50000
    p = line.build_particles(
        delta=delta0 + 0 * np.random.uniform(-1, 1, num_particles),
        x_norm=0, y_norm=0
    )

    # Need to take beta of actual particles to convert the distribution along the
    # circumference to a distribution in time
    p.zeta = (np.random.uniform(0, circumference, num_particles) / p.rvv
            + (zeta_max0 - circumference) / p.rvv)

    st.prepare_particles_for_sync_time(p, line)

    p.y[(p.zeta > 1) & (p.zeta < 2)] = 1e-3  # kick
    p.weight[(p.zeta > 5) & (p.zeta < 10)] *= 1.3

    p0 = p.copy()

    def particles(_, p):
        return p.copy()

    def intensity(line, particles):
        return (np.sum(particles.weight[particles.state > 0])
                    / ((zeta_max0 - zeta_min0)/tw.beta0/clight))

    def z_range(line, particles):
        mask_alive = particles.state > 0
        return particles.zeta[mask_alive].min(), particles.zeta[mask_alive].max()

    def long_density(line, particles):
        mask_alive = particles.state > 0
        if not(np.any(particles.at_turn[mask_alive] == 0)): # don't check at the first turn
            assert np.all(particles.zeta[mask_alive] > zeta_min0)
            assert np.all(particles.zeta[mask_alive] < zeta_max0)
        return np.histogram(particles.zeta[mask_alive], bins=200,
                            range=(zeta_min0, zeta_max0),
                            weights=particles.weight[mask_alive])

    def y_mean_hist(line, particles):

        mask_alive = particles.state > 0
        if not(np.any(particles.at_turn[mask_alive] == 0)): # don't check at the first turn
            assert np.all(particles.zeta[mask_alive] > zeta_min0)
            assert np.all(particles.zeta[mask_alive] < zeta_max0)
        return np.histogram(particles.zeta[mask_alive], bins=200,
                            range=(zeta_min0, zeta_max0), weights=particles.y[mask_alive])


    line.enable_time_dependent_vars = True
    num_turns=200
    line.track(p, num_turns=num_turns, log=xt.Log(intensity=intensity,
                                            long_density=long_density,
                                            y_mean_hist=y_mean_hist,
                                            z_range=z_range,
                                            particles=particles
                                            ), with_progress=10)

    inten = line.log_last_track['intensity']

    f_rev_ave = 1 / tw.T_rev0 * (1 - tw.slip_factor * p.delta.mean())
    t_rev_ave = 1 / f_rev_ave

    inten_exp =  np.sum(p0.weight) / t_rev_ave

    z_axis = line.log_last_track['long_density'][0][1]
    hist_mat = np.array([rr[0] for rr in line.log_last_track['long_density']])
    hist_y = np.array([rr[0] for rr in line.log_last_track['y_mean_hist']])

    dz = z_axis[1] - z_axis[0]
    y_vs_t = np.fliplr(hist_y).flatten() # need to flip because of the minus in z = -beta0 c t
    intensity_vs_t = np.fliplr(hist_mat).flatten()
    z_unwrapped = np.arange(0, len(y_vs_t)) * dz
    t_unwrapped = z_unwrapped / (tw.beta0 * clight)

    z_range_size = z_axis[-1] - z_axis[0]
    t_range_size = z_range_size / (tw.beta0 * clight)

    import nafflib
    intensity_no_ave = intensity_vs_t - np.mean(intensity_vs_t)
    f_harmons = nafflib.get_tunes(intensity_no_ave, N=50)[0] / (t_unwrapped[1] - t_unwrapped[0])
    f_nominal = 1 / tw.T_rev0
    dt_expected = -(twom.zeta[-1] - twom.zeta[0]) / tw.beta0 / clight
    f_expected = 1 / (tw.T_rev0 + dt_expected)

    f_measured = f_harmons[np.argmin(np.abs(f_harmons - f_nominal))]

    print('f_nominal:  ', f_nominal, ' Hz')
    print('f_expected: ', f_expected, ' Hz')
    print('f_measured: ', f_measured, ' Hz')
    print('Error:      ', f_measured - f_expected, 'Hz')

    xo.assert_allclose(f_expected, f_measured, rtol=0, atol=5.) # 5 Hz tolerance (to account for random fluctuations)
    xo.assert_allclose(np.mean(inten), inten_exp, rtol=1e-2, atol=0)
    xo.assert_allclose(p.at_turn, num_turns*0.9, rtol=3e-2, atol=0) #beta1 defaults to 0.1

    tt = line.get_table()
    tt_synch = tt.rows[tt.element_type=='SyncTime']
    assert len(tt_synch) == 12
    assert tt_synch.name[0] == 'synctime_start'
    assert tt_synch.name[-1] == 'synctime_end'
    assert np.all(tt_synch.name[5] == 'synctime_4')
    assert line['synctime_start'].at_start
    assert not line['synctime_end'].at_start
    assert not line['synctime_4'].at_start
    assert line['synctime_end'].at_end
    assert not line['synctime_start'].at_end
    assert not line['synctime_4'].at_end
