# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xtrack as xt
import numpy as np
from scipy.constants import c as clight
from scipy.signal import find_peaks
from scipy.signal.windows import flattop
from scipy.special import jv, ellipk

# The Schottky monitor is a plain-python (numpy) element, hence CPU only:
# the tests below are not run with for_all_test_contexts.
LHC_LENGTH = 26658.8831999989
LHC_SCHOTTKY_HARMONIC = 427_725


# Function used for testing
def extract_qs_spacings(f, psd, window_width=0.15, threshold=0.01):
    # Extract the distance between successive peaks in the spectrum
    # Only works for single particle spectra

    # Restrict to a window around central frequency
    mask = np.abs(f - np.mean(f)) < 0.5 * window_width
    f_win = f[mask]
    psd_win = psd[mask]

    # Find peaks in the windowed PSD with minimum heights threshold
    peaks, _ = find_peaks(psd_win, height=np.max(psd_win) * threshold)
    peak_freqs = np.sort(f_win[peaks])

    # Compute spacings
    spacings = np.diff(peak_freqs)
    return spacings


def test_qs_qx_qy_linear_rf():
    # Create a line with a linear RF and add Schottky monitor
    lmap = xt.LineSegmentMap(
        length=26658.8831999989,
        qx=0.27,
        qy=0.295,
        dqx=15,
        dqy=15,
        longitudinal_mode='linear_fixed_qs',
        qs=0.004,
        bets=1,
        betx=1,
        bety=1,
    )
    line = xt.Line(elements=[lmap])
    line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, energy0=450e9)
    twiss = line.twiss()
    schottky_monitor = xt.monitors.SchottkyMonitor(
        f_rev=1 / twiss.t_rev0, schottky_harmonic=427_725, n_taylor=4
    )
    line.append('Schottky monitor', schottky_monitor)
    line.build_tracker()
    # Track single particle
    particle = line.build_particles(
        x=1e-3,
        px=0,
        y=1e-3,
        py=0,
        zeta=1e-2,
        delta=0,
    )
    line.track(particle, num_turns=10000, with_progress=True)
    schottky_monitor.process_spectrum(
        inst_spectrum_len=10000,
        delta_q=5e-5,
        band_width=0.45,
        qx=line.elements[0].qx,
        qy=line.elements[0].qy,
    )

    # Test synchrotron tune on each band
    for region in ['lowerH', 'center', 'upperH', 'lowerV', 'upperV']:
        spacings = extract_qs_spacings(
            schottky_monitor.frequencies[region], schottky_monitor.PSD_avg[region]
        )
        # Assert that each spacing is close to qs or an integer multiple (some peaks may be zero)
        qs_ref = line.elements[0].qs
        multiples = np.round(spacings / qs_ref,)
        np.testing.assert_allclose(spacings, multiples * qs_ref, atol=1e-4) #atol = 2*delta_q (resolution of the spectra)

    # Test vertical and horizontal betatron tunes
    for region in ['lowerH', 'upperH', 'lowerV', 'upperV']:
        window_width = 0.45
        f = schottky_monitor.frequencies[region]
        psd = schottky_monitor.PSD_avg[region]

        # Restrict to a window around central frequency
        mask = np.abs(f - np.mean(f)) < 0.5 * window_width
        f_win = f[mask]
        psd_win = psd[mask]
        # Check that full band is used to compute com and no overlap with adjacent harmonics
        assert np.all(psd_win[:300] < 0.001 * np.max(psd_win))
        assert np.all(psd_win[-300:] < 0.001 * np.max(psd_win))

        # Center of mass using full spectrum
        f_com_full = np.sum(f_win * psd_win) / np.sum(psd_win)
        f_com_full = np.abs(f_com_full)
        if region in ['lowerH', 'upperH']:
            np.testing.assert_allclose(f_com_full, line.elements[0].qx, atol=5e-5) #atol = delta_q (resolution of the spectra)
        elif region in ['lowerV', 'upperV']:
            np.testing.assert_allclose(f_com_full, line.elements[0].qy, atol=5e-5)


# Function used for testing
def line_with_schottky_monitor(n_taylor, schottky_harmonic, n_turns=None, n_part=None,
                               **lmap_kwargs):
    # Build a line made of a single LineSegmentMap and append a Schottky monitor,
    # preceded by a particles monitor recording the coordinates if n_turns is given
    line = xt.Line(elements=[xt.LineSegmentMap(betx=1, bety=1, **lmap_kwargs)])
    line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, energy0=450e9)
    twiss = line.twiss()
    schottky_monitor = xt.monitors.SchottkyMonitor(
        f_rev=1 / twiss.t_rev0, schottky_harmonic=schottky_harmonic, n_taylor=n_taylor
    )
    particles_monitor = None
    if n_turns is not None:
        particles_monitor = xt.ParticlesMonitor(
            start_at_turn=0, stop_at_turn=n_turns, num_particles=n_part
        )
        line.append('Particles monitor', particles_monitor)
    line.append('Schottky monitor', schottky_monitor)
    line.build_tracker()
    return line, twiss, schottky_monitor, particles_monitor


# Function used for testing
def max_relative_difference(schottky_monitor, particles_monitor, window, regions):
    # Largest difference between the spectra of the monitor and a direct evaluation of
    # S(omega) = sum_n w_n sum_k q_k(n) exp(-i omega (n T_rev + tau_k(n))), without Taylor
    # expansion, with omega = omega_c + 2 pi f_rev f and q = 1 (long.), x or y (transv.)
    T_rev = 1 / schottky_monitor.f_rev
    tau = -np.asarray(particles_monitor.zeta) / (clight * np.asarray(particles_monitor.beta0))
    turns = np.arange(tau.shape[1])
    window = window / np.sum(window)
    max_difference = 0
    for region in regions:
        if region == 'center':
            signal = np.ones_like(tau)
        else:
            signal = np.asarray(particles_monitor.x if 'H' in region else particles_monitor.y)
        delta_omega = 2 * np.pi * schottky_monitor.f_rev * schottky_monitor.frequencies[region]
        omega = schottky_monitor.omega_c + delta_omega
        spectrum = np.zeros(len(omega), dtype=np.complex128)
        for k in range(tau.shape[0]):
            phase = np.outer(omega, turns * T_rev + tau[k])
            spectrum += np.exp(-1j * phase) @ (window * signal[k])
        psd_direct = np.abs(spectrum) ** 2 / schottky_monitor.N_macropart_max
        psd = np.asarray(schottky_monitor.PSD_avg[region])
        difference = np.max(np.abs(psd - psd_direct)) / np.max(psd_direct)
        max_difference = max(max_difference, difference)
    return max_difference


def test_direct_evaluation_of_the_spectra():
    # Compare the spectra of the monitor with a direct evaluation of the Schottky signal
    # for a small ring (f_rev = 3 MHz) with bunch lengths of a few metres, so that
    # delta_omega * tau ~ 0.1 and the Taylor terms of order >= 1 are needed
    n_turns, n_part = 2048, 3
    regions = ['lowerH', 'upperH', 'lowerV', 'upperV', 'center']
    ring = dict(length=100, qx=0.31, qy=0.32, dqx=3, dqy=-2,
                longitudinal_mode='linear_fixed_qs', qs=0.013, bets=200)
    processing_param = dict(inst_spectrum_len=n_turns, delta_q=0.5 / n_turns,
                            band_width=0.45, qx=0.31, qy=0.32)

    def track(n_taylor):
        line, _, schottky_monitor, particles_monitor = line_with_schottky_monitor(
            n_taylor, schottky_harmonic=20, n_turns=n_turns, n_part=n_part, **ring
        )
        # Track a few particles with large longitudinal amplitudes
        particles = line.build_particles(
            x=[1e-3, -0.5e-3, 0.2e-3], px=[0, 0.3e-3, 0], y=[0.4e-3, 0.8e-3, -0.6e-3],
            py=[0.1e-3, 0, 0], zeta=[2, -5, 8], delta=[0, 0.01, -0.02],
        )
        line.track(particles, num_turns=n_turns)
        return schottky_monitor, particles_monitor

    schottky_monitor, particles_monitor = track(n_taylor=6)
    # Check that the Taylor terms of order >= 1 are needed for this case
    tau_max = np.max(np.abs(particles_monitor.zeta)) / clight
    delta_omega_max = np.pi * processing_param['band_width'] * schottky_monitor.f_rev
    assert 0.1 < delta_omega_max * tau_max < 0.2

    # Test the flattop and the rectangular windows
    schottky_monitor.process_spectrum(**processing_param)
    assert max_relative_difference(schottky_monitor, particles_monitor,
                                   flattop(n_turns), regions) < 1e-5
    schottky_monitor.clear_spectrum()
    schottky_monitor.process_spectrum(**processing_param, flattop_window=False)
    assert max_relative_difference(schottky_monitor, particles_monitor,
                                   np.ones(n_turns), regions) < 1e-5

    # With a single Taylor term the spectra must be visibly wrong, otherwise the
    # comparison above would not be sensitive to the Taylor expansion
    schottky_monitor, particles_monitor = track(n_taylor=1)
    schottky_monitor.process_spectrum(**processing_param)
    assert max_relative_difference(schottky_monitor, particles_monitor,
                                   flattop(n_turns), regions) > 1e-2


def test_bessel_satellites_single_particle():
    # Create a line with a linear RF and a small chromaticity and add Schottky monitor
    qs, bets, dqx, dqy = 0.004, 1, 0.5, -1
    qx, qy = 0.27, 0.295
    x_hat, y_hat, zeta_hat = 1e-3, 0.5e-3, 1.5e-2
    n_turns = 5000
    line, _, schottky_monitor, _ = line_with_schottky_monitor(
        n_taylor=4, schottky_harmonic=LHC_SCHOTTKY_HARMONIC, length=LHC_LENGTH,
        qx=qx, qy=qy, dqx=dqx, dqy=dqy,
        longitudinal_mode='linear_fixed_qs', qs=qs, bets=bets,
    )
    # Track single particle
    particle = line.build_particles(x=x_hat, px=0, y=y_hat, py=0, zeta=zeta_hat, delta=0)
    beta0 = float(particle.beta0[0])
    line.track(particle, num_turns=n_turns)
    schottky_monitor.process_spectrum(inst_spectrum_len=n_turns, delta_q=1e-4,
                                      band_width=0.12, qx=qx, qy=qy)

    # The satellites are spaced by qs and their heights are given by Bessel functions:
    # longitudinal J_m(a)^2 and transverse (x_hat / 2)^2 J_m(a -+ b)^2 (upper / lower
    # sideband), with a = omega_c zeta_hat / (beta0 c) and b = Q' delta_hat / qs,
    # i.e. for Q' > 0 above transition the lower sideband is the wider one
    a = schottky_monitor.omega_c * zeta_hat / (beta0 * clight)
    delta_hat = zeta_hat / bets / beta0
    bx, by = dqx * delta_hat / qs, dqy * delta_hat / qs

    def satellite_heights(region, center, orders):
        # Maximum of the spectrum around each satellite
        f = schottky_monitor.frequencies[region]
        psd = np.asarray(schottky_monitor.PSD_avg[region])
        masks = [np.abs(f - (center + order * qs)) < 0.25 * qs for order in orders]
        assert all(np.any(mask) for mask in masks), f'satellites outside the {region} band'
        return np.array([np.max(psd[mask]) for mask in masks])

    def rms_width(region, center):
        f = schottky_monitor.frequencies[region] - center
        psd = np.asarray(schottky_monitor.PSD_avg[region])
        return np.sqrt(np.sum(f**2 * psd) / np.sum(psd))

    # Test the height of the satellites on each band. The tolerance is set by the
    # frequency resolution, the transverse satellites being the narrower ones
    orders = np.arange(-6, 7)
    for region, center, amplitude, index, rtol in [
        ('center', 0, 1, a, 1e-4),
        ('upperH', qx, x_hat**2 / 4, abs(a - bx), 2e-2),
        ('lowerH', -qx, x_hat**2 / 4, abs(a + bx), 2e-2),
        ('upperV', qy, y_hat**2 / 4, abs(a - by), 2e-2),
        ('lowerV', -qy, y_hat**2 / 4, abs(a + by), 2e-2),
    ]:
        theory = amplitude * jv(orders, index) ** 2
        selection = theory > 1e-4 * np.max(theory)
        assert np.sum(selection) >= 5
        np.testing.assert_allclose(satellite_heights(region, center, orders)[selection],
                                   theory[selection], rtol=rtol)
        # The satellites of a band sum up to the power of the band, sum_m J_m^2 = 1
        np.testing.assert_allclose(np.sum(satellite_heights(region, center,
                                                            np.arange(-12, 13))),
                                   amplitude, rtol=1e-3)

    # Test the chromatic asymmetry of the width of the sidebands (rms width in tune unit)
    assert rms_width('lowerH', -qx) > 2 * rms_width('upperH', qx)
    assert rms_width('upperV', qy) > 1.5 * rms_width('lowerV', -qy)


def test_qs_nonlinear_rf():
    # Create a line with a nonlinear RF and add Schottky monitor
    n_turns, frequency_rf, zeta_hat, delta_q = 5000, 400e6, 0.2, 5e-5
    qx, qy = 0.27, 0.295
    line, twiss, schottky_monitor, _ = line_with_schottky_monitor(
        n_taylor=4, schottky_harmonic=LHC_SCHOTTKY_HARMONIC, length=LHC_LENGTH,
        qx=qx, qy=qy, dqx=0, dqy=0, longitudinal_mode='nonlinear', voltage_rf=4e6,
        frequency_rf=frequency_rf, phase_rf=np.pi, momentum_compaction_factor=3.225e-04,
    )
    # Track single particle with a large longitudinal amplitude
    particle = line.build_particles(x=1e-3, px=0, y=1e-3, py=0, zeta=zeta_hat, delta=0)
    beta0 = float(particle.beta0[0])
    line.track(particle, num_turns=n_turns)
    schottky_monitor.process_spectrum(inst_spectrum_len=n_turns, delta_q=delta_q,
                                      band_width=0.2, qx=qx, qy=qy, x=False, y=False)

    # In a nonlinear RF bucket the satellites of a particle with a large amplitude are
    # spaced by the amplitude dependent synchrotron tune of the pendulum,
    # qs(phi_hat) = qs * pi / (2 K(sin^2(phi_hat / 2))), with K the complete elliptic
    # integral of the first kind
    phi_hat = 2 * np.pi * frequency_rf * zeta_hat / (beta0 * clight)
    qs_theory = twiss.qs * np.pi / (2 * ellipk(np.sin(phi_hat / 2) ** 2))
    assert abs(qs_theory / twiss.qs - 1) > 0.1  # Check that the case is clearly nonlinear

    # Find the satellites and refine their position with the center of mass of the
    # neighbouring bins
    f = schottky_monitor.frequencies['center']
    psd = np.asarray(schottky_monitor.PSD_avg['center'])
    peaks, _ = find_peaks(psd, height=1e-3 * np.max(psd),
                          distance=int(0.5 * qs_theory / delta_q))
    assert len(peaks) > 30
    peak_freqs = np.array([np.sum(f[i - 6:i + 7] * psd[i - 6:i + 7])
                           / np.sum(psd[i - 6:i + 7]) for i in peaks])

    # Test the spacing of the satellites against the amplitude dependent tune
    qs_measured = np.polyfit(np.round(peak_freqs / qs_theory), peak_freqs, 1)[0]
    np.testing.assert_allclose(qs_measured, qs_theory, rtol=5e-4)
    assert abs(qs_measured / twiss.qs - 1) > 0.1  # The small amplitude tune is far off
