import xtrack as xt
import numpy as np
from scipy.signal import find_peaks

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
		f_rev=1 / twiss.T_rev0, schottky_harmonic=427_725, n_taylor=4
	)
	line.discard_tracker()
	line.append_element(element=schottky_monitor, name='Schottky monitor')
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

# TODO: add test for nonlinear RF