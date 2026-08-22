"""
Schottky spectrum monitor

Author: Christophe Lannoy
Date: 2025-08-17
"""

import numpy as np
import scipy as sp
from scipy.constants import c
from scipy.signal import ZoomFFT


class SchottkyMonitor:
    def __init__(self, f_rev, schottky_harmonic, n_taylor):
        """
        Tracking element computing Schottky spectra.
        Equations based on JINST 19 P03017, C. Lannoy et al.

        Parameters
        ----------
        f_rev : float
            Revolution frequency.
        schottky_harmonic : int
            Harmonic of the Schottky monitor.
        n_taylor : int
            Number of terms used for the Taylor expansion (4 is enough for LHC conditions).
        """
        self.f_rev = f_rev
        if n_taylor < 1:
            raise ValueError('At least one coefficient for the Taylor expansion is needed')
        self.n_taylor = n_taylor
        # Taylor expansion around central Schottky frequency omega_c
        self.omega_c = 2 * np.pi * f_rev * schottky_harmonic
        self.x_coeff, self.y_coeff, self.z_coeff = [], [], []
        self.initialised_with_first_tracking = False

    def track(self, particles):
        mask_alive = particles.state > 0
        tau = -particles.zeta[mask_alive] / (c * particles.beta0[mask_alive])

        # First time: store bunch parameters for truncation error estimate
        if not self.initialised_with_first_tracking:
            self.tau_max = 4 * np.std(tau)
            self.x_max = 4 * np.std(particles.x[mask_alive])
            self.y_max = 4 * np.std(particles.y[mask_alive])
            self.N_macropart_max = len(tau)
            self.initialised_with_first_tracking = True

        # Calculates the longitudinal and transverse coefficients (L and T) as defined in Eqs (2.2) and (2.4)
        z_terms = np.empty((self.n_taylor, len(tau)), dtype=np.csingle)
        z_terms[0, :] = np.exp(1j * self.omega_c * tau)
        for l in range(1, self.n_taylor):
            z_terms[l, :] = z_terms[l - 1, :] * tau

        self.x_coeff.append(np.sum(z_terms * particles.x[mask_alive], axis=1))
        self.y_coeff.append(np.sum(z_terms * particles.y[mask_alive], axis=1))
        self.z_coeff.append(np.sum(z_terms, axis=1))

    def process_spectrum(self, inst_spectrum_len, delta_q, band_width, qx, qy,
                         x=True, y=True, z=True, flattop_window=True):
        """
        Compute Schottky spectra from stored longitudinal and transverse coefficients.

        Parameters
        ----------
        inst_spectrum_len : int
            Number of turns used to compute one instantaneous spectrum.
        delta_q : float
            Frequency resolution expressed as normalised frequency f/f_rev.
            Physical spacing of Fourier bins is delta_q * f_rev.
        band_width : float
            Width of each processed band in normalised frequency f/f_rev (0 < band_width < 1).
        qx, qy : float
            Transverse tunes used to set the central frequency around which the 
            transverse side-bands will be computed. 
        x, y, z: bool
            Which band are to be computed. x, y, z stand for, respectively, 
            transverse horizontal, transverse vertical and longitudinal bands.
        flattop_window: bool
            Multiply time signal by flattop window, else no windowing used (=rectangular window).
        """
        if inst_spectrum_len > len(self.x_coeff):
            raise ValueError(f'Not enough turns tracked to produce one instantaneous spectra \n \
                               Number of turns tracked: {len(self.x_coeff)} \n \
                               Length of instantaneous spectra: {inst_spectrum_len}')
        if band_width <= 0 or band_width >= 1:
            raise ValueError('band_width must be a normalised frequency f/f_rev with 0 < band_width < 1')
        
        # If it's the first time calling the method we need to initialise it.
        if not hasattr(self, 'processing_param'):
            self.processing_param = locals()
            self._init_processing(delta_q, qx, qy, band_width)
        
        # Not the first time calling this method, we will append the new instantaneous Schottky PSDs to the 
        # existing ones. In this case we need to confirm that the processing parameters are identical.
        elif any(self.processing_param[key] != value
                 for key, value in locals().items() if key not in ['x', 'y', 'z']):
            raise ValueError('Different parameters for the processing, ' +
                             'keep the same parameters (except x, y and z) or use "clear_spectrum()". \n' +
                             'Existing parameters:' + str(self.processing_param) + '\n New parameters:' + str(locals()))
        if flattop_window:
            window = sp.signal.windows.flattop(inst_spectrum_len)
        else:
            window = np.ones(inst_spectrum_len)
        window /= np.sum(window)  # Normalizing window

        region_to_process = []
        if x: region_to_process.extend(['lowerH', 'upperH'])
        if y: region_to_process.extend(['lowerV', 'upperV'])
        if z: region_to_process.extend(['center'])

        for region in region_to_process:
            freq = self.frequencies[region]
            if region == 'center':
                coeff = self.z_coeff
            elif region in ('lowerH', 'upperH'):
                coeff = self.x_coeff
            else:  # 'lowerV', 'upperV'
                coeff = self.y_coeff

            # Computing instantaneous Schottky spectra as defined in Eqs. (2.2) and (2.4):
            #   S(q_k) = sum_l alpha[l, k] * X_l[k],
            #   X_l[k] = sum_n w_n * C[n, l] * exp(+2*pi*i*q_k*n)
            # X_l is the discrete-time Fourier transform of the windowed coefficients
            # C[n, l] (turn n, Taylor order l) evaluated on the uniform normalised
            # frequency grid q_k = q_0 + k*delta_q. It is computed with the chirp-z
            # (Bluestein) algorithm, i.e. O((L + n_freq) log) time and memory per
            # order, instead of materialising the (L x n_freq) kernel w_n*exp(2*pi*i*q_k*n).
            n_freq = len(freq)
            delta_omega = freq * 2 * np.pi * self.f_rev
            alpha = np.empty((self.n_taylor, n_freq), dtype=np.complex128)
            alpha[0, :] = 1.0
            for i_taylor in range(1, self.n_taylor):
                alpha[i_taylor, :] = alpha[i_taylor - 1, :] * 1j * delta_omega / i_taylor
            # scipy evaluates sum_n y_n exp(-2*pi*i*q_k*n) on q_k = linspace(freq[0], freq[-1], n_freq)
            # (fs=1: frequencies per turn, i.e. normalised to f_rev); the sign is handled below.
            zoom_dtft = ZoomFFT(inst_spectrum_len, [freq[0], freq[-1]], m=n_freq,
                                fs=1.0, endpoint=True)
            n_inst_spectra = len(coeff) // inst_spectrum_len
            for i in range(len(self.instantaneous_PSDs[region]), n_inst_spectra):
                print(f'Processing {region} Schottky spectrum {i+1}/{n_inst_spectra}', end='\r')
                # Coefficients (x, y, or z) of the turns of the i-th instantaneous spectrum, (L, n_taylor)
                inst_coeff = np.asarray(
                    coeff[i * inst_spectrum_len : (i + 1) * inst_spectrum_len]
                )
                windowed = inst_coeff.T * window  # (n_taylor, L): w_n * C[n, l]
                # DTFT on the grid with the + sign convention: conj(sum_n conj(y_n) exp(-2*pi*i*q_k*n))
                dtft = np.conj(zoom_dtft(np.conj(windowed), axis=-1))  # (n_taylor, n_freq)
                spectrum = np.einsum('lk,lk->k', alpha, dtft)  # sum over the Taylor orders
                self.instantaneous_PSDs[region].append(
                    np.abs(spectrum) ** 2 / self.N_macropart_max
                )
            self.PSD_avg[region] = np.mean(self.instantaneous_PSDs[region], axis=0)
            print(f'{region} band of Schottky spectrum processed')
        self._check_Taylor_approx()

    def _init_processing(self, delta_q, qx, qy, band_width):
        '''
        For each region of the Schottky spectrum, create an array of normalised frequencies 
        from -band_width/2 to +band_width/2 around the center of the Schottky band. 
        '''
        n_freq = band_width / delta_q
        center_freq = np.arange(-(n_freq//2), (n_freq)//2) * band_width / n_freq
        self.frequencies = {
            'lowerH': center_freq - (qx % 1),
            'upperH': center_freq + (qx % 1),
            'lowerV': center_freq - (qy % 1),
            'upperV': center_freq + (qy % 1),
            'center': center_freq,
        }
        # Create dict where the instantaneous and averaged PSDs will be stored
        self.instantaneous_PSDs = {i: [] for i in self.frequencies.keys()}
        self.PSD_avg = {i: [] for i in self.frequencies.keys()}

    def _check_Taylor_approx(self):
    # Longitudinal band, Eq. (2.3)
        if self.processing_param['z']:
            delta_omega_max = max(self.frequencies['center']) * 2 * np.pi * self.f_rev
            max_error = self.N_macropart_max**0.5 * (delta_omega_max*self.tau_max)**self.n_taylor * \
                        np.exp(delta_omega_max*self.tau_max) / sp.special.factorial(self.n_taylor)
            if np.sqrt(self.PSD_avg['center'][0]) < 100 * max_error:
                print('Number of Taylor terms too low for the longitudinal band')
            print(f'Maximal Taylor truncation error in z plane to be compared against sqrt(PSD): {max_error}')

    # Transverse bands
        if self.processing_param['x']:
            delta_omega_max = max(self.frequencies['upperH']) * 2 * np.pi * self.f_rev
            max_error = self.N_macropart_max**0.5 * self.x_max * (delta_omega_max*self.tau_max)**self.n_taylor * \
                        np.exp(delta_omega_max*self.tau_max) / sp.special.factorial(self.n_taylor)
            if np.sqrt(self.PSD_avg['upperH'][0]) < 100 * max_error:
                print('Number of Taylor terms too low for the horizontal bands')
            print(f'Maximal Taylor truncation error in x plane to be compared against sqrt(PSD): {max_error}')
        if self.processing_param['y']:
            delta_omega_max = max(self.frequencies['upperV']) * 2 * np.pi * self.f_rev
            max_error = self.N_macropart_max**0.5 * self.y_max * (delta_omega_max*self.tau_max)**self.n_taylor * \
                        np.exp(delta_omega_max*self.tau_max) / sp.special.factorial(self.n_taylor)
            if np.sqrt(self.PSD_avg['upperV'][0]) < 100 * max_error:
                print('Number of Taylor terms too low for the vertical bands')
            print(f'Maximal Taylor truncation error in y plane to be compared against sqrt(PSD): {max_error}')

    def clear_spectrum(self):
        """
        Clear the instantaneous spectra but keep the coefficients L and T.
        Can be used to re-compute Schottky spectra for different processing 
        parameters (window, frequency resolution, band widths) without 
        tracking the particles again
        """
        if hasattr(self, 'processing_param'):
            delattr(self, 'processing_param')
            delattr(self, 'frequencies')
            delattr(self, 'instantaneous_PSDs')
            delattr(self, 'PSD_avg')

    def clear_all(self):
        """Fully reset monitor (coefficients and spectra)."""
        self.clear_spectrum()
        self.x_coeff = []
        self.y_coeff = []
        self.z_coeff = []

    # Plotting utility -------------------------------------------------
    def plot(self, regions=None, log=False):
        """
        Plot average Schottky spectra.

        Parameters
        ----------
        regions : list(str) or None
            Regions to include (default: all available among
            ['lowerH','center','upperH','lowerV','upperV']).
        log : bool
            If True use logarithmic y-scale.
        """
        import matplotlib.pyplot as plt
        if not hasattr(self, 'PSD_avg'):
            raise RuntimeError('No processed spectrum found. Call process_spectrum() first.')

        # Determine regions to plot
        default_order = ['lowerH', 'center', 'upperH', 'lowerV', 'upperV']
        available = [r for r in default_order if r in self.PSD_avg and len(self.PSD_avg[r]) > 0]
        if regions is None:
            regions = available
        else:
            # Keep original order but enforce strict presence
            missing = [r for r in regions if r not in available]
            if len(missing) > 0:
                raise ValueError(f'Requested region(s) not processed: {missing}. Available: {available}. Call process_spectrum() with the appropriate parameters first.')
        if len(regions) == 0:
            raise RuntimeError('No spectra available to plot.')

        fig, axes = plt.subplots(1, len(regions), figsize=(4 * len(regions), 4))
        if len(regions) == 1:
            axes = [axes]

        for ax, region in zip(axes, regions):
            freq = self.frequencies[region]
            psd = self.PSD_avg[region]
            ax.plot(freq, psd)
            if log:
                ax.set_yscale('log')
            ax.set_xlabel('Frequency [$f_0$]')
            ax.set_ylabel('PSD [arb. units]')
            ax.set_title(region)
            ax.grid(True, which='both', ls=':', alpha=0.5)
        fig.tight_layout()
        return fig, axes