import numpy as np

import xtrack as xt

class LinearRescale():

    def __init__(self, knob_name, v0, dv):
            self.knob_name = knob_name
            self.v0 = v0
            self.dv = dv

def _footprint_with_linear_rescale(linear_rescale_on_knobs, line,
                                   freeze_longitudinal=False,
                                   delta0=None, zeta0=None, kwargs={}):

        if isinstance (linear_rescale_on_knobs, LinearRescale):
            linear_rescale_on_knobs = [linear_rescale_on_knobs]

        assert len(linear_rescale_on_knobs) == 1, (
            'Only one linear rescale is supported for now')

        knobs_0 = {}
        for rr in linear_rescale_on_knobs:
            nn = rr.knob_name
            v0 = rr.v0
            knobs_0[nn] = v0

        with xt._temp_knobs(line, knobs_0):
            fp = line.get_footprint(
                freeze_longitudinal=freeze_longitudinal,
                delta0=delta0, zeta0=zeta0, **kwargs)

        qx0 = fp.qx
        qy0 = fp.qy

        for rr in linear_rescale_on_knobs:
            nn = rr.knob_name
            v0 = rr.v0
            dv = rr.dv

            knobs_1 = knobs_0.copy()
            knobs_1[nn] = v0 + dv

            with xt._temp_knobs(line, knobs_1):
                fp1 = line.get_footprint(freeze_longitudinal=freeze_longitudinal,
                                        delta0=delta0, zeta0=zeta0, **kwargs)
            delta_qx = (fp1.qx - qx0) / dv * (line.vars[nn]._value - v0)
            delta_qy = (fp1.qy - qy0) / dv * (line.vars[nn]._value - v0)

            fp.qx += delta_qx
            fp.qy += delta_qy

        return fp

class Footprint():

    def __init__(self, nemitt_x=None, nemitt_y=None, n_turns=256, n_fft=2**18,
            mode='polar', r_range=None, theta_range=None, n_r=None, n_theta=None,
            x_norm_range=None, y_norm_range=None, n_x_norm=None, n_y_norm=None,
            keep_fft=False):

        assert nemitt_x is not None and nemitt_y is not None, (
            'nemitt_x and nemitt_y must be provided')
        self.mode = mode

        self.n_turns = n_turns
        self.n_fft = n_fft
        self.keep_fft = keep_fft

        self.nemitt_x = nemitt_x
        self.nemitt_y = nemitt_y

        assert mode in ['polar', 'uniform_action_grid'], (
            'mode must be either polar or uniform_action_grid')

        if mode == 'polar':

            assert x_norm_range is None and y_norm_range is None, (
                'x_norm_range and y_norm_range must be None for mode polar')
            assert n_x_norm is None and n_y_norm is None, (
                'n_x_norm and n_y_norm must be None for mode polar')

            if r_range is None:
                r_range = (0.1, 6)
            if theta_range is None:
                theta_range = (0.05, np.pi/2-0.05)
            if n_r is None:
                n_r = 10
            if n_theta is None:
                n_theta = 10

            self.r_range = r_range
            self.theta_range = theta_range
            self.n_r = n_r
            self.n_theta = n_theta

            self.r_grid = np.linspace(*r_range, n_r)
            self.theta_grid = np.linspace(*theta_range, n_theta)
            self.R_2d, self.Theta_2d = np.meshgrid(self.r_grid, self.theta_grid)

            self.x_norm_2d = self.R_2d * np.cos(self.Theta_2d)
            self.y_norm_2d = self.R_2d * np.sin(self.Theta_2d)

        elif mode == 'uniform_action_grid':

            assert r_range is None and theta_range is None, (
                'r_range and theta_range must be None for mode uniform_action_grid')
            assert n_r is None and n_theta is None, (
                'n_r and n_theta must be None for mode uniform_action_grid')

            if x_norm_range is None:
                x_norm_range = (0.1, 6)
            if y_norm_range is None:
                y_norm_range = (0.1, 6)
            if n_x_norm is None:
                n_x_norm = 10
            if n_y_norm is None:
                n_y_norm = 10

            Jx_min = nemitt_x * x_norm_range[0]**2 / 2
            Jx_max = nemitt_x * x_norm_range[1]**2 / 2
            Jy_min = nemitt_y * y_norm_range[0]**2 / 2
            Jy_max = nemitt_y * y_norm_range[1]**2 / 2

            self.Jx_grid = np.linspace(Jx_min, Jx_max, n_x_norm)
            self.Jy_grid = np.linspace(Jy_min, Jy_max, n_y_norm)

            self.Jx_2d, self.Jy_2d = np.meshgrid(self.Jx_grid, self.Jy_grid)

            self.x_norm_2d = np.sqrt(2 * self.Jx_2d / nemitt_x)
            self.y_norm_2d = np.sqrt(2 * self.Jy_2d / nemitt_y)

    def _compute_footprint(self, line, freeze_longitudinal=False,
                           delta0=None, zeta0=None):

        if freeze_longitudinal is None:
            # In future we could detect if the line has frozen longitudinal plane
            freeze_longitudinal = False

        particles = line.build_particles(
            x_norm=self.x_norm_2d.flatten(), y_norm=self.y_norm_2d.flatten(),
            nemitt_x=self.nemitt_x, nemitt_y=self.nemitt_y,
            zeta=zeta0, delta=delta0,
            freeze_longitudinal=freeze_longitudinal,
            method={True: '4d', False: '6d'}[freeze_longitudinal]
            )

        print('Tracking particles for footprint...')
        line.track(particles, num_turns=self.n_turns, turn_by_turn_monitor=True,
                   freeze_longitudinal=freeze_longitudinal)
        print('Done tracking.')

        ctx2np = line._context.nparray_from_context_array
        assert np.all(ctx2np(particles.state == 1)), (
            'Some particles were lost during tracking')
        mon = line.record_last_track

        print('Computing footprint...')
        fft_x = np.fft.rfft(
            mon.x - np.atleast_2d(np.mean(mon.x, axis=1)).T, n=self.n_fft, axis=1)
        fft_y = np.fft.rfft(
            mon.y - np.atleast_2d(np.mean(mon.y, axis=1)).T, n=self.n_fft, axis=1)

        if self.keep_fft:
            self.fft_x = fft_x
            self.fft_y = fft_y

        freq_axis = np.fft.rfftfreq(self.n_fft)

        qx = freq_axis[np.argmax(np.abs(fft_x), axis=1)]
        qy = freq_axis[np.argmax(np.abs(fft_y), axis=1)]

        self.qx = np.reshape(qx, self.x_norm_2d.shape)
        self.qy = np.reshape(qy, self.x_norm_2d.shape)
        print ('Done computing footprint.')

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        if 'color' not in kwargs:
            kwargs['color'] = 'k'

        labels = [None] * self.qx.shape[1]

        if 'label' in kwargs:
            label_str = kwargs['label']
            kwargs.pop('label')
            labels[0] = label_str

        ax.plot(self.qx, self.qy, label=labels, **kwargs)
        ax.plot(self.qx.T, self.qy.T, **kwargs)

        ax.set_xlabel(r'$q_x$')
        ax.set_ylabel(r'$q_y$')
