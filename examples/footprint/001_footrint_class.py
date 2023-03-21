import numpy as np

import xtrack as xt
import xpart as xp

import matplotlib.pyplot as plt




class FootprintPolar():

    def __init__(self, r_range=(0.1, 6), theta_range=(0.2, np.pi/2-0.05),
                 n_r=10, n_theta=10, n_turns=512, n_fft=2**18,
                 nemitt_x=None, nemitt_y=None):

        assert nemitt_x is not None and nemitt_y is not None, (
            'nemitt_x and nemitt_y must be provided')

        self.r_range = r_range
        self.theta_range = theta_range
        self.n_r = n_r
        self.n_theta = n_theta
        self.n_turns = n_turns
        self.n_fft = n_fft

        self.nemitt_x = nemitt_x
        self.nemitt_y = nemitt_y

        self.r = np.linspace(*r_range, n_r)
        self.theta = np.linspace(*theta_range, n_theta)
        self.R, self.Theta = np.meshgrid(self.r, self.theta)

        self.x_norm = self.R * np.cos(self.Theta)
        self.y_norm = self.R * np.sin(self.Theta)

    def _compute_footprint(self, line):

        particles = line.build_particles(
            x_norm=self.x_norm.flatten(), y_norm=self.y_norm.flatten(),
            nemitt_x=self.nemitt_x, nemitt_y=self.nemitt_y)

        line.track(particles, num_turns=self.n_turns, turn_by_turn_monitor=True)

        assert np.all(particles.state == 1), (
            'Some particles were lost during tracking')
        mon = line.record_last_track

        fft_x = np.fft.rfft(
            mon.x - np.atleast_2d(np.mean(mon.x, axis=1)).T, n=self.n_fft, axis=1)
        fft_y = np.fft.rfft(
            mon.y - np.atleast_2d(np.mean(mon.y, axis=1)).T, n=self.n_fft, axis=1)

        freq_axis = np.fft.rfftfreq(self.n_fft)

        qx = freq_axis[np.argmax(np.abs(fft_x), axis=1)]
        qy = freq_axis[np.argmax(np.abs(fft_y), axis=1)]

        self.qx = np.reshape(qx, self.R.shape)
        self.qy = np.reshape(qy, self.R.shape)

    def plot(self, ax=None, **kwargs):

        if ax is None:
            ax = plt.gca()

        if 'color' not in kwargs:
            kwargs['color'] = 'k'

        ax.plot(self.qx, self.qy, **kwargs)
        ax.plot(self.qx.T, self.qy.T, **kwargs)

        ax.set_xlabel('qx')
        ax.set_ylabel('qy')

        return ax


nemitt_x = 1e-6
nemitt_y = 1e-6


fp = FootprintPolar(nemitt_x=nemitt_x, nemitt_y=nemitt_y)

line = xt.Line.from_json(
    '../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b1.json')
line.build_tracker()

fp._compute_footprint(line)
