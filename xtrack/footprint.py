import numpy as np
import xobjects as xo
import xtrack as xt

class LinearRescale():

    def __init__(self, knob_name, v0, dv):
            self.knob_name = knob_name
            self.v0 = v0
            self.dv = dv

def _footprint_with_linear_rescale(linear_rescale_on_knobs, line, kwargs):

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
            fp = line.get_footprint(**kwargs)

        _fp0_ref = fp.__dict__.copy() # for debugging
        qx0 = fp.qx
        qy0 = fp.qy

        for rr in linear_rescale_on_knobs:
            nn = rr.knob_name
            v0 = rr.v0
            dv = rr.dv

            knobs_1 = knobs_0.copy()
            knobs_1[nn] = v0 + dv

            with xt._temp_knobs(line, knobs_1):
                fp1 = line.get_footprint(**kwargs)

            delta_qx = (fp1.qx - qx0) / dv * (line.vars[nn]._value - v0)
            delta_qy = (fp1.qy - qy0) / dv * (line.vars[nn]._value - v0)

            fp.qx += delta_qx
            fp.qy += delta_qy

        return fp

class Footprint():

    def __init__(self, nemitt_x=None, nemitt_y=None, n_turns=256, n_fft=2**18,
            mode='polar', r_range=None, theta_range=None, n_r=None, n_theta=None,
            x_norm_range=None, y_norm_range=None, n_x_norm=None, n_y_norm=None,
            keep_fft=False,auto_to_numpy=True,fft_chunk_size=200):

        assert nemitt_x is not None and nemitt_y is not None, (
            'nemitt_x and nemitt_y must be provided')
        self.mode = mode
        self.auto_to_numpy = auto_to_numpy
        self.n_turns = n_turns
        self.n_fft = n_fft
        self.fft_chunk_size = fft_chunk_size
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

    def _compute_footprint(self, line):

        nplike_lib = line._context.nplike_lib

        particles = line.build_particles(
            x_norm=self.x_norm_2d.flatten(), y_norm=self.y_norm_2d.flatten(),
            nemitt_x=self.nemitt_x, nemitt_y=self.nemitt_y)

        line.track(particles, num_turns=self.n_turns, turn_by_turn_monitor=True)
        
        assert nplike_lib.all(particles.state == 1), (
            'Some particles were lost during tracking')
        mon = line.record_last_track
        mon.auto_to_numpy = False
        x_noCO = mon.x - nplike_lib.atleast_2d(nplike_lib.mean(mon.x, axis=1)).T
        y_noCO = mon.y - nplike_lib.atleast_2d(nplike_lib.mean(mon.y, axis=1)).T
        freq_axis = nplike_lib.fft.rfftfreq(self.n_fft)
        if self.keep_fft:
            self.fft_x = nplike_lib.zeros((npart,len(freq_axis)),dtype=complex)
            self.fft_y = nplike_lib.zeros((npart,len(freq_axis)),dtype=complex)
        npart = nplike_lib.shape(x_noCO)[0]
        self.qx = nplike_lib.zeros(npart,dtype=float)
        self.qy = nplike_lib.zeros(npart,dtype=float)
        iStart = 0
        while iStart < npart:
            iEnd = iStart + self.fft_chunk_size
            if iEnd > npart:
                iEnd = npart
            fft_x = nplike_lib.fft.rfft(x_noCO[iStart:iEnd,:], n=self.n_fft)
            fft_y = nplike_lib.fft.rfft(y_noCO[iStart:iEnd,:], n=self.n_fft)
            if self.keep_fft:
                self.fft_x[iStart:iEnd,:] = fft_x
                self.fft_y[iStart:iEnd,:] = fft_y
            qx = freq_axis[nplike_lib.argmax(nplike_lib.abs(fft_x), axis=1)]
            qy = freq_axis[nplike_lib.argmax(nplike_lib.abs(fft_y), axis=1)]
            self.qx[iStart:iEnd] = qx
            self.qy[iStart:iEnd] = qy
            iStart += self.fft_chunk_size

        self.qx = nplike_lib.reshape(self.qx, self.x_norm_2d.shape)
        self.qy = nplike_lib.reshape(self.qy, self.y_norm_2d.shape)

        if self.auto_to_numpy:
            ctx2np = line._context.nparray_from_context_array
            self.qx = ctx2np(self.qx)
            self.qy = ctx2np(self.qy)
            if self.keep_fft:
                self.fft_x = ctx2np(self.fft_x)
                self.fft_y = ctx2np(self.fft_y)

    def _compute_tune_shift(self,_context,J1_2d,J1_grid,J2_2d,J2_grid,q,coherent_tune,epsilon):
        nplike_lib = _context.nplike_lib
        ctx2np = _context.nparray_from_context_array
        np2ctx = _context.nparray_to_context_array

        integrand = -J1_2d*nplike_lib.exp(-J1_2d-J2_2d) / (coherent_tune - q + epsilon*1j)
        tune_shift = ctx2np(-1.0/nplike_lib.trapz(J2_grid,nplike_lib.trapz(J1_grid,integrand,1),0))
        return tune_shift

    def _compute_tune_shift_adaptive_epsilon(self,_context,J1_2d,J1_grid,J2_2d,J2_grid,q,coherent_tune,
                                             epsilon0,epsilon_factor,epsilon_rel_tol,max_iter,min_epsilon):
        tune_shift = self._compute_tune_shift(_context,J1_2d,J1_grid,J2_2d,J2_grid,q,coherent_tune,epsilon0)
        if epsilon_factor > 0.0:
            epsilon_ref = epsilon0
            epsilon = np.abs(np.imag(tune_shift)*epsilon_factor)
            if epsilon < min_epsilon:
                epsilon = min_epsilon
            count = 0
            while np.abs(1-epsilon/epsilon_ref) > epsilon_rel_tol and count < max_iter and epsilon >= min_epsilon:
                tune_shift = self._compute_tune_shift(_context,J1_2d,J1_grid,J2_2d,J2_grid,q,coherent_tune,epsilon)
                epsilon_ref = epsilon
                epsilon = np.abs(np.imag(tune_shift)*epsilon_factor)
                count += 1
        return tune_shift

    def get_stability_diagram(self,_context=None,n_points_stabiliy_diagram=100,epsilon0=1E-5,epsilon_factor=0.1,epsilon_rel_tol=0.1,max_iter = 10,min_epsilon = 1E-6,n_points_interpolate = 1000):
        if _context == None:
            _context = xo.ContextCpu()
        nplike_lib = _context.nplike_lib
        splike_lib = _context.splike_lib
        ctx2np = _context.nparray_from_context_array
        np2ctx = _context.nparray_to_context_array

        Jx_2d = np2ctx(self.Jx_2d/self.nemitt_x)
        Jx_grid = np2ctx(self.Jx_grid/self.nemitt_x)
        Jy_2d = np2ctx(self.Jy_2d/self.nemitt_y)
        Jy_grid = np2ctx(self.Jy_grid/self.nemitt_y)
        qx = np2ctx(self.qx)
        qy = np2ctx(self.qy)

        if n_points_interpolate > len(Jx_grid) or n_points_interpolate > len(Jy_grid):
            interpolator_x = splike_lib.interpolate.RegularGridInterpolator(points=[Jy_grid,Jx_grid],values=qx,bounds_error=True, fill_value=None)
            interpolator_y = splike_lib.interpolate.RegularGridInterpolator(points=[Jy_grid,Jx_grid],values=qy,bounds_error=True, fill_value=None)
            Jx_grid = nplike_lib.linspace(Jx_grid[0],Jx_grid[-1],n_points_interpolate)
            Jy_grid = nplike_lib.linspace(Jy_grid[0],Jy_grid[-1],n_points_interpolate)
            Jx_2d,Jy_2d = nplike_lib.meshgrid(Jx_grid,Jy_grid)
            qx = interpolator_x((Jy_2d,Jx_2d))
            qy = interpolator_y((Jy_2d,Jx_2d))

        coherent_tunes_x = np.linspace(np.min(self.qx),np.max(self.qx),n_points_stabiliy_diagram)
        coherent_tunes_y = np.linspace(np.min(self.qy),np.max(self.qy),n_points_stabiliy_diagram)
        tune_shifts_x = np.zeros_like(coherent_tunes_x,dtype=complex)
        tune_shifts_y = np.zeros_like(coherent_tunes_y,dtype=complex)
        for i in range(n_points_stabiliy_diagram):
            tune_shifts_x[i] = self._compute_tune_shift_adaptive_epsilon(_context=_context,
                                          J1_2d=Jx_2d,J1_grid=Jx_grid,J2_2d=Jy_2d,J2_grid=Jy_grid,
                                          q = qx,coherent_tune = coherent_tunes_x[i],
                                          epsilon0 = epsilon0,epsilon_factor = epsilon_factor,epsilon_rel_tol=epsilon_rel_tol,
                                          max_iter=max_iter,min_epsilon=min_epsilon)
            tune_shifts_y[i] = self._compute_tune_shift_adaptive_epsilon(_context=_context,
                                          J1_2d=Jy_2d,J1_grid=Jy_grid,J2_2d=Jx_2d,J2_grid=Jx_grid,
                                          q = qy,coherent_tune = coherent_tunes_y[i],
                                          epsilon0 = epsilon0,epsilon_factor = epsilon_factor,epsilon_rel_tol=epsilon_rel_tol,
                                          max_iter=max_iter,min_epsilon=min_epsilon)
        return tune_shifts_x,tune_shifts_y

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
