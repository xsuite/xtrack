import numpy as np
import xtrack as xt

def _compute_correction(x_iter, response_matrix, n_micado=None, rcond=None,
                        n_singular_values=None):

    if isinstance(response_matrix, (list, tuple)):
        assert len(response_matrix) == 3 # U, S, Vt
        U, S, Vh = response_matrix
        if n_singular_values is not None:
            U = U[:, :n_singular_values]
            S = S[:n_singular_values]
            Vh = Vh[:n_singular_values, :]
        response_matrix = U @ np.diag(S) @ Vh
    else:
        assert n_singular_values is None
        U = None
        S = None
        Vh = None

    n_hcorrectors = response_matrix.shape[1]

    if n_micado is not None:
        used_correctors = []

        for i_micado in range(n_micado):

            residuals = []
            for i_corr in range(n_hcorrectors):
                if i_corr in used_correctors:
                    residuals.append(np.nan)
                    continue
                mask_corr = np.zeros(n_hcorrectors, dtype=bool)
                mask_corr[i_corr] = True
                for i_used in used_correctors:
                    mask_corr[i_used] = True

                # Compute the correction with least squares
                _, residual_x, rank_x, sval_x = np.linalg.lstsq(
                            response_matrix[:, mask_corr], -x_iter, rcond=rcond)
                residuals.append(residual_x[0])
            used_correctors.append(np.nanargmin(residuals))

        mask_corr = np.zeros(n_hcorrectors, dtype=bool)
        mask_corr[used_correctors] = True
    else:
        mask_corr = np.ones(n_hcorrectors, dtype=bool)
        mask_corr[:] = True

    # Compute the correction with least squares
    if mask_corr.all() and S is not None:
        # Can reuse the SVD decomposition
        S_inv = np.zeros_like(S)
        S_inv[S > 0] = 1 / S[S > 0]
        if rcond is not None:
            S_inv[S < rcond * S[0]] = 0
        correction_masked = Vh.T.conj() @ (np.diag(S_inv) @ (U.T.conj() @ (-x_iter)))
    else:
        correction_masked, residual_x, rank_x, sval_x = np.linalg.lstsq(
                    response_matrix[:, mask_corr], -x_iter, rcond=rcond)
    correction_x = np.zeros(n_hcorrectors)
    correction_x[mask_corr] = correction_masked

    return correction_x


def _build_response_matrix(tw, monitor_names, corrector_names,
                           mode='closed', plane=None):

    assert mode in ['closed', 'open']
    assert plane in ['x', 'y']

    # Build response matrix
    indices_monitors = tw.rows.indices[monitor_names]
    indices_correctors = tw.rows.indices[corrector_names]
    bet_monitors = tw['bet' + plane][indices_monitors]
    bet_correctors = tw['bet' + plane][indices_correctors]

    mu_monitor = tw['mu' + plane][indices_monitors]
    mux_correctors = tw['mu' + plane][indices_correctors]

    n_monitors = len(monitor_names)
    n_correctors = len(corrector_names)

    bet_prod = np.atleast_2d(bet_monitors).T @ np.atleast_2d(bet_correctors)
    mu_diff = (np.tile(mu_monitor, (n_correctors, 1)).T
                        - np.tile(mux_correctors, (n_monitors, 1)))

    if mode == 'open':
        # Wille eq. 3.164
        mu_diff[mu_diff < 0] = 0 # use only correctors upstream of the monitor
        response_matrix = (np.sqrt(bet_prod) * np.sin(2*np.pi*np.abs(mu_diff)))
    elif mode == 'closed':
        # Slide 28
        # https://indico.cern.ch/event/1328128/contributions/5589794/attachments/2786478/4858384/linearimperfections_2024.pdf
        tune = tw['q' + plane]
        response_matrix = (np.sqrt(bet_prod) / 2 / np.sin(np.pi * tune)
                             * np.cos(np.pi * tune - 2*np.pi*np.abs(mu_diff)))

    return response_matrix


class OrbitCorrectionSinglePlane:

    def __init__(self, line, plane, monitor_names, corrector_names,
                 start=None, end=None, twiss_table=None, n_micado=None,
                 n_singular_values=None, rcond=None,
                 x_init=0, px_init=0, y_init=0, py_init=0, zeta_init=0, delta_init=0,
                 monitor_alignment=None):

        assert plane in ['x', 'y']

        self.x_init = x_init
        self.px_init = px_init
        self.y_init = y_init
        self.py_init = py_init
        self.zeta_init = zeta_init
        self.delta_init = delta_init

        self.twiss_table = twiss_table
        if twiss_table is not None:
            assert twiss_table.reference_frame == 'proper', (
                'Twiss table must be in the proper frame (`reverse` not supported)')

        if start is None:
            assert end is None
            self.mode = 'closed'
            if self.twiss_table is None:
                self.twiss_table = line.twiss4d(reverse=False)
        else:
            assert end is not None
            self.mode = 'open'
            if self.twiss_table is None:
                # Initialized with betx=1, bety=1 (use W_matrix to avoid compilation)
                self.twiss_table = line.twiss4d(start=start, end=end,
                    init=xt.TwissInit(W_matrix=np.eye(6),
                                      particle_on_co=line.build_particles(
                                          x=self.x_init, y=self.y_init,
                                          px=self.px_init, py=self.py_init,
                                          zeta=self.zeta_init, delta=self.delta_init),
                                      element_name=start),
                                      reverse=False)

        if corrector_names is None:
            corr_names_from_line = getattr(line, f'steering_correctors_{plane}')
            assert corr_names_from_line is not None, (
                f'No steering correctors found for plane {plane}')
            if start is not None:
                corrector_names = [nn for nn in corr_names_from_line
                                   if nn in self.twiss_table.name]
            else:
                corrector_names = corr_names_from_line

        if monitor_names is None:
            monitor_names = getattr(line, f'steering_monitors_{plane}')
            assert monitor_names is not None, (
                f'No monitors found for plane {plane}')
            if start is not None:
                monitor_names = [nn for nn in monitor_names
                                 if nn in self.twiss_table.name]
            else:
                monitor_names = monitor_names

        assert len(monitor_names) > 0
        assert len(corrector_names) > 0

        self.line = line
        self.plane = plane
        self.monitor_names = monitor_names
        self.corrector_names = corrector_names
        self.start = start
        self.end = end
        self.n_micado = n_micado
        self.rcond = rcond
        self.n_singular_values = n_singular_values

        self.response_matrix = _build_response_matrix(plane=self.plane,
            tw=self.twiss_table, monitor_names=self.monitor_names,
            corrector_names=self.corrector_names, mode=self.mode)

        U, S , Vt = np.linalg.svd(self.response_matrix, full_matrices=False)
        self.singular_values = S
        self.singular_vectors_out = U
        self.singular_vectors_in = Vt

        tw_table_local = self.twiss_table.rows[start:end]

        self._indices_monitor = tw_table_local.rows.indices[self.monitor_names]
        self._indices_correctors = tw_table_local.rows.indices[self.corrector_names]
        self.s_correctors = tw_table_local.s[self._indices_correctors]
        self.s_monitors = tw_table_local.s[self._indices_monitor]

        self.shift_x_monitors = 0*self.s_monitors
        self.shift_y_monitors = 0*self.s_monitors
        self.rot_s_rad_monitors = 0*self.s_monitors

        self.monitor_alignment = monitor_alignment
        if monitor_alignment is not None:
            for nn, alignment in monitor_alignment.items():
                for kk in alignment.keys():
                    assert kk in ['shift_x', 'shift_y', 'rot_s_rad']
                if nn in self.monitor_names:
                    i_monitor = self.monitor_names.index(nn)
                    self.shift_x_monitors[i_monitor] = alignment.get('shift_x', 0)
                    self.shift_y_monitors[i_monitor] = alignment.get('shift_y', 0)
                    self.rot_s_rad_monitors[i_monitor] = alignment.get('rot_s_rad', 0)

        self._add_correction_knobs()

    def correct(self, n_iter='auto', n_micado=None, n_singular_values=None,
                rcond=None, stop_iter_factor=0.1, verbose=True, _tw_orbit=None):

        if _tw_orbit is not None and n_iter !=1:
            raise ValueError('`_tw_orbit` can only be used with `n_iter=1`')

        assert n_iter == 'auto' or np.isscalar(n_iter)
        if n_iter == 'auto':
            assert stop_iter_factor > 0
            assert stop_iter_factor < 1

        pos_rms_prev = 0

        i_iter = 0
        while True:
            try:
                position = self._measure_position(tw_orbit=_tw_orbit)
            except xt.twiss.ClosedOrbitSearchError:
                raise RuntimeError('Closed orbit not found. '
                    'Please use the `thread(...)` method to obtain a first guess, '
                    'then call the `correct(...)` method again.')
            self._position_before_iter = position
            if i_iter == 0:
                self._position_before = position
            if verbose:
                print(
                    f'Trajectory correction - iter {i_iter}, rms: {position.std()}')

            if n_iter == 'auto':
                if i_iter > 0 and position.std() > (1. - stop_iter_factor) * pos_rms_prev:
                    break
                pos_rms_prev = position.std()

            correction = self._compute_correction(position=position,
                                    n_micado=n_micado, rcond=rcond,
                                    n_singular_values=n_singular_values)
            self._apply_correction(correction)

            i_iter += 1
            if n_iter != 'auto' and i_iter >= n_iter:
                break

        if _tw_orbit is None:
            position = self._measure_position()
            self._position_after = position
            if verbose:
                print(
                    f'Trajectory correction - iter {i_iter}, rms: {position.std()}')
        else:
            self._position_after = None

    def _compute_tw_orbit(self):
        if self.mode == 'open':
            # Initialized with betx=1, bety=1 (use W_matrix to avoid compilation)
            twinit = xt.TwissInit(W_matrix=np.eye(6),
                            particle_on_co=self.line.build_particles(
                                          x=self.x_init, y=self.y_init,
                                            px=self.px_init, py=self.py_init,
                                            zeta=self.zeta_init, delta=self.delta_init),
                            element_name=self.start)
        else:
            twinit = None
        tw_orbit = self.line.twiss4d(only_orbit=True, start=self.start, end=self.end,
                                     init=twinit, reverse=False)
        return tw_orbit

    def _measure_position(self, tw_orbit=None):

        if tw_orbit is None:
            tw_orbit = self._compute_tw_orbit()

        x_twiss_at_monitor = tw_orbit['x'][self._indices_monitor]
        y_twiss_at_monitor = tw_orbit['y'][self._indices_monitor]

        x_shifted = x_twiss_at_monitor - self.shift_x_monitors
        y_shifted = y_twiss_at_monitor - self.shift_y_monitors

        x_local = x_shifted * np.cos(self.rot_s_rad_monitors) + y_shifted * np.sin(self.rot_s_rad_monitors)
        y_local = -x_shifted * np.sin(self.rot_s_rad_monitors) + y_shifted * np.cos(self.rot_s_rad_monitors)

        position = {'x': x_local, 'y': y_local}[self.plane]

        return position

    def _compute_correction(self, position, n_micado=None,
                            n_singular_values=None, rcond=None):

        if rcond is None:
            rcond = self.rcond

        if n_singular_values is None:
            n_singular_values = self.n_singular_values

        if n_micado is None:
            n_micado = self.n_micado

        correction = _compute_correction(position,
            response_matrix=(self.singular_vectors_out, self.singular_values, self.singular_vectors_in),
            n_micado=n_micado,
            rcond=rcond, n_singular_values=n_singular_values)

        return correction

    def _add_correction_knobs(self):

        self.correction_knobs = []
        for nn_kick in self.corrector_names:
            corr_knob_name = f'orbit_corr_{nn_kick}_{self.plane}'
            assert hasattr(self.line[nn_kick], 'knl')
            assert hasattr(self.line[nn_kick], 'ksl')

            if corr_knob_name not in self.line.vars:
                self.line.vars[corr_knob_name] = 0

            if self.plane == 'x':
                if (self.line.element_refs[nn_kick].knl[0]._expr is None or
                    (self.line.vars[corr_knob_name]
                    not in self.line.element_refs[nn_kick].knl[0]._expr._get_dependencies())):
                    if self.line.element_refs[nn_kick].knl[0]._expr is not None:
                        self.line.element_refs[nn_kick].knl[0] -= ( # knl[0] is -kick
                            self.line.vars[f'orbit_corr_{nn_kick}_x'])
                    else:
                        # Workarond for https://github.com/xsuite/xsuite/issues/501
                        val = self.line.element_refs[nn_kick].knl[0]._value
                        if hasattr(val, 'get'):
                            val = val.get()
                        self.line.element_refs[nn_kick].knl[0] = val - (
                            self.line.vars[f'orbit_corr_{nn_kick}_x'])
            elif self.plane == 'y':
                if (self.line.element_refs[nn_kick].ksl[0]._expr is None or
                    (self.line.vars[corr_knob_name]
                    not in self.line.element_refs[nn_kick].ksl[0]._expr._get_dependencies())):
                    if  self.line.element_refs[nn_kick].ksl[0]._expr is not None:
                        self.line.element_refs[nn_kick].ksl[0] += ( # ksl[0] is +kick
                            self.line.vars[f'orbit_corr_{nn_kick}_y'])
                    else:
                        # Workarond for https://github.com/xsuite/xsuite/issues/501
                        val = self.line.element_refs[nn_kick].ksl[0]._value
                        if hasattr(val, 'get'):
                            val = val.get()
                        self.line.element_refs[nn_kick].ksl[0] = val + (
                            self.line.vars[f'orbit_corr_{nn_kick}_y'])

            self.correction_knobs.append(corr_knob_name)

    def _apply_correction(self, correction=None):

        if correction is None:
            correction = self.correction

        for nn_knob, kick in zip(self.correction_knobs, correction):
            self.line.vars[nn_knob] += kick

    def get_kick_values(self):
        return np.array([self.line.vv[nn_knob] for nn_knob in self.correction_knobs])

    def clear_correction_knobs(self):
        for nn_knob in self.correction_knobs:
            self.line.vars[nn_knob] = 0

class TrajectoryCorrection:

    def __init__(self, line,
                 start=None, end=None, twiss_table=None,
                 monitor_names_x=None, corrector_names_x=None,
                 monitor_names_y=None, corrector_names_y=None,
                 monitor_alignment=None,
                 x_init=0, px_init=0, y_init=0, py_init=0, zeta_init=0, delta_init=0,
                 n_micado=None, n_singular_values=None, rcond=None):

        '''
        Trajectory correction using linearized response matrix from optics
        table.

        Parameters
        ----------

        line : xtrack.Line
            Line object on which the trajectory correction is performed.
                start : str
            Start of the line range in which the correction is performed.
            If `start` is provided `end` must also be provided.
            If `start` is None, the correction is performed on the periodic
            solution (closed orbit).
        end : str
            End of the line range in which the correction is performed.
            If `end` is provided `start` must also be provided.
            If `start` is None, the correction is performed on the periodic
            solution (closed orbit).
        twiss_table : TwissTable
            Twiss table used to compute the response matrix for the correction.
            If None, the twiss table is computed from the line.
        monitor_names_x : list of str
            List of elements used as monitors in the horizontal plane.
        corrector_names_x : list of str
            List of elements used as correctors in the horizontal plane. They
            must have `knl` and `ksl` attributes.
        monitor_names_y : list of str
            List of elements used as monitors in the vertical plane.
        corrector_names_y : list of str
            List of elements used as correctors in the vertical plane. They
            must have `knl` and `ksl` attributes.
        n_micado : int
            If `n_micado` is not None, the MICADO algorithm is used for the
            correction. In that case, the number of correctors to be used is
            given by `n_micado`.
        n_singular_values : int
            Number of singular values used for the correction.
        rcond : float
            Cutoff for small singular values (relative to the largest singular
            value). Singular values smaller than `rcond` are considered zero.
        '''

        if isinstance(rcond, (tuple, list)):
            rcond_x, rcond_y = rcond
        else:
            rcond_x, rcond_y = rcond, rcond

        if isinstance(n_singular_values, (tuple, list)):
            n_singular_values_x, n_singular_values_y = n_singular_values
        else:
            n_singular_values_x, n_singular_values_y = n_singular_values, n_singular_values

        if isinstance(n_micado, (tuple, list)):
            n_micado_x, n_micado_y = n_micado
        else:
            n_micado_x, n_micado_y = n_micado, n_micado

        self.x_init = x_init
        self.px_init = px_init
        self.y_init = y_init
        self.py_init = py_init
        self.zeta_init = zeta_init
        self.delta_init = delta_init

        self.monitor_alignment = monitor_alignment

        if (monitor_names_x is not None or corrector_names_x is not None
            or line.steering_correctors_x is not None
            or line.steering_monitors_x is not None):
            self.x_correction = OrbitCorrectionSinglePlane(
                line=line, plane='x', monitor_names=monitor_names_x,
                corrector_names=corrector_names_x, start=start, end=end,
                twiss_table=twiss_table, n_micado=n_micado_x,
                n_singular_values=n_singular_values_x, rcond=rcond_x,
                x_init=x_init, px_init=px_init, y_init=y_init, py_init=py_init,
                zeta_init=zeta_init, delta_init=delta_init,
                monitor_alignment=monitor_alignment)
        else:
            self.x_correction = None

        if (monitor_names_y is not None or corrector_names_y is not None
           or line.steering_correctors_y is not None
           or line.steering_monitors_y is not None):
            self.y_correction = OrbitCorrectionSinglePlane(
                line=line, plane='y', monitor_names=monitor_names_y,
                corrector_names=corrector_names_y, start=start, end=end,
                twiss_table=twiss_table, n_micado=n_micado_y,
                n_singular_values=n_singular_values_y, rcond=rcond_y,
                x_init=x_init, px_init=px_init, y_init=y_init, py_init=py_init,
                zeta_init=zeta_init, delta_init=delta_init,
                monitor_alignment=monitor_alignment)
        else:
            self.y_correction = None

    def correct(self, planes=None, n_micado=None, n_singular_values=None,
                rcond=None, n_iter='auto', verbose=True, stop_iter_factor=0.1,
                tol_position_std=1e-10):

        '''
        Correct the trajectory in the horizontal and/or vertical plane.

        Parameters
        ----------
        planes : str
            Plane(s) in which the correction is performed. Possible values are
            'x', 'y', 'xy'.
        n_micado : int or tuple of int
            If `n_micado` is not None, the MICADO algorithm is used for the
            correction. In that case, the number of correctors to be used is
            given by `n_micado`.
        n_singular_values : int or tuple of int
            Number of singular values used for the correction.
        rcond : float or tuple of float
            Cutoff for small singular values (relative to the largest singular
            value). Singular values smaller than `rcond` are considered zero.
        n_iter : int or 'auto'
            Number of iterations for the correction. If 'auto', the correction
            stops when the rms of the position does not decrease by more than
            `stop_iter_factor` with respect to the previous iteration.
        verbose : bool
            If True, print the rms of the position at each iteration.
        stop_iter_factor : float
            If `n_iter` is 'auto', the correction stops when the rms of the
            position does not decrease by more than `stop_iter_factor` with
            respect to the previous iteration.
        '''

        assert n_iter == 'auto' or np.isscalar(n_iter)
        if n_iter == 'auto':
            assert stop_iter_factor > 0
            assert stop_iter_factor < 1

        if isinstance(rcond, (tuple, list)):
            rcond_x, rcond_y = rcond
        else:
            rcond_x, rcond_y = rcond, rcond

        if isinstance(n_singular_values, (tuple, list)):
            n_singular_values_x, n_singular_values_y = n_singular_values
        else:
            n_singular_values_x, n_singular_values_y = n_singular_values, n_singular_values

        if isinstance(n_micado, (tuple, list)):
            n_micado_x, n_micado_y = n_micado
        else:
            n_micado_x, n_micado_y = n_micado, n_micado

        if planes is None:
            planes = 'xy'
        assert planes in ['x', 'y', 'xy']

        i_iter = 0
        stop_x = self.x_correction is None or 'x' not in planes
        stop_y = self.y_correction is None or 'y' not in planes

        if stop_x and stop_y:
            return

        if self.x_correction is not None:
            a_correction = self.x_correction
        if self.y_correction is not None:
            a_correction = self.y_correction

        tw_orbit = a_correction._compute_tw_orbit()

        while True:

            if self.x_correction is not None and 'x' in planes:
                self.x_correction.correct(n_micado=n_micado_x,
                            n_singular_values=n_singular_values_x,
                            rcond=rcond_x, verbose=False, n_iter=1,
                            _tw_orbit=tw_orbit)

            if self.y_correction is not None and 'y' in planes:
                self.y_correction.correct(n_micado=n_micado_y,
                            n_singular_values=n_singular_values_y,
                            rcond=rcond_y, verbose=False, n_iter=1,
                            _tw_orbit=tw_orbit)

            tw_orbit_prev = tw_orbit
            tw_orbit = a_correction._compute_tw_orbit()

            if n_iter == 'auto' and self.x_correction is not None and 'x' in planes:
                new_position = self.x_correction._measure_position(tw_orbit)
                old_position = self.x_correction._measure_position(tw_orbit_prev)
                stop_x = (new_position.std() < tol_position_std or
                    new_position.std() > (1. - stop_iter_factor) * old_position.std())

            if n_iter == 'auto' and self.y_correction is not None and 'y' in planes:
                new_position = self.y_correction._measure_position(tw_orbit)
                old_position = self.y_correction._measure_position(tw_orbit_prev)
                stop_y = (new_position.std() < tol_position_std or
                    new_position.std() > (1. - stop_iter_factor) * old_position.std())

            if verbose:
                str_2print = f'Iteration {i_iter}, '
                if self.x_correction is not None and 'x' in planes:
                    new_position = self.x_correction._measure_position(tw_orbit)
                    old_position = self.x_correction._measure_position(tw_orbit_prev)
                    str_2print += (f'x_rms: {old_position.std():.2e}'
                        f' -> {new_position.std():.2e}, ')
                if self.y_correction is not None and 'y' in planes:
                    new_position = self.y_correction._measure_position(tw_orbit)
                    old_position = self.y_correction._measure_position(tw_orbit_prev)
                    str_2print += (f'y_rms: {old_position.std():.2e}'
                        f' -> {new_position.std():.2e}')
                print(str_2print)
            if stop_x and stop_y:
                break
            i_iter += 1
            if n_iter != 'auto' and i_iter >= n_iter:
                break

    def thread(self, ds_thread=None, rcond_short=None, rcond_long=None):

        '''
        Thread the trajectory along the line. The correction is performed in
        portions of length `ds_thread`. For each portion the correction is
        firs performed only on the new added part, then on the whole portion up
        to the end of the new added part.

        Parameters
        ----------
        ds_thread : float
            Length of the portion added at each iteration.
        rcond_long : float or tuple of float
            Cutoff for small singular values (relative to the largest singular
            value) used for the correction of the whole portion up to the end
            of the new added part.
        '''

        if self.start is not None or self.end is not None:
            raise NotImplementedError('Thread not implemented for line portions')

        threader = _thread(line=self.line, ds_thread=ds_thread, twiss_table=self.twiss_table,
                rcond_short=rcond_short, rcond_long=rcond_long,
                monitor_names_x=self.x_correction.monitor_names,
                monitor_names_y=self.y_correction.monitor_names,
                corrector_names_x=self.x_correction.corrector_names,
                corrector_names_y=self.y_correction.corrector_names,
                monitor_alignment=self.monitor_alignment)
        return threader

    def clear_correction_knobs(self):

        '''
        Set all correction knobs to zero. Erases all applied corrections.
        '''

        if self.x_correction is not None:
            self.x_correction.clear_correction_knobs()
        if self.y_correction is not None:
            self.y_correction.clear_correction_knobs()

    @property
    def start(self):
        if self.x_correction is not None:
            x_start = self.x_correction.start
        else:
            x_start = None
        if self.y_correction is not None:
            y_start = self.y_correction.start
        else:
            y_start = None

        if x_start is None:
            return y_start
        if y_start is None:
            return x_start
        if x_start is not None and y_start is not None:
            assert x_start == y_start
            return x_start

    @property
    def end(self):
        if self.x_correction is not None:
            x_end = self.x_correction.end
        else:
            x_end = None
        if self.y_correction is not None:
            y_end = self.y_correction.end
        else:
            y_end = None

        if x_end is None:
            return y_end
        if y_end is None:
            return x_end
        if x_end is not None and y_end is not None:
            assert x_end == y_end
            return x_end
    @property
    def line(self):
        if self.x_correction is not None:
            return self.x_correction.line
        if self.y_correction is not None:
            return self.y_correction.line

    @property
    def twiss_table(self):
        if self.x_correction is not None:
            return self.x_correction.twiss_table
        if self.y_correction is not None:
            return self.y_correction.twiss_table

def _thread(line, ds_thread, twiss_table=None, rcond_short = None, rcond_long = None,
            monitor_names_x=None, monitor_names_y=None,
            corrector_names_x=None, corrector_names_y=None,
            x_init=0, px_init=0, y_init=0, py_init=0, zeta_init=0, delta_init=0,
            monitor_alignment=None,
            verbose=True):

    assert ds_thread is not None

    # r_cond_short is not used anymore, see commented code below

    tt = line.get_table()
    line_length = tt.s[-1]

    if monitor_names_x is None:
        monitor_names_x = line.steering_monitors_x

    if monitor_names_y is None:
        monitor_names_y = line.steering_monitors_y

    if corrector_names_x is None:
        corrector_names_x = line.steering_correctors_x

    if corrector_names_y is None:
        corrector_names_y = line.steering_correctors_y

    i_win = 0
    end_loop = False
    s_corr_end = ds_thread
    while not end_loop:

        if s_corr_end > line_length:
            s_corr_end = line_length
            end_loop = True

        # ----- The following was used to correct only the newly added part
        # ----- It us not used anymore, as it was observed not to help
        # ----- We keep it in case it is needed in the futures
        # Correct only the new added portion
        # tt_new_part = tt.rows[s_corr_end-ds_thread:s_corr_end:'s']
        #
        # Get initial conditions for the new added portion
        # if i_win == 0:
        #     this_x_init = x_init
        #     this_y_init = y_init
        #     this_px_init = px_init
        #     this_py_init = py_init
        #     this_zeta_init = zeta_init
        #     this_delta_init = delta_init
        # else:
        #     # Initialized with betx=1, bety=1 (use W_matrix to avoid compilation)
        #     name_start = tt_new_part.name[0]
        #     tw_to_start = line.twiss4d(
        #         start=line.element_names[0], end=tt_new_part.name[0],
        #             init=xt.TwissInit(W_matrix=np.eye(6),
        #                                 particle_on_co=line.build_particles(
        #                                     x=x_init, y=y_init,
        #                                     px=px_init, py=py_init,
        #                                     zeta=zeta_init, delta=delta_init),
        #                                 element_name=line.element_names[0]),
        #                                 reverse=False)

        #     this_x_init = tw_to_start['x', name_start]
        #     this_y_init = tw_to_start['y', name_start]
        #     this_px_init = tw_to_start['px', name_start]
        #     this_py_init = tw_to_start['py', name_start]
        #     this_zeta_init = tw_to_start['zeta', name_start]
        #     this_delta_init = tw_to_start['delta', name_start]

        # ocorr_only_added_part = TrajectoryCorrection(
        #     line=line, start=tt_new_part.name[0], end=tt_new_part.name[-1],
        #     twiss_table=twiss_table,
        #     monitor_names_x=[nn for nn in monitor_names_x if nn in tt_new_part.name],
        #     monitor_names_y=[nn for nn in monitor_names_y if nn in tt_new_part.name],
        #     corrector_names_x=[nn for nn in corrector_names_x if nn in tt_new_part.name],
        #     corrector_names_y=[nn for nn in corrector_names_y if nn in tt_new_part.name],
        #     x_init=this_x_init, px_init=this_px_init,
        #     y_init=this_y_init, py_init=this_py_init,
        #     zeta_init=this_zeta_init, delta_init=this_delta_init,
        #     monitor_alignment=monitor_alignment
        # )
        # ocorr_only_added_part.correct(rcond=rcond_short, n_iter=1, verbose=False)

        # if verbose:
        #     ocprint = ocorr_only_added_part
        #     tw_orbit_print = ocprint.x_correction._compute_tw_orbit()
        #     x_meas_print = ocprint.x_correction._measure_position(tw_orbit_print)
        #     y_meas_print = ocprint.y_correction._measure_position(tw_orbit_print)
        #     str_2print = f'Stop at s={s_corr_end}, '
        #     str_2print += 'local rms  = ['
        #     str_2print += (f'x: {ocprint.x_correction._position_before.std():.2e}'
        #         f' -> {x_meas_print.std():.2e}, ')
        #     str_2print += (f'y: {ocprint.y_correction._position_before.std():.2e}'
        #         f' -> {y_meas_print.std():.2e}]')
        #     print(str_2print)

        # Correct from start line to end of new added portion
        tt_part = tt.rows[0:s_corr_end:'s']
        ocorr = TrajectoryCorrection(
            twiss_table=twiss_table,
            line=line, start=tt_part.name[0], end=tt_part.name[-1],
            monitor_names_x=[nn for nn in monitor_names_x if nn in tt_part.name],
            monitor_names_y=[nn for nn in monitor_names_y if nn in tt_part.name],
            corrector_names_x=[nn for nn in corrector_names_x if nn in tt_part.name],
            corrector_names_y=[nn for nn in corrector_names_y if nn in tt_part.name],
            x_init=x_init, px_init=px_init, y_init=y_init, py_init=py_init,
            zeta_init=zeta_init, delta_init=delta_init,
            monitor_alignment=monitor_alignment
        )
        ocorr.correct(rcond=rcond_long, n_iter=1, verbose=False)

        if verbose:
            ocprint = ocorr
            tw_orbit_print = ocprint.x_correction._compute_tw_orbit()
            x_meas_print = ocprint.x_correction._measure_position(tw_orbit_print)
            y_meas_print = ocprint.y_correction._measure_position(tw_orbit_print)
            str_2print = f'Stop at s={s_corr_end}, '
            str_2print += 'global rms = ['
            str_2print += (f'x: {ocprint.x_correction._position_before_iter.std():.2e}'
                f' -> {x_meas_print.std():.2e}, ')
            str_2print += (f'y: {ocprint.y_correction._position_before_iter.std():.2e}'
                f' -> {y_meas_print.std():.2e}]')
            print(str_2print)

        s_corr_end += ds_thread
        i_win += 1

    return ocorr
