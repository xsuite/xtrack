import numpy as np
from numpy.matlib import repmat

def _compute_correction(x_iter, response_matrix, n_micado=None, rcond=None,
                        n_singular_values=None):

    if isinstance(response_matrix, (list, tuple)):
        assert len(response_matrix) == 3 # U, S, Vt
        U, S, Vt = response_matrix
        if n_singular_values is not None:
            U = U[:, :n_singular_values]
            S = S[:n_singular_values]
            Vt = Vt[:n_singular_values, :]
        response_matrix = U @ np.diag(S) @ Vt
    else:
        assert n_singular_values is None

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
    bet_monitors = tw.rows[monitor_names]['bet' + plane]
    bet_correctors = tw.rows[corrector_names]['bet' + plane]

    mu_monitor = tw.rows[monitor_names]['mu' + plane]
    mux_correctors = tw.rows[corrector_names]['mu' + plane]

    n_monitors = len(monitor_names)
    n_correctors = len(corrector_names)

    bet_prod = np.atleast_2d(bet_monitors).T @ np.atleast_2d(bet_correctors)
    mu_diff = (repmat(mu_monitor, n_correctors, 1).T
                        - repmat(mux_correctors, n_monitors, 1))

    if mode == 'open':
        # Wille eq. 3.164
        mu_diff[mu_diff < 0] = 0 # use only correctors upstream of the monitor
        response_matrix = (np.sqrt(bet_prod) * np.sin(2*np.pi*np.abs(mu_diff)))
    elif mode == 'closed':
        # Slide 28
        # https://indico.cern.ch/event/1328128/contributions/5589794/attachments/2786478/4858384/linearimperfections_2024.pdf
        tune = tw.qx
        response_matrix = (np.sqrt(bet_prod) / 2 / np.sin(np.pi * tune)
                             * np.cos(np.pi * tune - 2*np.pi*np.abs(mu_diff)))

    return response_matrix


class OrbitCorrectionSinglePlane:

    def __init__(self, line, plane, monitor_names, corrector_names,
                 start=None, end=None, twiss_table=None, n_micado=None,
                 n_singular_values=None, rcond=None):

        assert plane in ['x', 'y']

        self.line = line
        self.plane = plane
        self.monitor_names = monitor_names
        self.corrector_names = corrector_names
        self.start = start
        self.end = end
        self.twiss_table = twiss_table
        self.n_micado = n_micado
        self.rcond = rcond
        self.n_singular_values = n_singular_values

        if start is None:
            assert end is None
            self.mode = 'closed'
            if self.twiss_table is None:
                self.twiss_table = line.twiss4d()
        else:
            assert end is not None
            self.mode = 'open'
            if self.twiss_table is None:
                self.twiss_table = line.twiss4d(start=start, end=end,
                                                betx=1, bety=1)

        self.response_matrix = _build_response_matrix(plane=self.plane,
            tw=self.twiss_table, monitor_names=self.monitor_names,
            corrector_names=self.corrector_names, mode=self.mode)

        U, S , Vt = np.linalg.svd(self.response_matrix, full_matrices=False)
        self.singular_values = S
        self.singular_vectors_out = U
        self.singular_vectors_in = Vt

        self.s_correctors = self.twiss_table.rows[self.corrector_names].s
        self.s_monitors = self.twiss_table.rows[self.monitor_names].s

        self._add_correction_knobs()

    def correct(self, n_micado=None, n_singular_values=None, rcond=None):
        self._measure_position()
        self._compute_correction(n_micado=n_micado, rcond=rcond,
                                 n_singular_values=n_singular_values)
        self._apply_correction()

    def _measure_position(self):
        if self.mode == 'open':
            betx=1
        else:
            betx=None
        tw_orbit = self.line.twiss4d(only_orbit=True, start=self.start, end=self.end,
                                     betx=betx, bety=betx)

        self.position = tw_orbit.rows[self.monitor_names][self.plane]

    def _compute_correction(self, position=None, n_micado=None,
                            n_singular_values=None, rcond=None):

        if rcond is None:
            rcond = self.rcond

        if n_singular_values is None:
            n_singular_values = self.n_singular_values

        if n_micado is None:
            n_micado = self.n_micado

        if position is None:
            position = self.position

        self.correction = _compute_correction(position,
            response_matrix=(self.singular_vectors_out, self.singular_values, self.singular_vectors_in),
            n_micado=n_micado,
            rcond=rcond, n_singular_values=n_singular_values)

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
                    self.line.element_refs[nn_kick].knl[0] -= ( # knl[0] is -kick
                        self.line.vars[f'orbit_corr_{nn_kick}_x'])
            elif self.plane == 'y':
                if (self.line.element_refs[nn_kick].ksl[0]._expr is None or
                    (self.line.vars[corr_knob_name]
                    not in self.line.element_refs[nn_kick].ksl[0]._expr._get_dependencies())):
                    self.line.element_refs[nn_kick].ksl[0] += ( # ksl[0] is +kick
                        self.line.vars[f'orbit_corr_{nn_kick}_y'])

            self.correction_knobs.append(corr_knob_name)

    def _apply_correction(self, correction=None):

        if correction is None:
            correction = self.correction

        for nn_knob, kick in zip(self.correction_knobs, correction):
            self.line.vars[nn_knob] += kick

    def get_kick_values(self):
        return np.array([self.line.vv[nn_knob] for nn_knob in self.correction_knobs])

    def _clean_correction_knobs(self):
        for nn_knob in self.correction_knobs:
            self.line.vars[nn_knob] = 0

class OrbitCorrection:

    def __init__(self, line,
                 start=None, end=None, twiss_table=None,
                 monitor_names_x=None, corrector_names_x=None,
                 monitor_names_y=None, corrector_names_y=None,
                 n_micado=None, n_singular_values=None, rcond=None):

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

        if monitor_names_x is not None or corrector_names_x is not None:
            if monitor_names_x is None:
                raise ValueError('monitor_names_x must be provided when '
                                 'corrector_names_x is provided')
            if corrector_names_x is None:
                raise ValueError('corrector_names_x must be provided when '
                                 'monitor_names_x is provided')
            self.x_correction = OrbitCorrectionSinglePlane(
                line=line, plane='x', monitor_names=monitor_names_x,
                corrector_names=corrector_names_x, start=start, end=end,
                twiss_table=twiss_table, n_micado=n_micado_x,
                n_singular_values=n_singular_values_x, rcond=rcond_x)
        else:
            self.x_correction = None

        if monitor_names_y is not None or corrector_names_y is not None:
            if monitor_names_y is None:
                raise ValueError('monitor_names_y must be provided when '
                                 'corrector_names_y is provided')
            if corrector_names_y is None:
                raise ValueError('corrector_names_y must be provided when '
                                 'monitor_names_y is provided')
            self.y_correction = OrbitCorrectionSinglePlane(
                line=line, plane='y', monitor_names=monitor_names_y,
                corrector_names=corrector_names_y, start=start, end=end,
                twiss_table=twiss_table, n_micado=n_micado_y,
                n_singular_values=n_singular_values_y, rcond=rcond_y)
        else:
            self.y_correction = None

    def correct(self, planes=None, n_micado=None, n_singular_values=None,
                rcond=None):

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
        if self.x_correction is not None and 'x' in planes:
            self.x_correction.correct(n_micado=n_micado_x,
                                      n_singular_values=n_singular_values_x,
                                      rcond=rcond_x)
        if self.y_correction is not None and 'y' in planes:
            self.y_correction.correct(n_micado=n_micado_y,
                                      n_singular_values=n_singular_values_y,
                                      rcond=rcond_y)
