import numpy as np
from numpy.matlib import repmat

def _compute_correction(x_iter, response_matrix_x, n_micado=None, rcond=None):

    n_hcorrectors = response_matrix_x.shape[1]

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
                            response_matrix_x[:, mask_corr], -x_iter, rcond=rcond)
                residuals.append(residual_x[0])
            used_correctors.append(np.nanargmin(residuals))

        mask_corr = np.zeros(n_hcorrectors, dtype=bool)
        mask_corr[used_correctors] = True
    else:
        mask_corr = np.ones(n_hcorrectors, dtype=bool)
        mask_corr[:] = True

    # Compute the correction with least squares
    correction_masked, residual_x, rank_x, sval_x = np.linalg.lstsq(
                response_matrix_x[:, mask_corr], -x_iter, rcond=rcond)
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


class OrbitCorrection:

    def __init__(self, line, plane, monitor_names, corrector_names,
                 start=None, end=None, twiss_table=None, n_micado=None,
                 rcond=None):

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

        self.s_correctors = self.twiss_table.rows[self.corrector_names].s
        self.s_monitors = self.twiss_table.rows[self.monitor_names].s

        self._add_correction_knobs()

    def correct(self):
        self._measure_position()
        self._compute_correction()
        self._apply_correction()

    def _measure_position(self):
        if self.mode == 'open':
            betx=1
        else:
            betx=None
        tw_orbit = self.line.twiss4d(only_orbit=True, start=self.start, end=self.end,
                                     betx=betx, bety=betx)

        self.position = tw_orbit.rows[self.monitor_names][self.plane]

    def _compute_correction(self, position=None, n_micado=None):

        if n_micado is None:
            n_micado = self.n_micado

        if position is None:
            position = self.position

        self.correction = _compute_correction(position, self.response_matrix, n_micado,
                                              rcond=self.rcond)

    def _add_correction_knobs(self):

        self.correction_knobs = []
        for nn_kick in self.corrector_names:
            corr_knob_name = f'orbit_corr_{nn_kick}'
            assert hasattr(self.line[nn_kick], 'knl')
            assert hasattr(self.line[nn_kick], 'ksl')

            if corr_knob_name not in self.line.vars:
                self.line.vars[corr_knob_name] = 0

            if self.plane == 'x':
                if (self.line.element_refs[nn_kick].knl[0]._expr is None or
                    (self.line.vars[corr_knob_name]
                    not in self.line.element_refs[nn_kick].knl[0]._expr._get_dependencies())):
                    self.line.element_refs[nn_kick].knl[0] -= ( # knl[0] is -kick
                        self.line.vars[f'orbit_corr_{nn_kick}'])
            elif self.plane == 'y':
                if (self.line.element_refs[nn_kick].ksl[0]._expr is None or
                    (self.line.vars[corr_knob_name]
                    not in self.line.element_refs[nn_kick].ksl[0]._expr._get_dependencies())):
                    self.line.element_refs[nn_kick].ksl[0] += ( # ksl[0] is +kick
                        self.line.vars[f'orbit_corr_{nn_kick}'])

            self.correction_knobs.append(corr_knob_name)

    def _apply_correction(self, correction=None):

        if correction is None:
            correction = self.correction

        for nn_knob, kick in zip(self.correction_knobs, correction):
            self.line.vars[nn_knob] += kick

    def get_kick_values(self):
        return np.array([self.line.vv[nn_knob] for nn_knob in self.correction_knobs])