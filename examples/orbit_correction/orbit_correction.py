import numpy as np
from numpy.matlib import repmat

def _compute_correction(x_iter, response_matrix_x, n_micado=None):

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
                            response_matrix_x[:, mask_corr], -x_iter, rcond=None)
                residuals.append(residual_x[0])
            used_correctors.append(np.nanargmin(residuals))

        mask_corr = np.zeros(n_hcorrectors, dtype=bool)
        mask_corr[used_correctors] = True
    else:
        mask_corr = np.ones(n_hcorrectors, dtype=bool)
        mask_corr[:] = True

    # Compute the correction with least squares
    correction_masked, residual_x, rank_x, sval_x = np.linalg.lstsq(
                response_matrix_x[:, mask_corr], -x_iter, rcond=None)
    correction_x = np.zeros(n_hcorrectors)
    correction_x[mask_corr] = correction_masked

    return correction_x


def _build_response_matrix(tw, h_monitor_names, h_corrector_names,
                           mode='closed'):

    assert mode in ['closed', 'open']

    # Build response matrix
    betx_monitors = tw.rows[h_monitor_names].betx
    betx_correctors = tw.rows[h_corrector_names].betx

    mux_monitor = tw.rows[h_monitor_names].mux
    mux_correctors = tw.rows[h_corrector_names].mux

    n_h_monitors = len(h_monitor_names)
    n_hcorrectors = len(h_corrector_names)

    bet_prod_x = np.atleast_2d(betx_monitors).T @ np.atleast_2d(betx_correctors)
    mux_diff = (repmat(mux_monitor, n_hcorrectors, 1).T
                        - repmat(mux_correctors, n_h_monitors, 1))
    mux_diff[mux_diff < 0] = 0

    if mode == 'open':
        # Wille eq. 3.164
        response_matrix_x = (np.sqrt(bet_prod_x) * np.sin(2*np.pi*np.abs(mux_diff)))
    elif mode == 'closed':
        # Slide 28
        # https://indico.cern.ch/event/1328128/contributions/5589794/attachments/2786478/4858384/linearimperfections_2024.pdf
        tune = tw.qx
        response_matrix_x = (np.sqrt(bet_prod_x) / 2 / np.sin(np.pi * tune)
                             * np.cos(np.pi * tune - 2*np.pi*np.abs(mux_diff)))

    return response_matrix_x


class OrbitCorrection:

    def __init__(self, line, h_monitor_names, h_corrector_names):

        self.line = line
        self.h_monitor_names = h_monitor_names
        self.h_corrector_names = h_corrector_names

    def add_correction_knobs(self):

        self.h_correction_knobs = []
        for nn_kick in self.h_corrector_names:
            corr_knob_name = f'orbit_corr_{nn_kick}'
            assert hasattr(self.line[nn_kick], 'knl')

            if corr_knob_name not in self.line.vars:
                self.line.vars[corr_knob_name] = 0

            if (self.line.element_refs[nn_kick].knl[0]._expr is None or
                  (self.line.vars[corr_knob_name]
                   not in self.line.element_refs[nn_kick].knl[0]._expr._get_dependencies())):
                self.line.element_refs[nn_kick].knl[0] += (
                    self.line.vars[f'orbit_corr_{nn_kick}'])
            self.h_correction_knobs.append(corr_knob_name)

    def apply_correction(self, correction_x):

        for nn_knob, kick in zip(self.h_correction_knobs, correction_x):
            self.line.vars[nn_knob] -= kick # knl[0] is -kick

    def get_kick_values(self):
        return np.array([self.line.vv[nn_knob] for nn_knob in self.h_correction_knobs])