import numpy as np

def _compute_correction_micado(x_iter, response_matrix_x, n_micado):

    n_hcorrectors = response_matrix_x.shape[1]

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
    # Compute the correction with least squares
    correction_masked, residual_x, rank_x, sval_x = np.linalg.lstsq(
                response_matrix_x[:, mask_corr], -x_iter, rcond=None)
    correction_x = np.zeros(n_hcorrectors)
    correction_x[mask_corr] = correction_masked

    return correction_x