"""Residual vector for checking the analytical coupling response."""

import numpy as np


def coupling_vector(tw, *, tw_ref, coupling_indices):
    return np.concatenate(
        [
            part
            for key in ('f1001', 'f1010')
            for part in (
                (tw[key] - tw_ref[key])[coupling_indices].real,
                (tw[key] - tw_ref[key])[coupling_indices].imag,
            )
        ]
        + [dispersion_errors(tw, tw_ref=tw_ref, coupling_indices=coupling_indices)]
    )


def dispersion_errors(tw, *, tw_ref, coupling_indices):
    # Ideal model observable, like the coupling RDTs; not a noisy RF scan.
    return (tw.dy - tw_ref.dy)[coupling_indices]
