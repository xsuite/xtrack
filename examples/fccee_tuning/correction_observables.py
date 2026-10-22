"""Observables and scalar diagnostics for the FCC-ee correction example."""

import numpy as np


def beta_errors(tw, *, tw_ref):
    return [tw['bet' + p] / tw_ref['bet' + p] - 1 for p in 'xy']


def horizontal_dispersion_errors(tw, *, tw_ref, dispersion_indices):
    # Ideal model observable, not a noisy RF scan. Preserve nominal Dx.
    return (tw.dx - tw_ref.dx)[dispersion_indices]


def optics_vector(tw, *, tw_ref, obs_indices, alpha_indices, dispersion_indices):
    return np.r_[
        *[np.log(tw['bet' + p][obs_indices] / tw_ref['bet' + p][obs_indices]) for p in 'xy'],
        *[
            (
                tw['alf' + p][alpha_indices]
                * tw_ref['bet' + p][alpha_indices]
                / tw['bet' + p][alpha_indices]
                - tw_ref['alf' + p][alpha_indices]
            )
            for p in 'xy'
        ],
        horizontal_dispersion_errors(tw, tw_ref=tw_ref, dispersion_indices=dispersion_indices),
        tw.qx - tw_ref.qx,
        tw.qy - tw_ref.qy,
    ]


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


def coupling_rms(tw, *, tw_ref, coupling_indices):
    return np.array(
        [
            np.sqrt(np.mean(np.abs(tw[key][coupling_indices] - tw_ref[key][coupling_indices]) ** 2))
            for key in ('f1001', 'f1010')
        ]
    )


def coupling_ip_max(tw, *, tw_ref, ip_indices):
    return max(
        np.max(np.abs(tw[key][ip_indices] - tw_ref[key][ip_indices])) for key in ('f1001', 'f1010')
    )


def orbit_errors(tw, fraction=None, *, monitor_indices, monitor_alignment):
    dx, dy, roll = monitor_alignment()
    x, y = tw.x[monitor_indices] - dx, tw.y[monitor_indices] - dy
    return [x * np.cos(roll) + y * np.sin(roll), -x * np.sin(roll) + y * np.cos(roll)]


def lab_orbit_errors(tw, *, monitor_indices, monitor_alignment):
    dx, dy, _ = monitor_alignment()
    return [tw.x[monitor_indices] - dx, tw.y[monitor_indices] - dy]


def correctable_orbit_errors(tw, *, lab_orbit_errors, orbit_svds, rcond):
    # Discarded SVD modes and the rectangular response's left null space
    # cannot be removed by the available correctors in the nominal model.
    residuals = []
    for error, svd in zip(lab_orbit_errors(tw), orbit_svds):
        u = svd.U[:, svd.s > rcond * svd.s[0]]
        residuals.append(u @ (u.T @ error))
    return residuals


def rms_arrays(arrays):
    return np.array([np.sqrt(np.mean(a * a)) for a in arrays])


def measure(tw, *, tw_ref, dispersion_indices, ip_indices, orbit_errors):
    """Diagnostics used for stopping criteria, history and progress reports."""
    beta = beta_errors(tw, tw_ref=tw_ref)
    dx, dy = tw.dx - tw_ref.dx, tw.dy - tw_ref.dy
    return dict(
        orbit_rms=rms_arrays(orbit_errors(tw)) if orbit_errors is not None else None,
        beta_rms=rms_arrays(beta),
        beta_max=np.array([np.max(np.abs(e)) for e in beta]),
        tunes=np.array([tw.qx, tw.qy]),
        tune_error=np.array([tw.qx - tw_ref.qx, tw.qy - tw_ref.qy]),
        dx_rms=float(np.sqrt(np.mean(dx[dispersion_indices] ** 2))),
        dx_ip=float(np.max(np.abs(dx[ip_indices]))),
        dy_rms=float(np.sqrt(np.mean(dy[dispersion_indices] ** 2))),
        dy_ip=float(np.max(np.abs(dy[ip_indices]))),
        coupling_rms=coupling_rms(tw, tw_ref=tw_ref, coupling_indices=dispersion_indices),
        coupling_ip=coupling_ip_max(tw, tw_ref=tw_ref, ip_indices=ip_indices),
        c_minus=float(tw.c_minus),
    )
