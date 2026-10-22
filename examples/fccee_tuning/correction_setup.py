"""Reference optics and response preparation for 003b.

The functions return ordinary Twiss tables, name lists and NumPy arrays.
Optimizer construction, SVD cutoffs and correction policy stay in 003b.
Only nominal_reference and the tracking probes change the line's settings.
"""

from functools import partial

import numpy as np

import correction_observables as obs
from correction_steps import set_values, values
from response_builders import normal_quadrupole_response, skew_quadrupole_response


def nominal_reference(line, *, full_twiss, log):
    """Set nominal knobs and return the reference with nonlinear magnets off.

    The input is the line saved by 003a, with nonlinear magnets on. Check that
    switching them off leaves the aligned optics unchanged, so the same fixed
    responses can be used throughout the ramp. Leave errors/nonlinearities off.
    """
    circuits = line.metadata['correction_knobs']
    orbit_knobs = circuits['orbit']['x']['knobs'] + circuits['orbit']['y']['knobs']
    set_values(line, orbit_knobs, np.zeros(len(orbit_knobs)))
    line['on_errors'] = 0.0
    for name in circuits['tunes']['knobs']:
        line[name] = circuits['tunes']['nominal_values'][name]
    line.twiss_default['co_guess'] = dict(x=0, px=0, y=0, py=0, zeta=0, delta=0)
    tw_nonlinear_on = full_twiss()
    line['on_nonlinear'] = 0.0
    tw_ref = full_twiss()
    nominal_beta_change = max(
        np.max(np.abs(tw_nonlinear_on['bet' + p] / tw_ref['bet' + p] - 1)) for p in 'xy'
    )
    log('Nominal nonlinear on/off: max beta change', nominal_beta_change)
    if nominal_beta_change > 1e-4:
        raise RuntimeError('Aligned optics depend on nonlinear setting; rebuild responses per stage')
    return tw_ref


def optics_observation_names(tw_ref, quadrupoles, interaction_points, *, phase_step):
    """Return beta and alpha observation names in their response-row order.

    Beta observations add phase samples in both planes: quadrupole entrances
    alone can hide beating in the intervening drifts and interaction regions.
    Alpha is observed at quadrupoles and interaction points.
    """
    selected = set(tw_ref.rows.indices[quadrupoles]) | set(
        tw_ref.rows.indices[interaction_points]
    )
    for plane in 'xy':
        mu = tw_ref['mu' + plane]
        selected.update(np.searchsorted(mu, np.arange(mu[0], mu[-1], phase_step)).tolist())
    return (
        list(tw_ref.name[np.array(sorted(selected))]),
        list(dict.fromkeys(quadrupoles + interaction_points)),
    )


def prepare_optics_response(
    line, tw_ref, quadrupoles, beta_at, alpha_at, dispersion_at, *,
    interaction_points, dispersion_ips, alpha_weight, ir_beta_weight,
    ir_half_length, dispersion_weight, dispersion_ip_weight, knobs, full_twiss, log,
):
    """Return matrix, knob scales, target weights and labels for optics/Dx/tunes.

    The matrix contains unweighted physical derivatives. RMS normalization and
    interaction-region emphasis are returned separately as Target weights.
    Check the response with two tracking probes, restoring the nominal trims.
    """
    matrix, scales = normal_quadrupole_response(
        line, tw_ref, quadrupoles, beta_at, alpha_at, dispersion_at
    )
    dx_weights = dispersion_weight * np.array(
        [dispersion_ip_weight if n in dispersion_ips else 1.0 for n in dispersion_at]
    )
    dx_weights /= np.sqrt(len(dispersion_at))
    weights = np.r_[
        np.full(2 * len(beta_at), 1 / np.sqrt(len(beta_at))),
        np.full(2 * len(alpha_at), alpha_weight / np.sqrt(len(alpha_at))),
        dx_weights,
        1000.0,
        1000.0,
    ]
    beta_indices = tw_ref.rows.indices[beta_at]
    ip_s = tw_ref.s[tw_ref.rows.indices[interaction_points]]
    ip_distance = np.abs(tw_ref.s[beta_indices, None] - ip_s[None, :])
    near_ip = np.min(np.minimum(ip_distance, tw_ref.line_length - ip_distance), axis=1) < ir_half_length
    for k in range(2):
        weights[k * len(beta_at) : (k + 1) * len(beta_at)] *= np.where(
            near_ip, ir_beta_weight, 1.0
        )
    log(f'Upweighting beta at {np.sum(near_ip)} interaction-region observations')
    labels = (
        [f'log_beta_{p}@{n}' for p in 'xy' for n in beta_at]
        + [f'normalized_alpha_{p}@{n}' for p in 'xy' for n in alpha_at]
        + [f'dx@{n}' for n in dispersion_at]
        + ['qx', 'qy']
    )
    dispersion_indices = tw_ref.rows.indices[dispersion_at]
    check_optics_response(
        line, quad_knobs=knobs, strength_scale=scales, nominal_response=matrix,
        optics_weights=weights, tw_ref=tw_ref, full_twiss=full_twiss,
        optics_vector=partial(
            obs.optics_vector, tw_ref=tw_ref, obs_indices=beta_indices,
            alpha_indices=tw_ref.rows.indices[alpha_at], dispersion_indices=dispersion_indices,
        ),
        horizontal_dispersion_errors=partial(
            obs.horizontal_dispersion_errors, tw_ref=tw_ref, dispersion_indices=dispersion_indices,
        ),
        n_dispersion=len(dispersion_at), log=log,
    )
    return matrix, scales, weights, labels


def prepare_coupling_response(
    line, tw_ref, skew_groups, observation_names, *, ip_names,
    coupling_ip_weight, dispersion_weight, dispersion_ip_weight, knobs, full_twiss, log,
):
    """Return matrix, knob scales, target weights and labels for RDTs and Dy.

    Check the response with one tracking probe, restoring the nominal trims.
    """
    matrix, scales = skew_quadrupole_response(line, tw_ref, skew_groups, observation_names)
    weights = np.tile(
        np.array([coupling_ip_weight if n in ip_names else 1.0 for n in observation_names]), 4
    )
    weights = np.r_[
        weights,
        dispersion_weight
        * np.array([dispersion_ip_weight if n in ip_names else 1.0 for n in observation_names]),
    ]
    weights /= np.sqrt(len(observation_names))
    labels = [
        f'{part}_{rdt}@{n}'
        for rdt in ('f1001', 'f1010')
        for part in ('real', 'imag')
        for n in observation_names
    ] + [f'dy@{n}' for n in observation_names]
    check_coupling_response(
        line, skew_knobs=knobs, skew_scale=scales, coupling_response=matrix,
        tw_ref=tw_ref, full_twiss=full_twiss,
        coupling_vector=partial(
            obs.coupling_vector, tw_ref=tw_ref,
            coupling_indices=tw_ref.rows.indices[observation_names],
        ),
        n_observations=len(observation_names), log=log,
    )
    return matrix, scales, weights, labels


def build_tune_response(tw, *, line, tune_knobs, full_twiss):
    """Numerical two-family tune response, with all probe settings restored."""
    baseline = values(line, tune_knobs)
    jac = np.empty((2, 2))
    try:
        for j in range(2):
            line[tune_knobs[j]] = baseline[j] + 1e-7
            probe = full_twiss(tw)
            jac[:, j] = (np.array([probe.qx, probe.qy]) - np.array([tw.qx, tw.qy])) / 1e-7
            set_values(line, tune_knobs, baseline)
    finally:
        set_values(line, tune_knobs, baseline)
    return jac


def check_optics_response(
    line,
    *,
    quad_knobs,
    strength_scale,
    nominal_response,
    optics_weights,
    tw_ref,
    full_twiss,
    optics_vector,
    horizontal_dispersion_errors,
    n_dispersion,
    log,
):
    # A central distributed probe checks the sign, integration and scaling in two
    # Twiss calls, without probing every quadrupole. The central difference cancels
    # quadratic tune shifts, which can dominate the small net first-order shift.
    probe = 1e-4 * strength_scale * np.sin(np.arange(len(quad_knobs)) + 0.3)
    set_values(line, quad_knobs, probe)
    try:
        probe_plus = full_twiss(tw_ref)
        set_values(line, quad_knobs, -probe)
        probe_minus = full_twiss(tw_ref)
        measured = (optics_vector(probe_plus) - optics_vector(probe_minus)) / 2
        prediction = nominal_response @ probe
        discrepancy = np.linalg.norm(optics_weights * (measured - prediction)) / np.linalg.norm(
            optics_weights * prediction
        )
        dx_prediction = nominal_response[-n_dispersion - 2 : -2] @ probe
        dx_measured = (
            horizontal_dispersion_errors(probe_plus) - horizontal_dispersion_errors(probe_minus)
        ) / 2
        dx_discrepancy = np.linalg.norm(dx_measured - dx_prediction) / np.linalg.norm(dx_prediction)
        log(
            'Normal optics/Dx response directional check, relative errors:',
            discrepancy,
            dx_discrepancy,
        )
        if max(discrepancy, dx_discrepancy) > 0.05:
            raise RuntimeError('Normal optics/dispersion response disagrees with tracking')
    finally:
        set_values(line, quad_knobs, np.zeros(len(quad_knobs)))


def check_coupling_response(
    line,
    *,
    skew_knobs,
    skew_scale,
    coupling_response,
    tw_ref,
    full_twiss,
    coupling_vector,
    n_observations,
    log,
):
    # Check the analytical response against tracking for a distributed probe.
    probe = 1e-5 * skew_scale * np.sin(np.arange(len(skew_knobs)) + 0.3)
    set_values(line, skew_knobs, probe)
    try:
        probe_tw = full_twiss(tw_ref)
        prediction = coupling_response @ probe
        discrepancy = np.linalg.norm(coupling_vector(probe_tw) - prediction) / np.linalg.norm(
            prediction
        )
        log('Coupling response directional check, relative error:', discrepancy)
        n_rdt = 4 * n_observations
        dy_discrepancy = np.linalg.norm(
            coupling_vector(probe_tw)[n_rdt:] - prediction[n_rdt:]
        ) / np.linalg.norm(prediction[n_rdt:])
        log('Dispersion response directional check, relative error:', dy_discrepancy)
        if max(discrepancy, dy_discrepancy) > 0.05:
            raise RuntimeError('Coupling response disagrees with tracking')
    finally:
        set_values(line, skew_knobs, np.zeros(len(skew_knobs)))
