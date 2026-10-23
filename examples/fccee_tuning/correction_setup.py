"""Nominal reference and coupling/tune response preparation."""

from functools import partial

import numpy as np

import correction_observables as obs
from correction_steps import set_values, values
from response_builders import skew_quadrupole_response


def nominal_reference(line, *, full_twiss, log):
    """Set nominal knobs and return the reference with nonlinear magnets off.

    The input is the line saved by 001, with nonlinear magnets on. Check that
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
