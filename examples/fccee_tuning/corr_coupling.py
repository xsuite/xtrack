"""Coupling and vertical dispersion: setup, metrics, goals and reporting."""

from copy import copy
from functools import partial
from types import SimpleNamespace
import numpy as np
import xtrack as xt

from correction_optimizer import CorrectionOptimizer
from response_builders import skew_quadrupole_response


DEFAULT_COUPLING_CORR_SETTINGS = SimpleNamespace(
    rcond=1e-6,
    coupling_ip_weight=20.,
    coupling_rms_tol=0.003,
    coupling_ip_tol=0.002,
    c_minus_tol=0.0003,
    dispersion_rms_tol=0.001,
    dispersion_ip_tol=0.00002,
    intermediate_c_minus_tol=0.003,
    intermediate_coupling_rms_tol=0.03,
    intermediate_coupling_ip_tol=0.03,
    intermediate_dispersion_rms_tol=0.005,
    intermediate_dispersion_ip_tol=0.0001,
    dispersion_weight=1.,
    orbit_tol=5e-8,
    step_scale=0.5,
)


def build_coupling_corr_optimizer(line, reference, *, observation_names, ip_names,
                                 full_twiss, settings=None, log=print):
    """Build a correction optimizer with local targets, response and diagnostics."""
    if settings is None:
        settings = copy(DEFAULT_COUPLING_CORR_SETTINGS)
    action = xt.Action(full_twiss)
    circuits = line.metadata['correction_knobs']['coupling']
    knobs = circuits['knobs']
    response, scales, weights, labels = prepare_coupling_response(
        line, reference, circuits['magnet_groups'], observation_names,
        ip_names=ip_names,
        coupling_ip_weight=settings.coupling_ip_weight,
        dispersion_weight=settings.dispersion_weight,
        dispersion_ip_weight=(settings.coupling_ip_weight * settings.dispersion_rms_tol
                              / settings.dispersion_ip_tol),
        knobs=knobs, full_twiss=full_twiss, log=log,
    )
    indices = reference.rows.indices[observation_names]
    observables = (
        [lambda tw, key=key, part=part, i=i:
            getattr(tw[key][i] - reference[key][i], part)
         for key in ('f1001', 'f1010') for part in ('real', 'imag') for i in indices]
        + [lambda tw, i=i: tw.dy[i] - reference.dy[i] for i in indices]
    )
    optimizer = CorrectionOptimizer(
        line, reference, settings=settings,
        compute_metrics=compute_metrics, goal=goal,
        print_metrics=print_metrics,
        name='coupling and vertical dispersion',
        vary=[xt.Vary(knob, step=1e-7, weight=scale) for knob, scale in zip(knobs, scales)],
        targets=[action.target(observable, value=0.0, tol=1e-15,
                               weight=weight, tag=label)
                 for observable, label, weight in zip(observables, labels, weights)],
    )
    optimizer.coupling_indices = indices
    optimizer.ip_indices = reference.rows.indices[ip_names]
    optimizer.rcond = settings.rcond
    optimizer.set_response(response)

    optimizer.step_scale = settings.step_scale
    optimizer.orbit_tol = settings.orbit_tol
    optimizer.rdt_rms_tol = settings.intermediate_coupling_rms_tol
    optimizer.rdt_ip_tol = settings.intermediate_coupling_ip_tol
    optimizer.c_minus_tol = settings.intermediate_c_minus_tol
    optimizer.dy_rms_tol = settings.intermediate_dispersion_rms_tol
    optimizer.dy_ip_tol = settings.intermediate_dispersion_ip_tol

    return optimizer


def compute_metrics(optimizer, tw):
    """RDT and vertical dispersion errors at BPMs/IPs, and global c_minus."""
    reference = optimizer.reference
    coupling_indices = optimizer.coupling_indices
    ip_indices = optimizer.ip_indices
    f1001 = tw['f1001'] - reference['f1001']
    f1010 = tw['f1010'] - reference['f1010']
    dy = tw.dy - reference.dy
    return dict(
        coupling_rms=np.array([np.sqrt(np.mean(np.abs(f1001[coupling_indices])**2)),
                               np.sqrt(np.mean(np.abs(f1010[coupling_indices])**2))]),
        coupling_ip=max(np.max(np.abs(f1001[ip_indices])),
                        np.max(np.abs(f1010[ip_indices]))),
        c_minus=float(tw.c_minus),
        dy_rms=float(np.sqrt(np.mean(dy[coupling_indices]**2))),
        dy_ip=float(np.max(np.abs(dy[ip_indices]))),
    )


def goal(optimizer, tw):
    m = optimizer.compute_metrics(tw)
    return bool(np.all(m['coupling_rms'] < optimizer.rdt_rms_tol)
                and m['coupling_ip'] < optimizer.rdt_ip_tol
                and abs(m['c_minus']) < optimizer.c_minus_tol
                and m['dy_rms'] < optimizer.dy_rms_tol
                and m['dy_ip'] < optimizer.dy_ip_tol)


def print_metrics(optimizer, tw, *, log=print):
    m = optimizer.compute_metrics(tw)
    log('Coupling: RDT RMS/IP', m['coupling_rms'], m['coupling_ip'],
        '|C-|', abs(m['c_minus']), 'Dy RMS/IP [mm]',
        1e3 * m['dy_rms'], 1e3 * m['dy_ip'])


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
            coupling_vector, tw_ref=tw_ref,
            coupling_indices=tw_ref.rows.indices[observation_names],
        ),
        n_observations=len(observation_names), log=log,
    )
    return matrix, scales, weights, labels


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
    line.set(skew_knobs, probe)
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
        line.set(skew_knobs, np.zeros(len(skew_knobs)))


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
