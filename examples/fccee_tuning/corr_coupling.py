"""Coupling and vertical dispersion: setup, metrics, goals and reporting."""

from copy import copy
from types import SimpleNamespace
import numpy as np
import xtrack as xt

import correction_setup as setup
from correction_optimizer import CorrectionOptimizer


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
    response, scales, weights, labels = setup.prepare_coupling_response(
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
