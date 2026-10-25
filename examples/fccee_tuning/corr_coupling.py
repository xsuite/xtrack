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
    step_scale=0.5,
)


def build_coupling_corr_optimizer(line, reference, *, observation_names, ip_names,
                                 settings=None, log=print, **twiss_kwargs):
    """Build a correction optimizer with local targets, response and diagnostics."""
    if settings is None:
        settings = copy(DEFAULT_COUPLING_CORR_SETTINGS)
    circuits = line.metadata['correction_knobs']['coupling']
    knobs = circuits['knobs']
    response, scales = prepare_coupling_response(
        line, reference, circuits['magnet_groups'], observation_names,
        knobs=knobs, log=log, **twiss_kwargs)

    targets = []
    for name in observation_names:
        weight = 1 / np.sqrt(len(observation_names))
        if name in ip_names:
            weight *= settings.coupling_ip_weight
        target = xt.Target('f1001.real', at=name,
            value=reference['f1001', name].real, tol=1e-15,
            weight=weight, tag=f'real_f1001@{name}')
        targets.append(target)
        target = xt.Target('f1001.imag', at=name,
            value=reference['f1001', name].imag, tol=1e-15,
            weight=weight, tag=f'imag_f1001@{name}')
        targets.append(target)

        target = xt.Target('f1010.real', at=name,
            value=reference['f1010', name].real, tol=1e-15,
            weight=weight, tag=f'real_f1010@{name}')
        targets.append(target)
        target = xt.Target('f1010.imag', at=name,
            value=reference['f1010', name].imag, tol=1e-15,
            weight=weight, tag=f'imag_f1010@{name}')
        targets.append(target)

        weight = settings.dispersion_weight / np.sqrt(len(observation_names))
        if name in ip_names:
            weight *= (settings.coupling_ip_weight * settings.dispersion_rms_tol
                       / settings.dispersion_ip_tol)
        target = xt.Target('dy', at=name, value=reference['dy', name],
                           tol=1e-15, weight=weight, tag=f'dy@{name}')
        targets.append(target)

    vary = []
    for knob, scale in zip(knobs, scales):
        vary.append(xt.Vary(knob, step=1e-7, weight=scale))
    optimizer = CorrectionOptimizer(
        line=line, reference_twiss=reference, settings=settings,
        method='4d', delta0=0., chrom=False, radiation_integrals=False,
        allow_twiss_failure=False,
        co_guess=dict(x=0, px=0, y=0, py=0, zeta=0, delta=0),
        **twiss_kwargs,
        compute_metrics=compute_metrics, goal=goal,
        print_metrics=print_metrics,
        name='coupling and vertical dispersion', vary=vary, targets=targets)
    optimizer.coupling_indices = reference.rows.indices[observation_names]
    optimizer.ip_indices = reference.rows.indices[ip_names]
    optimizer.rcond = settings.rcond
    optimizer.set_response(response)

    optimizer.step_scale = settings.step_scale
    optimizer.rdt_rms_tol = settings.intermediate_coupling_rms_tol
    optimizer.rdt_ip_tol = settings.intermediate_coupling_ip_tol
    optimizer.c_minus_tol = settings.intermediate_c_minus_tol
    optimizer.dy_rms_tol = settings.intermediate_dispersion_rms_tol
    optimizer.dy_ip_tol = settings.intermediate_dispersion_ip_tol

    return optimizer


def install_coupling_knobs(line):
    """Install new integrated skew trims, preserving existing sextupole expressions.

    The error metadata groups lattice elements belonging to the same physical
    magnet. A sextupole can have several pieces, so each group gets one knob
    for the total integrated skew-quadrupole strength, rather than one knob per
    piece. Each piece receives a fraction of that ksl[1] trim proportional to
    its length; summing over the group recovers the full trim. The contribution
    is outside on_nonlinear, so the skew trim remains active when k2 is off.
    """
    table = line.get_table()
    sextupoles = set(table.rows.match(element_type='Sextupole').env_name)
    groups = [record['names']
              for record in line.metadata['tuning_errors']['magnets'].values()
              if record['names'][0] in sextupoles]
    knobs = [f'dk1sl_coupling_{names[0]}' for names in groups]
    switch = 'on_coupling_corr'
    assert all(knob not in line.vars for knob in knobs), 'Coupling correction knobs already exist'
    line[switch] = 1.
    for names, knob in zip(groups, knobs):
        line[knob] = 0.
        length = sum(line.get(name).length for name in names)
        for name in names:
            line.ref[name].ksl[1] += (line.ref[switch] * line.ref[knob]
                                      * line.get(name).length / length)

    line.metadata.setdefault('correction_knobs', {})['coupling'] = dict(
        knobs=knobs, magnet_groups=groups, units='m^-1')
    return knobs


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
    line, tw_ref, skew_groups, observation_names, *, knobs, log, **twiss_kwargs
):
    """Return the response in observation order and the knob scales.

    Check the response with one tracking probe, restoring the nominal trims.
    """
    matrix, scales = skew_quadrupole_response(line, tw_ref, skew_groups, observation_names)
    check_coupling_response(
        line, skew_knobs=knobs, skew_scale=scales, coupling_response=matrix,
        tw_ref=tw_ref,
        coupling_vector=partial(
            coupling_vector, tw_ref=tw_ref,
            coupling_indices=tw_ref.rows.indices[observation_names],
        ),
        n_observations=len(observation_names), log=log, **twiss_kwargs,
    )
    # The analytical response groups rows by quantity; targets group them by location.
    matrix = matrix.reshape(5, len(observation_names), len(knobs))
    matrix = matrix.transpose(1, 0, 2).reshape(5 * len(observation_names), len(knobs))
    return matrix, scales


def check_coupling_response(
    line,
    *,
    skew_knobs,
    skew_scale,
    coupling_response,
    tw_ref,
    coupling_vector,
    n_observations,
    log,
    **twiss_kwargs,
):
    # Check the analytical response against tracking for a distributed probe.
    probe = 1e-5 * skew_scale * np.sin(np.arange(len(skew_knobs)) + 0.3)
    line.set(skew_knobs, probe)
    try:
        probe_tw = line.twiss(method='4d', delta0=0., chrom=False, radiation_integrals=False,
                              co_guess=tw_ref.particle_on_co, **twiss_kwargs)
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
