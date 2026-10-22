"""Build the coupling/vertical-dispersion optimizer and its response."""

import numpy as np
import xtrack as xt

import correction_setup as setup


def compute_metrics(tw, reference, *, coupling_indices, ip_indices):
    """RDT and vertical dispersion errors at BPMs/IPs, and global c_minus."""
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


def coupling_goal(tw, reference, *, coupling_indices, ip_indices,
                  settings, final_refinement):
    metrics = compute_metrics(tw, reference, coupling_indices=coupling_indices,
                              ip_indices=ip_indices)
    if final_refinement:
        rdt_rms_tol = settings.coupling_rms_tol
        rdt_ip_tol = settings.coupling_ip_tol
        c_minus_tol = settings.c_minus_tol
        dy_rms_tol = settings.dispersion_rms_tol
        dy_ip_tol = settings.dispersion_ip_tol
    else:
        rdt_rms_tol = settings.ramp_coupling_rms_tol
        rdt_ip_tol = settings.ramp_coupling_ip_tol
        c_minus_tol = settings.ramp_c_minus_tol
        dy_rms_tol = settings.ramp_dispersion_rms_tol
        dy_ip_tol = settings.ramp_dispersion_ip_tol
    return bool(
        np.all(metrics['coupling_rms'] < rdt_rms_tol)
        and metrics['coupling_ip'] < rdt_ip_tol
        and abs(metrics['c_minus']) < c_minus_tol
        and metrics['dy_rms'] < dy_rms_tol
        and metrics['dy_ip'] < dy_ip_tol
    )


def coupling_vector(tw, reference, *, coupling_indices):
    f1001 = (tw['f1001'] - reference['f1001'])[coupling_indices]
    f1010 = (tw['f1010'] - reference['f1010'])[coupling_indices]
    dy = (tw.dy - reference.dy)[coupling_indices]
    return np.r_[f1001.real, f1001.imag, f1010.real, f1010.imag, dy]


def build_optimizer(line, reference, *, observation_names, ip_names,
                    settings, full_twiss, log):
    """Return an optimizer with RDT/Dy targets and a fixed nominal response."""
    action = xt.Action(full_twiss)
    circuits = line.metadata['correction_knobs']['coupling']
    knobs = circuits['knobs']
    response, scales, weights, labels = setup.prepare_coupling_response(
        line, reference, circuits['magnet_groups'], observation_names,
        ip_names=ip_names,
        coupling_ip_weight=settings.coupling_ip_weight,
        dispersion_weight=settings.dispersion_weight,
        dispersion_ip_weight=settings.dispersion_ip_weight,
        knobs=knobs, full_twiss=full_twiss, log=log,
    )
    indices = reference.rows.indices[observation_names]
    observables = (
        [lambda tw, key=key, part=part, i=i:
            getattr(tw[key][i] - reference[key][i], part)
         for key in ('f1001', 'f1010') for part in ('real', 'imag') for i in indices]
        + [lambda tw, i=i: tw.dy[i] - reference.dy[i] for i in indices]
    )
    optimizer = line.match(
        solve=False,
        name='coupling and vertical dispersion',
        vary=[xt.Vary(knob, step=1e-7, weight=scale) for knob, scale in zip(knobs, scales)],
        targets=[action.target(observable, value=0.0, tol=1e-15,
                               weight=weight, tag=label)
                 for observable, label, weight in zip(observables, labels, weights)],
        action_results=[reference],
        assert_within_tol=False,
        show_call_counter=False,
    )
    optimizer.rcond = settings.coupling_rcond
    optimizer.set_response(response)
    return optimizer
