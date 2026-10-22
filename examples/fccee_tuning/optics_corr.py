"""Build the normal-optics/horizontal-dispersion optimizer and its response."""

import numpy as np
import xtrack as xt

import correction_setup as setup


def build_optimizer(line, reference, action, *, beta_at, alpha_at, dispersion_at,
                    interaction_points, ip_names, settings, full_twiss, log):
    """Return an optimizer with weighted targets and a fixed nominal response.

    Response checks use full_twiss; optimizer construction uses the supplied
    reference table. The caller owns subsequent evaluations and trial acceptance.
    """
    circuits = line.metadata['correction_knobs']['optics']
    knobs = circuits['knobs']
    response, scales, weights, labels = setup.prepare_optics_response(
        line, reference, circuits['quadrupoles'], beta_at, alpha_at, dispersion_at,
        interaction_points=interaction_points,
        dispersion_ips=ip_names,
        alpha_weight=settings.alpha_weight,
        ir_beta_weight=settings.ir_beta_weight,
        ir_half_length=settings.ir_half_length,
        dispersion_weight=settings.horizontal_dispersion_weight,
        dispersion_ip_weight=settings.horizontal_dispersion_ip_weight,
        knobs=knobs, full_twiss=full_twiss, log=log,
    )
    beta_indices = reference.rows.indices[beta_at]
    alpha_indices = reference.rows.indices[alpha_at]
    dispersion_indices = reference.rows.indices[dispersion_at]
    observables = (
        [lambda tw, p=p, i=i: np.log(tw['bet' + p][i] / reference['bet' + p][i])
         for p in 'xy' for i in beta_indices]
        + [lambda tw, p=p, i=i: (
            tw['alf' + p][i] * reference['bet' + p][i] / tw['bet' + p][i]
            - reference['alf' + p][i])
           for p in 'xy' for i in alpha_indices]
        + [lambda tw, i=i: tw.dx[i] - reference.dx[i] for i in dispersion_indices]
        + [lambda tw, q=q: tw[q] - reference[q] for q in ('qx', 'qy')]
    )
    optimizer = line.match(
        solve=False,
        name='normal optics and horizontal dispersion',
        vary=[xt.Vary(knob, step=1e-7, weight=scale, max_step=0.5)
              for knob, scale in zip(knobs, scales)],
        targets=[action.target(observable, value=0.0, tol=1e-15,
                               weight=weight, tag=label)
                 for observable, label, weight in zip(observables, labels, weights)],
        action_results={action: reference},
        assert_within_tol=False,
        show_call_counter=False,
    )
    optimizer.set_response(response)
    return optimizer
