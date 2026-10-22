"""Build the normal-optics/horizontal-dispersion optimizer and its response."""

from functools import partial

import numpy as np
import xtrack as xt

import correction_setup as setup


def _normalized_alpha(tw, *, plane, at, beta_ref):
    return tw['alf' + plane, at] * beta_ref / tw['bet' + plane, at]


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
    targets = (
        [xt.Target('bet' + p, at=name, value=reference['bet' + p, name], action=action)
         for p in 'xy' for name in beta_at]
        + [xt.Target(
            partial(_normalized_alpha, plane=p, at=name, beta_ref=reference['bet' + p, name]),
            value=reference['alf' + p, name], action=action)
           for p in 'xy' for name in alpha_at]
        + [xt.Target('dx', at=name, value=reference['dx', name], action=action)
           for name in dispersion_at]
        + [xt.Target(q, value=reference[q], action=action) for q in ('qx', 'qy')]
    )
    # Targets expose physical beta values. Convert the nominal relative-beta
    # response to d(beta)/d(knob), and normalize the cost by beta_ref.
    n_beta = 2 * len(beta_at)
    beta_ref = np.array([target.value for target in targets[:n_beta]])
    response[:n_beta] *= beta_ref[:, None]
    weights[:n_beta] /= beta_ref
    labels[:n_beta] = [f'bet{p}@{name}' for p in 'xy' for name in beta_at]
    for target, weight, label in zip(targets, weights, labels):
        target.tol = 1e-15
        target.weight = weight
        target.tag = label
    for target in targets[:n_beta]:
        target.tol *= target.value  # Preserve the relative tolerance.

    optimizer = line.match(
        solve=False,
        name='normal optics and horizontal dispersion',
        vary=[xt.Vary(knob, step=1e-7, weight=scale, max_step=0.5)
              for knob, scale in zip(knobs, scales)],
        targets=targets,
        action_results={action: reference},
        assert_within_tol=False,
        show_call_counter=False,
    )
    optimizer.set_response(response)
    return optimizer
