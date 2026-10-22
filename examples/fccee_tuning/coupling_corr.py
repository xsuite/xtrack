"""Build the coupling/vertical-dispersion optimizer and its response."""

import xtrack as xt

import correction_setup as setup


def build_optimizer(line, reference, action, *, observation_names, ip_names,
                    settings, full_twiss, log):
    """Return an optimizer with RDT/Dy targets and a fixed nominal response."""
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
        action_results={action: reference},
        assert_within_tol=False,
        show_call_counter=False,
    )
    optimizer.set_response(response)
    return optimizer
