"""Build horizontal and vertical orbit optimizers and their nominal responses."""

import xtrack as xt

from response_builders import orbit_response


def build_optimizers(line, reference, action):
    """Return the x/y optimizers; targets subtract live BPM alignment offsets."""
    circuits = line.metadata['correction_knobs']['orbit']
    alignment = line.metadata['tuning_errors']['bpm_alignment_variables']
    env = line.env
    optimizers = []
    for plane in 'xy':
        monitors = circuits[plane]['monitors']
        indices = reference.rows.indices[monitors]
        optimizer = line.match(
            solve=False,
            name=f'orbit {plane}',
            vary=[xt.Vary(knob, step=1e-7) for knob in circuits[plane]['knobs']],
            targets=[
                action.target(
                    lambda tw, p=plane, i=i, offset=alignment[name]['shift_' + plane]:
                        tw[p][i] - env[offset],
                    value=0.0, tol=1e-15, tag=f'{plane}@{name}')
                for i, name in zip(indices, monitors)
            ],
            action_results={action: reference},
            assert_within_tol=False,
            show_call_counter=False,
        )
        optimizer.set_response(
            orbit_response(reference, monitors, circuits[plane]['correctors'], plane))
        optimizers.append(optimizer)
    return tuple(optimizers)
