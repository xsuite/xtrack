"""Build horizontal and vertical orbit optimizers and their nominal responses."""

import numpy as np
import xtrack as xt

from response_builders import orbit_response


def orbit_errors(tw, *, monitor_indices, monitor_alignment):
    """Orbit relative to the BPM offsets, expressed in the BPM's rotated frame."""
    dx, dy, roll = monitor_alignment()
    x = tw.x[monitor_indices] - dx
    y = tw.y[monitor_indices] - dy
    return [x * np.cos(roll) + y * np.sin(roll),
            -x * np.sin(roll) + y * np.cos(roll)]


def lab_orbit_errors(tw, *, monitor_indices, monitor_alignment):
    dx, dy, _ = monitor_alignment()
    return [tw.x[monitor_indices] - dx, tw.y[monitor_indices] - dy]


def compute_metrics(tw, *, monitor_indices, monitor_alignment):
    x, y = orbit_errors(tw, monitor_indices=monitor_indices,
                        monitor_alignment=monitor_alignment)
    return dict(orbit_rms=np.array([np.sqrt(np.mean(x**2)), np.sqrt(np.mean(y**2))]))


def orbit_goal(tw, *, monitor_indices, monitor_alignment, error_fraction, settings):
    metrics = compute_metrics(tw, monitor_indices=monitor_indices,
                              monitor_alignment=monitor_alignment)
    tolerance = max(0.002e-6, error_fraction * settings.orbit_tol)
    return bool(np.all(metrics['orbit_rms'] <= tolerance))


def correctable_orbit_errors(tw, *, lab_orbit_errors, orbit_optimizers):
    """Project out discarded SVD modes and the response's left null space."""
    residuals = []
    for error, optimizer in zip(lab_orbit_errors(tw), orbit_optimizers):
        svd = optimizer.response.svd
        rcond = optimizer.rcond
        if rcond is None:
            rcond = svd.rcond
        u = svd.U[:, svd.s > rcond * svd.s[0]]
        residuals.append(u @ (u.T @ error))
    return residuals


def build_optimizers(line, reference, *, settings):
    """Return the x/y optimizers; targets subtract live BPM alignment offsets."""
    circuits = line.metadata['correction_knobs']['orbit']
    alignment = line.metadata['tuning_errors']['bpm_alignment_variables']
    env = line.env
    optimizers = []
    for plane in 'xy':
        action = xt.Action(line.twiss)
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
            action_results=[reference],
            assert_within_tol=False,
            show_call_counter=False,
        )
        optimizer.rcond = settings.orbit_rcond
        optimizer.set_response(
            orbit_response(reference, monitors, circuits[plane]['correctors'], plane))
        optimizers.append(optimizer)
    return tuple(optimizers)
