"""Orbit correction, including live BPM alignment and correctable-mode metrics."""

from functools import partial
from copy import copy
from types import SimpleNamespace
import numpy as np
import xtrack as xt

from correction_optimizer import CorrectionOptimizer
from response_builders import orbit_response


DEFAULT_ORBIT_CORR_SETTINGS = SimpleNamespace(
    orbit_tol=1e-5,
    rcond=1e-6,
)


def build_orbit_corr_optimizer(line, reference, *, plane, settings=None):
    """Build a correction optimizer with local targets, response and diagnostics."""
    if plane not in ('x', 'y'):
        raise ValueError('Orbit plane must be x or y')
    if settings is None:
        settings = copy(DEFAULT_ORBIT_CORR_SETTINGS)
    circuits = line.metadata['correction_knobs']['orbit']
    alignment = line.metadata['tuning_errors']['bpm_alignment_variables']
    monitors = circuits[plane]['monitors']
    indices = reference.rows.indices[monitors]
    action = xt.Action(line.twiss)
    env = line.env
    optimizer = CorrectionOptimizer(
        line, reference, settings=settings,
        compute_metrics=compute_metrics, goal=goal,
        print_metrics=print_metrics,
        name=f'orbit {plane}',
        vary=[xt.Vary(knob, step=1e-7) for knob in circuits[plane]['knobs']],
        targets=[action.target(
            lambda tw, p=plane, i=i, offset=alignment[name]['shift_' + plane]:
                tw[p][i] - env[offset],
            value=0., tol=1e-15, tag=f'{plane}@{name}')
            for i, name in zip(indices, monitors)])
    optimizer.plane = plane
    optimizer.monitor_names = list(circuits['x']['monitors'])
    optimizer.monitor_indices = reference.rows.indices[optimizer.monitor_names]
    optimizer.alignment_variables = alignment
    optimizer.rcond = settings.rcond
    optimizer.set_response(orbit_response(
        reference, monitors, circuits[plane]['correctors'], plane))
    optimizer.orbit_tol = settings.orbit_tol

    optimizer.correctable_error = partial(correctable_error, optimizer)
    return optimizer


def monitor_alignment(optimizer):
    return [np.array([optimizer.line.env[optimizer.alignment_variables[name][coordinate]]
                      for name in optimizer.monitor_names], dtype=float)
            for coordinate in ('shift_x', 'shift_y', 'rot_s_rad')]


def orbit_error(optimizer, tw):
    dx, dy, roll = monitor_alignment(optimizer)
    x = tw.x[optimizer.monitor_indices] - dx
    y = tw.y[optimizer.monitor_indices] - dy
    if optimizer.plane == 'x':
        return x * np.cos(roll) + y * np.sin(roll)
    return -x * np.sin(roll) + y * np.cos(roll)


def compute_metrics(optimizer, tw):
    error = orbit_error(optimizer, tw)
    return {f'orbit_{optimizer.plane}_rms': float(np.sqrt(np.mean(error**2)))}


def goal(optimizer, tw):
    return optimizer.compute_metrics(tw)[f'orbit_{optimizer.plane}_rms'] <= optimizer.orbit_tol


def correctable_error(optimizer, tw):
    """Lab-frame orbit projected onto retained nominal response modes."""
    offset = monitor_alignment(optimizer)[0 if optimizer.plane == 'x' else 1]
    error = tw[optimizer.plane][optimizer.monitor_indices] - offset
    svd = optimizer.response.svd
    rcond = optimizer.rcond if optimizer.rcond is not None else svd.rcond
    u = svd.U[:, svd.s > rcond * svd.s[0]]
    return u @ (u.T @ error)


def print_metrics(optimizer, tw, *, log=print):
    rms = optimizer.compute_metrics(tw)[f'orbit_{optimizer.plane}_rms']
    log(f'Orbit {optimizer.plane}: RMS {1e6 * rms:.3f} um '
        f'(goal {1e6 * optimizer.orbit_tol:.3f} um)')
