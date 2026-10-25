"""Orbit correction, including live BPM alignment and orbit metrics."""

from copy import copy
from types import SimpleNamespace
import numpy as np
import xtrack as xt

from correction_optimizer import CorrectionOptimizer
from response_builders import orbit_response
from error_recipe import get_bpm_alignment


DEFAULT_ORBIT_CORR_SETTINGS = SimpleNamespace(
    orbit_tol=1e-5,
    rcond=1e-6,
)


def build_orbit_corr_optimizer(line, reference, *, plane, settings=None, **twiss_kwargs):
    """Build a correction optimizer with local targets, response and diagnostics."""
    if plane not in ('x', 'y'):
        raise ValueError('Orbit plane must be x or y')
    if settings is None:
        settings = copy(DEFAULT_ORBIT_CORR_SETTINGS)
    circuits = line.metadata['correction_knobs']['orbit']
    alignment = line.metadata['tuning_errors']['bpm_alignment_variables']
    monitors = circuits[plane]['monitors']
    targets = []
    for name in monitors:
        offset = alignment[name]['shift_' + plane]
        # Keep the BPM target as ref: its alignment offset changes with the error ramp.
        target = xt.Target(plane, at=name, value=line.ref[offset],
                           tol=1e-15, weight=1., tag=f'{plane}@{name}')
        targets.append(target)

    vary = []
    for knob in circuits[plane]['knobs']:
        vary.append(xt.Vary(knob, step=1e-7))
    optimizer = CorrectionOptimizer(
        line=line, reference_twiss=reference, settings=settings,
        method='4d', delta0=0., chrom=False, radiation_integrals=False,
        allow_twiss_failure=False,
        co_guess=dict(x=0, px=0, y=0, py=0, zeta=0, delta=0),
        **twiss_kwargs,
        compute_metrics=compute_metrics, goal=goal,
        print_metrics=print_metrics,
        name=f'orbit {plane}', vary=vary, targets=targets)
    optimizer.plane = plane
    optimizer.monitor_names = list(circuits['x']['monitors'])
    optimizer.monitor_indices = reference.rows.indices[optimizer.monitor_names]
    optimizer.alignment_variables = alignment
    optimizer.rcond = settings.rcond
    optimizer.set_response(orbit_response(
        reference, monitors, circuits[plane]['correctors'], plane))
    optimizer.orbit_tol = settings.orbit_tol

    return optimizer


def install_orbit_knobs(line):
    """Install zero-valued orbit trims and record correctors and BPMs by plane.

    Complete the layout with horizontal and vertical correctors at every
    quadrupole entrance, preserving existing correctors and their expressions.
    The trims represent kicks in radians: -knl[0] for x and +ksl[0] for y.
    """
    env = line.env
    # Complete the presentation's corrector-in-every-quadrupole layout. Existing
    # correctors are preserved; missing zero-length trims go at the quad entrance.
    added_correctors = []
    placements = []
    layout = line.get_table()
    for name in dict.fromkeys(layout.rows.match(element_type='Quadrupole').env_name):
        for prefix in ('hcor_', 'vcor_'):
            corrector = prefix + name
            if corrector not in line.element_dict:
                placements.append(
                    env.new(corrector, xt.Multipole, knl=[0], ksl=[0], at=name + '@start')
                )
                added_correctors.append(corrector)
    if placements:
        line.insert(placements)
    print(
        'Added', len(added_correctors), 'H/V correctors to complete the quadrupole layout'
    )
    monitor_names = list(get_bpm_alignment(line))
    corrector_names = [
        [n for n in line.element_names if n.startswith(prefix)]
        for prefix in ('hcor_', 'vcor_')
    ]
    orbit_knobs = [f'orbit_corr_{name}_{plane}'
                   for plane, names in zip('xy', corrector_names) for name in names]
    assert all(knob not in line.vars for knob in orbit_knobs), 'Orbit correction knobs already exist'
    line['on_orbit_corr'] = 1.0
    for plane, names, field, sign in zip('xy', corrector_names, ('knl', 'ksl'), (-1, 1)):
        for name in names:
            knob = f'orbit_corr_{name}_{plane}'
            line[knob] = 0.0
            getattr(line.ref[name], field)[0] += (
                sign * line.ref['on_orbit_corr'] * line.ref[knob]
            )
    line.steering_monitors_x = line.steering_monitors_y = monitor_names
    line.steering_correctors_x, line.steering_correctors_y = corrector_names
    print('BPMs:', len(monitor_names), 'H/V correctors:', [len(n) for n in corrector_names])

    circuits = line.metadata.setdefault('correction_knobs', {})
    circuits['orbit'] = {
        plane: dict(
            knobs=[f'orbit_corr_{name}_{plane}' for name in names],
            correctors=names, monitors=monitor_names, units='rad')
        for plane, names in zip('xy', corrector_names)
    }
    circuits.setdefault('switches', {})['orbit'] = 'on_orbit_corr'
    circuits['added_correctors'] = added_correctors
    return orbit_knobs


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


def print_metrics(optimizer, tw, *, log=print):
    rms = optimizer.compute_metrics(tw)[f'orbit_{optimizer.plane}_rms']
    log(f'Orbit {optimizer.plane}: RMS {1e6 * rms:.3f} um '
        f'(goal {1e6 * optimizer.orbit_tol:.3f} um)')
