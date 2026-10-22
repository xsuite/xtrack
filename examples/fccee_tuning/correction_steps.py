"""Trial steps for 003b. Each step returns an accepted Twiss table.

Exhausting a trial search restores its trial knobs and raises RuntimeError.
The script owns the outer transaction, including rollback after orbit/tune
settling or a failed Twiss evaluation.
No response is recomputed here; get_correction uses the installed response.
"""

from time import perf_counter
import numpy as np
import xtrack as xt
from correction_observables import rms_arrays


def orbit_guess(tw):
    return dict(
        x=float(tw.x[0]),
        px=float(tw.px[0]),
        y=float(tw.y[0]),
        py=float(tw.py[0]),
        zeta=0.0,
        delta=0.0,
    )


def values(line, names):
    return np.array([line[name] for name in names])


def set_values(line, names, vals):
    line.vars.update({name: float(value) for name, value in zip(names, vals)})


def compute_twiss(
    seed=None, co_guess=None, radiation_integrals=False, only_orbit=False, *, line, stats, steps
):
    kwargs = {} if seed is None else dict(co_guess=orbit_guess(seed))
    if co_guess is not None:
        kwargs['co_guess'] = co_guess
    started = perf_counter()
    stats['calls'] += 1
    if only_orbit:
        stats['orbit_calls'] += 1
    try:
        return line.twiss(
            chrom=False,
            steps_R_matrix=steps.copy(),
            radiation_integrals=radiation_integrals,
            only_orbit=only_orbit,
            **kwargs,
        )
    finally:
        elapsed = perf_counter() - started
        stats['seconds'] += elapsed
        if only_orbit:
            stats['orbit_seconds'] += elapsed


def ensure_full_twiss(tw, *, full_twiss):
    # Call only while the line still has the settings used to compute tw.
    # Use the tracked orbit as a guess; keep the ordinary closed-orbit search.
    if 'betx' in tw._col_names:
        return tw
    return full_twiss(tw)


def tune_cleanup(
    tw, *, line, tune_knobs, tw_ref, tune_tol, tune_optimizer, ensure_full_twiss, full_twiss, log
):
    """Use the fixed nominal 2x2 response with the original trial acceptance."""
    tw = ensure_full_twiss(tw)
    target = np.array([tw_ref.qx, tw_ref.qy])
    for it in range(7):
        error = np.array([tw.qx, tw.qy]) - target
        if max(abs(error)) < tune_tol / 2:
            return tw
        baseline = values(line, tune_knobs)
        change = tune_optimizer.get_correction(error, rcond=0.0)
        for bt in range(4):
            set_values(line, tune_knobs, baseline + change * 0.5**bt)
            try:
                probe = full_twiss(tw)
            except (ValueError, xt.twiss.ClosedOrbitSearchError):
                continue
            new_error = np.array([probe.qx, probe.qy]) - target
            if np.linalg.norm(new_error) < np.linalg.norm(error):
                tw = probe
                log('  tune cached step', 0.5**bt, 'max error', max(abs(new_error)))
                break
        else:
            set_values(line, tune_knobs, baseline)
            raise RuntimeError('Tune cleanup failed with the fixed nominal response')
    if max(abs(np.array([tw.qx, tw.qy]) - target)) < tune_tol / 2:
        return tw
    raise RuntimeError('Tune cleanup iteration limit')


def orbit_step(
    tw,
    fraction,
    max_backtracks=13,
    only_orbit=False,
    *,
    line,
    orbit_knobs,
    steering_direction,
    orbit_errors,
    orbit_twiss,
    full_twiss,
    log,
):
    baseline = values(line, orbit_knobs)
    direction = steering_direction(tw, fraction)
    before = np.linalg.norm(np.concatenate(orbit_errors(tw, fraction)))
    for backtrack in range(max_backtracks):
        step = 0.5**backtrack
        set_values(line, orbit_knobs, baseline + step * direction)
        try:
            trial = orbit_twiss(tw) if only_orbit else full_twiss(tw)
        except (ValueError, xt.twiss.ClosedOrbitSearchError):
            continue
        if np.linalg.norm(np.concatenate(orbit_errors(trial, fraction))) < before * (1 - 1e-6):
            log(
                '  orbit step',
                step,
                'rms_um',
                rms_arrays(orbit_errors(trial, fraction)) * 1e6,
                flush=True,
            )
            return trial
    set_values(line, orbit_knobs, baseline)
    raise RuntimeError('No stable improving orbit step')


def optics_step(
    tw,
    *,
    line,
    quad_knobs,
    tune_knobs,
    orbit_knobs,
    optical_direction,
    optical_cost,
    strength_scale,
    orbit_twiss,
    orbit_step,
    tune_cleanup,
    measure,
    log,
):
    """Evaluate each quadrupole trial together with its orbit/tune correction."""
    baseline = values(line, quad_knobs)
    families = values(line, tune_knobs)
    steering = values(line, orbit_knobs)
    fraction = float(line['on_errors'])
    direction = optical_direction(tw)
    before = optical_cost(tw)
    largest = np.max(np.abs(direction / strength_scale))
    initial_step = min(1.0, 0.5 / max(largest, 1e-30))
    for backtrack in range(10):
        step = initial_step * 0.5**backtrack
        set_values(line, quad_knobs, baseline + step * direction)
        set_values(line, tune_knobs, families)
        set_values(line, orbit_knobs, steering)
        try:
            trial = orbit_twiss(tw)
            try:
                trial = orbit_step(trial, fraction)
            except RuntimeError:
                pass
            trial = tune_cleanup(trial)
        except (ValueError, RuntimeError):
            continue
        if optical_cost(trial) < before * (1 - 1e-6):
            metrics = measure(trial, orbit_errors=None)
            log(
                '  joint step',
                step,
                'cost',
                optical_cost(trial),
                'beta rms',
                metrics['beta_rms'],
                'beta max',
                metrics['beta_max'],
                'Dx error RMS/IP max [mm]',
                1e3 * metrics['dx_rms'],
                1e3 * metrics['dx_ip'],
                flush=True,
            )
            return trial
    set_values(line, quad_knobs, baseline)
    set_values(line, tune_knobs, families)
    set_values(line, orbit_knobs, steering)
    raise RuntimeError('No stable improving joint step')


def settle_coupling_orbit(
    tw, *, line, correctable_orbit_errors, coupling_orbit_tol, orbit_step, tune_cleanup
):
    # Small remaining orbit changes feed down in the sextupoles and can obscure
    # the improvement of a skew trial. Compare baseline and trials after the
    # same settling procedure; exclude the nominal response's discarded modes.
    for iteration in range(3):
        if np.max(rms_arrays(correctable_orbit_errors(tw))) < coupling_orbit_tol:
            break
        try:
            tw = orbit_step(
                tw, float(line['on_errors']), max_backtracks=4, only_orbit=iteration < 2
            )
        except RuntimeError:
            break
    return tune_cleanup(tw)


def coupling_step(
    tw,
    *,
    line,
    skew_knobs,
    orbit_knobs,
    tune_knobs,
    coupling_optimizer,
    coupling_vector,
    coupling_rcond,
    step_fraction,
    coupling_cost,
    coupling_goal,
    settle_coupling_orbit,
    orbit_twiss,
    measure,
    log,
):
    tw = settle_coupling_orbit(tw)
    if coupling_goal(tw):
        return tw
    baseline = values(line, skew_knobs)
    steering, families = values(line, orbit_knobs), values(line, tune_knobs)
    direction = coupling_optimizer.get_correction(coupling_vector(tw), rcond=coupling_rcond)
    before = coupling_cost(tw)
    for backtrack in range(10):
        step = step_fraction * 0.5**backtrack
        set_values(line, skew_knobs, baseline + step * direction)
        set_values(line, orbit_knobs, steering)
        set_values(line, tune_knobs, families)
        try:
            trial = orbit_twiss(tw)
            # Re-steer after the fractional skew update before comparing RDTs.
            trial = settle_coupling_orbit(trial)
        except (ValueError, RuntimeError, xt.twiss.ClosedOrbitSearchError):
            continue
        if coupling_cost(trial) < before * (1 - 1e-6):
            metrics = measure(trial, orbit_errors=None)
            log(
                '  coupling step',
                step,
                'RDT RMS',
                metrics['coupling_rms'],
                'c_minus',
                trial.c_minus,
                'IP max',
                metrics['coupling_ip'],
                'Dy RMS/IP max [mm]',
                1e3 * metrics['dy_rms'],
                1e3 * metrics['dy_ip'],
            )
            return trial
    set_values(line, skew_knobs, baseline)
    set_values(line, orbit_knobs, steering)
    set_values(line, tune_knobs, families)
    raise RuntimeError('No stable improving coupling step')
