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


def build_tune_response(tw, *, line, tune_knobs, full_twiss):
    """Numerical two-family tune response, with all probe settings restored."""
    baseline = values(line, tune_knobs)
    jac = np.empty((2, 2))
    try:
        for j in range(2):
            line[tune_knobs[j]] = baseline[j] + 1e-7
            probe = full_twiss(tw)
            jac[:, j] = (np.array([probe.qx, probe.qy]) - np.array([tw.qx, tw.qy])) / 1e-7
            set_values(line, tune_knobs, baseline)
    finally:
        set_values(line, tune_knobs, baseline)
    return jac


def tune_cleanup(
    tw, *, line, tune_knobs, tw_ref, tune_tol, tune_direction, ensure_full_twiss, full_twiss, log
):
    """Use the fixed nominal 2x2 response with the original trial acceptance."""
    tw = ensure_full_twiss(tw)
    target = np.array([tw_ref.qx, tw_ref.qy])
    for it in range(7):
        error = np.array([tw.qx, tw.qy]) - target
        if max(abs(error)) < tune_tol / 2:
            return tw
        baseline = values(line, tune_knobs)
        change = tune_direction(error)
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
    coupling_direction,
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
    direction = coupling_direction(tw)
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


def check_optics_response(
    line,
    *,
    quad_knobs,
    strength_scale,
    nominal_response,
    optics_weights,
    tw_ref,
    full_twiss,
    optics_vector,
    horizontal_dispersion_errors,
    n_dispersion,
    log,
):
    # A central distributed probe checks the sign, integration and scaling in two
    # Twiss calls, without probing every quadrupole. The central difference cancels
    # quadratic tune shifts, which can dominate the small net first-order shift.
    probe = 1e-4 * strength_scale * np.sin(np.arange(len(quad_knobs)) + 0.3)
    set_values(line, quad_knobs, probe)
    try:
        probe_plus = full_twiss(tw_ref)
        set_values(line, quad_knobs, -probe)
        probe_minus = full_twiss(tw_ref)
        measured = (optics_vector(probe_plus) - optics_vector(probe_minus)) / 2
        prediction = nominal_response @ probe
        discrepancy = np.linalg.norm(optics_weights * (measured - prediction)) / np.linalg.norm(
            optics_weights * prediction
        )
        dx_prediction = nominal_response[-n_dispersion - 2 : -2] @ probe
        dx_measured = (
            horizontal_dispersion_errors(probe_plus) - horizontal_dispersion_errors(probe_minus)
        ) / 2
        dx_discrepancy = np.linalg.norm(dx_measured - dx_prediction) / np.linalg.norm(dx_prediction)
        log(
            'Normal optics/Dx response directional check, relative errors:',
            discrepancy,
            dx_discrepancy,
        )
        if max(discrepancy, dx_discrepancy) > 0.05:
            raise RuntimeError('Normal optics/dispersion response disagrees with tracking')
    finally:
        set_values(line, quad_knobs, np.zeros(len(quad_knobs)))


def check_coupling_response(
    line,
    *,
    skew_knobs,
    skew_scale,
    coupling_response,
    tw_ref,
    full_twiss,
    coupling_vector,
    n_observations,
    log,
):
    # Check the analytical response against tracking for a distributed probe.
    probe = 1e-5 * skew_scale * np.sin(np.arange(len(skew_knobs)) + 0.3)
    set_values(line, skew_knobs, probe)
    try:
        probe_tw = full_twiss(tw_ref)
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
        set_values(line, skew_knobs, np.zeros(len(skew_knobs)))
