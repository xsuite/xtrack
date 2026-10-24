"""Explore arc quadrupole knl[3] trims, with the original dipole b4 restored.

This is a response/tradeoff study, not an accepted correction or a hardware
design. All original correction settings and IR octupoles remain unchanged.
Use 006 separately for 6D radiation/tapered acceptance validation of a candidate.
"""

from pathlib import Path
from time import perf_counter
import hashlib
import importlib.util
import json

import matplotlib.pyplot as plt
import nafflib
import numpy as np
from scipy.optimize import LinearConstraint, minimize
import xobjects as xo
import xtrack as xt

HERE = Path(__file__).resolve().parent
INPUT = HERE / 'response_api_class' / 'machine.json'
OUTPUT_DIR = HERE / 'analysis' / 'arc_octupoles_quad_knl'
SHOW_PLOTS = True
FAMILIES = {'dko_arc_f': 'qf2a', 'dko_arc_d': 'qd1a'}
PROBE_K3L = 0.02                    # Per physical quadrupole, m^-3; central differences.
FRACTIONS = [0.25, 0.5, 1.]         # Fractions of the linear Q'' correction.
DETUNING_ERROR_FRACTION = 0.9       # Guarded fit: reduce each initial error by 10%.
NUM_TURNS = 512
TRACKING_THREADS = 8
MAX_SECONDS_PER_CASE = 120.
EMITTANCES = np.array([0.71e-9, 1.4e-12])  # Geometric, m.
AMPLITUDES = np.array([[.5, .5], [1., .5], [2., .5], [3., .5],
                       [.5, 1.], [.5, 2.], [.5, 3.], [1., 1.],
                       [2., 2.], [3., 3.]])
DELTAS = np.linspace(-.012, .012, 25)
MAX_TUNE_STEP = .02                # Reject abrupt branch changes; halve delta step.

spec = importlib.util.spec_from_file_location('momentum_scan', HERE / '003_momentum_scan.py')
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)
scan.MAX_TUNE_STEP = MAX_TUNE_STEP
SETTINGS = dict(method='4d', chrom=False, steps_R_matrix=scan.STEPS_R_MATRIX)


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, default=lambda a: a.tolist()) + '\n')


def install_octupole_trims(line):
    """Add total K3L per physical quadrupole, preserving existing expressions."""
    records = line.metadata['tuning_errors']['magnets'].values()
    groups = {knob: [g['names'] for g in records if g['category'] == 'arc_quad'
                    and g['names'][0].split('.')[0] == family]
              for knob, family in FAMILIES.items()}
    if any(not magnets for magnets in groups.values()):
        raise ValueError('Missing regular arc quadrupole family')
    if 'on_arc_octupole_corr' in line.vars:
        raise ValueError('Arc octupole trims are already installed')
    line['on_arc_octupole_corr'] = 1.
    for knob, magnets in groups.items():
        line[knob] = 0.
        for names in magnets:
            length = sum(line[name].length for name in names)
            for name in names:
                line.ref[name].knl[3] += (line.ref['on_arc_octupole_corr']
                    * line.ref['on_nonlinear'] * line.ref[knob]
                    * line[name].length / length)
    line.metadata['correction_knobs']['switches']['arc_octupoles'] = 'on_arc_octupole_corr'
    return groups


def analytic_response(tw, names):
    """Entrance-optics estimate of [Qx''/2, Qy''/2, axx, axy, ayy] per K3L.

    The actual quadrupole body kicks and alignment are included in the measured
    response and tracking; this estimate does not integrate optics through them.
    """
    ii = tw.rows.indices[names]
    x, y, d, e = [tw[k][ii] for k in ('betx', 'bety', 'dx', 'dy')]
    return np.array([np.sum(x*(d*d-e*e))/(8*np.pi),
                     -np.sum(y*(d*d-e*e))/(8*np.pi),
                     np.sum(x*x)/(16*np.pi), -np.sum(x*y)/(8*np.pi),
                     np.sum(y*y)/(16*np.pi)])


def chromatic_vector(line, seed=None, h=.001):
    tw = line.twiss(co_search_at='ipd', **SETTINGS) if seed is None else line.twiss(
        co_guess=seed.particle_on_co, **SETTINGS)
    _, q, errors, _ = scan.scan_momentum(
        line, tw, h*np.arange(-2, 3), [], SETTINGS, 'Local chromatic fit')
    if any(errors):
        raise RuntimeError(str(errors))
    c = np.polynomial.polynomial.polyfit(np.arange(-2, 3), q, 4)
    return np.concatenate((c[0], c[1]/h, c[2]/h**2)), tw


def detuning_slopes(tunes, maximum):
    """Vary each action separately; keep the other amplitude fixed at 0.5 sigma."""
    slopes = np.empty((2, 2))  # rows: tune mode; columns: geometric action.
    for mode in range(2):
        selected = (AMPLITUDES[:, 1-mode] == .5) & (AMPLITUDES[:, mode] <= maximum)
        action_in_emittances = AMPLITUDES[selected, mode]**2/2
        slopes[:, mode] = np.polynomial.polynomial.polyfit(
            action_in_emittances, tunes[selected]-tunes[0], 1)[1]/EMITTANCES[mode]
    return slopes


def study_vector(case):
    a = np.asarray(case['tracking']['estimates'][str(NUM_TURNS)]['slopes']['1.0'])
    return np.r_[case['vector'][4:], a[0, 0], (a[0, 1]+a[1, 0])/2, a[1, 1]]


def guarded_step(report):
    """Fit chromaticity with linear amplitude-detuning guards; verify by tracking."""
    error = study_vector(report['cases']['corrected'])-study_vector(report['cases']['clean'])
    response = np.array(report['analytic_response'], copy=True)
    response[:2] = np.asarray(report['response'])[4:]
    scale = np.maximum(np.abs(error[:2]), 1.)
    guard = DETUNING_ERROR_FRACTION*np.abs(error[2:])
    constraints = LinearConstraint(response[2:]/1e5,
        (-guard-error[2:])/1e5, (guard-error[2:])/1e5)
    fit = minimize(lambda k: np.sum(((error[:2]+response[:2]@k)/scale)**2),
        np.zeros(response.shape[1]), method='SLSQP', constraints=[constraints],
        options=dict(ftol=1e-12, maxiter=200))
    if not fit.success:
        raise RuntimeError(f'Guarded fit failed: {fit.message}')
    prediction = error+response@fit.x
    if np.any(np.abs(prediction[2:]) > guard+1.):
        raise RuntimeError('Guarded solution violates linear detuning constraints')
    report['guarded_fit'] = dict(strengths=dict(zip(FAMILIES, fit.x)),
        initial_error=error, predicted_error=prediction,
        maximum_detuning_error=guard, error_fraction=DETUNING_ERROR_FRACTION,
        model='Measured chromatic response; analytic uncoupled detuning response')
    print('GUARDED STEP', fit.x, 'PREDICTED ERRORS', prediction, flush=True)
    return fit.x


def track_amplitudes(line, tw, turns, label):
    if line.element_names[0] != 'septum':
        raise ValueError('Cycle to septum before the amplitude diagnostic')
    particles = line.build_particles(particle_on_co=tw.particle_on_co,
        W_matrix=tw.W_matrix[0], method='4d',
        x_norm=AMPLITUDES[:, 0], px_norm=0., y_norm=AMPLITUDES[:, 1], py_norm=0.,
        gemitt_x=EMITTANCES[0], gemitt_y=EMITTANCES[1], zeta=0., delta=0.)
    flags = line.tracker.track_flags
    previous = flags.XS_FLAG_KILL_CAVITY_KICK
    started = perf_counter()
    try:
        flags.XS_FLAG_KILL_CAVITY_KICK = True
        line.track(particles, num_turns=turns, turn_by_turn_monitor=True)
    finally:
        flags.XS_FLAG_KILL_CAVITY_KICK = previous
    elapsed = perf_counter()-started
    mon = line.record_last_track
    order = np.argsort(np.asarray(particles.particle_id))
    state = np.asarray(particles.state)[order]
    if np.any(state <= 0):
        raise RuntimeError(f'{label}: amplitude diagnostic lost particles: {state}')
    np.testing.assert_allclose(mon.delta, 0., atol=1e-14, rtol=0.)
    physical = np.array([np.asarray(getattr(mon, key))-tw[key][0]
                         for key in ('x', 'px', 'y', 'py')])
    normal = np.einsum('ij,jpt->ipt', np.linalg.inv(tw.W_matrix[0, :4, :4]), physical)
    q0 = np.array([tw.qx, tw.qy])
    results = {}
    for n in sorted(set([turns//2, turns])):
        tunes = np.empty((len(AMPLITUDES), 2))
        for mode in range(2):
            signals = normal[2*mode, :, :n]-1j*normal[2*mode+1, :, :n]
            for i, signal in enumerate(signals):
                q = nafflib.get_tune(signal-signal.mean())
                tunes[i, mode] = q + np.round(q0[mode]-q)
        results[str(n)] = dict(tunes=tunes, slopes={str(a): detuning_slopes(tunes, a)
                                                   for a in (1., 2., 3.)})
    np.savez_compressed(OUTPUT_DIR / f'{label}_tracking.npz',
        normalized=normal, q0=q0, amplitudes=AMPLITUDES, emittances=EMITTANCES,
        state=state, delta=np.asarray(mon.delta), zeta=np.asarray(mon.zeta))
    print(label, 'tracking seconds', elapsed, 'detuning', results[str(turns)]['slopes']['1.0'], flush=True)
    return dict(seconds=elapsed, turns=turns, estimates=results)


def main():
    started = perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    line = xt.load(INPUT, with_progress=False)
    line.twiss_default.clear()
    line.configure_radiation(model=None)
    if any(getattr(el, 'delta_taper', 0.) != 0. for el in line.element_dict.values()):
        raise ValueError('Use the untapered standard corrected machine')
    line['on_error_arc_dipole_b4'] = 1.
    line.cycle('septum', inplace=True)
    initial = line.twiss(co_search_at='ipd', **SETTINGS)
    line.discard_tracker()
    groups = install_octupole_trims(line)
    sites = {knob: [names[0] for names in magnets] for knob, magnets in groups.items()}
    analytic = np.column_stack([analytic_response(initial, names) for names in sites.values()])
    line.build_tracker(_context=xo.ContextCpu(omp_num_threads=TRACKING_THREADS))
    baseline, tw = chromatic_vector(line)
    np.testing.assert_allclose(baseline[:2], [initial.qx, initial.qy], atol=1e-8, rtol=0.)
    pilot = track_amplitudes(line, tw, 64, 'pilot')
    estimate = pilot['seconds']*NUM_TURNS/64
    print('Estimated tracking seconds per case:', estimate, flush=True)
    if estimate > MAX_SECONDS_PER_CASE:
        raise RuntimeError('Tracking pilot exceeds configured runtime limit')

    circuits = line.metadata['correction_knobs']
    clean = {name: 0. for name in circuits['switches'].values()}
    clean.update(on_nonlinear=1., on_chromaticity_corr=0.)
    clean.update(circuits['tunes']['nominal_values'])
    clean.update(circuits['tunes']['family_nominal_values'])
    saved = {name: float(line[name]) for name in clean}
    saved['on_error_arc_dipole_b4'] = 1.
    try:
        line.vars.update(clean)
        target, _ = chromatic_vector(line)
    finally:
        line.vars.update(saved)
    response = np.empty((6, len(sites)))
    for i, knob in enumerate(sites):
        try:
            line[knob] = PROBE_K3L
            plus, _ = chromatic_vector(line, tw)
            line[knob] = -PROBE_K3L
            minus, _ = chromatic_vector(line, tw)
            response[:, i] = (plus-minus)/(2*PROBE_K3L)
        finally:
            line[knob] = 0.
    step = np.linalg.solve(response[4:], target[4:]-baseline[4:])
    report = dict(input=str(INPUT), sha256=digest, sites=sites, magnet_groups=groups,
        implementation='Additive quadrupole knl[3]; length-weighted across physical halves',
        strength_units='Total K3L trim per physical quadrupole [m^-3]', analytic_response=analytic,
        analytic_rows=['Qx_second/2', 'Qy_second/2', 'axx', 'axy', 'ayy'],
        response=response, response_rows=['Qx', 'Qy', 'Qprime_x', 'Qprime_y', 'Qx_second/2', 'Qy_second/2'],
        full_linear_step=dict(zip(sites, step)), pilot_seconds=pilot['seconds'],
        emittances=EMITTANCES, amplitudes=AMPLITUDES, launch='septum; px_norm=py_norm=0; delta=0',
        max_tune_step=MAX_TUNE_STEP,
        radiation=False, tapering=False, rf='XS_FLAG_KILL_CAVITY_KICK; no energy/zeta freeze',
        detuning_units='dQ_i/dJ_j [m^-1]; independent action scans',
        hardware_limits='Not specified; octupole components in existing arc quadrupoles', cases={})
    dump(OUTPUT_DIR / 'study.json', report)
    print('FULL LINEAR STEP', report['full_linear_step'], 'ANALYTIC DETUNING CHANGE', analytic[2:]@step, flush=True)
    cases = [('clean', None), ('corrected', 0.), ('b4_off', 0.)]
    cases += [(f'oct_{fraction:g}', fraction) for fraction in FRACTIONS]
    cases += [('oct_guarded', None)]
    try:
        for label, fraction in cases:
            line.vars.update(saved)
            for knob in sites:
                line[knob] = 0.
            if label == 'oct_guarded':
                for knob, value in zip(sites, guarded_step(report)):
                    line[knob] = value
            elif fraction is None:
                line.vars.update(clean)
            elif label == 'b4_off':
                line['on_error_arc_dipole_b4'] = 0.
            else:
                for knob, value in zip(sites, step*fraction):
                    line[knob] = value
            vector, tw = chromatic_vector(line)
            half, _ = chromatic_vector(line, tw, h=.0005)
            tracking = track_amplitudes(line, tw, NUM_TURNS, label)
            _, tunes, errors, reached = scan.scan_momentum(line, tw, DELTAS, [], SETTINGS, label)
            np.savez_compressed(OUTPUT_DIR / f'{label}_momentum.npz',
                delta=DELTAS, tunes=tunes, errors=errors, reached_delta=reached)
            report['cases'][label] = dict(fraction=fraction, vector=vector,
                half_step_vector=half, tracking=tracking, tunes=tunes,
                errors=errors, reached_delta=reached)
            if label == 'oct_1':
                # Experimental candidate for the separate 6D acceptance study.
                line.to_json(OUTPUT_DIR / 'trial_full_machine.json')
            if label == 'oct_guarded':
                measured = study_vector(report['cases'][label])-study_vector(report['cases']['clean'])
                report['guarded_fit']['measured_error'] = measured
                report['guarded_fit']['measured_errors_below_baseline'] = bool(np.all(
                    np.abs(measured[2:]) <= np.abs(report['guarded_fit']['initial_error'][2:])))
            report['seconds'] = perf_counter()-started
            dump(OUTPUT_DIR / 'study.json', report)
    finally:
        line.vars.update(saved)
        for knob in sites:
            line[knob] = 0.
    assert hashlib.sha256(INPUT.read_bytes()).hexdigest() == digest
    plot_results(report)
    print('COMPLETE', perf_counter()-started, 'seconds; artifacts:', OUTPUT_DIR, flush=True)


def plot_results(report):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for label, case in report['cases'].items():
        for mode, ax in enumerate(axes):
            q = np.asarray(case['tunes'])[:, mode]
            ln, = ax.plot(100*DELTAS, q, '.-', label=label)
            for delta, error in zip(DELTAS, case['errors']):
                if error:
                    ax.axvline(100*delta, color=ln.get_color(), alpha=.12)
            ax.set(xlabel='δ [%]', ylabel=f'Q{"xy"[mode]}', xlim=(-1.2, 1.2))
            ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle('Arc octupole study: radiation/tapering off; faint lines mark failed scans')
    for suffix in ('png', 'pdf'):
        fig.savefig(OUTPUT_DIR / f'tunes_vs_delta.{suffix}', dpi=180)
    fig2, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    for label, case in report['cases'].items():
        estimates = case['tracking']['estimates'][str(NUM_TURNS)]['slopes']
        matrices = np.array([estimates[str(a)] for a in (1., 2., 3.)])
        for ax, (i, j), title in zip(axes, [(0, 0), (0, 1), (1, 1)], ['axx', 'axy (solid), ayx (dotted)', 'ayy']):
            ln, = ax.plot([1, 2, 3], matrices[:, i, j]/1e5, '.-', label=label)
            if i != j:
                ax.plot([1, 2, 3], matrices[:, j, i]/1e5, ':', color=ln.get_color())
            ax.set(xlabel='Maximum amplitude in separate action fit [σ]',
                   ylabel='Detuning coefficient [10⁵ m⁻¹]', title=title, xticks=[1, 2, 3])
            ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig2.suptitle('Amplitude detuning: 0.5–3σ, other mode held at 0.5σ')
    for suffix in ('png', 'pdf'):
        fig2.savefig(OUTPUT_DIR / f'amplitude_detuning.{suffix}', dpi=180)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig2)


if __name__ == '__main__':
    main()
