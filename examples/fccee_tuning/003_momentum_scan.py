"""Clean/corrected tunes versus delta and on-momentum beta comparison (4D).

Clean and corrected machines are scanned on the same delta grid. Both beta
beating curves use the ideal delta=0 lattice, with nominal nonlinear strengths.
Orbit is in lattice coordinates, not the BPM measurement frame. Both are
sampled at BPM entrances and ipa/ipd/ipg/ipj. Failed Twiss points remain NaN.
"""

from pathlib import Path
from time import perf_counter
import json

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

# Configuration (delta = (p-p0)/p0).
HERE = Path(__file__).resolve().parent
INPUT = HERE / 'response_api_class' / 'machine.json'
OUTPUT_DIR = HERE / 'analysis' / 'momentum_scan'
DELTA_MAX = 0.01
NUM_POINTS = 21                     # Odd, so zero is included; spacing 0.001.
SHOW_PLOTS = True                   # Set False for batch runs.
MAX_CONTINUATION_STEP = 5e-4
MIN_CONTINUATION_STEP = 1e-6
MAX_TUNE_STEP = 0.25                # Reject ambiguous/discontinuous mode branches.
CO_SEARCH_AT = 'ipd'                # Zero-guess search at ipa fails on this seed.
IPS = ['ipa', 'ipd', 'ipg', 'ipj']
STEPS_R_MATRIX = dict(dx=1e-8, dpx=1e-9, dy=1e-8, dpy=1e-9,
                      dzeta=1e-7, ddelta=1e-8)


def scan_momentum(line, tw0, deltas, indices, settings, label):
    """Continue one machine independently in both delta directions."""
    zero = len(deltas) // 2
    data = {key: np.full((len(deltas), len(indices)), np.nan)
            for key in ('x', 'y', 'betx', 'bety')}
    tunes = np.full((len(deltas), 2), np.nan)
    errors = [''] * len(deltas)
    reached_delta = np.full(len(deltas), np.nan)
    failures = (ValueError, RuntimeError, xt.twiss.ClosedOrbitSearchError)

    # Follow each sign independently from zero. Halve a failed step, retaining
    # only accepted seeds; intermediate points are not substituted for grid points.
    for branch in ([zero] + list(range(zero + 1, len(deltas))),
                   list(range(zero - 1, -1, -1))):
        seed, current = tw0, 0.
        for position, ii in enumerate(branch):
            target = deltas[ii]
            step = min(MAX_CONTINUATION_STEP, abs(target - current))
            try:
                while abs(target - current) > 1e-14:
                    trial = current + np.sign(target - current) * step
                    try:
                        candidate = line.twiss(
                            delta0=trial, co_guess=seed.particle_on_co, **settings)
                        if not np.all(np.isfinite([candidate.qx, candidate.qy])):
                            raise ValueError('Nonfinite tunes')
                        tune_step = np.array([candidate.qx - seed.qx,
                                              candidate.qy - seed.qy])
                        if np.max(np.abs(tune_step)) > MAX_TUNE_STEP:
                            raise ValueError(f'Discontinuous tune branch: ΔQ={tune_step}')
                    except failures:
                        step /= 2
                        if step < MIN_CONTINUATION_STEP:
                            raise
                        continue
                    seed, current = candidate, trial
                    step = min(MAX_CONTINUATION_STEP, 2 * step, abs(target - current))
                tunes[ii] = seed.qx, seed.qy
                for key in data:
                    data[key][ii] = seed[key][indices]
                print(f'{label}: delta={target:+.6f}: Qx={seed.qx:.9f} Qy={seed.qy:.9f}', flush=True)
            except failures as err:
                errors[ii] = f'{type(err).__name__}: {err}'
                print(f'{label}: FAILED delta={target:+.6f}, last reached {current:+.8f}: '
                      f'{errors[ii]}', flush=True)
                reached_delta[ii] = current
                # Further points on this branch cannot be reached continuously.
                # Do not jump across an unresolved region to a different orbit.
                for pending in branch[position + 1:]:
                    errors[pending] = f'Not reached beyond failed delta={target:+.6f}'
                    reached_delta[pending] = current
                    print(f'{label}: delta={deltas[pending]:+.6f}: {errors[pending]}', flush=True)
                break
            reached_delta[ii] = current

    return data, tunes, errors, reached_delta


def main():
    if (not np.isfinite(DELTA_MAX) or DELTA_MAX <= 0
            or NUM_POINTS < 3 or NUM_POINTS % 2 != 1):
        raise ValueError('Require DELTA_MAX > 0 and an odd NUM_POINTS >= 3')
    started = perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Loading', INPUT, flush=True)
    line = xt.load(INPUT, with_progress=False)
    line.configure_radiation(model=None)
    # Remove any serialized zero guess; do not inherit correction-only defaults.
    line.twiss_default.clear()
    if any(getattr(el, 'delta_taper', 0.) != 0.
           for el in line.element_dict.values()):
        raise ValueError('Use the untapered standard machine')
    settings = dict(method='4d', chrom=False, steps_R_matrix=STEPS_R_MATRIX)
    tw0 = line.twiss(delta0=0., co_search_at=CO_SEARCH_AT, **settings)
    print(f'Independent closed orbit: Qx={tw0.qx:.9f}, Qy={tw0.qy:.9f}', flush=True)

    # Recover the ideal lattice on the identical element layout. Disable every
    # installed correction and zero the tune offsets without overwriting their families.
    circuits = line.metadata['correction_knobs']
    ideal_values = {name: 0. for name in circuits['switches'].values()}
    ideal_values.update(on_nonlinear=1., on_chromaticity_corr=0.)
    ideal_values.update(circuits['tunes']['nominal_values'])
    names = list(dict.fromkeys(circuits['orbit']['x']['monitors'] + IPS))
    indices = tw0.rows.indices[names]
    s = tw0.s[indices]
    deltas = np.linspace(-DELTA_MAX, DELTA_MAX, NUM_POINTS)
    zero = NUM_POINTS // 2
    deltas[zero] = 0.
    before = {name: float(line[name]) for name in ideal_values}
    try:
        line.vars.update(ideal_values)
        reference = line.twiss(delta0=0., co_search_at=CO_SEARCH_AT, **settings)
        clean_data, clean_tunes, clean_errors, clean_reached_delta = scan_momentum(
            line, reference, deltas, indices, settings, 'Clean')
    finally:
        line.vars.update(before)
    np.testing.assert_array_equal(tw0.name, reference.name)
    data, tunes, errors, reached_delta = scan_momentum(
        line, tw0, deltas, indices, settings, 'Corrected')

    beta_reference = np.array([reference.betx[indices], reference.bety[indices]])
    beating = np.stack([data['betx'], data['bety']], axis=1) / beta_reference - 1
    clean_beating = np.stack([clean_data['betx'], clean_data['bety']], axis=1) / beta_reference - 1
    np.savez_compressed(OUTPUT_DIR / 'momentum_scan.npz',
                        delta=deltas, name=np.array(names), s=s, tunes=tunes,
                        beta_reference=beta_reference, beta_beating=beating,
                        errors=np.array(errors), reached_delta=reached_delta, **data,
                        clean_tunes=clean_tunes, clean_beta_beating=clean_beating,
                        clean_errors=np.array(clean_errors),
                        clean_reached_delta=clean_reached_delta,
                        **{'clean_' + key: value for key, value in clean_data.items()})
    summary = dict(input=str(INPUT.resolve()), radiation=False, tapering=False,
                   method='4d', co_search_at=CO_SEARCH_AT,
                   observation_points='BPM entrances plus ipa/ipd/ipg/ipj',
                   orbit_coordinates='lattice frame [m], no BPM alignment subtraction',
                   beta_reference='ideal at delta=0; nominal nonlinear strengths',
                   ideal_tunes=[float(reference.qx), float(reference.qy)],
                   max_continuation_step=MAX_CONTINUATION_STEP,
                   min_continuation_step=MIN_CONTINUATION_STEP,
                   max_tune_step=MAX_TUNE_STEP,
                   failed_points=[dict(delta=float(d), last_reached_delta=float(r), error=e)
                                  for d, r, e in zip(deltas, reached_delta, errors) if e],
                   clean_failed_points=[dict(delta=float(d), last_reached_delta=float(r), error=e)
                                        for d, r, e in zip(deltas, clean_reached_delta, clean_errors) if e],
                   clean_comparison='clean and corrected evaluated at the same delta grid',
                   elapsed_seconds=perf_counter() - started)
    (OUTPUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, layout='constrained')
    for jj, plane in enumerate('xy'):
        ax = axes[jj]
        ax.plot(100 * deltas, tunes[:, jj], '.-', color='C0', label='Corrected')
        ax.plot(100 * deltas, clean_tunes[:, jj], '.--', color='C1', label='Clean')
        for machine_errors, color in ((errors, 'C0'), (clean_errors, 'C1')):
            for delta, err in zip(deltas, machine_errors):
                if err:
                    ax.axvline(100 * delta, color=color, alpha=0.15)
        ax.set_ylabel(f'Q{plane}')
        ax.set_xlabel('δ [%]')
        ax.set_xlim(-100 * DELTA_MAX, 100 * DELTA_MAX)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle('Tunes versus δ; radiation/tapering off\n'
                 'Blue/orange vertical lines: corrected/clean failed or unreached')
    fig.savefig(OUTPUT_DIR / 'tunes_vs_delta.png', dpi=180)
    if not SHOW_PLOTS:
        plt.close(fig)

    # Overlay absolute beta and show signed beating on separate panels. Use
    # the same BPM/IP observations as the scan, sorted by their ring position.
    order = np.argsort(s)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, layout='constrained')
    for jj, plane in enumerate('xy'):
        axes[0, jj].plot(s[order] * 1e-3, beta_reference[jj, order],
                         label='Ideal, δ=0', lw=1.5)
        axes[0, jj].plot(s[order] * 1e-3, data['bet' + plane][zero, order],
                         label='Corrected, δ=0', lw=0.8, ls='--')
        axes[0, jj].set_ylabel(f'β{plane} [m]')
        axes[1, jj].plot(s[order] * 1e-3, 100 * beating[zero, jj, order], lw=0.8)
        axes[1, jj].axhline(0, color='grey', lw=0.8)
        axes[1, jj].set_ylabel(f'(β{plane} / β{plane},ideal − 1) [%]')
        axes[1, jj].set_xlabel('s [km]')
        axes[0, jj].legend()
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    fig.suptitle('On-momentum beta at BPM entrances + four IPs; radiation/tapering off')
    fig.savefig(OUTPUT_DIR / 'beta_along_ring.png', dpi=180)
    if not SHOW_PLOTS:
        plt.close(fig)

    print(f'Saved {OUTPUT_DIR}; failed/unreached points: '
          f'corrected={sum(bool(e) for e in errors)}, clean={sum(bool(e) for e in clean_errors)}; '
          f'{perf_counter() - started:.1f} s', flush=True)
    if SHOW_PLOTS:
        plt.show()


if __name__ == '__main__':
    main()
