"""Inspect the saved result of 003b without rerunning the correction.

Set INPUT to machine.json or partial.json. This script computes the clean
reference, detailed diagnostics, radiation-integral estimates (for a completed
correction), and the correction/history plots. It does not modify the lattice.
"""

from functools import partial
from pathlib import Path
from time import perf_counter
import builtins

import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt

from correction_observables import orbit_errors as bpm_residuals, measure
from correction_io import report

# %% Settings
HERE = Path(__file__).resolve().parent
INPUT = HERE / 'response_api' / 'machine.json'
OUTPUT_DIR = INPUT.parent
PLOT = OUTPUT_DIR / 'correction.png'
HISTORY = INPUT.parent / 'history.csv'
SHOW_PLOTS = False
START_TIME = perf_counter()


def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    builtins.print(f'[+{perf_counter() - START_TIME:8.1f} s]', *args, **kwargs)


# %% Corrected optics and clean reference
print('Loading', INPUT)
line = xt.load(INPUT, with_progress=False)
metadata = line.metadata['orbit_optics_correction']
circuits = line.metadata['correction_knobs']
progress = metadata['progress']
print('Computing corrected optics')
tw_after = line.twiss(chrom=False, radiation_integrals=metadata['completed'])
masters = list(circuits['switches'].values())
original_tunes = circuits['tunes']['nominal_values']
before = {name: float(line[name]) for name in masters + list(original_tunes)}
try:
    line.vars.update({name: 0.0 for name in masters})
    line.vars.update(original_tunes)
    print('Computing clean reference')
    tw_ref = line.twiss(chrom=False, co_guess=dict(x=0, px=0, y=0, py=0, zeta=0, delta=0))
finally:
    line.vars.update(before)
np.testing.assert_array_equal(tw_after.name, tw_ref.name)
monitor_names = circuits['orbit']['x']['monitors']
monitor_indices = tw_ref.rows.indices[monitor_names]
ip_names = ['ipa', 'ipd', 'ipg', 'ipj']
ip_indices = tw_ref.rows.indices[ip_names]
dispersion_indices = tw_ref.rows.indices[list(dict.fromkeys(monitor_names + ip_names))]
quad_names = circuits['optics']['quadrupoles']
quad_indices = tw_ref.rows.indices[quad_names]


def monitor_alignment():
    variables = line.metadata['tuning_errors']['bpm_alignment_variables']
    return [
        np.array([line[variables[n][a]] for n in monitor_names], dtype=float)
        for a in ('shift_x', 'shift_y', 'rot_s_rad')
    ]


orbit_errors = partial(
    bpm_residuals, monitor_indices=monitor_indices, monitor_alignment=monitor_alignment
)
metrics = measure(
    tw_after,
    tw_ref=tw_ref,
    dispersion_indices=dispersion_indices,
    ip_indices=ip_indices,
    orbit_errors=orbit_errors,
)
report(print, 'CORRECTED', metrics)
summary = dict(
    input=str(INPUT), completed=metadata['completed'], progress=progress, metrics=metrics
)
if metadata['completed']:
    summary['final_4d_radiation_integrals'] = {
        key: float(tw_after[key])
        for key in (
            'rad_int_i2',
            'rad_int_i4x',
            'rad_int_i4y',
            'rad_int_i5x',
            'rad_int_i5y',
            'rad_int_eq_gemitt_x',
            'rad_int_eq_gemitt_y',
        )
    }
    print(
        'Radiation-integral emittance estimates x/y [pm]:',
        tw_after.rad_int_eq_gemitt_x * 1e12,
        tw_after.rad_int_eq_gemitt_y * 1e12,
    )
print(
    'Maximum quadrupole trim [m^-2]:', max(abs(line[name]) for name in circuits['optics']['knobs'])
)
print('One-turn eigenvalue moduli:', np.abs(np.linalg.eigvals(tw_after.R_matrix[:4, :4])))
for plane in 'xy':
    targets = np.array([getattr(line[name], 'shift_' + plane) for name in quad_names])
    residual = tw_after[plane][quad_indices] - targets
    print(
        f'Quadrupole entrance residual {plane}, RMS/max [um]:',
        np.sqrt(np.mean(residual**2)) * 1e6,
        np.max(np.abs(residual)) * 1e6,
    )
    dispersion = tw_after['d' + plane] - tw_ref['d' + plane]
    print(
        f'Dispersion error {plane}, RMS/max [m]:',
        np.sqrt(np.mean(dispersion**2)),
        np.max(np.abs(dispersion)),
    )
xt.json.dump(summary, OUTPUT_DIR / 'summary.json')
print('Saved', OUTPUT_DIR / 'summary.json')

# %% Plots
fig, axes = plt.subplots(3, 2, sharex=True, figsize=(13, 9), layout='constrained')
for j, plane in enumerate('xy'):
    axes[0, j].semilogy(
        tw_ref.s * 1e-3,
        np.maximum(1e-7, 100 * np.abs(tw_after['bet' + plane] / tw_ref['bet' + plane] - 1)),
        label='Interleaved correction',
        lw=0.8,
    )
    axes[0, j].set_ylabel(f'Absolute beta beating {plane} [%]')
    axes[0, j].legend()
    axes[1, j].plot(
        tw_ref.s[monitor_indices] * 1e-3,
        orbit_errors(tw_after, min(progress, 1.0))[j] * 1e6,
        '.',
        ms=2,
    )
    axes[1, j].set_ylabel(f'Orbit residual {plane} [µm]')
    axes[2, j].plot(tw_ref.s * 1e-3, tw_after['d' + plane] - tw_ref['d' + plane], lw=0.8)
    axes[2, j].set_ylabel(f'Dispersion error {plane} [m]')
    axes[2, j].set_xlabel('Nominal s [km]')
for axis in axes.flat:
    axis.grid(alpha=0.25)
fig.suptitle(f'4D correction: errors={line["on_errors"]:.3f}, nonlinear={line["on_nonlinear"]:.3f}')
fig.savefig(PLOT, dpi=180)
print('Saved', PLOT)
history = np.loadtxt(HISTORY, delimiter=',', skiprows=1, ndmin=2) if HISTORY.exists() else []
if len(history):
    history_fig, history_axes = plt.subplots(
        5, 1, sharex=True, figsize=(10, 12), layout='constrained'
    )
    for j, plane in enumerate('xy'):
        history_axes[0].plot(100 * history[:, 0], history[:, 1 + j] * 1e6, label=plane)
        history_axes[1].plot(100 * history[:, 0], 100 * history[:, 3 + j], label=f'{plane} RMS')
        history_axes[1].plot(
            100 * history[:, 0], 100 * history[:, 5 + j], ls='--', label=f'{plane} max'
        )
        history_axes[2].plot(
            100 * history[:, 0], history[:, 7 + j] - tw_ref['q' + plane], label=plane
        )
    history_axes[0].set_ylabel('Orbit RMS [um]')
    history_axes[1].set_ylabel('Beta beating [%]')
    history_axes[2].set_ylabel('Tune error')
    history_axes[3].plot(100 * history[:, 0], history[:, 15] * 1e3, '.-', label='BPMs + IPs RMS')
    history_axes[3].plot(100 * history[:, 0], history[:, 16] * 1e3, '.-', label='IP max')
    history_axes[3].set_ylabel('Vertical dispersion error [mm]')
    history_axes[4].plot(100 * history[:, 0], history[:, 17] * 1e3, '.-', label='BPMs + IPs RMS')
    history_axes[4].plot(100 * history[:, 0], history[:, 18] * 1e3, '.-', label='IP max')
    history_axes[4].set_ylabel('Horizontal dispersion error [mm]')
    history_axes[4].set_xlabel('Error fraction [%]; nonlinear strengths ramp together after 15%')
    for ax in history_axes:
        ax.legend()
        ax.grid(alpha=0.25)
        ax.axvline(100 * metadata['bootstrap_fraction'], color='grey', ls=':', lw=1)
    history_fig.savefig(OUTPUT_DIR / 'history.png', dpi=180)
if SHOW_PLOTS:
    plt.show()
