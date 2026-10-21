"""Plot the corrected FCC-ee machine without rerunning the correction.

Run: python 002_analyze_corrected_machine.py [--show]
Optional: --input machine.json --output-dir analysis_directory

Two 4D Twiss calls compare the saved machine with the clean reference used by
001: errors, correction masters and nonlinear magnets off, and the original
tune-family strengths restored. Tracking models and finite-difference steps
are taken from the saved machine. Radiation is off.

Older corrected files do not store the original tune-family values. For those,
read them from the companion fccee_z_lcc_errors.json, or --nominal-input.
No lattice JSON is modified. PNG figures and summary.json go into analysis/.
BPM residuals include their offsets and rolls; dispersion and coupling are
ideal model observables. The coupling RDTs are the tracked Edwards–Teng values.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt

from error_recipe import get_bpm_alignment


HERE = Path(__file__).resolve().parent
IP_NAMES = ['ipa', 'ipd', 'ipg', 'ipj']


def rms(values):
    return float(np.sqrt(np.mean(np.abs(values)**2)))


def clean_reference(env, line, metadata, nominal_input):
    """Temporarily turn off errors/trims, then restore the corrected settings."""
    tune_values = metadata.get('nominal_tune_knobs')
    if tune_values is None:
        with nominal_input.open() as stream:
            nominal_values = json.load(stream)['_var_management_data']['var_values']
        tune_values = {name: nominal_values[name] for name in ('kqf2', 'kqd1')}
        print('Original tune-family values from', nominal_input, flush=True)
    masters = ['on_errors', 'on_orbit_corr', 'on_optics_corr',
               'on_coupling_corr', 'on_nonlinear']
    saved = {name: float(env[name]) for name in masters + list(tune_values)}
    try:
        for name in masters:
            env[name] = 0.
        for name, value in tune_values.items():
            env[name] = value
        return line.twiss(method='4d', delta0=0., chrom=False,
                          co_guess={c: 0. for c in ('x', 'px', 'y', 'py', 'zeta', 'delta')})
    finally:
        for name, value in saved.items():
            env[name] = value


def mark_ips(axes, tw):
    positions = tw.s[tw.rows.indices[IP_NAMES]] * 1e-3
    for ax in np.asarray(axes).flat:
        for s in positions:
            ax.axvline(s, color='0.6', ls=':', lw=0.7)
        ax.grid(alpha=0.2)
    for ax in np.asarray(axes)[0].flat:
        for name, s in zip(IP_NAMES, positions):
            ax.text(s, 1.01, name, transform=ax.get_xaxis_transform(),
                    ha='center', fontsize=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE/'fccee_z_lcc_corrected.json')
    parser.add_argument('--nominal-input', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    output = args.output_dir or args.input.parent/'analysis'
    nominal_input = args.nominal_input or args.input.parent/'fccee_z_lcc_errors.json'

    print('Loading', args.input, flush=True)
    env = xt.load(args.input, with_progress=False)
    line = env.fccee_p_ring
    metadata = env.metadata['orbit_optics_correction']
    line.configure_radiation(model=None)
    print('Computing corrected 4D optics', flush=True)
    tw = line.twiss(method='4d', delta0=0., chrom=False)
    alignment = get_bpm_alignment(env)
    bpm_names = list(alignment)
    bpm_idx = tw.rows.indices[bpm_names]
    ip_idx = tw.rows.indices[IP_NAMES]
    obs_idx = tw.rows.indices[list(dict.fromkeys(bpm_names + IP_NAMES))]
    offsets = {key: np.array([alignment[n][key] for n in bpm_names])
               for key in ('shift_x', 'shift_y', 'rot_s_rad')}
    x = tw.x[bpm_idx] - offsets['shift_x']
    y = tw.y[bpm_idx] - offsets['shift_y']
    roll = offsets['rot_s_rad']
    bpm_orbit = {'x': x*np.cos(roll) + y*np.sin(roll),
                 'y': -x*np.sin(roll) + y*np.cos(roll)}

    print('Computing clean 4D reference', flush=True)
    ref = clean_reference(env, line, metadata, nominal_input)
    np.testing.assert_array_equal(tw.name, ref.name)
    beta = {p: tw['bet'+p]/ref['bet'+p] - 1 for p in 'xy'}
    dispersion = {p: tw['d'+p] - ref['d'+p] for p in 'xy'}
    rdts = {key: tw[key] - ref[key] for key in ('f1001', 'f1010')}
    s_km = tw.s*1e-3

    summary = {
        'input': str(args.input.resolve()),
        'completed_correction': bool(metadata['completed']),
        'error_fraction': float(env['on_errors']),
        'nonlinear_fraction': float(env['on_nonlinear']),
        'reference': 'errors, correction masters and nonlinear magnets off; original tune families',
        'c_minus': float(tw.c_minus),
        'c_minus_tolerance': metadata['c_minus_tolerance'],
        'planes': {}, 'rdts': {}, 'ips': {}, 'checks': {},
    }
    checks = summary['checks']
    checks['coupling_c_minus'] = bool(tw.c_minus < metadata['c_minus_tolerance'])
    for p in 'xy':
        prefix = 'horizontal_' if p == 'x' else ''
        peak = int(np.argmax(np.abs(beta[p])))
        stats = {
            'tune': float(tw['q'+p]),
            'tune_error': float(tw['q'+p] - ref['q'+p]),
            'bpm_orbit_rms_m': rms(bpm_orbit[p]),
            'beta_beating_rms': rms(beta[p]),
            'beta_beating_max_abs': float(np.max(np.abs(beta[p]))),
            'beta_peak_element': str(tw.name[peak]),
            'beta_peak_s_m': float(tw.s[peak]),
            'dispersion_error_rms_bpms_ips_m': rms(dispersion[p][obs_idx]),
            'dispersion_error_ip_max_abs_m': float(np.max(np.abs(dispersion[p][ip_idx]))),
        }
        summary['planes'][p] = stats
        checks[p+'_orbit'] = stats['bpm_orbit_rms_m'] <= metadata['orbit_rms_tolerance_m']
        checks[p+'_tune'] = abs(stats['tune_error']) < metadata['tune_tolerance']
        checks[p+'_beta_rms'] = stats['beta_beating_rms'] < metadata['beta_rms_tolerance']
        checks[p+'_beta_max'] = stats['beta_beating_max_abs'] < metadata['beta_max_tolerance']
        checks[p+'_dispersion_rms'] = stats['dispersion_error_rms_bpms_ips_m'] < metadata[prefix+'dispersion_rms_tolerance_m']
        checks[p+'_dispersion_ip'] = stats['dispersion_error_ip_max_abs_m'] < metadata[prefix+'dispersion_ip_tolerance_m']
        print(f'{p}: BPM orbit RMS {1e6*rms(bpm_orbit[p]):.4g} um; '
              f'beta RMS/max {100*rms(beta[p]):.3f}/{100*np.max(np.abs(beta[p])):.3f}%; '
              f'dispersion RMS {1e3*rms(dispersion[p][obs_idx]):.4f} mm', flush=True)
        print(f'  Largest beta error at {tw.name[peak]}, s={tw.s[peak]/1e3:.3f} km', flush=True)
    for key, values in rdts.items():
        peak = int(np.argmax(np.abs(values)))
        stats = {'error_rms_bpms_ips': rms(values[obs_idx]),
                 'error_ip_max_abs': float(np.max(np.abs(values[ip_idx]))),
                 'error_full_ring_max_abs': float(abs(values[peak])),
                 'peak_element': str(tw.name[peak]),
                 'peak_s_m': float(tw.s[peak])}
        summary['rdts'][key] = stats
        checks[key+'_rms'] = stats['error_rms_bpms_ips'] < metadata['coupling_rms_tolerance']
        checks[key+'_ip'] = stats['error_ip_max_abs'] < metadata['coupling_ip_tolerance']
        print(f'{key}: BPM + IP RMS {rms(values[obs_idx]):.4g}; '
              f'full-ring peak {abs(values[peak]):.4g} at {tw.name[peak]}, '
              f's={tw.s[peak]/1e3:.3f} km', flush=True)
    for name, idx in zip(IP_NAMES, ip_idx):
        summary['ips'][name] = {
            's_m': float(tw.s[idx]),
            **{f'beta_beating_{p}': float(beta[p][idx]) for p in 'xy'},
            **{f'dispersion_error_{p}_m': float(dispersion[p][idx]) for p in 'xy'},
            **{key+'_error_abs': float(abs(values[idx])) for key, values in rdts.items()},
        }
    print('c_minus:', tw.c_minus, flush=True)
    print('Final acceptance checks:', 'PASS' if all(checks.values()) else
          'FAIL: ' + ', '.join(key for key, ok in checks.items() if not ok), flush=True)

    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    figures = {}
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(13, 11),
                             height_ratios=[2, 1, 1], layout='constrained')
    for j, p in enumerate('xy'):
        axes[0, j].semilogy(s_km, ref['bet'+p], lw=1.2, label='Ideal')
        axes[0, j].semilogy(s_km, tw['bet'+p], '--', lw=0.9, label='Corrected')
        axes[0, j].set_ylabel(rf'$\beta_{p}$ [m]')
        axes[0, j].legend(loc='upper right')
        axes[1, j].plot(s_km, 100*beta[p], lw=0.8,
                        label=f'RMS {100*rms(beta[p]):.2f}%')
        axes[1, j].axhline(0, color='0.4', lw=0.6)
        axes[1, j].set_ylabel(rf'$\Delta\beta_{p}/\beta_{{{p},\mathrm{{ideal}}}}$ [%]')
        axes[1, j].legend(loc='upper right', fontsize=8)
    for j, key in enumerate(('f1001', 'f1010')):
        axes[2, j].plot(s_km, np.abs(ref[key]), lw=1.2, label='Ideal')
        axes[2, j].plot(s_km, np.abs(tw[key]), '--', lw=0.9, label='Corrected')
        axes[2, j].plot(s_km[ip_idx], np.abs(tw[key][ip_idx]), 'o',
                        color='tab:green', ms=4, label='Corrected at IPs')
        axes[2, j].set_ylabel(rf'$|f_{{{key[1:]}}}|$')
        axes[2, j].set_xlabel('s [km]')
        axes[2, j].legend(loc='upper right', fontsize=8)
    mark_ips(axes, tw)
    fig.suptitle('Ideal and corrected beta functions, beta beating and coupling RDTs')
    figures['beta_functions'] = fig

    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(13, 9), layout='constrained')
    for j, p in enumerate('xy'):
        axes[0, j].plot(s_km, 100*beta[p], lw=0.7,
                        label=f'RMS {100*rms(beta[p]):.2f}%')
        axes[0, j].set_ylabel(f'Beta beating {p} [%]')
        axes[1, j].plot(s_km[bpm_idx], 1e6*bpm_orbit[p], '.', ms=2,
                        label=f'RMS {1e6*rms(bpm_orbit[p]):.3g} µm')
        axes[1, j].set_ylabel(f'Measured BPM residual {p} [µm]')
        axes[2, j].plot(s_km, 1e3*dispersion[p], lw=0.7,
                        label=f'BPM + IP RMS {1e3*rms(dispersion[p][obs_idx]):.3f} mm')
        axes[2, j].set_ylabel(f'Dispersion error {p} [mm]')
        axes[2, j].set_xlabel('s [km]')
    for ax in axes.flat:
        ax.legend(loc='upper right', fontsize=8)
    mark_ips(axes, tw)
    fig.suptitle('Corrected machine relative to the clean reference')
    figures['orbit_optics_dispersion'] = fig

    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(13, 7), layout='constrained')
    for j, key in enumerate(rdts):
        axes[0, j].plot(s_km, np.abs(tw[key]), lw=0.8, label='Corrected')
        axes[0, j].plot(s_km, np.abs(ref[key]), '--', lw=0.8, label='Clean')
        axes[0, j].plot(s_km[ip_idx], np.abs(tw[key][ip_idx]), 'o', ms=4, label='IPs')
        axes[0, j].set_ylabel(f'|{key}|, full ring')
        axes[1, j].plot(s_km[obs_idx], rdts[key][obs_idx].real, '.', ms=2, label='Real')
        axes[1, j].plot(s_km[obs_idx], rdts[key][obs_idx].imag, '.', ms=2, label='Imaginary')
        axes[1, j].set_ylabel(f'Δ{key} at BPMs + IPs')
        axes[1, j].set_xlabel('s [km]')
    for ax in axes.flat:
        ax.legend(loc='upper right', fontsize=8)
    mark_ips(axes, tw)
    fig.suptitle(f'Tracked Edwards–Teng coupling RDTs; c_minus = {tw.c_minus:.3g}')
    figures['coupling'] = fig

    fig, axes = plt.subplots(3, 2, figsize=(11, 9), layout='constrained')
    for j, p in enumerate('xy'):
        axes[0, j].bar(IP_NAMES, 100*beta[p][ip_idx])
        axes[0, j].set_ylabel(f'Beta beating {p} [%]')
        axes[1, j].bar(IP_NAMES, 1e6*dispersion[p][ip_idx])
        axes[1, j].set_ylabel(f'Dispersion error {p} [µm]')
        prefix = 'horizontal_' if p == 'x' else ''
        limit = 1e6*metadata[prefix+'dispersion_ip_tolerance_m']
        axes[1, j].axhline(limit, color='tab:red', ls='--', label='Final IP limit')
        axes[1, j].axhline(-limit, color='tab:red', ls='--')
        axes[1, j].legend(fontsize=8)
    for j, key in enumerate(rdts):
        axes[2, j].bar(IP_NAMES, np.abs(rdts[key][ip_idx]))
        axes[2, j].set_ylabel(f'|{key} − clean reference|')
        axes[2, j].axhline(metadata['coupling_ip_tolerance'], color='tab:red',
                          ls='--', label='Final IP limit')
        axes[2, j].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(axis='y', alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle('Collision-point residuals')
    figures['interaction_points'] = fig

    for name, fig in figures.items():
        path = output/(name+'.png')
        fig.savefig(path, dpi=180)
        print('Saved', path, flush=True)
    print('Saved', output/'summary.json', flush=True)
    if args.show:
        plt.show()
    else:
        plt.close('all')


if __name__ == '__main__':
    main()
