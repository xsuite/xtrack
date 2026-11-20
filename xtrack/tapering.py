import numpy as np
from scipy.constants import c as clight

from .general import _print

import xtrack as xt
import xobjects as xo

def compensate_radiation_energy_loss(line, delta0='zero_mean', rtol_eneloss=1e-12,
                                     max_iter=100, verbose=True,
                                     co_search_at=None, taper_magnets=True,
                                     **kwargs):

    assert isinstance(line._context, xo.ContextCpu), "Only CPU context is supported"
    assert line.particle_ref is not None, "Particle reference is not set"
    assert np.abs(line.particle_ref.q0) == 1, "Only |q0| = 1 is supported (for now)"

    enames_no_autodrift = [nn for nn in line.element_names if not nn.startswith('||drift_')]
    if len(set(enames_no_autodrift)) != len(enames_no_autodrift):
        raise ValueError("Line must not contain repeated elements to use "
                         "`compensate_radiation_energy_loss(...)`. ")

    ele_type = line.tracker._tracker_data_base._line_table.element_type
    ele_type_set = set(ele_type)
    for et in ele_type_set:
        if 'SliceCavity' in et:
            raise ValueError(
                f"Element type '{et}' is not supported for radiation energy "
                "loss compensation.")

    if 'record_iterations' in kwargs:
        record_iterations = kwargs['record_iterations']
        kwargs.pop('record_iterations')
        line._tapering_iterations = []
    else:
        record_iterations = False

    cache = line.attr._cache
    delta_taper_mask = cache['delta_taper'].mask
    delta_taper_setter = cache['delta_taper'].multisetter
    delta_taper_original = delta_taper_setter.get_values().copy()
    taper_names = np.array(line.element_names)[delta_taper_mask]
    if isinstance(taper_magnets, (bool, np.bool_)):
        taper_all = bool(taper_magnets)
        selected = np.full(len(taper_names), taper_all, dtype=bool)
    elif isinstance(taper_magnets, (list, tuple, np.ndarray)):
        taper_all = False
        invalid = set(taper_magnets) - set(taper_names)
        if invalid:
            raise ValueError(f'Elements not available for tapering: {sorted(invalid)}')
        selected = np.isin(taper_names, taper_magnets)
    else:
        raise ValueError('taper_magnets must be True, False or a list of element names')
    if not taper_all and co_search_at is not None:
        raise ValueError('co_search_at requires taper_magnets=True')
    if line._radiation_model in ('quantum', 'quantum-kick'):
        raise ValueError('Energy compensation requires deterministic radiation')
    if max_iter < 1:
        raise ValueError('max_iter must be at least one')

    # Voltages, frequencies and harmonics stay at their configured values.
    voltage = cache['_own_voltage'].multisetter.get_values()
    active = voltage != 0
    cavity_mask = cache['_own_voltage'].mask
    phase_setter = cache['_own_phase_taper'].multisetter
    phase_original = phase_setter.get_values().copy()
    base_phase = (cache['_own_phase'].multisetter.get_values()
                  + np.deg2rad(cache['_own_lag'].multisetter.get_values()
                               + cache['_own_lag_taper'].multisetter.get_values()))
    beta0 = float(line.particle_ref.beta0[0])
    p0c = float(line.particle_ref.p0c[0])
    line_length = line.get_length()
    energy_tol = float(line.particle_ref.energy0[0]) * rtol_eneloss
    frequency = (cache['_own_frequency'].multisetter.get_values()
                 + cache['_own_harmonic'].multisetter.get_values()
                 * beta0 * clight / line_length)
    wave_number = 2 * np.pi * frequency / (beta0 * clight)
    zero_mean = delta0 == 'zero_mean'
    delta_start = 0. if zero_mean else float(delta0)
    delta_taper = delta_taper_original.copy()

    if verbose: _print('Compensating energy loss.')
    try:
        with xt.line._preserve_track_flags(line):
            line.tracker.track_flags.XS_FLAG_SR_TAPER = taper_all
            line.tracker.track_flags.XS_FLAG_KILL_CAVITY_KICK = True
            with xt.line._preserve_config(line):
                line.config.XTRACK_MULTIPOLE_NO_SYNRAD = True
                particle_on_co = line.find_closed_orbit(
                    method='4d', delta0=delta_start, zeta0=0,
                    co_search_at=co_search_at)

            # Estimate radiation loss and arrival times to seed the RF phases.
            line.track(particle_on_co.copy(), turn_by_turn_monitor='ONE_TURN_EBE')
            mon = line.record_last_track
            if np.any(mon.state[0] <= 0):
                raise RuntimeError('Particle lost while estimating radiation energy loss')
            loss = -(mon.ptau[0, -1] - mon.ptau[0, 0]) * p0c
            if not np.any(active) or voltage.sum() == 0:
                if abs(loss) < energy_tol:
                    if verbose: _print('  - No compensation needed')
                    return
                raise ValueError('Energy compensation requires nonzero total RF voltage')
            partition = voltage / voltage.sum()
            gain_ratio = loss * partition[active] / voltage[active]
            if np.any(np.abs(gain_ratio) >= 1):
                raise ValueError('Insufficient RF voltage to compensate radiation energy loss')
            phase = phase_original.copy()
            phase[active] = (np.pi - np.arcsin(gain_ratio)
                             + wave_number[active] * mon.zeta[0, :-1][cavity_mask][active]
                             - base_phase[active])
            phase_setter.set_values(phase)
            line.tracker.track_flags.XS_FLAG_KILL_CAVITY_KICK = False
            guess = particle_on_co.copy()

            for i_iter in range(max_iter):
                if taper_all:
                    # Automatic tapering follows the tracked momentum without
                    # repeating the transverse closed-orbit search.
                    orbit = particle_on_co.copy()
                    orbit.delta = delta_start
                    orbit.zeta = 0
                else:
                    orbit = line.find_closed_orbit(
                        method='5d', delta0=delta_start, co_guess=guess)
                line.track(orbit.copy(), turn_by_turn_monitor='ONE_TURN_EBE')
                mon = line.record_last_track
                if np.any(mon.state[0] <= 0):
                    raise RuntimeError('Particle lost during radiation energy compensation')
                if record_iterations:
                    line._tapering_iterations.append(mon)

                energy_change = np.diff(mon.ptau[0]) * p0c
                gain = energy_change[cavity_mask]
                loss = -energy_change[~cavity_mask].sum()
                target_gain = loss * partition
                delta_mean = xt.twiss.trapz(mon.delta[0], mon.s[0]) / line_length
                energy_error = energy_change.sum()
                sharing_error = np.max(np.abs(gain - target_gain))
                target_taper = (0.5 * (mon.delta[0, :-1] + mon.delta[0, 1:]))[delta_taper_mask]
                taper_error = 0.
                if not taper_all and np.any(selected):
                    taper_error = np.max(np.abs(target_taper[selected] - delta_taper[selected]))
                converged = (abs(energy_error) < energy_tol
                             and sharing_error < energy_tol
                             and taper_error < rtol_eneloss
                             and (not zero_mean or abs(delta_mean) < rtol_eneloss))
                if verbose:
                    _print(f'  - Iteration {i_iter + 1}: energy imbalance '
                           f'{energy_error:.3g} eV, mean delta {delta_mean:.6g}')

                # Put the reference at zeta=0, and adjust the energy gains on
                # the stable RF branch above transition.
                phase[active] -= wave_number[active] * orbit.zeta[0]
                if not converged:
                    target_ratio = target_gain[active] / voltage[active]
                    if np.any(np.abs(target_ratio) >= 1):
                        raise ValueError('Insufficient RF voltage to compensate radiation energy loss')
                    actual_ratio = np.clip(gain[active] / voltage[active], -1, 1)
                    phase[active] += np.arcsin(actual_ratio) - np.arcsin(target_ratio)
                phase_setter.set_values(phase)
                if converged:
                    if taper_all:
                        delta_taper_setter.set_values(target_taper)
                    return

                if not taper_all and np.any(selected):
                    delta_taper[selected] = target_taper[selected]
                    delta_taper_setter.set_values(delta_taper)
                if zero_mean:
                    delta_start -= delta_mean
                guess = orbit.copy()
                guess.zeta = 0

            raise RuntimeError('Maximum number of iterations reached')
    except Exception:
        phase_setter.set_values(phase_original)
        if np.any(selected):
            delta_taper_setter.set_values(delta_taper_original)
        raise
