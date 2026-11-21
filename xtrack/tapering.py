import numpy as np
from scipy.constants import c as clight

from .general import _print

import xtrack as xt
import xobjects as xo

def compensate_radiation_energy_loss(line, delta0='zero_mean', rtol_eneloss=1e-12,
                                     max_iter=100, verbose=True,
                                     co_search_at=None, taper_magnets=True, co_guess=None,
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

    # Resolve the magnet selection. True uses automatic tapering during tracking;
    # a list updates explicit delta_taper values between orbit searches.
    # The selection is indexed over taperable elements, not the full line.
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

    # Read the RF settings and save the values needed to restore the line on
    # failure. Only phase_taper is adjusted; base phases and RF strengths stay fixed.
    # cavity_mask selects elements in the line, cav_active selects powered cavities.
    voltage = cache['_own_voltage'].multisetter.get_values()
    cav_active = voltage != 0
    cavity_mask = cache['_own_voltage'].mask
    phase_setter = cache['_own_phase_taper'].multisetter
    phase_original = phase_setter.get_values().copy()
    base_phase = (cache['_own_phase'].multisetter.get_values()
                  + np.deg2rad(cache['_own_lag'].multisetter.get_values()
                               + cache['_own_lag_taper'].multisetter.get_values()))
    beta0 = float(line.particle_ref.beta0[0])
    p0c = float(line.particle_ref.p0c[0])
    line_length = line.tracker._tracker_data_base.line_length
    energy_tol = float(line.particle_ref.energy0[0]) * rtol_eneloss
    # Include both frequency and harmonic contributions. wave_number converts
    # a change in zeta into the corresponding RF phase shift.
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
            # Find a starting orbit without radiation or RF energy kicks.
            # Radiation is restored on leaving the config context below.
            line.tracker.track_flags.XS_FLAG_KILL_CAVITY_KICK = True
            with xt.line._preserve_config(line):
                line.config.XTRACK_MULTIPOLE_NO_SYNRAD = True
                particle_on_co_no_rad = line.find_closed_orbit(
                    method='4d', delta0=delta_start, zeta0=0,
                    co_search_at=co_search_at, co_guess=co_guess)

            # Track with radiation active and RF kicks disabled to estimate the
            # uncompensated loss and arrival times for the initial RF phases.
            line.track(particle_on_co_no_rad.copy(), turn_by_turn_monitor='ONE_TURN_EBE')
            mon = line.record_last_track
            if np.any(mon.state[0] <= 0):
                raise RuntimeError('Particle lost while estimating radiation energy loss')
            current_total_energy_loss = -(mon.ptau[0, -1] - mon.ptau[0, 0]) * p0c
            if not np.any(cav_active) or voltage.sum() == 0:
                if abs(current_total_energy_loss) < energy_tol:
                    if verbose: _print('  - No compensation needed')
                    return
                raise ValueError('Energy compensation requires nonzero total RF voltage')
            # Share the loss in proportion to cavity voltage. Choose the stable
            # phase above transition and account for arrival time at each cavity.
            partition = voltage / voltage.sum()
            target_rf_energy_gain = current_total_energy_loss * partition
            if np.any(np.abs(target_rf_energy_gain[cav_active]) >= np.abs(voltage[cav_active])):
                raise ValueError('Insufficient RF voltage to compensate radiation energy loss')
            phase_taper = phase_original.copy()
            phase_taper[cav_active] = (
                np.pi - np.arcsin(target_rf_energy_gain[cav_active] / voltage[cav_active])
                + wave_number[cav_active] * mon.zeta[0, :-1][cavity_mask][cav_active]
                - base_phase[cav_active])
            phase_setter.set_values(phase_taper)
            line.tracker.track_flags.XS_FLAG_KILL_CAVITY_KICK = False
            particle_on_co_guess = particle_on_co_no_rad.copy()

            for i_iter in range(max_iter):
                if taper_all:
                    # Automatic tapering follows the tracked momentum without
                    # repeating the transverse closed-orbit search.
                    particle_on_co = particle_on_co_no_rad.copy()
                    particle_on_co.delta = delta_start
                    particle_on_co.zeta = 0
                else:
                    # With fixed magnet settings, close x, px, y, py and delta.
                    # The starting zeta is free and will be absorbed into RF phases.
                    particle_on_co = line.find_closed_orbit(
                        method='5d', delta0=delta_start, co_guess=particle_on_co_guess)
                line.track(particle_on_co.copy(), turn_by_turn_monitor='ONE_TURN_EBE')
                mon = line.record_last_track
                if np.any(mon.state[0] <= 0):
                    raise RuntimeError('Particle lost during radiation energy compensation')
                if record_iterations:
                    line._tapering_iterations.append(mon)

                # Measure the actual cavity gains and the loss elsewhere in the
                # ring, then compute the requested gain for each cavity.
                energy_change = np.diff(mon.ptau[0]) * p0c
                current_rf_energy_gain = energy_change[cavity_mask]
                current_total_energy_loss = -energy_change[~cavity_mask].sum()
                target_rf_energy_gain = current_total_energy_loss * partition
                delta_mean = xt.twiss.trapz(mon.delta[0], mon.s[0]) / line_length
                energy_error = (mon.ptau[0, -1] - mon.ptau[0, 0]) * p0c
                sharing_error = np.max(np.abs(current_rf_energy_gain - target_rf_energy_gain))
                # Use the average entrance/exit delta for each magnet's taper.
                target_taper = (0.5 * (mon.delta[0, :-1] + mon.delta[0, 1:]))[delta_taper_mask]
                taper_error = 0.
                if not taper_all and np.any(selected):
                    taper_error = np.max(np.abs(target_taper[selected] - delta_taper[selected]))
                # Check the change before storing the new taper. Unselected
                # magnets keep their values; automatic tapering overrides these
                # stored values during tracking when taper_all is True.
                if np.any(selected):
                    delta_taper[selected] = target_taper[selected]
                    delta_taper_setter.set_values(delta_taper)

                if verbose:
                    _print(f'  - Iteration {i_iter + 1}: energy imbalance '
                           f'{energy_error:.3g} eV, mean delta {delta_mean:.6g}')

                # We have a degree of freedom to shift all phases
                # We can use this freedom to impose zeta=0 at the start line
                # and make the result deterministic
                phase_taper[cav_active] -= wave_number[cav_active] * particle_on_co.zeta[0]

                # Set the adjusted RF phases in the cavity elements
                phase_setter.set_values(phase_taper)

                converged = (abs(energy_error) < energy_tol
                             and sharing_error < energy_tol
                             and taper_error < rtol_eneloss
                             and (not zero_mean or abs(delta_mean) < rtol_eneloss))
                if converged:
                    break

                # Adjust the RF phases to match the target energy gain
                if np.any(np.abs(target_rf_energy_gain[cav_active]) >= np.abs(voltage[cav_active])):
                    raise ValueError('Insufficient RF voltage to compensate radiation energy loss')
                phase_current = np.pi - np.arcsin(np.clip(
                    current_rf_energy_gain[cav_active] / voltage[cav_active], -1, 1))
                phase_target = np.pi - np.arcsin(
                    target_rf_energy_gain[cav_active] / voltage[cav_active])
                phase_taper[cav_active] += phase_target - phase_current
                phase_setter.set_values(phase_taper)

                if zero_mean:
                    # Shift the starting momentum to remove the measured mean.
                    delta_start -= delta_mean
                # Reuse the previous orbit as a guess in the new RF phase frame.
                particle_on_co_guess = particle_on_co.copy()
                particle_on_co_guess.zeta = 0

            else:
                raise RuntimeError('Maximum number of iterations reached')
    except Exception:
        # Do not leave partially compensated RF or magnet settings on failure.
        phase_setter.set_values(phase_original)
        if np.any(selected):
            delta_taper_setter.set_values(delta_taper_original)
        raise
