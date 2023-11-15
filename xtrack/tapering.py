import numpy as np
from scipy.constants import c as clight

from .general import _print

import xtrack as xt
import xobjects as xo

def compensate_radiation_energy_loss(line, delta0=0, rtol_eneloss=1e-12,
                                     max_iter=100, verbose=True, **kwargs):

    assert isinstance(line._context, xo.ContextCpu), "Only CPU context is supported"
    assert line.particle_ref is not None, "Particle reference is not set"
    assert np.abs(line.particle_ref.q0) == 1, "Only |q0| = 1 is supported (for now)"

    if 'record_iterations' in kwargs:
        record_iterations = kwargs['record_iterations']
        kwargs.pop('record_iterations')
        line._tapering_iterations = []
    else:
        record_iterations = False

    if verbose: _print("Compensating energy loss.")

    line.config.XTRACK_MULTIPOLE_NO_SYNRAD = True
    with xt.freeze_longitudinal(line):
        particle_on_co = line.find_closed_orbit()
    line.config.XTRACK_MULTIPOLE_NO_SYNRAD = False

    beta0 = float(particle_on_co._xobject.beta0[0])

    # Check whether compensation is needed
    p_test = particle_on_co.copy()
    p_test.delta = delta0
    line.track(p_test, turn_by_turn_monitor='ONE_TURN_EBE')
    mon = line.record_last_track
    eloss = -(mon.ptau[0, -1] - mon.ptau[0, 0]) * p_test.p0c[0]
    if p_test.state[0] > 0 and abs(eloss) < p_test.energy0[0] * rtol_eneloss:
        if verbose: _print("  - No compensation needed")
        return

    # save voltages
    v_setter = line.attr._cache['voltage'].multisetter
    f_setter = line.attr._cache['frequency'].multisetter
    lag_setter = line.attr._cache['lag'].multisetter
    lag_taper_setter = line.attr._cache['lag_taper'].multisetter

    v0 = v_setter.get_values()
    f0 = f_setter.get_values()
    lag_zero = lag_setter.get_values()

    eneloss_partitioning = v0 / v0.sum()

    # Put all cavities on crest and at zero frequency
    lag_taper_setter.set_values(90. - lag_zero)
    v_setter.set_values(np.zeros_like(v_setter.get_values()))
    f_setter.set_values(np.zeros_like(f_setter.get_values()))

    if verbose: _print("Share energy loss among cavities (repeat until energy loss is zero)")
    with xt.line._preserve_config(line):
        line.config.XTRACK_MULTIPOLE_TAPER = True
        line.config.XTRACK_DIPOLEEDGE_TAPER = True

        i_iter = 0
        while True:
            p_test = particle_on_co.copy()
            p_test.delta = delta0
            line.track(p_test, turn_by_turn_monitor='ONE_TURN_EBE')
            mon = line.record_last_track

            if record_iterations:
                line._tapering_iterations.append(mon)

            eloss = -(mon.ptau[0, -1] - mon.ptau[0, 0]) * p_test.p0c[0]
            if verbose: _print(f"Energy loss: {eloss:.3f} eV             ")#, end='\r', flush=True)

            if eloss < p_test.energy0[0] * rtol_eneloss:
                break

            v_setter.set_values(v_setter.get_values()
                                + eloss * eneloss_partitioning)

            i_iter += 1
            if i_iter > max_iter:
                raise RuntimeError("Maximum number of iterations reached")
    if verbose: _print()
    delta_taper_full = 0.5*(mon.delta[0, :-1] + mon.delta[0, 1:]) # last point is added by the monitor

    if verbose: _print("  - Set delta_taper")
    delta_taper_mask = line.attr._cache['delta_taper'].mask
    delta_taper_setter = line.attr._cache['delta_taper'].multisetter
    delta_taper_setter.set_values(delta_taper_full[delta_taper_mask])

    if verbose: _print("  - Restore cavity voltage and frequency. Set cavity lag")
    v_synchronous = v_setter.get_values()

    mask_cav = line.attr._cache['voltage'].mask
    zeta_at_cav = np.atleast_1d(np.squeeze(mon.zeta[0, :-1]))[mask_cav]
    mask_active_cav = np.abs(v0) > 0
    v_ratio = v0 * 0
    v_ratio[mask_active_cav] = v_synchronous[mask_active_cav] / v0[mask_active_cav]
    assert np.all(np.abs(v_ratio[mask_active_cav]) < 1)
    inst_phase = np.arcsin(v_ratio)

    total_lag = 360.*(inst_phase / (2 * np.pi) - f0 * zeta_at_cav / beta0 / clight)
    total_lag = 180. - total_lag # we are above transition
    lag_taper = total_lag - lag_zero
    lag_taper[~mask_active_cav] = 0

    v_setter.set_values(v0)
    f_setter.set_values(f0)
    lag_taper_setter.set_values(lag_taper)

