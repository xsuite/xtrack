# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from functools import partial

import numpy as np

from ..general import _print
from .constants import (
    AT_TURN_FOR_TWISS,
    DEFAULT_CO_SEARCH_TOL,
    DEFAULT_NUM_TURNS_SEARCH_T_REV,
)

import xtrack as xt  # To avoid circular imports


class ClosedOrbitSearchError(Exception):
    pass


def find_closed_orbit_line(line, co_guess=None, particle_ref=None,
                      co_search_settings=None, zeta_shift=0,
                      delta0=None, zeta0=None,
                      start=None, end=None, num_turns=1,
                      co_search_at=None,
                      search_for_t_rev=False,
                      continue_on_closed_orbit_error=False,
                      num_turns_search_t_rev=None,
                      symmetrize=False,
                      spin=True):

    if search_for_t_rev:
        assert line.particle_ref is not None
        assert co_guess is None, '``co_guess`` not supported when ``search_for_t_rev`` is True'
        assert co_search_settings is None, '``co_search_settings`` not supported when ``search_for_t_rev`` is True'
        assert zeta_shift == 0, '``zeta_shift`` not supported when ``search_for_t_rev`` is True'
        assert delta0 is None, '``delta0`` not supported when ``search_for_t_rev`` is True'
        assert zeta0 is None, '``zeta0`` not supported when ``search_for_t_rev`` is True'
        assert start is None, '``start`` not supported when ``search_for_t_rev`` is True'
        assert end is None, '``end`` not supported when ``search_for_t_rev`` is True'
        assert num_turns == 1, '``num_turns`` not supported when ``search_for_t_rev`` is True'
        assert co_search_at is None, '``co_search_at`` not supported when ``search_for_t_rev`` is True'
        assert continue_on_closed_orbit_error is False, '``continue_on_closed_orbit_error`` not supported when ``search_for_t_rev`` is True'
        assert symmetrize is False, '``symmetrize`` not supported when ``search_for_t_rev`` is True'

        out = _find_closed_orbit_search_t_rev(line, num_turns_search_t_rev)
        return out

    if line.enable_time_dependent_vars:
        raise RuntimeError(
            'Time-dependent vars not supported in closed orbit search')

    if start is not None and end is not None:
        co_search_at = None # needs to be implemented

    if co_search_at is not None:
        assert not symmetrize, 'Symmetrize not supported when ``co_search_at`` is provided'
        kwargs = locals().copy()
        kwargs.pop('start')
        kwargs.pop('end')
        kwargs.pop('co_search_at')
        p_co_at_ele_co_search = find_closed_orbit_line(
            start=co_search_at, end=co_search_at,
            **kwargs)
        line.track(p_co_at_ele_co_search, ele_start=co_search_at, ele_stop=0)
        return p_co_at_ele_co_search

    if isinstance(start, str):
        start = line._element_names_unique.index(start)

    if isinstance(end, str):
        end = line._element_names_unique.index(end)

    if isinstance(co_guess, dict):
        co_guess = line.build_particles(**co_guess, include_collective=True)

    if co_guess is None:
        if particle_ref is None:
            if line.particle_ref is not None:
                particle_ref = line.particle_ref
            else:
                raise ValueError(
                    "Either ``co_guess`` or ``particle_ref`` must be provided")

        co_guess = particle_ref.copy()
        co_guess.x = 0
        co_guess.px = 0
        co_guess.y = 0
        co_guess.py = 0
        co_guess.zeta = 0
        co_guess.delta = 0
        co_guess.s = 0
        co_guess.at_element = (start or 0)
        co_guess.at_turn = 0
    else:
        particle_ref = co_guess

    if co_search_settings is None:
        co_search_settings = {}

    co_search_settings = co_search_settings.copy()
    if 'xtol' not in co_search_settings.keys():
        co_search_settings['xtol'] = 1e-6 # Relative error between calls

    co_guess = co_guess.copy(
                        _context=line._buffer.context)

    for shift_factor in [0, 1.]: # if not found at first attempt we shift slightly the starting point
        if shift_factor>0:
            _print('Warning! Need second attempt on closed orbit search')

        x0=np.array([co_guess._xobject.x[0] + shift_factor * 1e-5,
                    co_guess._xobject.px[0] + shift_factor * 1e-7,
                    co_guess._xobject.y[0] + shift_factor * 1e-5,
                    co_guess._xobject.py[0] + shift_factor * 1e-7,
                    co_guess._xobject.zeta[0] + shift_factor * 1e-4,
                    co_guess._xobject.delta[0] + shift_factor * 1e-5])
        if delta0 is not None and zeta0 is None:
            x0[5] = delta0
            _error_for_co = _error_for_co_search_4d_delta0
        elif delta0 is None and zeta0 is not None:
            x0[4] = zeta0
            _error_for_co = _error_for_co_search_4d_zeta0
        elif delta0 is not None and zeta0 is not None:
            _error_for_co = _error_for_co_search_4d_delta0_zeta0
        else:
            _error_for_co = _error_for_co_search_6d
        if zeta0 is not None:
            x0[4] = zeta0
        if np.all(np.abs(_error_for_co(
                x0, co_guess, line, zeta_shift, delta0, zeta0,
                start=start, end=end,
                num_turns=num_turns,
                symmetrize=symmetrize)) < DEFAULT_CO_SEARCH_TOL):
            res = x0
            fsolve_info = 'taken_guess'
            ier = 1
            break

        opt = xt.match.opt_from_callable(
            lambda p: _error_for_co(p, co_guess, line,
                            zeta_shift, delta0, zeta0, start=start,
                            end=end, num_turns=num_turns,
                            symmetrize=symmetrize),
                x0=x0, steps=[1e-8, 1e-9, 1e-8, 1e-9, 1e-7, 1e-8],
                tar=[0., 0., 0., 0., 0., 0.],
                tols=[1e-12, 1e-12, 1e-12, 1e-12, 1e-11, 1e-12])
        try:
            opt.solve(verbose=-1)
            ier = 1
        except Exception as e:
            ier = -1
        fsolve_info = opt

        res = opt.vary[0].container
        if ier == 1:
            break

    if ier != 1 and not(continue_on_closed_orbit_error):
        raise ClosedOrbitSearchError

    particle_on_co = co_guess.copy()
    particle_on_co.x = res[0]
    particle_on_co.px = res[1]
    particle_on_co.y = res[2]
    particle_on_co.py = res[3]
    particle_on_co.zeta = res[4] - zeta_shift
    particle_on_co.delta = res[5]

    particle_on_co._fsolve_info = fsolve_info

    if spin:
        from .spin import _find_spin_fixed_point
        spin_x, spin_y, spin_z = _find_spin_fixed_point(line, particle_on_co)
        particle_on_co.spin_x = spin_x
        particle_on_co.spin_y = spin_y
        particle_on_co.spin_z = spin_z

    return particle_on_co


def _one_turn_map(p, particle_ref, line, zeta_shift, start, end, num_turns, symmetrize):
    part = particle_ref.copy()
    part.x = p[0]
    part.px = p[1]
    part.y = p[2]
    part.py = p[3]
    part.zeta = p[4] - zeta_shift
    part.delta = p[5]
    part.at_turn = AT_TURN_FOR_TWISS

    if line.energy_program is not None:
        dp0c = line.energy_program.get_p0c_increse_per_turn_at_t_s(
                                                        line['t_turn_s'])
        part.update_p0c_and_energy_deviations(p0c = part._xobject.p0c[0] + dp0c)

    line.track(part, ele_start=start, ele_stop=end, num_turns=num_turns)
    if symmetrize:
        assert num_turns == 1
        with xt.line._preserve_config(line):
            line.config.XSUITE_MIRROR = True
            line.track(part, ele_start=start, ele_stop=end, num_turns=1)
    if part.state[0] < 0:
        raise ClosedOrbitSearchError(
            f'Particle lost, p.state = {part.state[0]}')
    p_res = np.array([
           part._xobject.x[0],
           part._xobject.px[0],
           part._xobject.y[0],
           part._xobject.py[0],
           part._xobject.zeta[0],
           part._xobject.delta[0]])
    return p_res


def _error_for_co_search_6d(p, co_guess, line, zeta_shift, delta0, zeta0, start, end, num_turns, symmetrize):
    return p - _one_turn_map(p, co_guess, line, zeta_shift, start, end, num_turns, symmetrize)


def _error_for_co_search_4d_delta0(p, co_guess, line, zeta_shift, delta0, zeta0, start, end, num_turns, symmetrize):
    one_turn_res = _one_turn_map(p, co_guess, line, zeta_shift, start, end, num_turns, symmetrize)
    return np.array([
        p[0] - one_turn_res[0],
        p[1] - one_turn_res[1],
        p[2] - one_turn_res[2],
        p[3] - one_turn_res[3],
        0,
        p[5] - delta0])


def _error_for_co_search_4d_zeta0(p, co_guess, line, zeta_shift, delta0, zeta0, start, end, num_turns, symmetrize):
    one_turn_res = _one_turn_map(p, co_guess, line, zeta_shift, start, end, num_turns, symmetrize)
    return np.array([
        p[0] - one_turn_res[0],
        p[1] - one_turn_res[1],
        p[2] - one_turn_res[2],
        p[3] - one_turn_res[3],
        p[4] - zeta0,
        0])


def _error_for_co_search_4d_delta0_zeta0(p, co_guess, line, zeta_shift, delta0, zeta0, start, end, num_turns, symmetrize):
    one_turn_res = _one_turn_map(p, co_guess, line, zeta_shift, start, end, num_turns, symmetrize)
    return np.array([
        p[0] - one_turn_res[0],
        p[1] - one_turn_res[1],
        p[2] - one_turn_res[2],
        p[3] - one_turn_res[3],
        p[4] - zeta0,
        p[5] - delta0])


def _merit_function_co_t_rev(x, line, num_turns):
    p = line.build_particles(x=x[0], px=x[1], y=x[2], py=x[3], zeta=x[4], delta=x[5])
    line.track(p, num_turns=num_turns, turn_by_turn_monitor=True)
    rec = line.record_last_track
    dx = rec.x[0, :] - rec.x[0, 0]
    dpx = rec.px[0, :] - rec.px[0, 0]
    dy = rec.y[0, :] - rec.y[0, 0]
    dpy = rec.py[0, :] - rec.py[0, 0]
    ddelta = rec.delta[0, :] - rec.delta[0, 0]

    out = np.array(list(dx) + list(dpx) + list(dy) + list(dpy) + list(ddelta))
    return out


def _find_closed_orbit_search_t_rev(line, num_turns_search_t_rev=None):

    if num_turns_search_t_rev is None:
        num_turns_search_t_rev = DEFAULT_NUM_TURNS_SEARCH_T_REV

    opt = xt.match.opt_from_callable(partial(_merit_function_co_t_rev,
                        line=line, num_turns=num_turns_search_t_rev),
                        x0=np.array(6*[0.]),
                        steps=[1e-9, 1e-10, 1e-9, 1e-10, 1e-4, 1e-7],
                        tar=np.array(5*num_turns_search_t_rev*[0.]),
                        tols=np.array(5*num_turns_search_t_rev*[1e-10]))
    opt.solve()
    x_sol = opt.get_knob_values()
    particle_on_co = line.build_particles(
        x=x_sol[0], px=x_sol[1], y=x_sol[2], py=x_sol[3], zeta=x_sol[4],
        delta=x_sol[5])

    return particle_on_co
