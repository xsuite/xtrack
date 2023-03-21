from functools import partial

import numpy as np
from scipy.optimize import fsolve, minimize

from .jacobian import jacobian
from .twiss import TwissInit
import xtrack as xt

class OrbitOnly:
    def __init__(self, x=0, px=0, y=0, py=0, zeta=0, delta=0):
        self.x = x
        self.px = px
        self.y = y
        self.py = py
        self.zeta = zeta
        self.delta = delta

def _error_for_match(knob_values, vary, targets, tracker, return_scalar,
                     call_counter, verbose, tw_kwargs):

    print(f"Matching: twiss call n. {call_counter['n']}       ", end='\r', flush=True)
    call_counter['n'] += 1

    for kk, vv in zip(vary, knob_values):
        tracker.vars[kk.name] = vv
    tw = tracker.twiss(**tw_kwargs)

    res_values = []
    target_values = []
    for tt in targets:
        if isinstance(tt.tar, str):
            if tt.at is not None:
                res_values.append(tw[tt.at, tt.tar])
            else:
                res_values.append(tw[tt.tar])
        else:
            assert tt.at is None, '`at` cannot be provided if target is a function'
            res_values.append(tt.tar(tw))
        target_values.append(tt.value)

    res_values = np.array(res_values)
    target_values = np.array(target_values)
    err_values = res_values - target_values

    tols = 0 * err_values
    for ii, tt in enumerate(targets):
        if tt.tol is not None:
            tols[ii] = tt.tol
        else:
            tols[ii] = 1e-14

    if np.all(np.abs(err_values) < tols):
        err_values *= 0
        if verbose:
            print('Found point within tolerance!')

    for ii, tt in enumerate(targets):
        if tt.scale is not None:
            err_values[ii] *= tt.scale

    if verbose:
        print(f'x = {knob_values}   f(x) = {res_values}')

    if return_scalar:
        return np.sum(err_values * err_values)
    else:
        return np.array(err_values)

def _jacobian(x, steps, fun):
    x = np.array(x).copy()
    steps = np.array(steps).copy()
    assert len(x) == len(steps)
    f0 = fun(x)
    if np.isscalar(f0):
        jac = np.zeros((1, len(x)))
    else:
        jac = np.zeros((len(f0), len(x)))
    for ii in range(len(x)):
        x[ii] += steps[ii]
        jac[:, ii] = (fun(x) - f0) / steps[ii]
        x[ii] -= steps[ii]
    return jac

class Vary:
    def __init__(self, name, limits=None, step=None):
        self.name = name
        self.limits = limits
        self.step = step


class Target:
    def __init__(self, tar, value, at=None, tol=None, scale=None):
        self.tar = tar
        self.value = value
        self.tol = tol
        self.at = at
        self.scale = scale

def match_tracker(tracker, vary, targets, restore_if_fail=True, solver=None,
                  verbose=False, **kwargs):

    if 'twiss_init' in kwargs and kwargs['twiss_init'] is not None:
        twiss_init = kwargs['twiss_init']
        assert 'ele_start' in kwargs and kwargs['ele_start'] is not None, (
            'ele_start must be provided if twiss_init is provided')
        if isinstance(twiss_init, OrbitOnly):
            if not isinstance(kwargs['ele_start'], str):
                element_name = tracker.line.element_names[kwargs['ele_start']]
            else:
                element_name = kwargs['ele_start']
            particle_on_co=tracker.build_particles(
                x=twiss_init.x, px=twiss_init.px,
                y=twiss_init.y, py=twiss_init.py,
                zeta=twiss_init.zeta, delta=twiss_init.delta)
            particle_on_co.at_element = tracker.line.element_names.index(
                                                                element_name)
            kwargs['twiss_init'] = TwissInit(
                particle_on_co=particle_on_co,
                W_matrix=np.eye(6),
                element_name=element_name)

    if isinstance(vary, (str, Vary)):
        vary = [vary]

    for ii, rr in enumerate(vary):
        if isinstance(rr, Vary):
            pass
        elif isinstance(rr, str):
            vary[ii] = Vary(rr)
        elif isinstance(rr, (list, tuple)):
            vary[ii] = Vary(*rr)
        else:
            raise ValueError(f'Invalid vary setting {rr}')

    for ii, tt in enumerate(targets):
        if isinstance(tt, Target):
            pass
        elif isinstance(tt, (list, tuple)):
            targets[ii] = Target(*tt)
        else:
            raise ValueError(f'Invalid target element {tt}')

    if 'ele_stop' in kwargs and kwargs['ele_stop'] is not None:
        ele_stop = kwargs['ele_stop']
        for tt in targets:
            if tt.at is not None and tt.at == ele_stop:
                tt.at = '_end_point'

    if solver is None:
        if len(targets) != len(vary):
            solver = 'bfgs'
        elif np.any([vv.limits is not None for vv in vary]):
            solver = 'bfgs'
        else:
            solver = 'fsolve'

    if verbose:
        print(f'Using solver {solver}')

    steps = []
    for vv in vary:
        if vv.step is not None:
            steps.append(vv.step)
        else:
            steps.append(1e-10)

    assert solver in ['fsolve', 'bfgs', 'jacobian'], (
                      f'Invalid solver {solver}.')

    return_scalar = {'fsolve': False, 'bfgs': True, 'jacobian': False}[solver]

    call_counter = {'n': 0}
    _err = partial(_error_for_match, vary=vary, targets=targets,
                call_counter=call_counter, verbose=verbose,
                tracker=tracker, return_scalar=return_scalar, tw_kwargs=kwargs)
    _jac= partial(_jacobian, steps=steps, fun=_err)
    x0 = [tracker.vars[vv.name]._value for vv in vary]
    try:
        if solver == 'fsolve':
            (res, infodict, ier, mesg) = fsolve(_err, x0=x0.copy(),
                full_output=True, fprime=_jac)
            if ier != 1:
                raise RuntimeError("fsolve failed: %s" % mesg)
            result_info = {
                'res': res, 'info': infodict, 'ier': ier, 'mesg': mesg}
        elif solver == 'bfgs':
            bounds =[]
            for vv in vary:
                if vv.limits is not None:
                    bounds.append(vv.limits)
                else:
                    bounds.append((-1e30, 1e30))
            optimize_result = minimize(_err, x0=x0.copy(), method='L-BFGS-B',
                        bounds=([vv.limits for vv in vary]),
                        jac=_jac, options={'gtol':0})
            result_info = {'optimize_result': optimize_result}
            res = optimize_result.x
        elif solver == 'jacobian':
            raise NotImplementedError # To be finalized and tested
            options = {}
            if step is not None:
                options['eps'] = step
            res, info = jacobian(
                                _err,
                                xstart=x0.copy(),
                                maskstart=None,
                                maxsteps=100,
                                bisec=5,
                                tol=1e-20,
                                maxcalls=10000,
                                debug=False,
                                **options
                                )
            result_info = info
            if not info['success']:
                raise RuntimeError("jacobian failed: %s" % info['message'])
        for vv, rr in zip(vary, res):
            tracker.vars[vv.name] = rr
    except Exception as err:
        if restore_if_fail:
            for ii, rr in enumerate(vary):
                tracker.vars[rr] = x0[ii]
        print('\n')
        raise err
    print('\n')
    return result_info

def closed_orbit_correction(tracker, tracker_co_ref, correction_config,
                            solver=None, verbose=False, restore_if_fail=True):

    for corr_name, corr in correction_config.items():
        print('Correcting', corr_name)
        with xt.tracker._temp_knobs(tracker, corr['ref_with_knobs']):
            tw_ref = tracker_co_ref.twiss(method='4d', zeta0=0, delta0=0)
        vary = [xt.Vary(vv, step=1e-9) for vv in corr['vary']]
        targets = []
        for tt in corr['targets']:
            assert isinstance(tt, str), 'For now only strings are supported for targets'
            for kk in ['x', 'px', 'y', 'py']:
                targets.append(xt.Target(kk, at=tt, value=tw_ref[tt, kk], tol=1e-9))

        tracker.match(
            solver=solver,
            verbose=verbose,
            restore_if_fail=restore_if_fail,
            vary=vary,
            targets=targets,
            twiss_init=xt.OrbitOnly(
                x=tw_ref[corr['start'], 'x'],
                px=tw_ref[corr['start'], 'px'],
                y=tw_ref[corr['start'], 'y'],
                py=tw_ref[corr['start'], 'py'],
                zeta=tw_ref[corr['start'], 'zeta'],
                delta=tw_ref[corr['start'], 'delta'],
            ),
            ele_start=corr['start'], ele_stop=corr['end'])