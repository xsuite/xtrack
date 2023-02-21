from functools import partial

import numpy as np
from scipy.optimize import fsolve, minimize

from .jacobian import jacobian

def _error_for_match(knob_values, vary, targets, tracker, return_norm,
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
            res_values.append(tw[tt.tar])
        else:
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
        print(f'x = {knob_values}   f(x) = {res_values}')

    if return_norm:
        return np.sqrt((err_values*err_values).sum())
    else:
        return np.array(err_values)

class Vary:
    def __init__(self, name, limits=None, step=None):
        self.name = name
        if limits is None:
            limits = [-1e30, 1e30]
        self.limits = limits
        self.step = step


class Target:
    def __init__(self, tar, value, tol=None):
        self.tar = tar
        self.value = value
        self.tol = tol

def match_tracker(tracker, vary, targets, restore_if_fail=True, solver=None,
                  verbose=False, **kwargs):

    if isinstance(vary, (str, Vary)):
        vary = [vary]

    for ii, vv in enumerate(vary):
        if isinstance(vv, Vary):
            pass
        elif isinstance(vv, str):
            vary[ii] = Vary(vv)
        elif isinstance(vv, (list, tuple)):
            vary[ii] = Vary(*vv)
        else:
            raise ValueError(f'Invalid vary element {vv}')

    for ii, tt in enumerate(targets):
        if isinstance(tt, Target):
            pass
        elif isinstance(tt, (list, tuple)):
            targets[ii] = Target(*tt)
        else:
            raise ValueError(f'Invalid target element {tt}')

    if solver is None:
        if len(targets) == len(vary):
            solver = 'fsolve'
        else:
            solver = 'bfgs'

    # Assert that if one vary has a step, all vary have a step
    if any([vv.step is not None for vv in vary]):
        if not all([vv.step is not None for vv in vary]):
            raise NotImplementedError('All vary must have the same step (for now).')

    # Assert that all vary have the same step
    steps = [vv.step for vv in vary]
    if (steps[0] is not None
            and not np.all(np.isclose(steps, steps[0], atol=0, rtol=1e-14))):
        raise NotImplementedError('All vary must have the same step (for now).')
    step = steps[0]

    assert solver in ['fsolve', 'bfgs', 'jacobian'], f'Invalid solver {solver}.'

    if solver == 'fsolve':
        return_norm = False
    elif solver == 'bfgs':
        return_norm = True
    elif solver == 'jacobian':
        return_norm = False

    call_counter = {'n': 0}
    _err = partial(_error_for_match, vary=vary, targets=targets,
                   call_counter=call_counter, verbose=verbose,
                   tracker=tracker, return_norm=return_norm, tw_kwargs=kwargs)
    x0 = [tracker.vars[vv.name]._value for vv in vary]
    try:
        if solver == 'fsolve':
            options = {}
            if step is not None:
                options['epsfcn'] = step
            (res, infodict, ier, mesg) = fsolve(_err, x0=x0.copy(),
                full_output=True, **options)
            if ier != 1:
                raise RuntimeError("fsolve failed: %s" % mesg)
            result_info = {
                'res': res, 'info': infodict, 'ier': ier, 'mesg': mesg}
        elif solver == 'bfgs':
            options = {}
            if step is not None:
                options['eps'] = step
            optimize_result = minimize(_err, x0=x0.copy(), method='L-BFGS-B',
                        bounds=([vv.limits for vv in vary]), options=options)
            result_info = {'optimize_result': optimize_result}
            res = optimize_result.x
        elif solver == 'jacobian':
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
        for kk, vv in zip(vary, res):
            tracker.vars[kk] = vv
    except Exception as err:
        if restore_if_fail:
            for ii, vv in enumerate(vary):
                tracker.vars[vv] = x0[ii]
        print('\n')
        raise err
    print('\n')
    return result_info
