from functools import partial

import numpy as np
from scipy.optimize import fsolve, minimize

from .jacobian import jacobian
from .twiss import TwissInit
from .general import _print
import xtrack as xt

class OrbitOnly:
    def __init__(self, x=0, px=0, y=0, py=0, zeta=0, delta=0):
        self.x = x
        self.px = px
        self.y = y
        self.py = py
        self.zeta = zeta
        self.delta = delta

def _error_for_match(knob_values, vary, targets, line, return_scalar,
                     call_counter, verbose, tw_kwargs):

    _print(f"Matching: twiss call n. {call_counter['n']}       ", end='\r', flush=True)
    call_counter['n'] += 1

    for kk, vv in zip(vary, knob_values):
        line.vars[kk.name] = vv
    tw = line.twiss(**tw_kwargs)

    res_values = []
    target_values = []
    for tt in targets:
        res_values.append(tt.eval(tw))
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
            _print('Found point within tolerance!')

    for ii, tt in enumerate(targets):
        if tt.scale is not None:
            err_values[ii] *= tt.scale

    if verbose:
        _print(f'x = {knob_values}   f(x) = {res_values}')

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

class TargetList:
    def __init__(self, tars, **kwargs):
        self.targets = [Target(tt, **kwargs) for tt in tars]

class VaryList:
    def __init__(self, vars, **kwargs):
        self.vary_objects = [Vary(vv, **kwargs) for vv in vars]

class Vary:
    def __init__(self, name, limits=None, step=None):
        self.name = name
        self.limits = limits
        self.step = step

class Target:
    def __init__(self, tar, value, at=None, tol=None, scale=None, line=None):
        self.tar = tar
        self.value = value
        self.tol = tol
        self.at = at
        self.scale = scale
        self.line = line

    def eval(self, tw):

        if isinstance (tw, xt.MultiTwiss) and not callable(self.tar):
            if self.line is None:
                raise ValueError('When using `Multiline.match`, '
                    'a `line` must associated to each non-callable target.')

        if self.line is not None:
            assert isinstance(tw, xt.MultiTwiss), (
                'The line associated to a target can be provided only when '
                'using `Multiline.match`')
            this_tw = tw[self.line]
        else:
            this_tw = tw

        if isinstance(self.tar, str):
            if self.at is not None:
                return this_tw[self.tar, self.at]
            else:
                return this_tw[self.tar]
        elif callable(self.tar):
            assert self.at is None, '`at` cannot be provided if target is a function'
            return self.tar(this_tw)

def match_line(line, vary, targets, restore_if_fail=True, solver=None,
                  verbose=False, **kwargs):

    if 'twiss_init' in kwargs and kwargs['twiss_init'] is not None:
        twiss_init = kwargs['twiss_init']
        assert 'ele_start' in kwargs and kwargs['ele_start'] is not None, (
            'ele_start must be provided if twiss_init is provided')
        if isinstance(twiss_init, OrbitOnly):
            if not isinstance(kwargs['ele_start'], str):
                element_name = line.element_names[kwargs['ele_start']]
            else:
                element_name = kwargs['ele_start']
            particle_on_co=line.build_particles(
                x=twiss_init.x, px=twiss_init.px,
                y=twiss_init.y, py=twiss_init.py,
                zeta=twiss_init.zeta, delta=twiss_init.delta)
            particle_on_co.at_element = line.element_names.index(
                                                                element_name)
            kwargs['twiss_init'] = TwissInit(
                particle_on_co=particle_on_co,
                W_matrix=np.eye(6),
                element_name=element_name)

    if isinstance(vary, (str, Vary)):
        vary = [vary]

    input_vary = vary
    vary = []
    for ii, rr in enumerate(input_vary):
        if isinstance(rr, Vary):
            vary.append(rr)
        elif isinstance(rr, str):
            vary.append(Vary(rr))
        elif isinstance(rr, (list, tuple)):
            raise ValueError('Not anymore supported')
        elif isinstance(rr, VaryList):
            vary += rr.vary_objects
        else:
            raise ValueError(f'Invalid vary setting {rr}')

    if 'twiss_init' in kwargs and kwargs['twiss_init'] == 'preserve':
        full_twiss_kwargs = kwargs.copy()
        full_twiss_kwargs.pop('twiss_init')
        full_twiss_kwargs.pop('ele_start')
        full_twiss_kwargs.pop('ele_stop')
        tw0_full = line.twiss(**full_twiss_kwargs)
        if isinstance(tw0_full, xt.MultiTwiss):
            kwargs['twiss_init'] = []
            for ll, nn in zip(tw0_full._line_names, kwargs['ele_start']):
                kwargs['twiss_init'].append(tw0_full[ll].get_twiss_init(at_element=nn))
        else:
            kwargs['twiss_init'] = tw0_full.get_twiss_init(at_element=kwargs['ele_start'])

    tw0 = line.twiss(**kwargs)

    input_targets = targets
    targets = []
    for ii, tt in enumerate(input_targets):
        if isinstance(tt, Target):
            targets.append(tt)
        elif isinstance(tt, (list, tuple)):
            targets.append(Target(*tt))
        elif isinstance(tt, TargetList):
            targets += tt.targets
        else:
            raise ValueError(f'Invalid target element {tt}')

    for tt in targets:
        if tt.value == 'preserve':
            tt.value = tt.eval(tw0)

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
        _print(f'Using solver {solver}')

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
                line=line, return_scalar=return_scalar, tw_kwargs=kwargs)
    _jac= partial(_jacobian, steps=steps, fun=_err)
    x0 = [line.vars[vv.name]._value for vv in vary]
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
            line.vars[vv.name] = rr
    except Exception as err:
        if restore_if_fail:
            for ii, rr in enumerate(vary):
                line.vars[rr.name] = x0[ii]
        _print('\n')
        raise err
    _print('\n')
    return result_info

def closed_orbit_correction(line, line_co_ref, correction_config,
                            solver=None, verbose=False, restore_if_fail=True):

    for corr_name, corr in correction_config.items():
        _print('Correcting', corr_name)
        with xt.line._temp_knobs(line, corr['ref_with_knobs']):
            tw_ref = line_co_ref.twiss(method='4d', zeta0=0, delta0=0)
        vary = [xt.Vary(vv, step=1e-9) for vv in corr['vary']]
        targets = []
        for tt in corr['targets']:
            assert isinstance(tt, str), 'For now only strings are supported for targets'
            for kk in ['x', 'px', 'y', 'py']:
                targets.append(xt.Target(kk, at=tt, value=tw_ref[kk, tt], tol=1e-9))

        line.match(
            solver=solver,
            verbose=verbose,
            restore_if_fail=restore_if_fail,
            vary=vary,
            targets=targets,
            twiss_init=xt.OrbitOnly(
                x=tw_ref['x', corr['start']],
                px=tw_ref['px', corr['start']],
                y=tw_ref['y', corr['start']],
                py=tw_ref['py', corr['start']],
                zeta=tw_ref['zeta', corr['start']],
                delta=tw_ref['delta', corr['start']],
            ),
            ele_start=corr['start'], ele_stop=corr['end'])