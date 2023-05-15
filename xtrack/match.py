from functools import partial

import numpy as np
from scipy.optimize import fsolve, minimize

from .jacobian import JacobianSolver
from .twiss import TwissInit
from .general import _print
import xtrack as xt

DEFAULT_WEIGHTS = {
    # For quantities not specified here the default weight is 1
    'x': 10,
    'px': 100,
    'y': 10,
    'py': 100,
    'zeta': 10,
    'delta': 100,
    'pzeta': 100,
    'ptau': 100,
    'alfx': 10.,
    'alfy': 10.,
    'mux': 10.,
    'muy': 10.,
    'q1': 10.,
    'q2': 10.,
}

class OrbitOnly:
    def __init__(self, x=0, px=0, y=0, py=0, zeta=0, delta=0):
        self.x = x
        self.px = px
        self.y = y
        self.py = py
        self.zeta = zeta
        self.delta = delta

class MeritFunctionForMatch:

    def __init__(self, vary, targets, line, actions, return_scalar,
                 call_counter, verbose, tw_kwargs, steps_for_jacobian):

        self.vary = vary
        self.targets = targets
        self.line = line
        self.actions = actions
        self.return_scalar = return_scalar
        self.call_counter = call_counter
        self.verbose = verbose
        self.tw_kwargs = tw_kwargs
        self.steps_for_jacobian = steps_for_jacobian

    def _x_to_knobs(self, x):
        knob_values = np.array(x).copy()
        for ii, vv in enumerate(self.vary):
            if vv.weight is not None:
                knob_values[ii] *= vv.weight
        return knob_values

    def _knobs_to_x(self, knob_values):
        x = np.array(knob_values).copy()
        for ii, vv in enumerate(self.vary):
            if vv.weight is not None:
                x[ii] /= vv.weight
        return x

    def __call__(self, x):

        _print(f"Matching: model call n. {self.call_counter}       ",
                end='\r', flush=True)
        self.call_counter += 1

        knob_values = self._x_to_knobs(x)

        for kk, vv in zip(self.vary, knob_values):
            self.line.vars[kk.name] = vv

        if self.verbose:
            _print(f'x = {knob_values}')

        res_data = {}
        failed = False
        for aa in self.actions:
            res_data[aa] = aa.compute()
            if res_data[aa] == 'failed':
                failed = True
                break

        if failed:
            err_values = [1e100 for tt in self.targets]
        else:
            res_values = []
            target_values = []
            for tt in self.targets:
                res_values.append(tt.eval(res_data))
                target_values.append(tt.value)
            self._last_data = res_data # for debugging

            res_values = np.array(res_values)
            target_values = np.array(target_values)
            err_values = res_values - target_values

            if self.verbose:
                _print(f'   f(x) = {res_values}')

            tols = 0 * err_values
            for ii, tt in enumerate(self.targets):
                if tt.tol is not None:
                    tols[ii] = tt.tol
                else:
                    tols[ii] = 1e-14

            if self.verbose:
                _print(f'   err/tols = {err_values/tols}')

            if np.all(np.abs(err_values) < tols):
                err_values *= 0
                if self.verbose:
                    _print('Found point within tolerance!')

            for ii, tt in enumerate(self.targets):
                if tt.weight is not None:
                    err_values[ii] *= tt.weight

        if self.return_scalar:
            return np.sum(err_values * err_values)
        else:
            return np.array(err_values)

    def get_jacobian(self, x):
        x = np.array(x).copy()
        steps = self._knobs_to_x(self.steps_for_jacobian)
        assert len(x) == len(steps)
        f0 = self(x)
        if np.isscalar(f0):
            jac = np.zeros((1, len(x)))
        else:
            jac = np.zeros((len(f0), len(x)))
        for ii in range(len(x)):
            x[ii] += steps[ii]
            jac[:, ii] = (self(x) - f0) / steps[ii]
            x[ii] -= steps[ii]
        return jac

class TargetList:
    def __init__(self, tars, **kwargs):
        self.targets = [Target(tt, **kwargs) for tt in tars]

class VaryList:
    def __init__(self, vars, **kwargs):
        self.vary_objects = [Vary(vv, **kwargs) for vv in vars]

class Vary:
    def __init__(self, name, limits=None, step=None, weight=None):

        if weight is None:
            weight = 1.

        if limits is None:
            limits = (-1e200, 1e200)
        else:
            assert len(limits) == 2, '`limits` must have length 2.'

        if step is None:
            step = 1e-10

        assert weight > 0, '`weight` must be positive.'

        self.name = name
        self.limits = np.array(limits)
        self.step = step
        self.weight = weight

class Action:
    def prepare(self):
        pass
    def compute(self):
        return dict()

class ActionTwiss(Action):

    def __init__(self, line, **kwargs):
        self.line = line
        self.kwargs = kwargs

    def prepare(self):
        line = self.line
        kwargs = self.kwargs

        if 'twiss_init' in kwargs and kwargs['twiss_init'] is not None:
            twinit = kwargs['twiss_init']
            assert 'ele_start' in kwargs and kwargs['ele_start'] is not None, (
                'ele_start must be provided if twiss_init is provided')
            if isinstance(twinit, OrbitOnly):
                if not isinstance(kwargs['ele_start'], str):
                    element_name = line.element_names[kwargs['ele_start']]
                else:
                    element_name = kwargs['ele_start']
                particle_on_co=line.build_particles(
                    x=twinit.x, px=twinit.px,
                    y=twinit.y, py=twinit.py,
                    zeta=twinit.zeta, delta=twinit.delta)
                particle_on_co.at_element = line.element_names.index(
                                                                    element_name)
                kwargs['twiss_init'] = TwissInit(
                    particle_on_co=particle_on_co,
                    W_matrix=np.eye(6),
                    element_name=element_name)

        if 'twiss_init' in kwargs and isinstance(kwargs['twiss_init'], str):
            assert kwargs['twiss_init'] in (
                ['preserve', 'preserve_start', 'preserve_end', 'periodic'])
            if kwargs['twiss_init'] in ['preserve', 'preserve_start', 'preserve_end']:
                full_twiss_kwargs = kwargs.copy()
                full_twiss_kwargs.pop('twiss_init')
                full_twiss_kwargs.pop('ele_start')
                full_twiss_kwargs.pop('ele_stop')
                tw0_full = line.twiss(**full_twiss_kwargs)
                if (kwargs['twiss_init'] == 'preserve'
                    or kwargs['twiss_init'] == 'preserve_start'):
                    init_at = kwargs['ele_start']
                elif kwargs['twiss_init'] == 'preserve_end':
                    init_at = kwargs['ele_stop']
                if isinstance(tw0_full, xt.MultiTwiss):
                    kwargs['twiss_init'] = []
                    for ll, nn in zip(tw0_full._line_names, init_at):
                        kwargs['twiss_init'].append(tw0_full[ll].get_twiss_init(at_element=nn))
                else:
                    kwargs['twiss_init'] = tw0_full.get_twiss_init(at_element=init_at)


        _keep_initial_particles = (
                'twiss_init' in kwargs and kwargs['twiss_init'] is not None)

        tw0 = line.twiss(_keep_initial_particles=_keep_initial_particles,
                        _keep_tracking_data=True, **kwargs)

        if isinstance(line, xt.Multiline):
            ebe_monitor = []
            for llnn in tw0._line_names:
                if tw0[llnn].tracking_data is not None:
                    ebe_monitor.append(tw0[llnn].tracking_data)
        else:
            ebe_monitor = tw0.tracking_data
        kwargs['_ebe_monitor'] = ebe_monitor

        if 'twiss_init' in kwargs and kwargs['twiss_init'] is not None: # open line mode
            if isinstance(line, xt.Multiline):
                for llnn in tw0._line_names:
                    kwargs['_initial_particles'] = tw0[llnn]._initial_particles
            else:
                kwargs['_initial_particles'] = tw0._initial_particles

        self.kwargs = kwargs

    def compute(self):
        return self.line.twiss(**self.kwargs)

class Target:
    def __init__(self, tar, value, at=None, tol=None, weight=None, scale=None,
                 line=None, action=None):

        if scale is not None and weight is not None:
            raise ValueError(
                'Cannot specify both `weight` and `scale` for a target.')

        if scale is not None:
            weight = scale

        if weight is None:
            if isinstance(tar, str):
                weight = DEFAULT_WEIGHTS.get(tar, 1.)
            else:
                weight = 1.

        if weight <= 0:
            raise ValueError('`weight` must be positive.')

        self.tar = tar
        self.action = action
        self.value = value
        self.tol = tol
        self.at = at
        self.weight = weight
        self.line = line
        self._at_index = None

    @property
    def scale(self):
        return self.weight

    @scale.setter
    def scale(self, value):
        self.weight = value

    def eval(self, data):

        res = data[self.action]

        if isinstance (res, xt.MultiTwiss) and not callable(self.tar):
            if self.line is None:
                raise ValueError('When using `Multiline.match`, '
                    'a `line` must associated to each non-callable target.')

        if self.line is not None:
            this_res = res[self.line]
        else:
            this_res = res

        if isinstance(self.tar, str):
            if self.at is not None:
                if self._at_index is not None:
                    return this_res[self.tar, self._at_index]
                else:
                    return this_res[self.tar, self.at]
            else:
                return this_res[self.tar]
        elif callable(self.tar):
            assert self.at is None, '`at` cannot be provided if target is a function'
            return self.tar(this_res)

class TargetInequality(Target):

    def __init__(self, tar, ineq_sign, rhs, at=None, tol=None, scale=None, line=None):
        super().__init__(tar, value=0, at=at, tol=tol, scale=scale, line=line)
        assert ineq_sign in ['<', '>'], ('ineq_sign must be either "<" or ">"')
        self.ineq_sign = ineq_sign
        self.rhs = rhs

    def eval(self, tw):
        val = super().eval(tw)
        if self.ineq_sign == '<' and val < self.rhs:
            return 0
        elif self.ineq_sign == '>' and val > self.rhs:
            return 0
        else:
            return val - self.rhs

def match_line(line, vary, targets, restore_if_fail=True, solver=None,
                  verbose=False, **kwargs):



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

    actions = []
    for tt in targets:
        if tt.action not in actions:
            actions.append(tt.action)

    if None in actions:
        actions.remove(None)
        actiontwiss = ActionTwiss(line, **kwargs)
        actions.append(actiontwiss)
        for tt in targets:
            if tt.action is None:
                tt.action = actiontwiss

    for aa in actions:
        aa.prepare()

    data0 = {}
    for aa in actions:
        data0[aa] = aa.compute()

    for tt in targets:
        if tt.value == 'preserve':
            tt.value = tt.eval(data0[tt.action])

    # Cache index of at for faster evaluation
    for tt in targets:
        if tt.at is not None and isinstance(tt.action, ActionTwiss):
            if tt.line is None:
                tt._at_index = list(data0[tt.action].name).index(tt.at)
            else:
                tt._at_index = list(data0[tt.action][tt.line].name).index(tt.at)


    if solver is None:
        solver = 'jacobian'
        #old logic from before jacobian implementation
        # if len(targets) != len(vary):
        #     solver = 'bfgs'
        # elif np.any([vv.limits is not None for vv in vary]):
        #     solver = 'bfgs'
        # else:
        #     solver = 'fsolve'

    if verbose:
        _print(f'Using solver {solver}')

    steps = []
    knob_limits = []
    for vv in vary:
        steps.append(vv.step)
        knob_limits.append(vv.limits)

    assert solver in ['fsolve', 'bfgs', 'jacobian'], (
                      f'Invalid solver {solver}.')

    return_scalar = {'fsolve': False, 'bfgs': True, 'jacobian': False}[solver]

    _err = MeritFunctionForMatch(
                vary=vary, targets=targets,
                line=line, actions=actions,
                return_scalar=return_scalar, call_counter=0, verbose=verbose,
                tw_kwargs=kwargs, steps_for_jacobian=steps)

    knob_limits = np.array(knob_limits)
    x_lim_low = _err._knobs_to_x(np.squeeze(knob_limits[:, 0]))
    x_lim_high = _err._knobs_to_x(np.squeeze(knob_limits[:, 1]))
    x_limits = [(hh, ll) for hh, ll in zip(x_lim_low, x_lim_high)]

    _jac= _err.get_jacobian

    x0 = _err._knobs_to_x([line.vars[vv.name]._value for vv in vary])
    try:
        if solver == 'fsolve':
            (res, infodict, ier, mesg) = fsolve(_err, x0=x0.copy(),
                full_output=True, fprime=_jac)
            if ier != 1:
                raise RuntimeError("fsolve failed: %s" % mesg)
            result_info = {
                'res': res, 'info': infodict, 'ier': ier, 'mesg': mesg}
        elif solver == 'bfgs':
            optimize_result = minimize(_err, x0=x0.copy(), method='L-BFGS-B',
                        bounds=x_limits,
                        jac=_jac, options={'gtol':0})
            result_info = {'optimize_result': optimize_result}
            res = optimize_result.x
        elif solver == 'jacobian':
            jac_solver = JacobianSolver(func=_err, limits=x_limits, verbose=verbose)
            res = jac_solver.solve(x0=x0.copy())
            result_info = {'jac_solver': jac_solver, 'res': res}

        for vv, rr in zip(vary, _err._x_to_knobs(res)):
            line.vars[vv.name] = rr
    except Exception as err:
        if restore_if_fail:
            knob_values0 = _err._x_to_knobs(x0)
            for ii, rr in enumerate(vary):
                line.vars[rr.name] = knob_values0[ii]
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