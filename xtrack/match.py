from functools import partial

import numpy as np
from scipy.optimize import fsolve, minimize

from .twiss import TwissInit
from .general import _print
import xtrack as xt
import xdeps as xd

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
    'qx': 10.,
    'qy': 10.,
}

Action = xd.Action

class ActionTwiss(xd.Action):

    def __init__(self, line, allow_twiss_failure, **kwargs):
        self.line = line
        self.kwargs = kwargs
        self.allow_twiss_failure = allow_twiss_failure

    def prepare(self):
        line = self.line
        kwargs = self.kwargs

        ismultiline = isinstance(line, xt.Multiline)

        if 'twiss_init' in kwargs:
            if ismultiline:
                twinit_list = kwargs['twiss_init']
                ele_start_list = kwargs['ele_start']
                ele_stop_list = kwargs['ele_stop']
                line_names = kwargs.get('lines', line.line_names)
                line_list = [line[nn] for nn in line_names]
                assert isinstance(twinit_list, list)
                assert isinstance(ele_start_list, list)
                assert isinstance(ele_stop_list, list)
            else:
                twinit_list = [kwargs['twiss_init']]
                ele_start_list = [kwargs['ele_start']]
                ele_stop_list = [kwargs['ele_stop']]
                line_list = [line]

            _keep_ini_particles_list = [False] * len(twinit_list)
            for ii, (twinit, ele_start, ele_stop) in enumerate(zip(
                    twinit_list, ele_start_list, ele_stop_list)):
                if isinstance(twinit, xt.TwissInit):
                    _keep_ini_particles_list[ii] = True
                elif isinstance(twinit, str):
                    assert twinit in (
                        ['preserve', 'preserve_start', 'preserve_end', 'periodic'])
                    if twinit in ['preserve', 'preserve_start', 'preserve_end']:
                        full_twiss_kwargs = kwargs.copy()
                        full_twiss_kwargs.pop('twiss_init')
                        full_twiss_kwargs.pop('ele_start')
                        full_twiss_kwargs.pop('ele_stop')
                        if 'lines' in full_twiss_kwargs:
                            full_twiss_kwargs.pop('lines')
                        tw0_full = line_list[ii].twiss(**full_twiss_kwargs)
                        if (twinit == 'preserve' or twinit == 'preserve_start'):
                            init_at = ele_start
                        elif kwargs['twiss_init'] == 'preserve_end':
                            init_at = ele_stop
                        assert not isinstance(tw0_full, xt.MultiTwiss)
                        twinit_list[ii] = tw0_full.get_twiss_init(at_element=init_at)
                        _keep_ini_particles_list[ii] = True

            if ismultiline:
                kwargs['twiss_init'] = twinit_list
                kwargs['_keep_initial_particles'] = _keep_ini_particles_list
            else:
                kwargs['twiss_init'] = twinit_list[0]
                kwargs['_keep_initial_particles'] = _keep_ini_particles_list[0]

            tw0 = line.twiss(**kwargs)

            if ismultiline:
                kwargs['_initial_particles'] = [
                    tw0[llnn]._data.get('_initial_particles', None) for llnn in line_names]
            else:
                kwargs['_initial_particles'] = tw0._data.get(
                                        '_initial_particles', None)

        self.kwargs = kwargs

    def run(self, allow_failure=True):
        if not self.allow_twiss_failure or not allow_failure:
            return self.line.twiss(**self.kwargs)
        else:
            try:
                return self.line.twiss(**self.kwargs)
            except Exception as ee:
                if allow_failure:
                    return 'failed'
                else:
                    raise ee

class Target(xd.Target):
    def __init__(self, tar, value, at=None, tol=None, weight=None, scale=None,
                 line=None, action=None):
        if at is not None:
            xdtar = (tar, at)
        else:
            xdtar = tar
        xd.Target.__init__(self, tar=xdtar, value=value, tol=tol,
                            weight=weight, scale=scale, action=action)
        self.line = line

    def eval(self, data):
        res = data[self.action]
        if self.line is not None:
            res = res[self.line]
        if callable(self.tar):
            return self.tar(res)
        else:
            return res[self.tar]

class Vary(xd.Vary):
    def __init__(self, name, limits=None, step=None, weight=None):
        xd.Vary.__init__(self, name=name, container=None, limits=limits,
                         step=step, weight=weight)

class VaryList(xd.VaryList):
    def __init__(self, vars, **kwargs):
        self.vary_objects = [Vary(vv, **kwargs) for vv in vars]

class TargetList(xd.TargetList):
    def __init__(self, tars, action=None, **kwargs):
        self.targets = [Target(tt, action=action, **kwargs) for tt in tars]

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
                  verbose=False, assert_within_tol=True,
                  solver_options={}, allow_twiss_failure=True, **kwargs):

    targets_flatten = []
    for tt in targets:
        if isinstance(tt, xd.TargetList):
            for tt1 in tt.targets:
                targets_flatten.append(tt1)
        else:
            targets_flatten.append(tt)

    action_twiss = None
    for tt in targets_flatten:
        if tt.action is None:
            if action_twiss is None:
                action_twiss = ActionTwiss(
                    line, allow_twiss_failure=allow_twiss_failure, **kwargs)
            tt.action = action_twiss
        if tt.weight is None:
            if isinstance(tt.tar, tuple):
                tt_name = tt.tar[0]
            else:
                tt_name = tt.tar
            tt.weight = DEFAULT_WEIGHTS.get(tt_name, 1.)

    vary_flatten = []
    for vv in vary:
        if isinstance(vv, xd.VaryList):
            for vv1 in vv.vary_objects:
                vary_flatten.append(vv1)
        else:
            vary_flatten.append(vv)
    for vv in vary_flatten:
        if vv.container is None:
            vv.container = line.vars

    opt = xd.Optimize(vary=vary, targets=targets, solver=solver,
                        verbose=verbose, assert_within_tol=assert_within_tol,
                        solver_options=solver_options,
                        restore_if_fail=restore_if_fail)
    res =  opt.solve()
    res['optimizer'] = opt

    return res


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

        assert isinstance(corr['start'], str)

        line.match(
            solver=solver,
            verbose=verbose,
            restore_if_fail=restore_if_fail,
            vary=vary,
            targets=targets,
            twiss_init=xt.TwissInit(
                line=line,
                element_name=corr['start'],
                x=tw_ref['x', corr['start']],
                px=tw_ref['px', corr['start']],
                y=tw_ref['y', corr['start']],
                py=tw_ref['py', corr['start']],
                zeta=tw_ref['zeta', corr['start']],
                delta=tw_ref['delta', corr['start']],
            ),
            ele_start=corr['start'], ele_stop=corr['end'])

def match_knob_line(line, knob_name, vary, targets,
                    knob_value_start, knob_value_end, **kwargs):

    vary_aux = []
    for vv in vary:
        if vv.limits[0] != -1e200 or vv.limits[1] != 1e200:
            raise ValueError('Cannot match knobs with limits')
        line.vars[vv.name + '_from_' + knob_name] = 0
        line.vars[vv.name] += line.vars[vv.name + '_from_' + knob_name]
        vary_aux.append(xt.Vary(vv.name + '_from_' + knob_name, step=vv.step))

    line.match(vary=vary_aux, targets = targets, **kwargs)

    line.vars[knob_name] = knob_value_end
    for vv in vary_aux:
        line.vars[vv.name] = (line.vars[vv.name]._value
                              * (line.vars[knob_name] - knob_value_start)
                              / (knob_value_end - knob_value_start))

    line.vars[knob_name] = knob_value_start

    _print('Matched knob: ', knob_name)