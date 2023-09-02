from collections.abc import Iterable
from functools import partial

import numpy as np
from scipy.optimize import fsolve, minimize

from .twiss import TwissInit
from .general import _print
import xtrack as xt
import xdeps as xd

XTRACK_DEFAULT_TOL = 1e-10

XTRACK_DEFAULT_WEIGHTS = {
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

ALLOWED_TARGET_KWARGS= ['x', 'px', 'y', 'py', 'zeta', 'delta', 'pzata', 'ptau',
                        'betx', 'bety', 'alfx', 'alfy', 'gamx', 'gamy',
                        'mux', 'muy', 'dx', 'dpx', 'dy', 'dpy',
                        'qx', 'qy', 'dqx', 'dqy',
                        'eq_gemitt_x', 'eq_gemitt_y', 'eq_gemitt_zeta',
                        'eq_nemitt_x', 'eq_nemitt_y', 'eq_nemitt_zeta']

Action = xd.Action

class _LOC:
    def __init__(self, name=None):
        self.name = name
    def __repr__(self):
        return self.name

START = _LOC('START')
END = _LOC('END')

class ActionTwiss(xd.Action):

    def __init__(self, line, allow_twiss_failure, table_for_twiss_init=None,
                 compensate_radiation_energy_loss=True, **kwargs):
        self.line = line
        self.kwargs = kwargs
        self.table_for_twiss_init = table_for_twiss_init
        self.allow_twiss_failure = allow_twiss_failure
        self.compensate_radiation_energy_loss = compensate_radiation_energy_loss

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
                if self.table_for_twiss_init is not None:
                    if isinstance(self.table_for_twiss_init, xt.multiline.MultiTwiss):
                        table_for_twinit_list = [self.table_for_twiss_init[nn] for nn in line_names]
                    else:
                        assert isinstance(self.table_for_twiss_init, (list, tuple)), (
                            'table_for_twiss_init for a Multiline match must be either a MultiTwiss, '
                            'a list or a tuple')
                        table_for_twinit_list = self.table_for_twiss_init
                else:
                    table_for_twinit_list = [None] * len(twinit_list)
            else:
                twinit_list = [kwargs['twiss_init']]
                ele_start_list = [kwargs['ele_start']]
                ele_stop_list = [kwargs['ele_stop']]
                line_list = [line]
                table_for_twinit_list = [self.table_for_twiss_init]

            _keep_ini_particles_list = [False] * len(twinit_list)
            for ii, (twinit, ele_start, ele_stop) in enumerate(zip(
                    twinit_list, ele_start_list, ele_stop_list)):
                if isinstance(twinit, xt.TwissInit):
                    twinit_list[ii] = twinit.copy()
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
                        tab_twinit = table_for_twinit_list[ii]
                        if tab_twinit is None:
                            tab_twinit = line_list[ii].twiss(**full_twiss_kwargs)
                        if (twinit == 'preserve' or twinit == 'preserve_start'):
                            init_at = ele_start
                        elif kwargs['twiss_init'] == 'preserve_end':
                            init_at = ele_stop
                        assert not isinstance(tab_twinit, xt.MultiTwiss)
                        twinit_list[ii] = tab_twinit.get_twiss_init(at_element=init_at)
                        _keep_ini_particles_list[ii] = True

            for twini, ln, eest in zip(twinit_list, line_list, ele_start_list):
                if isinstance(twini, xt.TwissInit) and twini._needs_complete():
                    assert isinstance(eest, str)
                    twini._complete(line=ln, element_name=eest)

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
        if self.compensate_radiation_energy_loss:
            if isinstance(self.line, xt.Multiline):
                raise NotImplementedError(
                    'Radiation energy loss compensation is not yet supported'
                    ' for Multiline')
            self.line.compensate_radiation_energy_loss(verbose=False)
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
    def __init__(self, tar=None, value=None, at=None, tol=None, weight=None, scale=None,
                 line=None, action=None, tag='', optimize_log=False, **kwargs):

        for kk in kwargs:
            assert kk in ALLOWED_TARGET_KWARGS, (
                f'Unknown keyword argument {kk}. '
                f'Allowed keywords are {ALLOWED_TARGET_KWARGS}')

        if len(kwargs) > 1:
            raise ValueError(f'{list(kwargs.keys())} cannot be specified '
                                'together in a single Target. Please use '
                                'multiple Targets or a TargetSet.')

        if len(kwargs) == 1:
            tar = list(kwargs.keys())[0]
            value = list(kwargs.values())[0]

        if at is not None:
            xdtar = (tar, at)
        else:
            xdtar = tar
        xd.Target.__init__(self, tar=xdtar, value=value, tol=tol,
                            weight=weight, scale=scale, action=action, tag=tag,
                            optimize_log=optimize_log)
        self.line = line

    def __repr__(self):
        out = xd.Target.__repr__(self)
        if self.line is not None:
            out = out.replace('Target(', f'Target(line={self.line}, ')
        return out

    def eval(self, data):
        res = data[self.action]
        if self.line is not None:
            res = res[self.line]
        if callable(self.tar):
            return self.tar(res)
        else:
            return res[self.tar]

class Vary(xd.Vary):
    def __init__(self, name, container=None, limits=None, step=None, weight=None,
                 max_step=None, active=True, tag=''):
        xd.Vary.__init__(self, name=name, container=container, limits=limits,
                         step=step, weight=weight, max_step=max_step, tag=tag,
                         active=active)

class VaryList(xd.VaryList):
    def __init__(self, vars, **kwargs):
        self.vary_objects = [Vary(vv, **kwargs) for vv in vars]

class TargetSet(xd.TargetList):
    def __init__(self, tars=None, action=None, **kwargs):
        vnames = []
        vvalues = []
        for kk in ALLOWED_TARGET_KWARGS:
            if kk in kwargs:
                vnames.append(kk)
                vvalues.append(kwargs[kk])
                kwargs.pop(kk)
        self.targets = []
        if tars is not None:
            self.targets += [Target(tt, action=action, **kwargs) for tt in tars]
        self.targets += [
            Target(tar=tar, value=val, action=action, **kwargs) for tar, val in zip(vnames, vvalues)]
        if len(self.targets) == 0:
            raise ValueError('No targets specified')

TargetList = TargetSet # for backward compatibility

class TargetInequality(Target):

    def __init__(self, tar, ineq_sign, rhs, at=None, tol=None, scale=None,
                 line=None, weight=None, tag=''):
        Target.__init__(self, tar, value=0, at=at, tol=tol, scale=scale, line=line,
                         weight=weight, tag=tag)
        assert ineq_sign in ['<', '>'], ('ineq_sign must be either "<" or ">"')
        self.ineq_sign = ineq_sign
        self.rhs = rhs

    def __repr__(self):
        return f'TargetInequality({self.tar} {self.ineq_sign} {self.rhs}, tol={self.tol}, weight={self.weight})'

    def eval(self, tw):
        val = super().eval(tw)
        if self.ineq_sign == '<' and val < self.rhs:
            return 0
        elif self.ineq_sign == '>' and val > self.rhs:
            return 0
        else:
            return val - self.rhs

class TargetRelPhaseAdvance(Target):

    def __init__(self, tar, value, at_1=None, at_0=None, tag='',  **kwargs):

        Target.__init__(self, tar=self.compute, value=value, tag=tag, **kwargs)

        assert tar in ['mux', 'muy'], 'Only mux and muy are supported'
        self.var = tar
        if at_1 is None:
            at_1 = '__ele_stop__'
        if at_0 is None:
            at_0 = '__ele_start__'
        self.at_1 = at_1
        self.at_0 = at_0

    def __repr__(self):
        return f'TargetPhaseAdvance({self.var}({self.at_1}) - {self.var}({self.at_0}), value={self.value}, tol={self.tol}, weight={self.weight})'

    def compute(self, tw):

        if self.at_1 == '__ele_stop__':
            mu_1 = tw[self.var, -1]
        else:
            mu_1 = tw[self.var, self.at_1]

        if self.at_0 == '__ele_start__':
            mu_0 = tw[self.var, 0]
        else:
            mu_0 = tw[self.var, self.at_0]

        return mu_1 - mu_0

def match_line(line, vary, targets, restore_if_fail=True, solver=None,
                  verbose=False, assert_within_tol=True,
                  solver_options={}, allow_twiss_failure=True,
                  n_steps_max=20, default_tol=None,
                  solve=True, compensate_radiation_energy_loss=False,**kwargs):

    targets_flatten = []
    for tt in targets:
        if isinstance(tt, xd.TargetList):
            for tt1 in tt.targets:
                targets_flatten.append(tt1.copy())
        else:
            targets_flatten.append(tt.copy())

    action_twiss = None
    for tt in targets_flatten:

        # Handle action
        if tt.action is None:
            if action_twiss is None:
                action_twiss = ActionTwiss(
                    line, allow_twiss_failure=allow_twiss_failure,
                    compensate_radiation_energy_loss=compensate_radiation_energy_loss,
                    **kwargs)
            tt.action = action_twiss

        # Handle at
        if isinstance(tt.tar, tuple):
            tt_name = tt.tar[0] # `at` is  present
            tt_at = tt.tar[1]
        else:
            tt_name = tt.tar
            tt_at = None
        if tt_at is not None and isinstance(tt_at, _LOC):
            tt_at = _at_from_placeholder(tt_at, line=line, line_name=tt.line,
                    ele_start=kwargs['ele_start'], ele_stop=kwargs['ele_stop'])
            tt.tar = (tt_name, tt_at)

        # Handle value
        if isinstance(tt.value, xt.multiline.MultiTwiss):
            tt.value=tt.value[line][tt.tar]
        if isinstance(tt.value, xt.TwissTable):
            tt.value=tt.value[tt.tar]
        if isinstance(tt.value, np.ndarray):
            raise ValueError('Target value must be a scalar')

        # Handle weight
        if tt.weight is None:
            tt.weight = XTRACK_DEFAULT_WEIGHTS.get(tt_name, 1.)
        if tt.tol is None:
            if default_tol is None:
                tt.tol = XTRACK_DEFAULT_TOL
            elif isinstance(default_tol, dict):
                tt.tol = default_tol.get(tt_name,
                                    default_tol.get(None, XTRACK_DEFAULT_TOL))
            else:
                tt.tol = default_tol

    if not isinstance(vary, (list, tuple)):
        vary = [vary]

    vary_flatten = _flatten_vary(vary)
    _complete_vary_with_info_from_line(vary_flatten, line)

    opt = xd.Optimize(vary=vary_flatten, targets=targets_flatten, solver=solver,
                        verbose=verbose, assert_within_tol=assert_within_tol,
                        solver_options=solver_options,
                        n_steps_max=n_steps_max,
                        restore_if_fail=restore_if_fail)

    if solve:
        opt.solve()

    return opt

def _flatten_vary(vary):
    vary_flatten = []
    for vv in vary:
        if isinstance(vv, xd.VaryList):
            for vv1 in vv.vary_objects:
                vary_flatten.append(vv1)
        else:
            vary_flatten.append(vv)
    return vary_flatten

def _complete_vary_with_info_from_line(vary, line):
    for vv in vary:
        if vv.container is None:
            vv.container = line.vars
            vv._complete_limits_and_step_from_defaults()

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

def match_knob_line(line, knob_name, vary, targets, knob_value_start,
                    knob_value_end, run=True, **kwargs):

    knob_opt = KnobOptimizer(line, knob_name, vary, targets,
                    knob_value_start, knob_value_end,
                    **kwargs)
    if run:
        knob_opt.solve()
        knob_opt.generate_knob()
    return knob_opt

class KnobOptimizer:

    def __init__(self, line, knob_name, vary, targets,
                    knob_value_start, knob_value_end,
                    **kwargs):

        if not isinstance (vary, (list, tuple)):
            vary = [vary]

        vary_flatten = _flatten_vary(vary)
        _complete_vary_with_info_from_line(vary_flatten, line)

        vary_aux = []
        for vv in vary_flatten:
            aux_name = vv.name + '_from_' + knob_name
            if (aux_name in line.vars
                and (line.vars[aux_name] in
                         line.vars[vv.name]._expr._get_dependencies())):
                # reset existing term in expression
                line.vars[aux_name] = 0
            else:
                # create new term in expression
                line.vars[aux_name] = 0
                line.vars[vv.name] += line.vars[aux_name]

            vv_aux = vv.__dict__.copy()
            vv_aux['name'] = aux_name
            vary_aux.append(xt.Vary(**vv_aux))

        opt = line.match(vary=vary_aux, targets = targets, solve=False, **kwargs)

        object.__setattr__(self, 'opt', opt)
        self.line = line
        self.knob_name = knob_name
        self.knob_value_start = knob_value_start
        self.knob_value_end = knob_value_end

    def __getattr__(self, attr):
        return getattr(self.opt, attr)

    def __setattr__(self, attr, value):
        if hasattr(self.opt, attr):
            setattr(self.opt, attr, value)
        else:
            object.__setattr__(self, attr, value)

    def __dir__(self):
        return object.__dir__(self) + dir(self.opt)

    def generate_knob(self):
        self.line.vars[self.knob_name] = self.knob_value_end
        for vv in self.vary:
            var_value = self.line.vars[vv.name]._value
            self.line.vars[vv.name] = (
                var_value / (self.knob_value_end - self.knob_value_start)
                * (self.line.vars[self.knob_name]))
            if self.knob_value_start != 0:
                self.line.vars[vv.name] -= (
                    var_value / (self.knob_value_end - self.knob_value_start)
                    * self.knob_value_start)

        self.line.vars[self.knob_name] = self.knob_value_start

        _print('Generated knob: ', self.knob_name)

def _at_from_placeholder(tt_at, line, line_name, ele_start, ele_stop):
    assert isinstance(tt_at, _LOC)
    if isinstance(line, xt.Multiline):
        assert line is not None, (
            'For a Multiline, the line must be specified if the target '
            'is `ele_start`')
        assert line_name in line.line_names
        i_line = line.line_names.index(line_name)
        this_line = line[line_name]
    else:
        i_line = None
        this_line = line
    if tt_at.name == 'START':
        if i_line is not None:
            tt_at = ele_start[i_line]
        else:
            tt_at = ele_start
    elif tt_at.name == 'END':
        if i_line is not None:
            tt_at = ele_stop[i_line]
        else:
            tt_at = ele_stop
    else:
        raise ValueError(f'Unknown location {tt_at.name}')
    if not isinstance(tt_at, str):
        tt_at = this_line.element_names[tt_at]

    return tt_at