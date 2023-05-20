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

class OrbitOnly:
    def __init__(self, x=0, px=0, y=0, py=0, zeta=0, delta=0):
        self.x = x
        self.px = px
        self.y = y
        self.py = py
        self.zeta = zeta
        self.delta = delta

Action = xd.Action

class ActionTwiss(xd.Action):

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
                'twiss_init' in kwargs and kwargs['twiss_init'] is not None
                and kwargs['twiss_init'] != 'periodic')

        tw0 = line.twiss(_keep_initial_particles=_keep_initial_particles,
                         **kwargs)

        if _keep_initial_particles:
            if isinstance(line, xt.Multiline):
                for llnn in tw0._line_names:
                    kwargs['_initial_particles'] = tw0[llnn]._initial_particles
            else:
                kwargs['_initial_particles'] = tw0._initial_particles

        self.kwargs = kwargs

    def compute(self, allow_failure=True):
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
        xd.Target.__init__(self, tar=(tar, at), value=value, tol=tol,
                            weight=weight, scale=scale, action=action)
        self.line = line

class Vary(xd.Vary):
    def __init__(self, name, limits=None, step=None, weight=None):
        xd.Vary.__init__(self, name=name, container=None, limits=limits,
                         step=step, weight=weight)

class VaryList:
    def __init__(self, vars, container, **kwargs):
        self.vary_objects = [Vary(vv, **kwargs) for vv in vars]

class TargetList:
    def __init__(self, tars, **kwargs):
        self.targets = [Target(tt, **kwargs) for tt in tars]

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
                  solver_options={}, **kwargs):

    twiss_actions = {}
    for tt in targets:
        if tt.action is None:
            if tt.line is not None:
                ln_twiss = line[tt.line]
            else:
                ln_twiss = line
            if ln_twiss not in twiss_actions:
                twiss_actions[ln_twiss] = ActionTwiss(ln_twiss, **kwargs)
            tt.action = twiss_actions[ln_twiss]

    for vv in vary:
        if vv.container is None:
            vv.container = line.vars

    opt = xd.Optimize(vary=vary, targets=targets, solver=solver,
                        verbose=verbose, assert_within_tol=assert_within_tol,
                        solver_options=solver_options,
                        restore_if_fail=restore_if_fail)
    return opt.solve()


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

