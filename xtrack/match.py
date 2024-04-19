from collections.abc import Iterable
from functools import partial

import numpy as np
from scipy.optimize import fsolve, minimize

from .twiss import TwissInit, VARS_FOR_TWISS_INIT_GENERATION, _complete_twiss_init
from .general import _print
import xtrack as xt
import xdeps as xd

XTRACK_DEFAULT_TOL = 1e-9
XTRACK_DEFAULT_SIGMA_REL = 0.01

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
                        'ax_chrom', 'bx_chrom', 'ay_chrom', 'by_chrom',
                        'wx_chrom', 'wy_chrom',
                        'ddqx', 'ddqy', 'ddx', 'ddpx', 'ddy', 'ddpy',
                        'betx1', 'bety1', 'betx2', 'bety2',
                        'alfx1', 'alfy1', 'alfx2', 'alfy2',
                        'gamx1', 'gamy1', 'gamx2', 'gamy2',
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

    def __init__(self, line, allow_twiss_failure=False,
                 compensate_radiation_energy_loss=False,
                 **kwargs):
        self.line = line
        self.kwargs = kwargs
        self.allow_twiss_failure = allow_twiss_failure
        self.compensate_radiation_energy_loss = compensate_radiation_energy_loss

    def prepare(self):
        line = self.line
        kwargs = self.kwargs

        ismultiline = isinstance(line, xt.Multiline)

        # Forbit specifying init through kwargs for Multiline
        if ismultiline:
            for kk in VARS_FOR_TWISS_INIT_GENERATION:
                if kk in kwargs:
                    raise ValueError(
                        f'`{kk}` cannot be specified for a Multiline match. '
                        f'Please specify provide a TwissInit object for each line instead.')

        # Handle init from table
        if ismultiline:

            line_names = kwargs.get('lines', line.line_names)
            none_list = [None] * len(line_names)
            twinit_list = kwargs.get('init', none_list)
            ele_start_list = kwargs.get('start', none_list)
            ele_stop_list = kwargs.get('end', none_list)
            ele_init_list = kwargs.get('init_at', none_list)
            line_list = [line[nn] for nn in line_names]

            assert isinstance(twinit_list, list)
            assert isinstance(ele_start_list, list)
            assert isinstance(ele_stop_list, list)

            for ii, twinit in enumerate(twinit_list):
                if isinstance(twinit, xt.MultiTwiss):
                    twinit_list[ii] = twinit[line_names[ii]]

        else:
            twinit_list = [kwargs.get('init', None)]
            ele_start_list = [kwargs.get('start', None)]
            ele_stop_list = [kwargs.get('end', None)]
            ele_init_list = [kwargs.get('init_at', None)]
            line_list = [line]

            for ii, (twinit, start, end, init_at) in enumerate(
                    zip(twinit_list, ele_start_list, ele_stop_list, ele_init_list)):
                if isinstance(twinit, xt.TwissInit):
                    twinit_list[ii] = twinit.copy()
                elif isinstance(twinit, str):
                    assert twinit == 'periodic'

        # Handle init_at as xt.START or xt.END
        for ii, init_at in enumerate(ele_init_list):
            if isinstance(init_at, _LOC):
                if init_at.name == 'START':
                    ele_init_list[ii] = ele_start_list[ii]
                elif init_at.name == 'END':
                    ele_init_list[ii] = ele_stop_list[ii]

        # Handle init from table
        for ii, (twinit, start, end, init_at) in enumerate(
                zip(twinit_list, ele_start_list, ele_stop_list, ele_init_list)):
            if isinstance(twinit, xt.TwissInit):
                continue
            elif isinstance(twinit, xt.TwissTable):
                if init_at is None:
                    init_at = start
                init_at = init_at
                twinit_list[ii] = twinit.get_twiss_init(at_element=init_at)
                ele_init_list[ii] = None
            else:
                assert twinit is None or twinit == 'periodic'

        if not ismultiline:
            # Handle case in which twiss init is defined through kwargs
            init = _complete_twiss_init(
                    start=ele_start_list[0],
                    end=ele_stop_list[0],
                    init_at=ele_init_list[0],
                    init=twinit_list[0],
                    line=line,
                    reverse=None, # will be handled by the twiss
                    x=kwargs.get('x', None),
                    px=kwargs.get('px', None),
                    y=kwargs.get('y', None),
                    py=kwargs.get('py', None),
                    zeta=kwargs.get('zeta', None),
                    delta=kwargs.get('delta', None),
                    alfx=kwargs.get('alfx', None),
                    alfy=kwargs.get('alfy', None),
                    betx=kwargs.get('betx', None),
                    bety=kwargs.get('bety', None),
                    bets=kwargs.get('bets', None),
                    dx=kwargs.get('dx', None),
                    dpx=kwargs.get('dpx', None),
                    dy=kwargs.get('dy', None),
                    dpy=kwargs.get('dpy', None),
                    dzeta=kwargs.get('dzeta', None),
                    mux=kwargs.get('mux', None),
                    muy=kwargs.get('muy', None),
                    muzeta=kwargs.get('muzeta', None),
                    ax_chrom=kwargs.get('ax_chrom', None),
                    bx_chrom=kwargs.get('bx_chrom', None),
                    ay_chrom=kwargs.get('ay_chrom', None),
                    by_chrom=kwargs.get('by_chrom', None),
                    ddx=kwargs.get('ddx', None),
                    ddy=kwargs.get('ddy', None),
                    ddpx=kwargs.get('ddpx', None),
                    ddpy=kwargs.get('ddpy', None),
                    )
            for kk in VARS_FOR_TWISS_INIT_GENERATION + ['init_at']:
                if kk in kwargs:
                    kwargs.pop(kk)
            twinit_list[0] = init

        _keep_ini_particles_list = []
        for tt in twinit_list:
            _keep_ini_particles_list.append(isinstance(tt, xt.TwissInit))

        for ii, tt in enumerate(twinit_list):
            if isinstance(tt, xt.TwissInit):
                twinit_list[ii] = tt.copy()

        for twini, ln, eest in zip(twinit_list, line_list, ele_start_list):
            if isinstance(twini, xt.TwissInit) and twini._needs_complete():
                assert isinstance(eest, str)
                twini._complete(line=ln, element_name=eest)

        if ismultiline:
            kwargs['init'] = twinit_list
            kwargs['_keep_initial_particles'] = _keep_ini_particles_list
        else:
            kwargs['init'] = twinit_list[0]
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
            out = self.line.twiss(**self.kwargs)
        else:
            try:
                out = self.line.twiss(**self.kwargs)
            except Exception as ee:
                if allow_failure:
                    return 'failed'
                else:
                    raise ee
        out.line = self.line
        return out

# Alternative transitions functions
# def _transition_sigmoid_integral(x):
#     x_shift = x - 3
#     if x_shift > 10:
#         return x_shift
#     else:
#         return np.log(1 + np.exp(x_shift))

# def _transition_sin(x):
#     if x < 0:
#         return 0
#     if x < 1.:
#         return 2 /np.pi - 2 /np.pi * np.cos(np.pi * x / 2)
#     else:
#         return x + 2 / np.pi - 1

def _poly(x):
     return 3 * x**3 - 2 * x**4

def _transition_poly(x):
        x_cut = 1/16 + np.sqrt(33)/16
        if x < 0:
            return 0
        if x < x_cut:
            return _poly(x)
        else:
            return x - x_cut + _poly(x_cut)

class GreaterThan:

    _transition = staticmethod(_transition_poly)

    def __init__(self, lower, mode='step', sigma=None,
                 sigma_rel=XTRACK_DEFAULT_SIGMA_REL):
        assert mode in ['step', 'smooth']
        self.lower = lower
        self._value = 0.
        self.mode=mode
        if mode == 'smooth':
            assert sigma is not None or sigma_rel is not None
            if sigma is not None:
                assert sigma_rel is None
                self.sigma = sigma
            else:
                assert sigma_rel is not None
                self.sigma = np.abs(self.lower) * sigma_rel

    def auxtarget(self, res):
        '''Transformation applied to target value to obtain the corresponding
        cost function.
        '''
        if self.mode == 'step':
            if res < self.lower:
                return res - self.lower
            else:
                return 0
        elif self.mode == 'smooth':
            return self.sigma * self._transition((self.lower - res) / self.sigma)
        elif self.mode == 'auxvar':
            raise NotImplementedError # experimental
            return res - self.lower - self.vary.container[self.vary.name]**2
        else:
            raise ValueError(f'Unknown mode {self.mode}')

    def __repr__(self):
        return f'GreaterThan({self.lower:4g})'

    # Part of the `auxvar` experimental code
    # def _set_value(self, val, target):
    #     self.lower = val
    #     aux_vary_container = self.vary.container
    #     aux_vary_container[self.vary.name] = 0
    #     val = target.runeval()
    #     if val > 0:
    #         aux_vary_container[self.vary.name] = np.sqrt(val)
    # def gen_vary(self, container):
    #     self.vary = _gen_vary(container)
    #     return self.vary

class LessThan:

    _transition = staticmethod(_transition_poly)

    def __init__(self, upper, mode='step', sigma=None,
                 sigma_rel=XTRACK_DEFAULT_SIGMA_REL):
        assert mode in ['step', 'smooth']
        self.upper = upper
        self._value = 0.
        self.mode=mode
        if mode == 'smooth':
            assert sigma is not None or sigma_rel is not None
            if sigma is not None:
                assert sigma_rel is None
                self.sigma = sigma
            else:
                assert sigma_rel is not None
                self.sigma = np.abs(self.upper) * sigma_rel

    def auxtarget(self, res):
        if self.mode == 'step':
            if res > self.upper:
                return self.upper - res
            else:
                return 0
        elif self.mode == 'smooth':
            return self.sigma * self._transition((res - self.upper) / self.sigma)
        elif self.mode == 'auxvar':
            raise NotImplementedError # experimental
            return self.upper - res - self.vary.container[self.vary.name]**2
        else:
            raise ValueError(f'Unknown mode {self.mode}')

    def __repr__(self):
        return f'LessThan({self.upper:4g})'

# part of the `auxvar` experimental code
# def _gen_vary(container):
#     for ii in range(10000):
#         if f'auxvar_{ii}' not in container:
#             vv = f'auxvar_{ii}'
#             break
#     else:
#         raise RuntimeError('Too many auxvary variables')
#     container[vv] = 0
#     return xt.Vary(name=vv, container=container, step=1e-3)


class Target(xd.Target):

    def __init__(self, tar=None, value=None, at=None, tol=None, weight=None, scale=None,
                 line=None, action=None, tag='', optimize_log=False,
                 **kwargs):

        """
        Target object for matching. Usage examples:

        .. code-block:: python

            Target('betx', 0.15, at='ip1', tol=1e-3)
            Target(betx=0.15, at='ip1', tol=1e-3)
            Target('betx', LessThan(0.15), at='ip1', tol=1e-3)
            Target('betx', GreaterThan(0.15), at='ip1', tol=1e-3)


        Parameters
        ----------
        tar : str or callable
            Name of the quantity to be matched or callable computing the
            quantity to be matched from the output of the action (by default the
            action is the Twiss action). Basic targets can also be specified
            using keyword arguments.
        value : float or xdeps.GreaterThan or xdeps.LessThan or xtrack.TwissTable
            Value to be matched. Inequality constraints can also be specified.
            If a TwissTable is specified, the value is obtained from the
            table using the specified tar and at.
        at : str, optional
            Element at which the quantity is evaluated. Needs to be specified
            if the quantity to be matched is not a scalar.
        tol : float, optional
            Tolerance below which the target is considered to be met.
        weight : float, optional
            Weight used for this target in the cost function.
        line : Line, optional
            Line in which the quantity is defined. Needs to be specified if the
            match involves multiple lines.
        action : Action, optional
            Action used to compute the quantity to be matched. By default the
            action is the Twiss action.
        tag : str, optional
            Tag associated to the target. Default is ''.
        optimize_log : bool, optional
            If True, the logarithm of the quantity is used in the cost function
            instead of the quantity itself. Default is False.
        """


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

        self._freeze_value = None

        xd.Target.__init__(self, tar=xdtar, value=value, tol=tol,
                            weight=weight, scale=scale, action=action, tag=tag,
                            optimize_log=optimize_log)
        self.line = line

    def __repr__(self):
        out = xd.Target.__repr__(self)
        if self.line is not None:
            lname = self.line
        elif hasattr(self.action, 'line') and hasattr(self.action.line , 'name'):
            lname = self.action.line.name
        else:
            lname = None
        if lname is not None:
            out = out.replace('Target(', f'Target(line={lname}, ')
        return out

    def eval(self, data):
        res = data[self.action]
        if self.line is not None:
            res = res[self.line]
        if callable(self.tar):
            out = self.tar(res)
        else:
            out = res[self.tar]

        if self._freeze_value is not None:
            return out

        return out

    def transform(self, val):
        if hasattr(self.value, 'auxtarget'):
            return self.value.auxtarget(val)
        else:
            return val

    @property
    def value(self):
        if self._freeze_value is not None:
            return self._freeze_value
        else:
            return self._user_value

    @value.setter
    def value(self, val):
        self._user_value = val

    def freeze(self):
        self._freeze_value = True # to bypass inequality logic
        self._freeze_value = self.runeval()

    def unfreeze(self):
        self._freeze_value = None

class TargetSet(xd.TargetList):

    def __init__(self, tars=None, value=None, at=None, tol=None, weight=None,
                 scale=None, line=None, action=None, tag='', optimize_log=False,
                 **kwargs):

        """
        TargetSet object for matching, specifying a set of targets to be matched.

        Examples:

        .. code-block:: python
                TargetSet(['betx', 'bety'], 0.15, at='ip1', tol=1e-3)
                TargetSet(betx=0.15, bety=0.2, at='ip1', tol=1e-3)

        Parameters
        ----------
        tars : list, optional
            List of quantities to be matched. Basic targets can also be
            specified using keyword arguments.
        value : float or xdeps.GreaterThan or xdeps.LessThan
            Value to be matched. Inequality constraints can also be specified.
        at : str, optional
            Element at which the quantity is evaluated. Needs to be specified
            if the quantity to be matched is not a scalar.
        tol : float, optional
            Tolerance below which the target is considered to be met.
        weight : float, optional
            Weight used for this target in the cost function.
        line : Line, optional
            Line in which the quantity is defined. Needs to be specified if the
            match involves multiple lines.
        action : Action, optional
            Action used to compute the quantity to be matched. By default the
            action is the Twiss action.
        tag : str, optional
            Tag associated to the target. Default is ''.
        optimize_log : bool, optional
            If True, the logarithm of the quantity is used in the cost function
            instead of the quantity itself. Default is False.
        """

        if tars is not None and not isinstance(tars, (list, tuple)):
            tars = [tars]

        common_kwargs = locals().copy()
        common_kwargs.pop('self')
        common_kwargs.pop('kwargs')
        common_kwargs.pop('tars')
        common_kwargs.pop('value')

        vnames = []
        vvalues = []
        for kk in ALLOWED_TARGET_KWARGS:
            if kk in kwargs:
                vnames.append(kk)
                vvalues.append(kwargs[kk])
                kwargs.pop(kk)

        self.targets = []
        if tars is not None:
            self.targets += [Target(tt, value=value, **common_kwargs) for tt in tars]
        self.targets += [
            Target(tar=tar, value=val, **common_kwargs) for tar, val in zip(vnames, vvalues)]
        if len(self.targets) == 0:
            raise ValueError('No targets specified')

TargetList = TargetSet # for backward compatibility

class Vary(xd.Vary):

    def __init__(self, name, container=None, limits=None, step=None, weight=None,
                 max_step=None, active=True, tag=''):
        """
        Vary object for matching.

        Parameters
        ----------
        name : str
            Name of the variable to be varied.
        container : dict, optional
            Container in which the variable is defined. If not specified,
            line.vars is used.
        limits : tuple or None, optional
            Limits in which the variable is allowed to vary. Default is None.
        step : float, optional
            Step size used to compute the derivative of the cost function
            with respect to the variable.
        weight : float, optional
            Weight used for this vary in the cost function.
        max_step : float, optional
            Maximum allowed change in the variable per iteration.
        active : bool, optional
            Whether the variable is active in the optimization. Default is True.
        tag : str, optional
            Tag associated to the variable. Default is ''.

        """

        xd.Vary.__init__(self, name=name, container=container, limits=limits,
                         step=step, weight=weight, max_step=max_step, tag=tag,
                         active=active)

class VaryList(xd.VaryList):

    def __init__(self, vars, container=None, limits=None, step=None, weight=None,
                 max_step=None, active=True, tag=''):
        """
        VaryList object for matching specifying a list of variables to be varied.

        Parameters
        ----------
        vars : list
            List of variables to be varied.
        container : dict, optional
            Container in which the variables are defined. If not specified,
            line.vars is used.
        limits : tuple or None, optional
            Limits in which the variables are allowed to vary. Default is None.
        step : float, optional
            Step size used to compute the derivative of the cost function
            with respect to the variables.
        weight : float, optional
            Weight used for these variables in the cost function.
        max_step : float, optional
            Maximum allowed change in the variables per iteration.
        active : bool, optional
            Whether the variables are active in the optimization. Default is True.
        tag : str, optional
            Tag associated to the variables. Default is ''.
        """

        kwargs = dict(container=container, limits=limits, step=step,
                      weight=weight, max_step=max_step, active=active, tag=tag)
        self.vary_objects = [Vary(vv, **kwargs) for vv in vars]

class TargetInequality(Target):

    def __init__(self, tar, ineq_sign, rhs, at=None, tol=None, scale=None,
                 line=None, weight=None, tag=''):

        raise NotImplementedError('TargetInequality is not anymore supported. '
            'Please use Target with `GreaterThan` `LessThan` instead. '
            'For example, instead of '
            'TargetInequality("x", "<", 0.1, at="ip1") '
            'use '
            'Target("x", LessThan(0.1), at="ip1")')

class TargetRelPhaseAdvance(Target):

    def __init__(self, tar, value, end=None, start=None, tag='',  **kwargs):

        """
        Target object for matching the relative phase advance between two
        elements in a line computed as mu(end) - mu(start).

        Parameters
        ----------
        tar : str
            Phase advance to be matched. Can be either 'mux' or 'muy'.
        value : float or GreaterThan or LessThan or TwissTable
            Value to be matched. Inequality constraints can also be specified.
            If a TwissTable is specified, the target obtained from the table
            using the specified tar and at.
        end : str, optional
            Final element at which the phase advance is evaluated. Default is the
            last element of selected twiss range.
        start : str, optional
            Initali wlement at which the phase advance is evaluated. Default is the
            first element of the selected twiss range.
        tol : float, optional
            Tolerance below which the target is considered to be met.
        weight : float, optional
            Weight used for this target in the cost function.
        line : Line, optional
            Line in which the phase advance is defined. Needs to be specified if the
            match involves multiple lines.
        tag : str, optional
            Tag associated to the target. Default is ''.
        """

        Target.__init__(self, tar=self.compute, value=value, tag=tag, **kwargs)

        assert tar in ['mux', 'muy'], 'Only mux and muy are supported'
        self.var = tar
        if end is None:
            end = '__ele_stop__'
        if start is None:
            start = '__ele_start__'
        self.end = end
        self.start = start

    def __repr__(self):
        return f'TargetPhaseAdv({self.var}({self.end} - {self.start}), val={self.value}, tol={self.tol}, weight={self.weight})'

    def compute(self, tw):

        if self.end == '__ele_stop__':
            mu_1 = tw[self.var, -1]
        else:
            mu_1 = tw[self.var, self.end]

        if self.start == '__ele_start__':
            mu_0 = tw[self.var, 0]
        else:
            mu_0 = tw[self.var, self.start]

        return mu_1 - mu_0


class TargetRmatrixTerm(Target):

    def __init__(self, tar, value, start=None, end=None, tag='',  **kwargs):

        """
        Target object for matching terms of the R-matrix between two
        elements in a line.

        Parameters
        ----------
        tar : str
            Term to be matched. Can be "r11", "r12", "r21", "r22", etc
        value : float or GreaterThan or LessThan or TwissTable
            Value to be matched. Inequality constraints can also be specified.
            If a TwissTable is specified, the target obtained from the table
            using the specified tar and at.
        start : str
            First element of the range for which the R-matrix is computed.
        end : str
            End element of the range for which the R-matrix is computed.
        tol : float, optional
            Tolerance below which the target is considered to be met.
        weight : float, optional
            Weight used for this target in the cost function.
        line : Line, optional
            Line in which the R matrix is calculated. Needs to be specified if the
            match involves multiple lines.
        tag : str, optional
            Tag associated to the target. Default is ''.
        """

        assert isinstance(tar, str), 'Only strings are supported for tar'
        assert len(tar) == 3, (
            'Only terms of the R-matrix in the form "r11", "r12", "r21", "r22", etc'
            ' are supported')

        Target.__init__(self, tar=self.compute, value=value, tag=tag, **kwargs)

        self.term = tar
        if end is None:
            raise NotImplementedError('end cannot be None')
            end = '__ele_stop__'
        if start is None:
            raise NotImplementedError('start cannot be None')
            start = '__ele_start__'
        self.end = end
        self.start = start

    def __repr__(self):
        return f'{self.term}({self.start}, {self.end}, val={self.value}, tol={self.tol}, weight={self.weight})'

    def compute(self, tw):

        assert isinstance(self.term, str), 'Only strings are supported for tar'
        assert len(self.term) == 3, (
            'Only terms of the R-matrix in the form "r11", "r12", "r21", "r22", etc'
            ' are supported')

        rmat = tw.get_R_matrix(self.start, self.end)

        ii = int(self.term[1]) - 1
        jj = int(self.term[2]) - 1

        assert ii >= 0 and ii < 6, 'Invalid R-matrix term'
        assert jj >= 0 and jj < 6, 'Invalid R-matrix term'

        return rmat[ii, jj]

class TargetRmatrix(TargetSet):

    def __init__(self, tars=None, value=None,start=None, end=None,
        r11=None, r12=None, r13=None, r14=None, r15=None, r16=None,
        r21=None, r22=None, r23=None, r24=None, r25=None, r26=None,
        r31=None, r32=None, r33=None, r34=None, r35=None, r36=None,
        r41=None, r42=None, r43=None, r44=None, r45=None, r46=None,
        r51=None, r52=None, r53=None, r54=None, r55=None, r56=None,
        r61=None, r62=None, r63=None, r64=None, r65=None, r66=None,
        **kwargs):

        if tars is not None:
            raise NotImplementedError
        if value is not None:
            raise NotImplementedError

        r_elems = {
            'r11': r11, 'r12': r12, 'r13': r13, 'r14': r14, 'r15': r15, 'r16': r16,
            'r21': r21, 'r22': r22, 'r23': r23, 'r24': r24, 'r25': r25, 'r26': r26,
            'r31': r31, 'r32': r32, 'r33': r33, 'r34': r34, 'r35': r35, 'r36': r36,
            'r41': r41, 'r42': r42, 'r43': r43, 'r44': r44, 'r45': r45, 'r46': r46,
            'r51': r51, 'r52': r52, 'r53': r53, 'r54': r54, 'r55': r55, 'r56': r56,
            'r61': r61, 'r62': r62, 'r63': r63, 'r64': r64, 'r65': r65, 'r66': r66,
        }

        tol = kwargs.pop('tol', None)

        self.targets = []
        for kk, vv in r_elems.items():
            thistol = tol
            if thistol is not None:
                if kk[1] in ['2', '4']:
                    thistol *= 1e-2
                if kk[2] in ['2', '4']:
                    thistol *= 1e+2
            if vv is not None:
                self.targets.append(TargetRmatrixTerm(kk, vv, start=start, end=end,
                                                      tol=thistol, **kwargs))


def match_line(line, vary, targets, solve=True, assert_within_tol=True,
                  compensate_radiation_energy_loss=False,
                  solver_options={}, allow_twiss_failure=True,
                  restore_if_fail=True, verbose=False,
                  n_steps_max=20, default_tol=None,
                  solver=None, check_limits=True, **kwargs):

    if not isinstance(targets, (list, tuple)):
        targets = [targets]

    targets_flatten = []
    for tt in targets:
        if isinstance(tt, xd.TargetList):
            for tt1 in tt.targets:
                targets_flatten.append(tt1.copy())
        else:
            targets_flatten.append(tt.copy())

    aux_vary = []

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
            tt_at = _at_from_placeholder(tt_at, line=tt.action.line,
                    line_name=tt.line, start=tt.action.kwargs['start'],
                    end=tt.action.kwargs['end'])
            tt.tar = (tt_name, tt_at)

        # Handle value
        if isinstance(tt.value, xt.multiline.MultiTwiss):
            tt.value=tt.value[tt.line][tt.tar]
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

        # part of the `auxvar` experimental code
        # if isinstance(tt.value, (GreaterThan, LessThan)):
        #     if tt.value.mode == 'auxvar':
        #         aux_vary.append(tt.value.gen_vary(aux_vary_container))
        #         aux_vary_container[aux_vary[-1].name] = 0
        #         val = tt.runeval()
        #         if val > 0:
        #             aux_vary_container[aux_vary[-1].name] = np.sqrt(val)

    if not isinstance(vary, (list, tuple)):
        vary = [vary]

    vary = list(vary) + aux_vary

    vary_flatten = _flatten_vary(vary)
    _complete_vary_with_info_from_line(vary_flatten, line)

    opt = xd.Optimize(vary=vary_flatten, targets=targets_flatten, solver=solver,
                        verbose=verbose, assert_within_tol=assert_within_tol,
                        solver_options=solver_options,
                        n_steps_max=n_steps_max,
                        restore_if_fail=restore_if_fail,
                        check_limits=check_limits)

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
            init=xt.TwissInit(
                line=line,
                element_name=corr['start'],
                x=tw_ref['x', corr['start']],
                px=tw_ref['px', corr['start']],
                y=tw_ref['y', corr['start']],
                py=tw_ref['py', corr['start']],
                zeta=tw_ref['zeta', corr['start']],
                delta=tw_ref['delta', corr['start']],
            ),
            start=corr['start'], end=corr['end'])

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

def _at_from_placeholder(tt_at, line, line_name, start, end):
    assert isinstance(tt_at, _LOC)
    if isinstance(line, xt.Multiline):
        assert line is not None, (
            'For a Multiline, the line must be specified if the target '
            'is `start`')
        assert line_name in line.line_names
        i_line = line.line_names.index(line_name)
        this_line = line[line_name]
    else:
        i_line = None
        this_line = line
    if tt_at.name == 'START':
        if i_line is not None:
            tt_at = start[i_line]
        else:
            tt_at = start
    elif tt_at.name == 'END':
        if i_line is not None:
            tt_at = end[i_line]
        else:
            tt_at = end
    else:
        raise ValueError(f'Unknown location {tt_at.name}')
    if not isinstance(tt_at, str):
        tt_at = this_line.element_names[tt_at]

    return tt_at

def opt_from_callable(function, x0, steps, tar, tols):

    '''Optimize a generic callable'''

    x0 = np.array(x0)
    x = x0.copy()
    vary = [xt.Vary(ii, container=x, step=steps[ii]) for ii in range(len(x))]

    opt = xd.Optimize(
        vary=vary,
        targets=ActionCall(function, vary).get_targets(tar),
        show_call_counter=False,
    )

    for ii, tt in enumerate(opt.targets):
        tt.tol = tols[ii]

    return opt

class ActionCall(Action):
    def __init__(self, function, vary):
        self.vary = vary
        self.function = function

    def run(self):
        x = [vv.container[vv.name] for vv in self.vary]
        return self.function(x)

    def get_targets(self, ftar):
        tars = []
        for ii in range(len(ftar)):
            tars.append(xt.Target(ii, ftar[ii], action=self))

        return tars