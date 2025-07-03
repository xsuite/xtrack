import numpy as np

from .twiss import VARS_FOR_TWISS_INIT_GENERATION
from .general import _print, _LOC
import xtrack as xt
import xdeps as xd
import sympy
from xtrack.autodiff import compute_param_derivatives

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
    'dx' : 10.,
    'dpx': 100.,
    'dy' : 10.,
    'dpy': 100.,
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
                        'eq_nemitt_x', 'eq_nemitt_y', 'eq_nemitt_zeta',
                        'spin_x', 'spin_y', 'spin_z',
                        'c_minus_re_0', 'c_minus_im_0',
                        'c_minus_re', 'c_minus_im']


AD_QTY_IDX = {
    "betx": 0,
    "bety": 1,
    "alfx": 2,
    "alfy": 3,
    "mux": 4,
    "muy": 5,
    "dx": 6,
    "dy": 7,
    "dpx": 8,
    "dpy": 9,
}

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
                lower_val = self.lower
                if xd.refs.is_ref(self.lower):
                    lower_val = self.lower._value
                self.sigma = np.abs(lower_val) * sigma_rel

    def auxtarget(self, res):
        '''Transformation applied to target value to obtain the corresponding
        cost function.
        '''
        if xd.refs.is_ref(self.lower):
            lower_val = self.lower._value
        else:
            lower_val = self.lower
        if self.mode == 'step':
            if res < lower_val:
                return res - lower_val
            else:
                return 0
        elif self.mode == 'smooth':
            return self.sigma * self._transition((lower_val - res) / self.sigma)
        elif self.mode == 'auxvar':
            raise NotImplementedError # experimental
            return res - lower_val - self.vary.container[self.vary.name]**2
        else:
            raise ValueError(f'Unknown mode {self.mode}')

    def __repr__(self):
        val = self.lower
        if xd.refs.is_ref(self.lower):
            val = self.lower._value
        return f'GreaterThan({val:.4g})'

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
                upper_val = self.upper
                if xd.refs.is_ref(self.upper):
                    upper_val = self.upper._value
                self.sigma = np.abs(upper_val) * sigma_rel

    def auxtarget(self, res):
        if xd.refs.is_ref(self.upper):
            upper_val = self.upper._value
        else:
            upper_val = self.upper
        if self.mode == 'step':
            if res > upper_val:
                return upper_val - res
            else:
                return 0
        elif self.mode == 'smooth':
            return self.sigma * self._transition((res - upper_val) / self.sigma)
        elif self.mode == 'auxvar':
            raise NotImplementedError # experimental
            return upper_val - res - self.vary.container[self.vary.name]**2
        else:
            raise ValueError(f'Unknown mode {self.mode}')

    def __repr__(self):
        val = self.upper
        if xd.refs.is_ref(self.upper):
            val = self.upper._value
        return f'LessThan({val:.4g})'

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
                 line=None, action=None, tag=None, optimize_log=False,
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

        if tag is None:
            tag_parts = []
            if line is not None:
                tag_parts.append(line)
            if at is not None:
                tag_parts.append(str(at))
            if isinstance(tar, str):
                tag_parts.append(tar)
            tag = '_'.join(tag_parts)

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
                 scale=None, line=None, action=None, tag=None, optimize_log=False,
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

        if self.start is xt.START:
            self.start = tw.name[0]

        if self.end is xt.END:
            self.end = '_end_point'

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

        tag = kwargs.pop('tag', None)

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
                if tag is not None:
                    this_tag = tag
                else:
                    this_tag = kk
                self.targets.append(TargetRmatrixTerm(kk, vv, start=start, end=end,
                                                      tol=thistol, tag=this_tag,
                                                      **kwargs))


def match_line(line, vary, targets, solve=True, assert_within_tol=True,
                  compensate_radiation_energy_loss=False,
                  solver_options={}, allow_twiss_failure=True,
                  restore_if_fail=True, verbose=False,
                  n_steps_max=20, default_tol=None,
                  solver=None, check_limits=True,
                  name="", use_ad=False,
                  **kwargs):

    opt = OptimizeLine(line, vary, targets,
                        assert_within_tol=assert_within_tol,
                        compensate_radiation_energy_loss=compensate_radiation_energy_loss,
                        solver_options=solver_options,
                        allow_twiss_failure=allow_twiss_failure,
                        restore_if_fail=restore_if_fail, verbose=verbose,
                        n_steps_max=n_steps_max, default_tol=default_tol,
                        solver=solver, check_limits=check_limits,
                        name=name, use_ad=use_ad,
                        **kwargs)

    if solve:
        opt.solve()

    return opt


class Action(xd.Action):

    _target_class = Target

    def __init__(self, callable, **kwargs):
        self.callable = callable
        self.kwargs = kwargs

    def run(self, allow_failure=False):
        return self.callable(**self.kwargs)

    def __call__(self, **kwargs):
        return self.run(**kwargs)

class ActionTwiss(xd.Action):

    def __init__(self, line, allow_twiss_failure=False,
                 compensate_radiation_energy_loss=False,
                 **kwargs):
        self.line = line
        self.kwargs = kwargs
        self.allow_twiss_failure = allow_twiss_failure
        self.compensate_radiation_energy_loss = compensate_radiation_energy_loss
        self._alredy_prepared = False

    def prepare(self, force=False):

        if self._alredy_prepared and not force:
            return

        line = self.line
        kwargs = self.kwargs

        ismultiline = isinstance(line, (xt.Multiline, xt.Environment, xt.MultilineLegacy))

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

            for ii, twinit in enumerate(twinit_list):
                if isinstance(twinit, xt.TwissInit):
                    twinit_list[ii] = twinit.copy()
                elif isinstance(twinit, str):
                    assert twinit == 'periodic' or twinit == 'periodic_symmetric'

        if ismultiline:
            kwargs['init'] = twinit_list
            kwargs['_keep_initial_particles'] = len(line_names) * [True]
        else:
            kwargs['init'] = twinit_list[0]
            kwargs['_keep_initial_particles'] = True

        tw0 = line.twiss(**kwargs)
        self._tw0 = tw0

        if ismultiline:
            kwargs['_initial_particles'] = len(line_names) * [None]
            for ii, llnn in enumerate(line_names):
                self.kwargs['init'][ii] = tw0[llnn].completed_init
                if not tw0[llnn].periodic:
                    kwargs['_initial_particles'][ii] = tw0[llnn]._data.get('_initial_particles', None)
        else:
            self.kwargs['init'] = tw0.completed_init
            for kk in VARS_FOR_TWISS_INIT_GENERATION:
                kwargs.pop(kk, None)
            if not(tw0.periodic):
                kwargs['_initial_particles'] = tw0._data.get(
                                        '_initial_particles', None)

        self.kwargs = kwargs

    def run(self, allow_failure=True):
        if self.compensate_radiation_energy_loss:
            if isinstance(self.line, (xt.Multiline, xt.Environment, xt.MultilineLegacy)):
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

class MeritFunctionLine(xd.MeritFunctionForMatch):
    def __init__(
        self,
        merit_function_match,
        use_ad=False
    ):

        self.vary = merit_function_match.vary
        self.targets = merit_function_match.targets
        self.actions = merit_function_match.actions
        self.return_scalar = merit_function_match.return_scalar
        self.call_counter = merit_function_match.call_counter
        self.verbose = merit_function_match.verbose
        self.tw_kwargs = merit_function_match.tw_kwargs
        self.steps_for_jacobian = merit_function_match.steps_for_jacobian
        self.found_point_within_tol = merit_function_match.found_point_within_tol
        self.zero_if_met = merit_function_match.zero_if_met
        self.show_call_counter = merit_function_match.show_call_counter
        self.check_limits = merit_function_match.check_limits
        self.use_ad = use_ad

    def get_derivatives_elements_knobs(self):
        """
        Compute the derivatives of quadrupole k1 with respect to knobs.
        This is done by executing the symbolic expressions for each knob
        and extracting the derivatives of k1 with respect to the symbolic variable.

        Yields
        -------
        dkq_dvv : dict
            A dictionary mapping knob names to another dictionary that maps
            quadrupole names to the derivative of the quadrupole k1 with respect
            to the knob.
        quad_sources_ord : list
            An ordered list of quadrupole names that appear in the derivatives.
        target_places : list
            An ordered list of target locations used in the optimization.
        """

        class DummyElement:
            """Placeholder object for injecting symbolic attributes."""
            pass

        dkq_dvv = {}  # Mapping: knob name -> {quad name -> d(quad)/d(knob)}

        for vary_entry in self.vary:
            knob_name = vary_entry.name
            symbolic_var = sympy.var("a")

            # Find all quadrupole k1 dependencies on this knob
            quad_exprs = []
            dummy_quads = {}

            for dep in self.actions[0].line.ref_manager.find_deps([self.actions[0].line.vars[knob_name]]):
                if dep.__class__.__name__ == "AttrRef" and dep._key == "k1":
                    quad_name = dep._owner._key
                    quad_exprs.append((quad_name, dep._expr))
                    dummy_quads[quad_name] = DummyElement()

            # Build symbolic expression function for the knob
            func_code = self.actions[0].line.ref_manager.mk_fun("myfun", a=self.actions[0].line.vars[knob_name])
            func_globals = {
                "vars": self.actions[0].line.ref_manager.containers["vars"]._owner.copy(),
                "element_refs": dummy_quads,
            }
            func_locals = {}

            ################### myfun ################################
            # def myfun(a):
            #    knob_name = a
            #    element_refs[quad_name].k1 = (1.0 * knob_name) -> SymPy expression
            #    ...
            ##########################################################

            exec(func_code, func_globals, func_locals) # Create function, stored in func_locals
            func_locals["myfun"](symbolic_var) # Execute function

            # Extract derivatives of k1 with respect to this knob
            k1_derivs = {}
            for quad_name, _ in quad_exprs:
                derivative = func_globals["element_refs"][quad_name].k1.diff(symbolic_var)
                k1_derivs[quad_name] = derivative

            dkq_dvv[knob_name] = k1_derivs

        # Set of all quadrupole names appearing in the derivatives
        quad_sources = set()
        for derivs in dkq_dvv.values():
            quad_sources.update(derivs.keys())

        # Set of all target locations used in optimization and sort them by order
        target_places = set()
        for target in self.targets:
            if isinstance(target.tar, tuple):
                target_places.add(target.tar[1])
            elif hasattr(target, "start") and hasattr(target, "end"):
                if target.start != '__ele_start__':
                    target_places.add(target.start)
                if target.end != '__ele_stop__':
                    target_places.add(target.end)
                else:
                    if self.actions[0]._tw0.name[-2] not in target_places:
                        target_places.add(self.actions[0]._tw0.name[-2])
                    # Assumption: Point before _end_point is same as endpoint given in opt
            else:
                raise ValueError(f"Unknown target type: {type(target)}")
        # Convert to ordered list based on appearance of name in opt.line
        index_map = {name: i for i, name in enumerate(self.actions[0]._tw0.name)}
        target_places = sorted(target_places, key=index_map.get)

        # Ordered list of quadrupole sources (based on their position in the beamline)
        quad_sources_ordered = [
            name for name in self.actions[0]._tw0.name if name in quad_sources
        ]

        self.quad_sources_ord = quad_sources_ordered
        self.target_places = target_places
        self.dkq_dvv = dkq_dvv

    def get_jacobian(self, x, f0=None):
        if self.use_ad:
            return self.get_jacobian_ad(x)
        else:
            return super().get_jacobian(x, f0=f0)

    def get_jacobian_ad(self, x):
        if not hasattr(self, "quad_sources_ord") or not hasattr(self, "target_places") or not hasattr(self, "dkq_dvv"):
            self.get_derivatives_elements_knobs()
        x = np.array(x).copy()
        #jacobian = get_jac(opt, all_quad_sources, target_places, dkq_dvv)

        opt_tw = self.actions[0].run()
        #opt_tw = opt.action_twiss._tw0
        # Initial conditions for first derivative calculation
        init_cond = np.array([opt_tw.betx[0], opt_tw.bety[0], opt_tw.alfx[0], opt_tw.alfy[0],
                            opt_tw.mux[0], opt_tw.muy[0], opt_tw.dx[0], opt_tw.dy[0],
                            opt_tw.dpx[0], opt_tw.dpy[0]])
        beta0 = opt_tw.particle_on_co.beta0[0]
        gamma0 = opt_tw.particle_on_co.gamma0[0]

        twiss_derivs = {}
        for place in self.target_places: # ip1, ip8 in order of appearance
            # Calc derivative for all quadrupoles for target place
            # Source point = qqnn, Observation point = target
            twiss_derivs[place] = {}
            trunc_elements = np.array([self.actions[0].line.element_dict[name] for name in opt_tw.rows[:place].name])
            nonzero_qq = []
            nonzero_qqn = []
            for qqnn in self.quad_sources_ord:
                if opt_tw['s', place] < opt_tw['s', qqnn]:
                    twiss_derivs[place][qqnn] = np.zeros(10) # batch after
                else:
                    nonzero_qqn.append(qqnn)
                    nonzero_qq.append(self.actions[0].line.element_dict[qqnn]) # first elements
                    # add to list to be calculated
            nonzero_deriv, _ = compute_param_derivatives(trunc_elements, nonzero_qq, init_cond, beta0, gamma0)

            for i, qqn in enumerate(nonzero_qqn):
                twiss_derivs[place][qqn] = nonzero_deriv[i]
            for qqn, deriv in zip(nonzero_qqn, nonzero_deriv.T):
                twiss_derivs[place][qqn] = deriv

        jac_estim = np.zeros((len(self.targets), len(self.vary)))
        for itt, tt in enumerate(self.targets):

            tar_start = None
            if isinstance(tt.tar, tuple):
                tar_quantity = tt.tar[0]
                tar_place = tt.tar[1]
            else:
                tar_quantity = tt.var
                tar_place = self.target_places[-1] if tt.end == '__ele_stop__' else tt.end
                tar_start = None if tt.start == '__ele_start__' else tt.start
                tar_weight = tt.weight

            tar_weight = tt.weight
            quantity_idx = AD_QTY_IDX[tar_quantity]
            for ivv in range(len(self.vary)):
                vv = self.vary[ivv].name
                quad_names = self.dkq_dvv[vv].keys()

                dtar_dvv = 0
                for qqnn in quad_names:
                    if qqnn in twiss_derivs[tar_place].keys():
                        dtar_dvv += (twiss_derivs[tar_place][qqnn][quantity_idx]) * float(self.dkq_dvv[vv][qqnn])
                        if tar_start is not None:
                            dtar_dvv -= (twiss_derivs[tar_start][qqnn][quantity_idx]) * float(self.dkq_dvv[vv][qqnn])

                dtar_dvv *= tar_weight

                jac_estim[itt, ivv] = dtar_dvv

        #return jac_estim

        self._last_jac = jac_estim
        return jac_estim

class OptimizeLine(xd.Optimize):

    def __init__(self, line, vary, targets, assert_within_tol=True,
                    compensate_radiation_energy_loss=False,
                    solver_options={}, allow_twiss_failure=True,
                    restore_if_fail=True, verbose=False,
                    n_steps_max=20, default_tol=None,
                    solver=None, check_limits=True,
                    action_twiss=None,
                    name="", use_ad=False,
                    **kwargs):

        if hasattr(targets, 'values'): # dict like
            targets = list(targets.values())

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

        for tt in targets_flatten:

            # Handle action
            if tt.action is None:
                if action_twiss is None:
                    action_twiss = ActionTwiss(
                        line, allow_twiss_failure=allow_twiss_failure,
                        compensate_radiation_energy_loss=compensate_radiation_energy_loss,
                        **kwargs)
                    action_twiss.prepare()
                tt.action = action_twiss


            # Handle at
            if isinstance(tt.tar, tuple):
                tt_name = tt.tar[0] # `at` is  present
                tt_at = tt.tar[1]
                if use_ad == True and tt_name not in ['betx', 'bety', 'alfx', 'alfy', 'mux', 'muy', 'dx', 'dy', 'dpx', 'dpy']:
                    print("Warning: use_ad is set to True, but the target {} is not supported for automatic differentiation.")
                    use_ad = False
            else:
                tt_name = tt.tar
                tt_at = None
                if use_ad == True and not isinstance(tt, TargetRelPhaseAdvance):
                    print("Warning: use_ad is set to True, but the target {} is not supported for automatic differentiation.")
                    use_ad = False

            if tt_at is not None and isinstance(tt_at, _LOC):
                assert isinstance(tt.action, ActionTwiss)
                tt.action.prepare() # does nothing if already prepared
                tw0 = tt.action._tw0[tt.line] if tt.line else tt.action._tw0
                this_line = tt.action.line[tt.line] if tt.line else tt.action.line
                if isinstance(tt_at, _LOC):
                    tt_at= tw0['name', {'START':0, 'END':-1}[tt_at.name]]
                    # If _end_point preceded by a marker, use the marker
                    if tt_at == '_end_point' and len(tw0.name) > 1:
                        nn_prev = tw0['name', -2]
                        nn_env_prev = tw0['name_env', -2]
                        if isinstance(this_line[nn_env_prev], xt.Marker):
                            tt_at= nn_prev
                tt.tar = (tt_name, tt_at)

            # Handle value
            if isinstance(tt.value, xt.MultiTwiss):
                tt.value=tt.value[tt.line][tt.tar]
            if isinstance(tt.value, xt.TwissTable):
                if isinstance(tt.tar, tuple) and tt.tar[1] == '_end_point':
                    # '_end_point' of the tar table might be different from the one of the action
                    raise ValueError('TwissTable target value cannot be used with at=_end_point')
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

        xd.Optimize.__init__(self,
                        vary=vary_flatten, targets=targets_flatten, solver=solver,
                        verbose=verbose, assert_within_tol=assert_within_tol,
                        solver_options=solver_options,
                        n_steps_max=n_steps_max,
                        restore_if_fail=restore_if_fail,
                        check_limits=check_limits,
                        name=name, line=line)

        _err = MeritFunctionLine(self._err, use_ad=use_ad)
        self.line = line
        self.action_twiss = action_twiss
        self.default_tol = default_tol
        self._err = _err

    def clone(self, add_targets=None, add_vary=None,
              remove_targets=None, remove_vary=None,
              name=None):

        if hasattr(add_targets, 'copy'):
            add_targets = add_targets.copy()

        if hasattr(add_vary, 'copy'):
            add_vary = add_vary.copy()

        if hasattr(add_targets, 'values'): # dict like
            add_targets = list(add_targets.values())

        if hasattr(add_vary, 'values'): # dict like
            add_vary = list(add_vary.values())

        if name is None:
            name = self.name

        assert remove_targets in [None, True, False]
        assert remove_vary in [None, True, False]

        targets = list(self.targets.copy())
        if remove_targets:
            targets = []
        if add_targets is not None:
            if not isinstance(add_targets, (list, tuple)):
                add_targets = [add_targets]
            targets.extend(add_targets)

        vary = list(self.vary.copy())
        if remove_vary:
            vary = []
        if add_vary is not None:
            if not isinstance(add_vary, (list, tuple)):
                add_vary = [add_vary]
            vary.extend(add_vary)

        out = self.__class__(
            line = self.line,
            vary=vary,
            targets=targets,
            default_tol=self.default_tol,
            restore_if_fail=self.restore_if_fail,
            verbose=self._err.verbose,
            assert_within_tol=self.assert_within_tol,
            n_steps_max=self.n_steps_max,
            check_limits=self.check_limits,
            action_twiss=self.action_twiss,
            name=name,
        )
        return out

    def plot(self, *args, **kwargs):
        return self.action_twiss.run().plot(*args, **kwargs)

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

def opt_from_callable(function, x0, steps, tar, tols):

    '''Optimize a generic callable'''

    opt = xd.Optimize.from_callable(function, x0, tar, steps=steps, tols=tols,
                                    show_call_counter=False)
    return opt
