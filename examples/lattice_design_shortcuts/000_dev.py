import xtrack as xt
import xdeps as xd
import xobjects as xo

class Expr:
    def __init__(self, expr):
        self.expr = expr

# line._xdeps_vref._eval("a * b")


def _new_element(line, name, cls, **kwargs):

    _eval = line._eval_obj.eval

    evaluated_kwargs = {}
    value_kwargs = {}
    for kk in kwargs:
        if isinstance(kwargs[kk], Expr):
            evaluated_kwargs[kk] = _eval(kwargs[kk].expr)
            value_kwargs[kk] = evaluated_kwargs[kk]._value
        elif hasattr(kwargs[kk], '_value'):
            evaluated_kwargs[kk] = kwargs[kk]
            value_kwargs[kk] = kwargs[kk]._value
        elif (isinstance(kwargs[kk], str) and hasattr(cls, '_xofields')
             and kk in cls._xofields and cls._xofields[kk].__name__ != 'String'):
            evaluated_kwargs[kk] = _eval(kwargs[kk])
            value_kwargs[kk] = evaluated_kwargs[kk]._value
        else:
            evaluated_kwargs[kk] = kwargs[kk]
            value_kwargs[kk] = kwargs[kk]

    element = cls(**value_kwargs)
    line.element_dict[name] = element
    for kk in kwargs:
        setattr(line.element_refs[name], kk, evaluated_kwargs[kk])

def _call_vars(vars, *args, **kwargs):
    _eval = vars.line._eval_obj.eval
    if len(args) > 0:
        assert len(kwargs) == 0
        assert len(args) == 1
        if isinstance(args[0], str):
            return vars[args[0]]
        elif isinstance(args[0], dict):
            kwargs.update(args[0])
        else:
            raise ValueError('Invalid argument')
    for kk in kwargs:
        if isinstance(kwargs[kk], Expr):
            vars[kk] = _eval(kwargs[kk].expr)
        elif isinstance(kwargs[kk], str):
            vars[kk] = _eval(kwargs[kk])
        else:
            vars[kk] = kwargs[kk]

xt.Line.newele = _new_element
xt.line.LineVars.__call__ = _call_vars

line = xt.Line()

line._eval_obj = xd.madxutils.MadxEval(variables=line._xdeps_vref,
                                       functions=line._xdeps_fref,
                                       elements=line.element_dict)

line.vars({
    'kqf': 0.027,
    'kqd': -0.0271,
    'kqf.1': 'kqf / 2',
    'kqd.1': 'kqd / 2',
})

line.newele('qf.1', xt.Quadrupole, k1='kqf.1', length=1.)
line.newele('qd.1', xt.Quadrupole, k1='kqd.1', length=1.)


