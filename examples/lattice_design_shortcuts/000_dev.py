import xtrack as xt

line = xt.Line()

class Expr:
    def __init__(self, expr):
        self.expr = expr

# line._xdeps_vref._eval("a * b")

def _eval(line, expr):
    return line._xdeps_vref._eval(expr)

def _new_element(line, name, cls, **kwargs):

    evaluated_kwargs = {}
    value_kwargs = {}
    for kk in kwargs:
        if isinstance(kwargs[kk], Expr):
            evaluated_kwargs[kk] = _eval(line, kwargs[kk].expr)
            value_kwargs[kk] = evaluated_kwargs[kk]._value
        else:
            evaluated_kwargs[kk] = kwargs[kk]
            value_kwargs[kk] = kwargs[kk]

    element = cls(**value_kwargs)
    line.element_dict[name] = element
    for kk in kwargs:
        setattr(line.element_refs[name], kk, evaluated_kwargs[kk])

line.vars['a'] = 2.

_new_element(line, 'd', xt.Drift, length=Expr('a'))
