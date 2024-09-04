import xtrack as xt
import xdeps as xd
import xobjects as xo

import numpy as np

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

    return name

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

def _flatten_components(components):
    flatten_components = []
    for nn in components:
        if isinstance(nn, Section):
            flatten_components += _flatten_components(nn.components)
        else:
            flatten_components.append(nn)
    return flatten_components

class Section(xt.Line):
    def __init__(self, line, components, name=None):
        self.line = line
        xt.Line.__init__(self, elements=line.element_dict,
                         element_names=_flatten_components(components))
        self._var_management = line._var_management
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def particle_ref(self):
        return self.line.particle_ref

    @particle_ref.setter
    def particle_ref(self, value):
        assert value is None

    @property
    def components(self):
        return self.element_names

    def mirror(self):
        self.element_names = self.element_names[::-1]

    def replicate(self, name):
        new_components = []
        for nn in self.components:
            new_nn = nn + '.' + name
            self.line.element_dict[new_nn] = xt.Replica(nn)
            new_components.append(new_nn)
        return Section(self.line, new_components, name=name)

def _section(line, components, name=None):
    return Section(line, components, name=name)

def _append_section(line, section):
    line.element_names += section.components

xt.Line.new_element = _new_element
xt.line.LineVars.__call__ = _call_vars
xt.Line.new_section = _section
xt.Line.append_section = _append_section

line = xt.Line()
line.particle_ref = xt.Particles(p0c=2e9)

line._eval_obj = xd.madxutils.MadxEval(variables=line._xdeps_vref,
                                       functions=line._xdeps_fref,
                                       elements=line.element_dict)

line.vars({
    'k1l.qf': 0.027 / 2,
    'k1l.qd': -0.0271 / 2,
    'l.mq': 0.5,
    'kqf.1': 'k1l.qf / l.mq',
    'kqd.1': 'k1l.qd / l.mq',
    'l.mb': 12,
    'angle.mb': 2 * np.pi / 48 ,
    'k0.mb': 'angle.mb / l.mb',
})

halfcell = line.new_section(components=[
    line.new_element('drift.1', xt.Drift,      length='l.mq / 2'),
    line.new_element('qf',      xt.Quadrupole, k1='kqf.1', length='l.mq'),
    line.new_element('drift.2', xt.Replica,    parent_name='drift.1'),
    line.new_element('mb.1',    xt.Bend,       k0='k0.mb', h='k0.mb', length='l.mb'),
    line.new_element('mb.2',    xt.Replica,    parent_name='mb.1'),
    line.new_element('mb.3',    xt.Replica,    parent_name='mb.1'),
    line.new_element('drift.3', xt.Replica,    parent_name='drift.1'),
    line.new_element('qd',      xt.Quadrupole, k1='kqd.1', length='l.mq'),
    line.new_element('drift.4', xt.Replica,    parent_name='drift.1'),
])

hcell_left = halfcell.replicate('l')
hcell_right = halfcell.replicate('r')
hcell_right.mirror()

cell= line.new_section(components=[
    line.new_element('start', xt.Marker),
    hcell_left,
    line.new_element('mid', xt.Marker),
    hcell_right,
    line.new_element('end', xt.Marker),
])

cell.twiss4d().plot()



