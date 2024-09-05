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
        self._element_dict = line.element_dict # Avoid copying
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

def _append(line, section):
    line.element_names += section.components

def _replace_replica(line, name):
    name_parent = line[name].resolve(line, get_name=True)
    line.element_dict[name] = line[name_parent].copy()

    pars_with_expr = list(
        line._xdeps_manager.tartasks[line.element_refs[name_parent]].keys())

    for rr in pars_with_expr:
        assert isinstance(rr, xd.refs.AttrRef)
        setattr(line.element_refs[name], rr._key, rr._expr)

def _replace_all_replicas(line):
    for nn in line.element_names:
        if isinstance(line[nn], xt.Replica):
            _replace_replica(line, nn)

xt.Line.new_element = _new_element
xt.line.LineVars.__call__ = _call_vars
xt.Line.new_section = _section
xt.Line.append = _append
xt.Line.replace_replica = _replace_replica
xt.Line.replace_all_replicas = _replace_all_replicas

line = xt.Line()
line.particle_ref = xt.Particles(p0c=2e9)

line._eval_obj = xd.madxutils.MadxEval(variables=line._xdeps_vref,
                                       functions=line._xdeps_fref,
                                       elements=line.element_dict)

n_bends_per_cell = 6
n_cells_par_arc = 3
n_arcs = 3

n_bends = n_bends_per_cell * n_cells_par_arc * n_arcs


line.vars({
    'k1l.qf': 0.027 / 2,
    'k1l.qd': -0.0271 / 2,
    'l.mq': 0.5,
    'kqf.1': 'k1l.qf / l.mq',
    'kqd.1': 'k1l.qd / l.mq',
    'l.mb': 12,
    'angle.mb': 2 * np.pi / n_bends,
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

hcell_left = halfcell.replicate(name='l')
hcell_right = halfcell.replicate(name='r') # could add mirror=True
hcell_right.mirror()

cell = line.new_section(components=[
    line.new_element('start', xt.Marker),
    hcell_left,
    line.new_element('mid', xt.Marker),
    hcell_right,
    line.new_element('end', xt.Marker),
])

arc = line.new_section(components=[
    cell.replicate(name='cell.1'),
    cell.replicate(name='cell.2'),
    cell.replicate(name='cell.3'),
])

cell_ss = cell.replicate('ss')
line.new_element('drift_ss', xt.Drift, length='l.mb')
for ii, nn in enumerate(cell_ss.components):
    if nn.startswith('mb'):
        cell_ss.components[ii] = line.new_element(
            f'drift.{ii}.ss', xt.Replica, parent_name='drift_ss')

ss = line.new_section(components=[
    cell_ss.replicate('cell.1'),
    cell_ss.replicate('cell.2'),
])

arc1 = arc.replicate(name='arc.1')
arc2 = arc.replicate(name='arc.2')
arc3 = arc.replicate(name='arc.3')

ss1 = ss.replicate(name='ss.1')
ss2 = ss.replicate(name='ss.2')
ss3 = ss.replicate(name='ss.3')

line.append(arc1) # Rename to append
line.append(ss1)
line.append(arc2)
line.append(ss2)
line.append(arc3)
line.append(ss3)

line.replace_all_replicas()

opt = cell.match(
    method='4d',
    vary=xt.VaryList(['k1l.qf', 'k1l.qd'], step=1e-5),
    targets=xt.TargetSet(
        qx=0.333,
        qy=0.333,
    ))

tw = line.twiss4d()

line.vars({
    'k1l.q1': 0.012,
    'k1l.q2': -0.012,
    'k1l.q3': 0.012,
    'k1l.q4': -0.012,
    'k1l.q5': 0.012,
    'k1.q1': 'k1l.q1 / l.mq',
    'k1.q2': 'k1l.q2 / l.mq',
    'k1.q3': 'k1l.q3 / l.mq',
    'k1.q4': 'k1l.q4 / l.mq',
    'k1.q5': 'k1l.q5 / l.mq',
})

ss_left = line.new_section(components=[
    line.new_element('ip', xt.Marker),
    line.new_element('dd.0', xt.Drift, length=10),
    line.new_element('mq.1', xt.Quadrupole, k1='k1l.q1', length='l.mq'),
    line.new_element('dd.1', xt.Drift, length=6),
    line.new_element('mq.2', xt.Quadrupole, k1='k1l.q2', length='l.mq'),
    line.new_element('dd.2', xt.Drift, length=18),
    line.new_element('mq.3', xt.Quadrupole, k1='k1l.q3', length='l.mq'),
    line.new_element('dd.3', xt.Drift, length=14),
    line.new_element('mq.4', xt.Quadrupole, k1='k1l.q4', length='l.mq'),
    line.new_element('dd.4', xt.Drift, length=14),
    line.new_element('mq.5', xt.Quadrupole, k1='k1l.q5', length='l.mq'),
    line.new_element('dd.5', xt.Drift, length=7.5),
    line.new_element('e.ss.r', xt.Marker),
])
ss_left.build_tracker()

tw_arc = arc.twiss4d()

bet_ip = 300.

opt = ss_left.match(
    solve=False,
    betx=tw_arc.betx[0], bety=tw_arc.bety[0],
    alfx=tw_arc.alfx[0], alfy=tw_arc.alfy[0],
    init_at='e.ss.r',
    start='ip', end='e.ss.r',
    vary=xt.VaryList(['k1l.q1', 'k1l.q2', 'k1l.q3', 'k1l.q4'], step=1e-5),
    targets=xt.TargetSet(
        alfx=0, alfy=0,
        at='ip'
    ))


opt.step(40)




tw_arc = arc.twiss4d()
ss_arc = line.new_section(components=[ss_left, arc])
ss_arc.cut_at_s(np.arange(0, ss_arc.get_length(), 0.5))
tw_ss_arc = ss_arc.twiss4d(betx=tw_arc.betx[-1], bety=tw_arc.bety[-1],
                           alfx=tw_arc.alfx[-1], alfy=tw_arc.alfy[-1],
                           init_at=xt.END)
tw_ss_arc.plot()

prrrr

import matplotlib.pyplot as plt
plt.close('all')
fig = plt.figure(1)
ax1 = fig.add_subplot(2, 1, 1)
tw.plot('betx bety', ax=ax1)
ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
tw.plot('dx', ax=ax2)

plt.show()



