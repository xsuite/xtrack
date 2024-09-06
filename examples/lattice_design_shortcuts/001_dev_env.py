import xtrack as xt
import numpy as np

def _flatten_components(components):
    flatten_components = []
    for nn in components:
        if isinstance(nn, xt.Line):
            flatten_components += nn.element_names
        else:
            flatten_components.append(nn)
    return flatten_components

class Environment:
    def __init__(self, element_dict=None, particle_ref=None):
        self._element_dict = element_dict or {}
        self.particle_ref = particle_ref

        self._init_var_management()

    def new_line(self, components, name=None):
        out = xt.Line()
        out.particle_ref = self.particle_ref
        out.line = self
        out._element_dict = self.element_dict # Avoid copying
        out.element_names = _flatten_components(components)
        out._var_management = self._var_management
        out._name = name

        return out

Environment.element_dict = xt.Line.element_dict
Environment._init_var_management = xt.Line._init_var_management
Environment._xdeps_vref = xt.Line._xdeps_vref
Environment._xdeps_fref = xt.Line._xdeps_fref
Environment._xdeps_manager = xt.Line._xdeps_manager
Environment._xdeps_eval = xt.Line._xdeps_eval
Environment.element_refs = xt.Line.element_refs
Environment.vars = xt.Line.vars
Environment.varval = xt.Line.varval
Environment.vv = xt.Line.vv
Environment.new_element = xt.Line.new_element

env = Environment(particle_ref=xt.Particles(p0c=2e9))

n_bends_per_cell = 6
n_cells_par_arc = 3
n_arcs = 3

n_bends = n_bends_per_cell * n_cells_par_arc * n_arcs

env.vars({
    'k1l.qf': 0.027 / 2,
    'k1l.qd': -0.0271 / 2,
    'l.mq': 0.5,
    'kqf.1': 'k1l.qf / l.mq',
    'kqd.1': 'k1l.qd / l.mq',
    'l.mb': 12,
    'angle.mb': 2 * np.pi / n_bends,
    'k0.mb': 'angle.mb / l.mb',
})

halfcell = env.new_line(components=[
    env.new_element('drift.1', xt.Drift,      length='l.mq / 2'),
    env.new_element('qf',      xt.Quadrupole, k1='kqf.1', length='l.mq'),
    env.new_element('drift.2', xt.Replica,    parent_name='drift.1'),
    env.new_element('mb.1',    xt.Bend,       k0='k0.mb', h='k0.mb', length='l.mb'),
    env.new_element('mb.2',    xt.Replica,    parent_name='mb.1'),
    env.new_element('mb.3',    xt.Replica,    parent_name='mb.1'),
    env.new_element('drift.3', xt.Replica,    parent_name='drift.1'),
    env.new_element('qd',      xt.Quadrupole, k1='kqd.1', length='l.mq'),
    env.new_element('drift.4', xt.Replica,    parent_name='drift.1'),
])

hcell_left = halfcell.replicate(name='l')
hcell_right = halfcell.replicate(name='r', mirror=True)

cell = env.new_line(components=[
    env.new_element('start', xt.Marker),
    hcell_left,
    env.new_element('mid', xt.Marker),
    hcell_right,
    env.new_element('end', xt.Marker),
])