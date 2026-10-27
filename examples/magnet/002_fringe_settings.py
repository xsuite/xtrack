"""
Edge (fringe) settings of magnets.

- Each face has its own settings: edge_entry_<param>, edge_exit_<param>.
- edge_<face>_active: the face is tracked or not.
- edge_<face>_model: which fringe maps are used.
- get_available_edge_models() lists the models of an element.
"""

import xtrack as xt

p0 = xt.Particles(p0c=1e9, x=1e-3, px=2e-4, y=-2e-3, py=1e-4, delta=1e-3)


def px_after(el):
    p = p0.copy()
    el.track(p)
    return p.px[0]


# --- Quadrupole ---
# - Edges are off by default.
# - Models: "full" (default), "no-mult", "suppressed".
quad = xt.Quadrupole(k1=0.1, length=1.0)
print('quadrupole models:', quad.get_available_edge_models())

# - Switch on the hard-edge fringe at both faces.
hard = quad.copy()
hard.edge_entry_active = True
hard.edge_exit_active = True

# - Add the SAD soft fringe: set qint1, qint2.
soft = hard.copy()
soft.edge_entry_qint1 = 0.2
soft.edge_entry_qint2 = 0.1
# - Same at the exit.
soft.edge_exit_qint1 = 0.2
soft.edge_exit_qint2 = 0.1

# - Drop the hard-edge multipole fringe, keep the soft one.
no_mult = soft.copy()
no_mult.edge_entry_model = 'no-mult'
no_mult.edge_exit_model = 'no-mult'

for name, el in [
    ('no edges', quad),
    ('hard edge', hard),
    ('hard + soft', soft),
    ('no-mult', no_mult),
]:
    print(f'quad, {name:<12} px = {px_after(el):.10e}')

# --- Bend ---
# - Edges are on by default, model "linear".
# - Models: "linear", "full", "no-mult", "suppressed".
bend = xt.Bend(
    length=1.0,
    angle=0.1,
    edge_entry_angle=0.05,
    edge_entry_fint=0.5,
    edge_entry_hgap=0.03,
)
# - Same face settings at the exit.
bend.edge_exit_angle = 0.05
bend.edge_exit_fint = 0.5
bend.edge_exit_hgap = 0.03

# - Nonlinear edges at both faces.
full = bend.copy()
full.edge_entry_model = 'full'
full.edge_exit_model = 'full'

# - No edges at all.
off = bend.copy()
off.edge_entry_active = False
off.edge_exit_active = False

for name, el in [('linear', bend), ('full', full), ('off', off)]:
    print(f'bend, {name:<7} px = {px_after(el):.10e}')

# --- Solenoid ---
# - Edges are on, model "full": the fringe of knl, ksl.
# - There is no fringe for ks.
sol = xt.UniformSolenoid(length=1.0, ks=0.1, knl=[0, 0.05])
print('solenoid models:', sol.get_available_edge_models())
