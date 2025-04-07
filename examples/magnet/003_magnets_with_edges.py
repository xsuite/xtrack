import numpy as np

import xtrack as xt
import xobjects as xo
from xtrack.beam_elements.magnets import Magnet

p0 = xt.Particles(kinetic_energy0=50e6,
                  x=[1e-3, -1e-3], y=2e-3, zeta=1e-2, px=10e-3, py=20e-3, delta=1e-2)


# Test entry linear edge alone
bb = xt.Bend(h=0.1, k0=0.11, length=0,
             edge_entry_angle=0.02, edge_exit_angle=0.03,
             edge_entry_hgap=0.04, edge_exit_hgap=0.05,
             edge_entry_fint=0.1, edge_exit_fint=0.2)

bb.edge_entry_active = 1
bb.edge_exit_active = 0
bb.model = 'rot-kick-rot'
bb.num_multipole_kicks = 10


mm = Magnet(h=0.1, k0=0.11, length=0,
             edge_entry_angle=0.02, edge_exit_angle=0.03,
             edge_entry_hgap=0.04, edge_exit_hgap=0.05,
             edge_entry_fint=0.1, edge_exit_fint=0.2)
mm.edge_entry_active = 1
mm.edge_exit_active = 0
mm.model = 'rot-kick-rot'
mm.num_multipole_kicks = 10


p_test = p0.copy()
p_ref = p0.copy()

mm.track(p_test)
bb.track(p_ref)

xo.assert_allclose(p_test.x, p_ref.x, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.y, p_ref.y, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.zeta, p_ref.zeta, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.px, p_ref.px, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.py, p_ref.py, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.delta, p_ref.delta, atol=1e-15, rtol=0)

# Test backtracking
line = xt.Line(elements=[mm])
line.track(p_test, backtrack=True)
xo.assert_allclose(p_test.s, 0.0, atol=1e-7, rtol=0)
xo.assert_allclose(p_test.x, p0.x, atol=5e-14, rtol=0)
xo.assert_allclose(p_test.y, p0.y, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.zeta, p0.zeta, atol=1e-14, rtol=0)
xo.assert_allclose(p_test.px, p0.px, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.py, p0.py, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.delta, p0.delta, atol=1e-15, rtol=0)

# ===============================================================================

# Test exit linear edge alone
bb = xt.Bend(h=0.1, k0=0.11, length=0,
             edge_entry_angle=0.02, edge_exit_angle=0.03,
             edge_entry_hgap=0.04, edge_exit_hgap=0.05,
             edge_entry_fint=0.1, edge_exit_fint=0.2)

bb.edge_entry_active = 0
bb.edge_exit_active = 1
bb.model = 'rot-kick-rot'
bb.num_multipole_kicks = 10

mm = Magnet(h=0.1, k0=0.11, length=0,
                edge_entry_angle=0.02, edge_exit_angle=0.03,
                edge_entry_hgap=0.04, edge_exit_hgap=0.05,
                edge_entry_fint=0.1, edge_exit_fint=0.2)

mm.edge_entry_active = 0
mm.edge_exit_active = 1
mm.model = 'rot-kick-rot'
mm.num_multipole_kicks = 10

p_test = p0.copy()
p_ref = p0.copy()

mm.track(p_test)
bb.track(p_ref)

xo.assert_allclose(p_test.x, p_ref.x, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.y, p_ref.y, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.zeta, p_ref.zeta, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.px, p_ref.px, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.py, p_ref.py, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.delta, p_ref.delta, atol=1e-15, rtol=0)

# Test backtracking
line = xt.Line(elements=[mm])
line.track(p_test, backtrack=True)
xo.assert_allclose(p_test.s, 0.0, atol=1e-7, rtol=0)
xo.assert_allclose(p_test.x, p0.x, atol=5e-14, rtol=0)
xo.assert_allclose(p_test.y, p0.y, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.zeta, p0.zeta, atol=1e-14, rtol=0)
xo.assert_allclose(p_test.px, p0.px, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.py, p0.py, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.delta, p0.delta, atol=1e-15, rtol=0)


# ==============================================================================
# Testa a full bend with linear edges

bb = xt.Bend(h=0.1, k0=0.11, length=10,
                edge_entry_angle=0.02, edge_exit_angle=0.03,
                edge_entry_hgap=0.04, edge_exit_hgap=0.05,
                edge_entry_fint=0.1, edge_exit_fint=0.2)

bb.edge_entry_active = 1
bb.edge_exit_active = 1
bb.model = 'rot-kick-rot'
bb.num_multipole_kicks = 10
bb.edge_entry_model = 'linear'
bb.edge_exit_model = 'linear'

mm = Magnet(h=0.1, k0=0.11, length=10,
                edge_entry_angle=0.02, edge_exit_angle=0.03,
                edge_entry_hgap=0.04, edge_exit_hgap=0.05,
                edge_entry_fint=0.1, edge_exit_fint=0.2)

mm.edge_entry_active = 1
mm.edge_exit_active = 1
mm.model = 'rot-kick-rot'
mm.num_multipole_kicks = 10
mm.edge_entry_model = 'linear'
mm.edge_exit_model = 'linear'

p_test = p0.copy()
p_ref = p0.copy()

mm.track(p_test)
bb.track(p_ref)

xo.assert_allclose(p_test.x, p_ref.x, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.y, p_ref.y, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.zeta, p_ref.zeta, atol=1e-13, rtol=0)
xo.assert_allclose(p_test.px, p_ref.px, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.py, p_ref.py, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.delta, p_ref.delta, atol=1e-15, rtol=0)

# Test backtracking
line = xt.Line(elements=[mm])
line.track(p_test, backtrack=True)
xo.assert_allclose(p_test.s, 0.0, atol=1e-7, rtol=0)
xo.assert_allclose(p_test.x, p0.x, atol=5e-14, rtol=0)
xo.assert_allclose(p_test.y, p0.y, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.zeta, p0.zeta, atol=1e-14, rtol=0)
xo.assert_allclose(p_test.px, p0.px, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.py, p0.py, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.delta, p0.delta, atol=1e-15, rtol=0)

# ===============================================================================

# Test entry non-linear edge alone
bb = xt.Bend(h=0.1, k0=0.11, length=0,
             edge_entry_angle=0.02, edge_exit_angle=0.03,
             edge_entry_hgap=0.04, edge_exit_hgap=0.05,
             edge_entry_fint=0.1, edge_exit_fint=0.2)

bb.edge_entry_active = 1
bb.edge_exit_active = 0
bb.model = 'rot-kick-rot'
bb.num_multipole_kicks = 10
bb.edge_entry_model = 'full'

mm = Magnet(h=0.1, k0=0.11, length=0,
                edge_entry_angle=0.02, edge_exit_angle=0.03,
                edge_entry_hgap=0.04, edge_exit_hgap=0.05,
                edge_entry_fint=0.1, edge_exit_fint=0.2)

mm.edge_entry_active = 1
mm.edge_exit_active = 0
mm.model = 'rot-kick-rot'
mm.num_multipole_kicks = 10
mm.edge_entry_model = 'full'

p_test = p0.copy()
p_ref = p0.copy()

mm.track(p_test)
bb.track(p_ref)

xo.assert_allclose(p_test.x, p_ref.x, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.y, p_ref.y, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.zeta, p_ref.zeta, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.px, p_ref.px, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.py, p_ref.py, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.delta, p_ref.delta, atol=1e-15, rtol=0)

# ===============================================================================
# Test exit non-linear edge alone
bb = xt.Bend(h=0.1, k0=0.11, length=0,
             edge_entry_angle=0.02, edge_exit_angle=0.03,
             edge_entry_hgap=0.04, edge_exit_hgap=0.05,
             edge_entry_fint=0.1, edge_exit_fint=0.2)

bb.edge_entry_active = 0
bb.edge_exit_active = 1
bb.model = 'rot-kick-rot'
bb.num_multipole_kicks = 10
bb.edge_exit_model = 'full'

mm = Magnet(h=0.1, k0=0.11, length=0,
                edge_entry_angle=0.02, edge_exit_angle=0.03,
                edge_entry_hgap=0.04, edge_exit_hgap=0.05,
                edge_entry_fint=0.1, edge_exit_fint=0.2)

mm.edge_entry_active = 0
mm.edge_exit_active = 1
mm.model = 'rot-kick-rot'
mm.num_multipole_kicks = 10
mm.edge_exit_model = 'full'

p_test = p0.copy()
p_ref = p0.copy()

mm.track(p_test)
bb.track(p_ref)

xo.assert_allclose(p_test.x, p_ref.x, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.y, p_ref.y, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.zeta, p_ref.zeta, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.px, p_ref.px, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.py, p_ref.py, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.delta, p_ref.delta, atol=1e-15, rtol=0)

# ==============================================================================
# Testa a full bend with non-linear edges

bb = xt.Bend(h=0.1, k0=0.11, length=10,
                edge_entry_angle=0.02, edge_exit_angle=0.03,
                edge_entry_hgap=0.04, edge_exit_hgap=0.05,
                edge_entry_fint=0.1, edge_exit_fint=0.2)

bb.edge_entry_active = 1
bb.edge_exit_active = 1
bb.model = 'rot-kick-rot'
bb.num_multipole_kicks = 10
bb.edge_entry_model = 'full'
bb.edge_exit_model = 'full'

mm = Magnet(h=0.1, k0=0.11, length=10,
                edge_entry_angle=0.02, edge_exit_angle=0.03,
                edge_entry_hgap=0.04, edge_exit_hgap=0.05,
                edge_entry_fint=0.1, edge_exit_fint=0.2)

mm.edge_entry_active = 1
mm.edge_exit_active = 1
mm.model = 'rot-kick-rot'
mm.num_multipole_kicks = 10
mm.edge_entry_model = 'full'
mm.edge_exit_model = 'full'

p_test = p0.copy()
p_ref = p0.copy()

mm.track(p_test)
bb.track(p_ref)

xo.assert_allclose(p_test.x, p_ref.x, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.y, p_ref.y, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.zeta, p_ref.zeta, atol=3e-15, rtol=0)
xo.assert_allclose(p_test.px, p_ref.px, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.py, p_ref.py, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.delta, p_ref.delta, atol=1e-15, rtol=0)


# ==============================================================================
# Test a quadrupole with non-linear fringes

qq = xt.Quadrupole(k1=0.11, length=3)
qq.edge_entry_active = 1
qq.edge_exit_active = 1

mm = Magnet(k0=0, k1=0.11, length=3,
            edge_entry_model='full', edge_exit_model='full',
            edge_entry_fint=0.1, edge_exit_fint=0.2, # should be ignored
            edge_entry_hgap=0.04, edge_exit_hgap=0.05) # should be ignored
mm.edge_entry_active = 1
mm.edge_exit_active = 1
mm.model='mat-kick-mat'

p_test = p0.copy()
p_ref = p0.copy()

mm.track(p_test)
qq.track(p_ref)

xo.assert_allclose(p_test.x, p_ref.x, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.y, p_ref.y, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.zeta, p_ref.zeta, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.px, p_ref.px, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.py, p_ref.py, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.delta, p_ref.delta, atol=1e-15, rtol=0)

# ==============================================================================
# Test a sextupole with non-linear fringes

ss = xt.Sextupole(k2=0.11 + 0.03/3., k2s=0.12 -0.04/3, length=3)
ss.edge_entry_active = 1
ss.edge_exit_active = 1

mm = Magnet(k0=0, k2=0.11, k2s=0.12, length=3,
            knl=[0, 0, 0.03], ksl=[0, 0, -0.04],
            edge_entry_model='full', edge_exit_model='full',
            edge_entry_fint=0.1, edge_exit_fint=0.2, # should be ignored
            edge_entry_hgap=0.04, edge_exit_hgap=0.05) # should be ignored
mm.edge_entry_active = 1
mm.edge_exit_active = 1
mm.model = 'drift-kick-drift-expanded'
mm.num_multipole_kicks = 1
mm.integrator = 'uniform'

p_test = p0.copy()
p_ref = p0.copy()

mm.track(p_test)
ss.track(p_ref)

xo.assert_allclose(p_test.x, p_ref.x, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.y, p_ref.y, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.zeta, p_ref.zeta, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.px, p_ref.px, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.py, p_ref.py, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.delta, p_ref.delta, atol=1e-15, rtol=0)

# ==============================================================================
# Test a octupole with non-linear fringes

oo = xt.Octupole(k3=0.3 + 0.02/3., k3s=0.4 - 0.07/3, length=3)
oo.edge_entry_active = 1
oo.edge_exit_active = 1

mm = Magnet(k0=0, k3=0.3, k3s=0.4, length=3,
            knl=[0, 0, 0, 0.02], ksl=[0, 0, 0, -0.07],
            edge_entry_model='full', edge_exit_model='full',
            edge_entry_fint=0.1, edge_exit_fint=0.2, # should be ignored
            edge_entry_hgap=0.04, edge_exit_hgap=0.05) # should be ignored
mm.edge_entry_active = 1
mm.edge_exit_active = 1
mm.model = 'drift-kick-drift-expanded'
mm.num_multipole_kicks = 1
mm.integrator = 'uniform'

p_test = p0.copy()
p_ref = p0.copy()

mm.track(p_test)
oo.track(p_ref)

xo.assert_allclose(p_test.x, p_ref.x, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.y, p_ref.y, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.zeta, p_ref.zeta, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.px, p_ref.px, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.py, p_ref.py, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.delta, p_ref.delta, atol=1e-15, rtol=0)

line = xt.Line(elements=[mm])
line.track(p_test, backtrack=True)
assert np.all(p_test.state == -32)