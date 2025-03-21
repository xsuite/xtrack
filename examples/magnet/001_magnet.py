import xtrack as xt
import xobjects as xo

from xtrack.beam_elements.magnets import Magnet

mm = Magnet(length=2.0, k0=0.0, k1=0.0, h=0.0)
mm.integrator = 'teapot'

p0 = xt.Particles(kinetic_energy0=50e6,
                  x=[1e-3, -1e-3], y=2e-3, zeta=1e-2, px=10e-3, py=20e-3, delta=1e-2)

mm.compile_kernels()

# Expanded drift
mm.model = 'drift-kick-drift-expanded'
p_test = p0.copy()
p_ref = p0.copy()

eref = xt.Drift(length=2.0)
mm.track(p_test)
eref.track(p_ref)

xo.assert_allclose(p_test.s, 2.0, atol=0, rtol=1e-7)
xo.assert_allclose(p_ref.s, 2.0, atol=0, rtol=1e-7)
xo.assert_allclose(p_test.x, p_ref.x, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.y, p_ref.y, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.zeta, p_ref.zeta, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.px, p_ref.px, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.py, p_ref.py, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.delta, p_ref.delta, atol=1e-15, rtol=0)

# Exact drift
mm.model = 'drift-kick-drift-exact'
p_test = p0.copy()
p_ref = p0.copy()

eref = xt.Solenoid(length=2.0) # Solenoid is exact drift when off
mm.track(p_test)
eref.track(p_ref)

xo.assert_allclose(p_test.s, 2.0, atol=0, rtol=1e-7)
xo.assert_allclose(p_ref.s, 2.0, atol=0, rtol=1e-7)
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

# Sextupole
mm.model = 'drift-kick-drift-expanded'
mm.k2 = 3.
mm.k2s = 5.

p_test = p0.copy()
p_ref = p0.copy()

eref = xt.Sextupole(length=2.0, k2=3., k2s=5.)
mm.track(p_test)
eref.track(p_ref)

xo.assert_allclose(p_test.s, 2.0, atol=0, rtol=1e-7)
xo.assert_allclose(p_ref.s, 2.0, atol=0, rtol=1e-7)
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

# Sextupole more kicks
mm.num_multipole_kicks = 5
eref.num_multipole_kicks = 5
mm.edge_entry_active = False
mm.edge_exit_active = False
mm.integrator = 'teapot'
eref.integrator = 'teapot'

p_test = p0.copy()
p_ref = p0.copy()

mm.track(p_test)
eref.track(p_ref)

xo.assert_allclose(p_test.s, 2.0, atol=0, rtol=1e-7)
xo.assert_allclose(p_ref.s, 2.0, atol=0, rtol=1e-7)
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

# Sextupole done with knl
mm.k2 = 0.
mm.knl = [0., 0., 3.*2., 0., 0., 0.]
mm.num_multipole_kicks = 5
eref.num_multipole_kicks = 5

p_test = p0.copy()
p_ref = p0.copy()

mm.track(p_test)
eref.track(p_ref)

xo.assert_allclose(p_test.s, 2.0, atol=0, rtol=1e-7)
xo.assert_allclose(p_ref.s, 2.0, atol=0, rtol=1e-7)
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


# Add skew sextupole component in ksl
mm.ksl[2] = -2.
eref.ksl[2] = -2.

p_test = p0.copy()
p_ref = p0.copy()

mm.track(p_test)
eref.track(p_ref)

xo.assert_allclose(p_test.s, 2.0, atol=0, rtol=1e-7)
xo.assert_allclose(p_ref.s, 2.0, atol=0, rtol=1e-7)
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

# Quadrupole
mm.model = 'mat-kick-mat'
mm.k0 = 0
mm.k1 = 3.
mm.k2 = 0.
mm.k2s = 0.
mm.knl = 0.
mm.ksl = 0.
mm.num_multipole_kicks = 1

eref = xt.Quadrupole(length=2.0, k1=3.)
eref.num_multipole_kicks = 1

p_test = p0.copy()
p_ref = p0.copy()

mm.track(p_test)
eref.track(p_ref)

xo.assert_allclose(p_test.s, 2.0, atol=0, rtol=1e-7)
xo.assert_allclose(p_ref.s, 2.0, atol=0, rtol=1e-7)
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

# Bend
mm.model = 'rot-kick-rot'
mm.integrator = 'yoshida4'
mm.num_multipole_kicks = 15
mm.length=2.0
mm.h = 0.05
mm.k1 = -0.3

eref = xt.Bend(length=2.0, k1=-0.3, h=0.05)
eref.num_multipole_kicks = 15

p_test = p0.copy()
p_ref = p0.copy()

mm.track(p_test)
eref.track(p_ref)

xo.assert_allclose(p_test.s, 2.0, atol=0, rtol=1e-7)
xo.assert_allclose(p_ref.s, 2.0, atol=0, rtol=1e-7)
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

# Bend auto no kicks
mm.model = 'bend-kick-bend'
mm.integrator = 'yoshida4'
mm.num_multipole_kicks = 0
mm.h = 0.05
mm.k1 = 0

eref = xt.Bend(length=2.0, h=0.05)
eref.num_multipole_kicks = 0

p_test = p0.copy()
p_ref = p0.copy()

mm.track(p_test)
eref.track(p_ref)

xo.assert_allclose(p_test.s, 2.0, atol=0, rtol=1e-7)
xo.assert_allclose(p_ref.s, 2.0, atol=0, rtol=1e-7)
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

# Bend auto quad kick
mm.model = 'bend-kick-bend'
mm.integrator = 'yoshida4'
mm.num_multipole_kicks = 1
mm.h = 0.05
mm.k1 = 0.3

eref = xt.Bend(length=2.0, h=0.05, k1=0.3)
eref.num_multipole_kicks = 1
eref.edge_entry_active = False
eref.edge_exit_active = False
eref.model = 'bend-kick-bend'

p_test = p0.copy()
p_ref = p0.copy()

mm.track(p_test)
eref.track(p_ref)

xo.assert_allclose(p_test.s, 2.0, atol=0, rtol=1e-7)
xo.assert_allclose(p_ref.s, 2.0, atol=0, rtol=1e-7)
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

# Bend dip quad kick
mm.integrator = 'yoshida4'
mm.num_multipole_kicks = 10
mm.h = 0.1
mm.k1 = 0.3
mm.k0 = 0.2

eref = xt.Bend(length=2.0, h=0.1, k1=0.3, k0=0.2)
eref.num_multipole_kicks = 10
eref.edge_entry_active = False
eref.edge_exit_active = False

for model in ['bend-kick-bend', 'rot-kick-rot', 'expanded']:
    mm.model = model
    eref.model = model

    p_test = p0.copy()
    p_ref = p0.copy()

    mm.track(p_test)
    eref.track(p_ref)

    xo.assert_allclose(p_test.s, 2.0, atol=0, rtol=1e-7)
    xo.assert_allclose(p_ref.s, 2.0, atol=0, rtol=1e-7)
    xo.assert_allclose(p_test.x, p_ref.x, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test.y, p_ref.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test.zeta, p_ref.zeta, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test.px, p_ref.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test.py, p_ref.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test.delta, p_ref.delta, atol=1e-15, rtol=0)

    # Test backtracking
    line = xt.Line(elements=[mm])
    line.track(p_test, backtrack=True)
    xo.assert_allclose(p_test.s, 0.0, rtol=0, atol=1e-7)
    xo.assert_allclose(p_test.x, p0.x, atol=5e-14, rtol=0)
    xo.assert_allclose(p_test.y, p0.y, atol=5e-14, rtol=0)
    xo.assert_allclose(p_test.zeta, p0.zeta, atol=5e-14, rtol=0)
    xo.assert_allclose(p_test.px, p0.px, atol=5e-14, rtol=0)
    xo.assert_allclose(p_test.py, p0.py, atol=5e-14, rtol=0)
    xo.assert_allclose(p_test.delta, p0.delta, atol=1e-15, rtol=0)

# Bend dip quad kick with multipoles
mm.integrator = 'yoshida4'
mm.num_multipole_kicks = 10
mm.h = 0.1
mm.k1 = 0.3
mm.k0 = 0.2

mm.k2 = 0.1
mm.k3 = 0.15

mm.k0s = 0.02
mm.k1s = 0.03
mm.k2s = 0.01
mm.k3s = 0.02

mm.knl = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
mm.ksl = [0.6, 0.5, 0.4, 0.15, 0.2, 0.1]

eref = xt.Bend(length=2.0, h=0.1, k1=0.3, k0=0.2)
eref.num_multipole_kicks = 10
eref.edge_entry_active = False
eref.edge_exit_active = False
eref.knl = [0.1, 0.2, 0.3 + 0.1*2, 0.4 + 0.15*2, 0.5, 0.6]
eref.ksl = [0.6 + 0.02*2, 0.5 + 0.03*2, 0.4 + 0.01*2, 0.15 + 0.02*2, 0.2, 0.1]

for model in ['bend-kick-bend', 'rot-kick-rot']:
    mm.model = model
    eref.model = model

    p_test = p0.copy()
    p_ref = p0.copy()

    mm.track(p_test)
    eref.track(p_ref)

    xo.assert_allclose(p_test.s, 2.0, atol=0, rtol=1e-7)
    xo.assert_allclose(p_ref.s, 2.0, atol=0, rtol=1e-7)
    xo.assert_allclose(p_test.x, p_ref.x, atol=1e-14, rtol=0)
    xo.assert_allclose(p_test.y, p_ref.y, atol=1e-14, rtol=0)
    xo.assert_allclose(p_test.zeta, p_ref.zeta, atol=1e-14, rtol=0)
    xo.assert_allclose(p_test.px, p_ref.px, atol=1e-14, rtol=0)
    xo.assert_allclose(p_test.py, p_ref.py, atol=1e-14, rtol=0)
    xo.assert_allclose(p_test.delta, p_ref.delta, atol=1e-14, rtol=0)

    # Test backtracking
    line = xt.Line(elements=[mm])
    line.track(p_test, backtrack=True)
    xo.assert_allclose(p_test.s, 0.0, rtol=0, atol=1e-7)
    xo.assert_allclose(p_test.x, p0.x, atol=5e-14, rtol=0)
    xo.assert_allclose(p_test.y, p0.y, atol=5e-14, rtol=0)
    xo.assert_allclose(p_test.zeta, p0.zeta, atol=5e-14, rtol=0)
    xo.assert_allclose(p_test.px, p0.px, atol=5e-14, rtol=0)
    xo.assert_allclose(p_test.py, p0.py, atol=5e-14, rtol=0)
    xo.assert_allclose(p_test.delta, p0.delta, atol=1e-15, rtol=0)

# Check uniform integrator
mm1 = Magnet(h=0.1, k1=0.3, k0=0.2, length=2.0)
mm2 = mm1.copy()

mm1.edge_entry_active = False
mm1.edge_exit_active = False
mm2.edge_entry_active = False
mm2.edge_exit_active = False

mm1.integrator = 'uniform'
mm2.integrator = 'teapot'
mm1.num_multipole_kicks = 1
mm2.num_multipole_kicks = 1

p_test = p0.copy()
p_ref = p0.copy()

mm1.track(p_test)
mm2.track(p_ref)

xo.assert_allclose(p_test.s, 2.0, atol=0, rtol=1e-7)
xo.assert_allclose(p_ref.s, 2.0, atol=0, rtol=1e-7)
xo.assert_allclose(p_test.x, p_ref.x, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.y, p_ref.y, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.zeta, p_ref.zeta, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.px, p_ref.px, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.py, p_ref.py, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.delta, p_ref.delta, atol=1e-15, rtol=0)

# Test backtracking
line = xt.Line(elements=[mm1])
line.track(p_test, backtrack=True)
xo.assert_allclose(p_test.s, 0.0, atol=1e-7, rtol=0)
xo.assert_allclose(p_test.x, p0.x, atol=5e-14, rtol=0)
xo.assert_allclose(p_test.y, p0.y, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.zeta, p0.zeta, atol=1e-14, rtol=0)
xo.assert_allclose(p_test.px, p0.px, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.py, p0.py, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.delta, p0.delta, atol=1e-15, rtol=0)

# more kicks (needs loser thresholds)
mm1.num_multipole_kicks = 10
mm2.num_multipole_kicks = 10

p_test = p0.copy()
p_ref = p0.copy()

mm1.track(p_test)
mm2.track(p_ref)

xo.assert_allclose(p_test.s, 2.0, atol=0, rtol=1e-7)
xo.assert_allclose(p_ref.s, 2.0, atol=0, rtol=1e-7)
xo.assert_allclose(p_test.x, p_ref.x, atol=0, rtol=5e-3)
xo.assert_allclose(p_test.y, p_ref.y, atol=0, rtol=5e-3)
xo.assert_allclose(p_test.zeta, p_ref.zeta, atol=0, rtol=1e-2)
xo.assert_allclose(p_test.px, p_ref.px, atol=0, rtol=5e-3)
xo.assert_allclose(p_test.py, p_ref.py, atol=0, rtol=5e-3)
xo.assert_allclose(p_test.delta, p_ref.delta, atol=0, rtol=5e-3)

# Test backtracking
line = xt.Line(elements=[mm1])
line.track(p_test, backtrack=True)
xo.assert_allclose(p_test.s, 0.0, atol=1e-7, rtol=0)
xo.assert_allclose(p_test.x, p0.x, atol=5e-14, rtol=0)
xo.assert_allclose(p_test.y, p0.y, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.zeta, p0.zeta, atol=1e-14, rtol=0)
xo.assert_allclose(p_test.px, p0.px, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.py, p0.py, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.delta, p0.delta, atol=1e-15, rtol=0)

