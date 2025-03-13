import xtrack as xt
import xobjects as xo
from xtrack.beam_elements.magnets import Magnet

bb = xt.Bend(h=0.1, k0=0.11, length=2.0,
             edge_entry_angle=0.02, edge_exit_angle=0.03,
             edge_entry_hgap=0.04, edge_exit_hgap=0.05,
             edge_entry_fint=0.1, edge_exit_fint=0.2)

bb.edge_entry_active = 1
bb.edge_exit_active = False
bb.model = 'rot-kick-rot'
bb.num_multipole_kicks = 10


mm = Magnet(h=0.1, k0=0.11, length=2.0,
             edge_entry_angle=0.02, edge_exit_angle=0.03,
             edge_entry_hgap=0.04, edge_exit_hgap=0.05,
             edge_entry_fint=0.1, edge_exit_fint=0.2)
mm.edge_entry_active = 1
mm.edge_exit_active = False
mm.model = 'rot-kick-rot'
mm.num_multipole_kicks = 10
# , edge_entry_angle=0.02, edge_exit_angle=0.03,
#              edge_entry_hgap=0.04, edge_exit_hgap=0.05,
#              edge_entry_fint=0.1, edge_exit_fint=0.2)

p0 = xt.Particles(kinetic_energy0=50e6,
                  x=1e-3, y=2e-3, zeta=1e-2, px=10e-3, py=20e-3, delta=1e-2)

p_test = p0.copy()
p_ref = p0.copy()

mm.track(p_test)
bb.track(p_ref)

xo.assert_allclose(p_test.s, 2.0, atol=0, rtol=1e-7)
xo.assert_allclose(p_ref.s, 2.0, atol=0, rtol=1e-7)
xo.assert_allclose(p_test.x, p_ref.x, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.y, p_ref.y, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.zeta, p_ref.zeta, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.px, p_ref.px, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.py, p_ref.py, atol=1e-15, rtol=0)
xo.assert_allclose(p_test.delta, p_ref.delta, atol=1e-15, rtol=0)