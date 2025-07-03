import xtrack as xt
import numpy as np

# TODO:
# - Handle edges (ax, ay)
# - Handle radiation
# - Handle spin
# - Add field from dks_ds
# - Make sure radiation is in the direction of the KINETIC momentum (add test)
# - Slicing (remember to worry about edges)

length = 3.
ks = 2.

sol = xt.UniformSolenoid(length=length, ks=ks)
# sol.edge_exit_active = False
# sol.edge_entry_active = False
ref_sol = xt.Solenoid(length=length, ks=ks) # Old solenoid

p0 = xt.Particles(p0c=1e9, x=1e-3, y=2e-3)

p = p0.copy()
p_ref = p0.copy()

sol.track(p)
ref_sol.track(p_ref)

# Try slicing
line = xt.Line(elements=[sol])
line.cut_at_s(np.linspace(0, length, 10))