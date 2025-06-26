import xtrack as xt

# TODO:
# - Handle edges (ax, ay)
# - Handle radiation
# - Handle spin
# - Add field from dks_ds
# - Make sure radiation is in the direction of the KINETIC momentum (add test)
# - Slicing (remember to worry about edges)

length = 3.
ks = 2.

sol = xt.Slnd(length=length, ks=ks)
ref_sol = xt.Solenoid(length=length, ks=ks)

p0 = xt.Particles(p0c=1e9, x=1e-3, y=2e-3)

p = p0.copy()
p_ref = p0.copy()

sol.track(p)
ref_sol.track(p_ref)