import xtrack as xt
import numpy as np

from scipy.constants import c as clight
from spin import estimate_magnetic_field

p0c = 700e6

By_T = 0.023349486663870645
Bx_T = 0.01
length = 0.02
hx = 0.03
hy = 0.0

p_ref = xt.Particles(p0c=p0c, delta=0, mass0=xt.ELECTRON_MASS_EV)
brho_ref = p_ref.p0c[0] / clight / p_ref.q0

k0 = By_T / brho_ref
k0s = Bx_T / brho_ref

bb = xt.Magnet(h=hx, k0=k0, k0s=k0s, length=length)
bb.edge_entry_active = False
bb.edge_exit_active = False

p = p_ref.copy()
p.delta = 0.01
p.px = 1e-2
p.py = 2e-3
p.x = 1e-2
p.y = 2e-2

p_before = p.copy()
bb.track(p)

Bx_meas, By_meas = estimate_magnetic_field(
    p_before=p_before, p_after=p, hx=hx, hy=hy, length=length
)

print('By_T   ', By_T)
print('By_meas', By_meas)
print('Bx_T   ', Bx_T)
print('Bx_meas', Bx_meas)

