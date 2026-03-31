"""
Minimal introduction to the `SplineBoris` API.

We build a single `SplineBoris` element with:

- a **longitudinal** field `Bs(s)` (here zero),
- a **normal dipole** component `By0(s)`,
- a **normal quadrupole** component `By1(s) · x`,

and show:

- how to pass multiple Hermite splines via `by=(..., ...)`,
- basic tracking through the element,
- simple field plots using `SplineBoris.get_field(...)`.
"""

import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt



# ---------------------------------------------------------------
# SplineBoris field definition
# ---------------------------------------------------------------

s_start = 0.0
s_end = 1.0
length = s_end - s_start
n_steps = 100

# Constant longitudinal field (zero in this example)
Bs = xt.Spline4(
    val_start=0.0,
    der_start=0.0,
    val_end=0.0,
    der_end=0.0,
    integral=0.0,
)

# Slightly asymmetric normal dipole field By0(s)
B0 = 0.1  # [T]
By0 = xt.Spline4(
    val_start=0.7 * B0,
    der_start=0.0,
    val_end=1.3 * B0,
    der_end=0.0,
    integral=B0,
)

# Constant normal quadrupole gradient By1(s) = G, giving By(x) = B0 + G * x
G = 100.0  # [T/m]
By1 = xt.Spline4(
    val_start=G,
    der_start=0.0,
    val_end=G,
    der_end=0.0,
    integral=G,
)

# We do not use any skew components in this example (bx defaults to zero).
sb = xt.SplineBoris(
    bs=Bs,
    # by takes a Spline4 or a tuple/list of Spline4/None:
    #   index 0 -> dipole, index 1 -> quadrupole, index 2 -> sextupole, ...
    by=(By0, By1),
    s_start=s_start,
    length=length,
    n_steps=n_steps,
)



# ---------------------------------------------------------------
# Basic tracking example
# ---------------------------------------------------------------

line = xt.Line(elements=[sb])
line.particle_ref = xt.Particles(
    mass0=xt.ELECTRON_MASS_EV,
    q0=1.0,
    energy0=1e9,
)

part = line.particle_ref.copy()
part.x = 1e-3  # 1 mm horizontal offset
part.px = 1e-3

line.track(part)

print("Final coordinates after SplineBoris element:")
print(f"  x  = {part.x[0]:+.6e} m")
print(f"  y  = {part.y[0]:+.6e} m")
print(f"  px = {part.kin_px[0]:+.6e}")
print(f"  py = {part.kin_py[0]:+.6e}")



# ---------------------------------------------------------------
# Twiss
# ---------------------------------------------------------------
twiss = line.twiss(betx=1, bety=1)
twiss.plot('x', 'y')



# ---------------------------------------------------------------
# Field evaluation
# ---------------------------------------------------------------

"""Plot B(s) along the element and By(x) at its center using `get_field`."""

# 1) Bx, By, Bs as a function of s at (x, y) = (0, 0)
s_grid = np.linspace(sb.s_start, sb.s_end, 200)
Bx_s = []
By_s = []
Bs_s = []
for ss in s_grid:
    Bx_val, By_val, Bs_val = sb.get_field(0.0, 0.0, ss)
    Bx_s.append(Bx_val)
    By_s.append(By_val)
    Bs_s.append(Bs_val)

# 2) By(x) at the center of the element (shows dipole + quadrupole behaviour)
s_mid = 0.5 * (sb.s_start + sb.s_end)
x_grid = np.linspace(-2e-3, 2e-3, 200)  # +/- 2 mm
By_x = []
for xx in x_grid:
    _, By_val, _ = sb.get_field(xx, 0.0, s_mid)
    By_x.append(By_val)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# B(s) at x=y=0
ax = axes[0]
ax.plot(s_grid, Bx_s, label="Bx(s)")
ax.plot(s_grid, By_s, label="By(s)")
ax.plot(s_grid, Bs_s, label="Bs(s)")
ax.set_xlabel("s [m]")
ax.set_ylabel("B [T]")
ax.set_title("Field along s at x=y=0")
ax.grid(True, alpha=0.3)
ax.legend()

# By(x) at element center
ax = axes[1]
ax.plot(x_grid * 1e3, By_x)
ax.set_xlabel("x [mm]")
ax.set_ylabel("By [T]")
ax.set_title(f"By(x) at s = {s_mid:.3f} m")
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()
