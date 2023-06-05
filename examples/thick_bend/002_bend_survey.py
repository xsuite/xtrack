import matplotlib.pyplot as plt
import xtrack as xt
import xpart as xp
import math
import numpy as np

from matplotlib.widgets import Slider

angle = 2 * math.pi
L = 27_000
rho = L / angle
h = 1 / rho
step = 500
k = 1 / rho

p0_ref = xp.Particles(p0c=7e12, mass0=xp.PROTON_MASS_EV, x=0.0, px=0.0, delta=0.0)
p0 = xp.Particles(p0c=7e12, mass0=xp.PROTON_MASS_EV, x=0.0, px=0.0, delta=0.0)
el_cfd = xt.CombinedFunctionMagnet(h=h, length=L, k0=k, order=1, num_multipole_kicks=1)
line_cfd = xt.Line(elements=[el_cfd])
line_cfd.reset_s_at_end_turn = False
line_cfd.build_tracker()

el_bend = xt.TrueBend(h=h, length=L, k0=k, order=1, num_multipole_kicks=1)
line_bend = xt.Line(elements=[el_bend])
line_bend.reset_s_at_end_turn = False
line_bend.build_tracker()

fig, ax = plt.subplots()

def update(x, px):
    if x is not None:
        p0.x = x

    if px is not None:
        p0.px = px

    s_array = np.array(list(range(0, L, step)) + [L])

    X0_array = np.zeros_like(s_array)
    Z0_array = np.zeros_like(s_array)

    X_array = np.zeros_like(s_array)
    Z_array = np.zeros_like(s_array)

    X_array_cfd = np.zeros_like(s_array)
    Z_array_cfd = np.zeros_like(s_array)

    for ii, s in enumerate(s_array):
        p_ref = p0_ref.copy()
        p_cfd = p0.copy()
        p_bend = p0.copy()

        el_cfd.length = s
        # el_cfd.knl = np.array([3e-4, 4e-4, 0, 0, 0]) * s / L
        line_cfd.track(p_cfd)

        el_bend.length = s
        el_bend.knl = np.array([3e-4, 4e-4, 0, 0, 0]) * s / L
        line_bend.track(p_bend)
        line_bend.track(p_ref)

        theta = s / rho

        X0 = -rho * (1 - np.cos(theta))
        Z0 = rho * np.sin(theta)

        ex_X = np.cos(theta)
        ex_Z = np.sin(theta)

        X0_array[ii] = X0
        Z0_array[ii] = Z0

        X_array[ii] = X0 + p_bend.x[0] * ex_X
        Z_array[ii] = Z0 + p_bend.x[0] * ex_Z

        X_array_cfd[ii] = X0 + p_cfd.x[0] * ex_X
        Z_array_cfd[ii] = Z0 + p_cfd.x[0] * ex_Z

    ax.cla()
    ax.plot(X0_array, Z0_array, label='Reference')
    ax.plot(X_array, Z_array, label='True Bend')
    ax.plot(X_array_cfd, Z_array_cfd, label='CFD Bend')
    ax.legend()
    ax.set_aspect(1)


update(0, 0)

ax_slide_x = fig.add_axes([0.1, 0.05, 0.8, 0.03])
ax_slide_px = fig.add_axes([0.1, 0.0, 0.8, 0.03])

slide_px = Slider(
    ax=ax_slide_px,
    label='p_x',
    valmin=-1.0,
    valmax=1.0,
    valinit=0.0,
)
slide_x = Slider(
    ax=ax_slide_x,
    label='x',
    valmin=-rho,
    valmax=rho,
    valinit=0.0,
)
slide_px.on_changed(lambda px: update(None, px))
slide_x.on_changed(lambda x: update(x, None))

plt.show()
