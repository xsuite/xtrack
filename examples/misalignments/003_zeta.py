import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

# Element parameters
length = 20
angle = 0.3  # rad
k0 = 0
k1 = 0

# Misalignment parameters
dx = -15
dy = 7
ds = 23
theta = -0.1  # rad
phi = 0.11  # rad
psi = 0.7  # rad
f = 0.0  # fraction of the element length for the misalignment

# element = xt.Drift(length=length)
element = xt.Bend(k0=0, angle=angle, length=length)

line = xt.Line(elements=[
    xt.Misalignment(
        dx=dx, dy=dy, ds=ds,
        theta=theta, phi=phi, psi=psi,
        length=length, angle=angle,
        anchor=f, is_exit=False,
    ),
    element,
    xt.Misalignment(
        dx=dx, dy=dy, ds=ds,
        theta=theta, phi=phi, psi=psi,
        length=length, angle=angle,
        anchor=f, is_exit=True,
    ),
])
line.reset_s_at_end_turn = False
line.config.XTRACK_GLOBAL_XY_LIMIT = 100

p0 = xt.Particles(
    x=[0, 0.2, -0.4, -0.6, 0.8],
    y=[0, 0.2, 0.4, -0.6, -0.8],
    px=[0, -0.01, -0.01, 0.01, 0.01],
    py=[0, -0.01, 0.01, -0.01, 0.01],
)

p_ref = p0.copy()
element.track(p_ref)

p_mis = p0.copy()
line.track(p_mis, turn_by_turn_monitor='ONE_TURN_EBE')
all_mis = line.record_last_track

zetas_ref = np.array([p0.zeta, p_ref.zeta])
zetas_mis = all_mis.zeta.T
diff_zetas_ref = np.diff(zetas_ref, axis=0)
diff_zetas_mis = np.diff(zetas_mis, axis=0)

ss_ref = np.array([p0.s, p_ref.s])
ss_mis = all_mis.s.T
diff_ss_ref = np.diff(ss_ref, axis=0)
diff_ss_mis = np.diff(ss_mis, axis=0)

# Two plots on top of each other
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.set_title(r'$\zeta$ at element')
ax1.plot(zetas_mis, c='orange')
ax1.plot(np.arange(3) + 0.5, diff_zetas_mis, '.', c='orange')
ax1.plot([0, 3], zetas_ref, c='blue')
ax1.plot([1.5], diff_zetas_ref, '.', c='blue')

ax2.set_title('$s$ at element')
ax2.plot(ss_mis, c='orange')
ax2.plot(np.arange(3) + 0.5, diff_ss_mis, '.', c='orange')
ax2.plot([0, 3], ss_ref, c='blue')
ax2.plot([1.5], diff_ss_ref, '.', c='blue')

plt.show()
