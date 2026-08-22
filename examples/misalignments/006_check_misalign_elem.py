import xtrack as xt

# Misalignment parameters
dx = -15  # m
dy = 7  # m
ds = 20  # m
theta = -0.1  # rad
phi = 0.11  # rad
psi = 0.7  # rad
anchor = 8  # m

length = 10.
angle = 0
tilt=0.2

mis_entry = xt.Misalignment(
        dx=dx, dy=dy, ds=ds,
        theta=theta, phi=phi, psi=psi,
        length=length, angle=angle, tilt=tilt,
        anchor=anchor, is_exit=False,
    )

mis_exit = xt.Misalignment(
        dx=dx, dy=dy, ds=ds,
        theta=theta, phi=phi, psi=psi,
        length=length, angle=angle, tilt=tilt,
        anchor=anchor, is_exit=True,
    )
# drift = xt.Drift(length=length, model='exact')
drift = xt.DriftExact(length=length)

s = xt.Solenoid(length=length,
                shift_x=dx,
                shift_y=dy,
                shift_s=ds,
                rot_x_rad=phi,
                rot_y_rad=theta,
                rot_s_rad=tilt,
                rot_s_rad_no_frame=psi,
                rot_shift_anchor=anchor

)

# Initial particle coordinates
p0 = xt.Particles(
    x=1e-3,
    y=2e-3,
    px=1e-3,
    py=0,
)

p1 = p0.copy()
mis_entry.track(p1)
drift.track(p1)
mis_exit.track(p1)

p2 = p0.copy()
drift.track(p2)

