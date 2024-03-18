import xtrack as xt
import numpy as np

theta_tilt_deg = 20
theta_tilt_rad = theta_tilt_deg * np.pi / 180

l_solenoid = 3

l_beam = l_solenoid / np.cos(theta_tilt_rad)

tilt_entry = xt.YRotation(angle=-theta_tilt_deg)
tilt_exit = xt.YRotation(angle=+theta_tilt_deg)
shift_entry = xt.XYShift(dx=l_solenoid/2 * np.tan(theta_tilt_rad))
shift_exit = xt.XYShift(dx=l_solenoid/2 * np.tan(theta_tilt_rad))
solenoid = xt.Drift(length=l_solenoid)

line = xt.Line(
    elements=[xt.Marker(), tilt_entry, shift_entry, solenoid, shift_exit, tilt_exit, xt.Marker()],
    element_names=['start', 'tilt_entry', 'shift_entry', 'solenoid', 'shift_exit', 'tilt_exit', 'end']
)
line.config.XTRACK_USE_EXACT_DRIFTS = True
line.config.XTRACK_GLOBAL_XY_LIMIT = 1000.

line2 = xt.Line(
    elements=[xt.Marker(), xt.Drift(length=l_beam), xt.Marker()],
    element_names=['start', 'solenoid', 'end']
)
line2.config.XTRACK_USE_EXACT_DRIFTS = True
line2.config.XTRACK_GLOBAL_XY_LIMIT = 1000.
line2.build_tracker()



line.build_tracker()
line.tracker.skip_end_turn_actions = True


# particle on the solenoid axis
p0 = xt.Particles(p0c=6500e9, px=-np.sin(theta_tilt_rad),
                  x=l_beam/2 * np.tan(theta_tilt_rad),
                  delta=0.1, py=0.1)
p = p0.copy()
line.track(p)

p2 = p0.copy()
line2.track(p2)


pref = xt.Particles(p0c=6500e9)
line.particle_ref = pref
line2.particle_ref = pref
tw = line.twiss(betx=1, bety=1, x=p0.x[0], px=p0.px[0], start='start', end='end')
tw2 = line2.twiss(betx=1, bety=1, x=p0.x[0], px=p0.px[0], start='start', end='end')

pco = pref.copy()
R = line.compute_one_turn_matrix_finite_differences(particle_on_co=pco)['R_matrix']
