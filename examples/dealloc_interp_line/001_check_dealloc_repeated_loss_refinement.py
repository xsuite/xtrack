import xtrack as xt
import xpart as xp
import xobjects as xo
import numpy as np
import logging

def _occupied_size(buffer):
    return buffer.capacity - buffer.get_free()

n_part=10000
shift_x = 0.3e-2
shift_y = 0.5e-2
sandwitch_aper = True

ctx = xo.context_default
buf = ctx.new_buffer()

logger = logging.getLogger('xtrack')
logger.setLevel(logging.DEBUG)

# Define aper_0
aper_0 = xt.LimitEllipse(_buffer=buf, a=2e-2, b=2e-2)
shift_aper_0 = (shift_x, shift_y)
rot_deg_aper_0 = 10.

# Define aper_1
aper_1 = xt.LimitEllipse(_buffer=buf, a=1e-2, b=1e-2)
shift_aper_1 = (shift_x, shift_y)
rot_deg_aper_1 = 10.

# aper_0_sandwitch
if sandwitch_aper:
    line_aper_0 = xt.Line(
        elements=[xt.XYShift(_buffer=buf, dx=shift_aper_0[0], dy=shift_aper_0[1]),
                xt.SRotation(_buffer=buf, angle=rot_deg_aper_0),
                aper_0,
                xt.Multipole(_buffer=buf, knl=[0.00]),
                xt.SRotation(_buffer=buf, angle=-rot_deg_aper_0),
                xt.XYShift(_buffer=buf, dx=-shift_aper_0[0], dy=-shift_aper_0[1])])
    line_aper_0.build_tracker(_buffer=buf)
    # aper_1_sandwitch
    line_aper_1 = xt.Line(
        elements=[xt.XYShift(_buffer=buf, dx=shift_aper_1[0], dy=shift_aper_1[1]),
                xt.SRotation(_buffer=buf, angle=rot_deg_aper_1),
                aper_1,
                xt.Multipole(_buffer=buf, knl=[0.00]),
                xt.SRotation(_buffer=buf, angle=-rot_deg_aper_1),
                xt.XYShift(_buffer=buf, dx=-shift_aper_1[0], dy=-shift_aper_1[1])])
else:
    aper_0.shift_x = shift_aper_0[0]
    aper_0.shift_y = shift_aper_0[1]
    aper_0.rot_s_rad = np.deg2rad(rot_deg_aper_0)
    line_aper_0 = xt.Line(
        elements=[aper_0, xt.Multipole(_buffer=buf, knl=[0.0])])
    line_aper_0.build_tracker(_buffer=buf)
    aper_1.shift_x = shift_aper_1[0]
    aper_1.shift_y = shift_aper_1[1]
    aper_1.rot_s_rad = np.deg2rad(rot_deg_aper_1)
    line_aper_1 = xt.Line(
        elements=[aper_1, xt.Multipole(_buffer=buf, knl=[0.00])])

line_aper_1.build_tracker(_buffer=buf)

# Build example line
line=xt.Line(
    elements = ((xt.Drift(_buffer=buf, length=0.5),)
                + line_aper_0.elements
                + (xt.Drift(_buffer=buf, length=1),
                    xt.Multipole(_buffer=buf, knl=[0.]),
                    xt.Quadrupole(_buffer=buf, length=1),
                    xt.Cavity(_buffer=buf, voltage=3e6, frequency=400e6),
                    xt.ParticlesMonitor(_buffer=buf,
                        start_at_turn=0, stop_at_turn=10, num_particles=3),
                    xt.Drift(_buffer=buf, length=1.),
                    xt.Marker())
                + line_aper_1.elements))
line.build_tracker(_buffer=buf)

# Test on full line
r = np.linspace(0, 0.018, n_part)
theta = np.linspace(0, 8*np.pi, n_part)
particles0 = xp.Particles(_context=ctx,
        p0c=6500e9,
        x=r*np.cos(theta)+shift_x,
        y=r*np.sin(theta)+shift_y)



print('occupied size before refinement', _occupied_size(buf))

loss_loc_refinement = xt.LossLocationRefinement(line,
                        n_theta = 360,
                        r_max = 0.5, # m
                        dr = 50e-6,
                        ds = 0.1,
                        save_refine_lines=False,
                        allowed_backtrack_types=[
                            xt.Multipole,
                            xt.Cavity
                            ])
print('occupied size after init refinement', _occupied_size(buf))

n_repetitions = 20


import time
for i_iter in range(n_repetitions):

    particles = particles0.copy()

    line.track(particles)

    t0 = time.time()

    loss_loc_refinement.refine_loss_location(particles)
    print('occupied size after refinement', _occupied_size(buf))

    if i_iter==0:
        occupied_size_first_iter = _occupied_size(buf)
    else:
        assert _occupied_size(buf) == occupied_size_first_iter, \
            'Buffer size increased after refinement! Memory leak?'

    t1 = time.time()
    print(f'Took\t{(t1-t0)*1e3:.2f} ms')

# Automatic checks
mask_lost = particles.state == 0
r_calc = np.sqrt((particles.x-shift_x)**2 + (particles.y-shift_y)**2)
assert np.all(r_calc[~mask_lost]<1e-2)
assert np.all(r_calc[mask_lost]>1e-2)
i_aper_1 = line.elements.index(aper_1)
assert np.all(particles.at_element[mask_lost]==i_aper_1)
assert np.all(particles.at_element[~mask_lost]==0)
s0 = line.get_s_elements()[line.elements.index(aper_0)]
s1 = line.get_s_elements()[line.elements.index(aper_1)]
r0 = np.sqrt(aper_0.a_squ)
r1 = np.sqrt(aper_1.a_squ)
s_expected = s0 + (r_calc-r0)/(r1 - r0)*(s1 - s0)
# TODO This threshold is a bit large
xo.assert_allclose(particles.s[mask_lost], s_expected[mask_lost], atol=0.11)