import numpy as np
import xtrack as xt

tilt_machine_by_90_degrees = True
wiggler_on = False
vertical_orbit_distortion = True

line = xt.Line.from_json('fccee_h_thin.json')

line.build_tracker()

if wiggler_on:
    line.vars['on_wiggler_v'] = 0.4

if vertical_orbit_distortion:
    line['qf4.232..4'].ksl[0] = 1.5e-6

# Make sure there is no vertical bend nor skew element
for ee in line.elements:
    if isinstance(ee, xt.Multipole):
        assert np.all(ee.ksl[1:] == 0)

if tilt_machine_by_90_degrees:

    tw_before_tilt = line.twiss()

    # Bring the machine to the vertical plane
    for ee in line.elements:
        print(type(ee))
        if isinstance(ee, xt.Multipole):
            knl = ee.knl.copy()
            ksl = ee.ksl.copy()
            hxl = ee.hxl
            hyl = ee.hyl
            ee.hxl = hyl
            ee.hyl = hxl

            ee.knl[0] = ksl[0]
            ee.ksl[0] = knl[0]
            if len(knl) > 1:
                ee.knl[1] = -knl[1]
                ee.ksl[1] = 0
            if len(knl) > 2:
                ee.knl[2] = 0
                ee.ksl[2] = -knl[2]

        if isinstance(ee, xt.DipoleEdge):
            ee._r21, ee._r43 = ee._r43, ee._r21

    tw_after_tilt = line.twiss()

    assert np.isclose(tw_before_tilt.qx, tw_after_tilt.qy, rtol=0, atol=3e-6)
    assert np.isclose(tw_before_tilt.qy, tw_after_tilt.qx, rtol=0, atol=3e-6)

    prrrr

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Add checks on tilted optics!

tw_no_rad = line.twiss(method='4d')

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()

tw_rad = line.twiss(eneloss_and_damping=True)
ex = tw_rad.eq_gemitt_x
ey = tw_rad.eq_gemitt_y
ez = tw_rad.eq_gemitt_zeta

# for regression testing
if not tilt_machine_by_90_degrees and not vertical_orbit_distortion and not wiggler_on:
    assert np.isclose(ex, 6.98840e-10, atol=0,     rtol=1e-4)
    assert np.isclose(ey, 0,           atol=1e-14, rtol=0)
    assert np.isclose(ez, 3.56339e-6,  atol=0,     rtol=1e-4)
elif tilt_machine_by_90_degrees and not vertical_orbit_distortion and not wiggler_on:
    assert np.isclose(ex, 0,           atol=1e-14, rtol=0)
    assert np.isclose(ey, 6.98840e-10, atol=0,     rtol=1e-4)
    assert np.isclose(ez, 3.56339e-6,  atol=0,     rtol=1e-4)
elif not tilt_machine_by_90_degrees and not vertical_orbit_distortion and wiggler_on:
    assert np.isclose(ex, 6.92528e-10, atol=0,     rtol=1e-4)
    assert np.isclose(ey, 1.71377e-12, atol=0,     rtol=1e-4)
    assert np.isclose(ez, 3.82023e-6,  atol=0,     rtol=1e-4)
elif tilt_machine_by_90_degrees and not vertical_orbit_distortion and wiggler_on:
    assert np.isclose(ex, 1.71163e-12, atol=0,     rtol=1e-4)
    assert np.isclose(ey, 6.92580e-10, atol=0,     rtol=1e-4)
    assert np.isclose(ez, 3.82025e-6,  atol=0,     rtol=1e-4)
elif not tilt_machine_by_90_degrees and vertical_orbit_distortion and not wiggler_on:
    assert np.isclose(ex, 6.98798e-10, atol=0,     rtol=1e-4)
    assert np.isclose(ey, 1.12360e-12, atol=0,     rtol=1e-4)
    assert np.isclose(ez, 3.57778e-6,  atol=0,     rtol=1e-4)
elif tilt_machine_by_90_degrees and vertical_orbit_distortion and not wiggler_on:
    assert np.isclose(ex, 1.20001e-12, atol=0,     rtol=1e-4) #????
    assert np.isclose(ey, 6.98852e-10, atol=0,     rtol=1e-4)
    assert np.isclose(ez, 3.57779e-6,  atol=0,     rtol=1e-4)
else:
    raise ValueError('Unknown configuration')

prrr

line.configure_radiation(model='quantum')
p = line.build_particles(num_particles=30)
line.track(p, num_turns=400, turn_by_turn_monitor=True, time=True)
mon = line.record_last_track
print(f'Tracking time: {line.time_last_track}')

import matplotlib.pyplot as plt
plt.close('all')
fig = plt.figure(1)
spx = fig. add_subplot(3, 1, 1)
spx.plot(np.std(mon.x, axis=0))
spx.axhline(np.sqrt(ex * tw_rad.betx[0] + ey * tw_rad.betx2[0] + (np.std(p.delta) * tw_rad.dx[0])**2), color='red')
# spx.axhline(np.sqrt(ex_hof * tw.betx[0] + (np.std(p.delta) * tw.dx[0])**2), color='green')

spy = fig. add_subplot(3, 1, 2, sharex=spx)
spy.plot(np.std(mon.y, axis=0))
spy.axhline(np.sqrt(ex * tw_rad.bety1[0] + ey * tw_rad.bety[0] + (np.std(p.delta) * tw_rad.dy[0])**2), color='red')
# spy.axhline(np.sqrt(ey_hof * tw.bety[0] + (np.std(p.delta) * tw.dy[0])**2), color='green')

spz = fig. add_subplot(3, 1, 3, sharex=spx)
spz.plot(np.std(mon.zeta, axis=0))
spz.axhline(np.sqrt(ez * tw_rad.betz0), color='red')

plt.show()