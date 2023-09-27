import numpy as np
import xtrack as xt


tilt_machine_by_90_degrees = True
wiggler_on = True
vertical_orbit_distortion = False
check_against_tracking = True

line = xt.Line.from_json('fccee_h_thin.json')
line.build_tracker()

print('Done building tracker')

if wiggler_on:
    line.vars['on_wiggler_v'] = 0.4

if vertical_orbit_distortion:
    line['mwi.e5rg'].ksl[0] = 2e-7

# Make sure there is no vertical bend nor skew element
for ee in line.elements:
    if isinstance(ee, xt.Multipole):
        assert np.all(ee.ksl[1:] == 0)

if tilt_machine_by_90_degrees:

    tw_before_tilt = line.twiss()

    # Bring the machine to the vertical plane
    for ee in line.elements:
        if isinstance(ee, xt.Multipole):
            knl = ee.knl.copy()
            ksl = ee.ksl.copy()
            hxl = ee.hxl
            hyl = ee.hyl

            ee.hxl = -hyl
            ee.hyl = -hxl

            ee.knl[0] = -ksl[0]
            ee.ksl[0] = -knl[0]
            if len(knl) > 1:
                ee.knl[1] = -knl[1]
                ee.ksl[1] = 0
            if len(knl) > 2:
                ee.knl[2] = 0
                ee.ksl[2] = knl[2]

        if isinstance(ee, xt.DipoleEdge):
            ee._r21, ee._r43 = ee._r43, ee._r21

    tw_after_tilt = line.twiss()

    assert np.isclose(tw_after_tilt.qy, tw_before_tilt.qx, rtol=0, atol=1e-8)
    assert np.isclose(tw_after_tilt.qx, tw_before_tilt.qy, rtol=0, atol=1e-8)
    assert np.isclose(tw_after_tilt.dqy, tw_before_tilt.dqx, rtol=0, atol=1e-4)
    assert np.isclose(tw_after_tilt.dqx, tw_before_tilt.dqy, rtol=0, atol=1e-4)

    assert np.allclose(tw_after_tilt.bety, tw_before_tilt.betx, rtol=1e-5, atol=0)
    assert np.allclose(tw_after_tilt.betx, tw_before_tilt.bety, rtol=1e-5, atol=0)

    assert np.allclose(tw_after_tilt.y, tw_before_tilt.x, rtol=0, atol=1e-9)
    assert np.allclose(tw_after_tilt.x, tw_before_tilt.y, rtol=0, atol=1e-9)

    assert np.allclose(tw_after_tilt.dy, tw_before_tilt.dx, rtol=0, atol=5e-6)
    assert np.allclose(tw_after_tilt.dx, tw_before_tilt.dy, rtol=0, atol=5e-6)

tw_no_rad = line.twiss(method='4d')

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()

tw_rad = line.twiss(eneloss_and_damping=True)
ex = tw_rad.eq_gemitt_x
ey = tw_rad.eq_gemitt_y
ez = tw_rad.eq_gemitt_zeta
