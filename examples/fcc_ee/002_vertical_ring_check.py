import numpy as np
import xtrack as xt


tilt_machine_by_90_degrees = True
wiggler_on = True
vertical_orbit_distortion = True
check_against_tracking = True

line = xt.Line.from_json('fccee_h_thin.json')
line.build_tracker()

print('Done building tracker')

if wiggler_on:
    line.vars['on_wiggler_v'] = 0.4

if vertical_orbit_distortion:
    line['mwi.e5rg'].knl[0] = 2e-7

if tilt_machine_by_90_degrees:

    tw_before_tilt = line.twiss()

    # Bring the machine to the vertical plane
    already_tilted = []
    for nn in line.element_names:
        if hasattr(line[nn], 'rot_s_rad'):
            if nn not in already_tilted:
                line[nn].rot_s_rad += np.pi/2
                already_tilted.append(nn)
        if hasattr(line[nn], 'parent_name'):
            nn_parent = line[nn].parent_name
            if hasattr(line[nn_parent], 'rot_s_rad'):
                if nn_parent not in already_tilted:
                    line[nn_parent].rot_s_rad += np.pi/2
                    already_tilted.append(nn_parent)

    tw_after_tilt = line.twiss()

    assert_allclose = np.testing.assert_allclose
    assert_allclose(tw_after_tilt.qy, tw_before_tilt.qx, rtol=0, atol=1e-8)
    assert_allclose(tw_after_tilt.qx, tw_before_tilt.qy, rtol=0, atol=1e-8)
    assert_allclose(tw_after_tilt.dqy, tw_before_tilt.dqx, rtol=0, atol=1e-4)
    assert_allclose(tw_after_tilt.dqx, tw_before_tilt.dqy, rtol=0, atol=1e-4)

    assert_allclose(tw_after_tilt.bety, tw_before_tilt.betx, rtol=1e-5, atol=0)
    assert_allclose(tw_after_tilt.betx, tw_before_tilt.bety, rtol=1e-5, atol=0)

    assert_allclose(tw_after_tilt.y, tw_before_tilt.x, rtol=0, atol=1e-9)
    assert_allclose(tw_after_tilt.x, -tw_before_tilt.y, rtol=0, atol=1e-9)

    assert_allclose(tw_after_tilt.dy, tw_before_tilt.dx, rtol=0, atol=5e-6)
    assert_allclose(tw_after_tilt.dx, -tw_before_tilt.dy, rtol=0, atol=5e-6)

tw_no_rad = line.twiss(method='4d')

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()

tw_rad = line.twiss(eneloss_and_damping=True)
ex = tw_rad.eq_gemitt_x
ey = tw_rad.eq_gemitt_y
ez = tw_rad.eq_gemitt_zeta
