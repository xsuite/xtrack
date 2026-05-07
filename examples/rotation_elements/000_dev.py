import xtrack as xt
import xobjects as xo
import numpy as np

from xtrack.survey import advance_element as survey_advance_element






rot = xt.Rotation(rot_s_rad=0.1, rot_x_rad=0.2, rot_y_rad=0.3)
assert rot.seq == 'yxs'
assert rot._first_rot == 1
assert rot._second_rot == 0
assert rot._third_rot == 2

rot.seq = 'sxy'
assert rot._first_rot == 2
assert rot._second_rot == 0
assert rot._third_rot == 1
assert rot.seq == 'sxy'


dct = rot.to_dict()
assert set(rot.to_dict().keys()) == {'__class__', 'rot_s_rad', 'rot_x_rad', 'rot_y_rad', 'seq'}
rot2 = xt.Rotation.from_dict(dct)
assert rot2.rot_s_rad == rot.rot_s_rad
assert rot2.rot_x_rad == rot.rot_x_rad
assert rot2.rot_y_rad == rot.rot_y_rad
assert rot2.seq == rot.seq



for seq in ['yxs', 'xsy', 'sxy', 'syx', 'xys', 'ysx']:

    env = xt.Environment()
    env.new('end_marker', xt.Marker)
    env.elements['rot'] = xt.Rotation(rot_s_rad=0.1, rot_x_rad=0.2, rot_y_rad=0.3, seq=seq)

    line = env.new_line(components=['rot', 'end_marker'])
    line.particle_ref = xt.Particles(p0c=1e9)

    rot = line['rot']
    legacy_rots = []
    for ax in rot.seq:
        if ax == 'x':
            legacy_rots.append(xt.XRotation(angle=np.rad2deg(rot.rot_x_rad)))
        elif ax == 'y':
            legacy_rots.append(xt.YRotation(angle=np.rad2deg(rot.rot_y_rad)))
        elif ax == 's':
            legacy_rots.append(xt.SRotation(angle=np.rad2deg(rot.rot_s_rad)))

    env.elements['r1'] = legacy_rots[0]
    env.elements['r2'] = legacy_rots[1]
    env.elements['r3'] = legacy_rots[2]
    line_legacy = env.new_line(components=['r1', 'r2', 'r3', 'end_marker'])
    line_legacy.particle_ref = xt.Particles(p0c=1e9)

    sv = line.survey(X0=0.1, Y0=0.2, Z0=0.3, theta0=0.4, phi0=0.5, psi0=0.6)
    sv_legacy = line_legacy.survey(X0=0.1, Y0=0.2, Z0=0.3, theta0=0.4, phi0=0.5, psi0=0.6)

    sv_back = line.survey(element0='end_marker', X0=sv.X[-1], Y0=sv.Y[-1], Z0=sv.Z[-1],
                        theta0=sv.theta[-1], phi0=sv.phi[-1], psi0=sv.psi[-1])

    xo.assert_allclose(sv.X[-1], sv_legacy.X[-1], atol=1e-12)
    xo.assert_allclose(sv.Y[-1], sv_legacy.Y[-1], atol=1e-12)
    xo.assert_allclose(sv.Z[-1], sv_legacy.Z[-1], atol=1e-12)
    xo.assert_allclose(sv.theta[-1], sv_legacy.theta[-1], atol=1e-12)
    xo.assert_allclose(sv.phi[-1], sv_legacy.phi[-1], atol=1e-12)
    xo.assert_allclose(sv.psi[-1], sv_legacy.psi[-1], atol=1e-12)

    xo.assert_allclose(sv.X, sv_back.X, atol=1e-12)
    xo.assert_allclose(sv.Y, sv_back.Y, atol=1e-12)
    xo.assert_allclose(sv.Z, sv_back.Z, atol=1e-12)
    xo.assert_allclose(sv.theta, sv_back.theta, atol=1e-12)
    xo.assert_allclose(sv.phi, sv_back.phi, atol=1e-12)
    xo.assert_allclose(sv.psi, sv_back.psi, atol=1e-12)

    tw = line.twiss(betx=1, bety=1, x=0.01, px=0.02, y=0.03, py=0.04)
    tw_legacy = line_legacy.twiss(betx=1, bety=1, x=0.01, px=0.02, y=0.03, py=0.04)

    xo.assert_allclose(tw.x[-1], tw_legacy.x[-1], atol=1e-12)
    xo.assert_allclose(tw.px[-1], tw_legacy.px[-1], atol=1e-12)
    xo.assert_allclose(tw.y[-1], tw_legacy.y[-1], atol=1e-12)
    xo.assert_allclose(tw.py[-1], tw_legacy.py[-1], atol=1e-12)

    tw_back = line.twiss(init_at='end_marker', x=tw.x[-1], px=tw.px[-1], y=tw.y[-1], py=tw.py[-1],
                         betx=tw.betx[-1], bety=tw.bety[-1], alfx=tw.alfx[-1], alfy=tw.alfy[-1])
    xo.assert_allclose(tw.x, tw_back.x, atol=1e-12)
    xo.assert_allclose(tw.px, tw_back.px, atol=1e-12)
    xo.assert_allclose(tw.y, tw_back.y, atol=1e-12)
    xo.assert_allclose(tw.py, tw_back.py, atol=1e-12)