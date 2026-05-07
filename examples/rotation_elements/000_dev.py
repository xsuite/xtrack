import xtrack as xt
import xobjects as xo
import numpy as np

from xtrack.survey import advance_element as survey_advance_element

_ROT_AX_TO_ID = {'x': 0, 'y': 1, 's': 2}
_ROT_ID_TO_AX = {0: 'x', 1: 'y', 2: 's'}

class Rotation(xt.BeamElement):

    allow_rot_and_shift = False
    has_backtrack = True

    _extra_c_sources = [
        '#include "rotation.h"',
    ]

    _noexpr_fields = ['seq']

    _skip_in_to_dict = ['_first_rot', '_second_rot', '_third_rot']
    _store_in_to_dict = ['seq']


    _xofields = {
        'rot_s_rad': xo.Float64,
        'rot_x_rad': xo.Float64,
        'rot_y_rad': xo.Float64,
        '_first_rot': xo.Field(xo.Int8, default=1),  # default to 'y' rotation
        '_second_rot': xo.Field(xo.Int8, default=0),  # default to 'x' rotation
        '_third_rot': xo.Field(xo.Int8, default=2),  # default to 's' rotation
    }

    def __init__(self, rot_s_rad=0, rot_x_rad=0, rot_y_rad=0, seq='yxs', **kwargs):

        """"
        3D rotation element.

        Parameters
        ----------
        rot_s_rad : float
            Rotation around the longitudinal axis applied to the element [rad].
        rot_x_rad : float
            Rotation around the horizontal axis applied to the element [rad].
        rot_y_rad : float
            Rotation around the vertical axis applied to the element [rad].
        seq : str
            Sequence of rotations, as a permutation of 'x', 'y', 's'.
            Default is 'yxs', which means that the first rotation applied to
            the element is around y, then around x, and finally around s.

        """

        super().__init__(**kwargs)
        self.rot_s_rad = rot_s_rad
        self.rot_x_rad = rot_x_rad
        self.rot_y_rad = rot_y_rad
        self.seq = seq  # this will set the _first_rot, _second_rot, _third_rot fields

    def __repr__(self):
        return (f"Rotation(rot_s_rad={self.rot_s_rad}, rot_x_rad={self.rot_x_rad}, "
                f"rot_y_rad={self.rot_y_rad}, seq='{self.seq}')")

    @property
    def seq(self):
        out = (_ROT_ID_TO_AX[self._first_rot] +
               _ROT_ID_TO_AX[self._second_rot] +
               _ROT_ID_TO_AX[self._third_rot])
        return out

    @seq.setter
    def seq(self, value):
        if len(value) != 3 or set(value) != {'x', 'y', 's'}:
            raise ValueError("Sequence must be a permutation of 'x', 'y', 's'")
        self._first_rot = _ROT_AX_TO_ID[value[0]]
        self._second_rot = _ROT_AX_TO_ID[value[1]]
        self._third_rot = _ROT_AX_TO_ID[value[2]]

    def _propagate_survey(self, v, w, backtrack):

        seq = self.seq
        fback = 1
        if backtrack:
            seq = seq[::-1]  # reverse the sequence for backtracking
            fback = -1

        for ax in seq:
            if ax == 'x':
                rx, ry, rs = self.rot_x_rad, 0, 0
            elif ax == 'y':
                rx, ry, rs = 0, self.rot_y_rad, 0
            elif ax == 's':
                rx, ry, rs = 0, 0, self.rot_s_rad
            else:
                raise ValueError(f"Invalid rotation axis '{ax}' in sequence '{self.seq}'")

            v, w = survey_advance_element(
                        v               = v,
                        w               = w,
                        length          = 0,
                        angle           = 0,
                        tilt            = 0,
                        ref_shift_x     = 0,
                        ref_shift_y     = 0,
                        ref_rot_x_rad   = fback * rx,
                        ref_rot_y_rad   = -fback * ry,
                        ref_rot_s_rad   = fback * rs,
                    )
        return v, w


rot = Rotation(rot_s_rad=0.1, rot_x_rad=0.2, rot_y_rad=0.3)
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
rot2 = Rotation.from_dict(dct)
assert rot2.rot_s_rad == rot.rot_s_rad
assert rot2.rot_x_rad == rot.rot_x_rad
assert rot2.rot_y_rad == rot.rot_y_rad
assert rot2.seq == rot.seq



for seq in ['yxs', 'xsy', 'sxy', 'syx', 'xys', 'ysx']:

    env = xt.Environment()
    env.new('end_marker', xt.Marker)
    env.elements['rot'] = Rotation(rot_s_rad=0.1, rot_x_rad=0.2, rot_y_rad=0.3, seq=seq)

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