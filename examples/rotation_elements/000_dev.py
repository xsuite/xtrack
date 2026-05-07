import xtrack as xt
import xobjects as xo
import numpy as np

from xtrack.survey import advance_element as survey_advance_element

ROT_AX_TO_ID = {'x': 0, 'y': 1, 's': 2}
MAPPING_ID_TO_AX = {0: 'x', 1: 'y', 2: 's'}

class Rotation(xt.BeamElement):

    allow_rot_and_shift = False

    _extra_c_sources = [
        '#include "rotation.h"',
    ]

    _xofields = {
        'rot_s_rad': xo.Float64,
        'rot_x_rad': xo.Float64,
        'rot_y_rad': xo.Float64,
        '_first_rot': xo.Field(xo.Int8, default=1),  # default to 'y' rotation
        '_second_rot': xo.Field(xo.Int8, default=0),  # default to 'x' rotation
        '_third_rot': xo.Field(xo.Int8, default=2),  # default to 's' rotation
    }

    def __init__(self, rot_s_rad=0, rot_x_rad=0, rot_y_rad=0, seq='yxs', **kwargs):
        super().__init__(**kwargs)
        self.rot_s_rad = rot_s_rad
        self.rot_x_rad = rot_x_rad
        self.rot_y_rad = rot_y_rad
        self.seq = seq  # this will set the _first_rot, _second_rot, _third_rot fields

    @property
    def seq(self):
        out = (MAPPING_ID_TO_AX[self._first_rot] +
               MAPPING_ID_TO_AX[self._second_rot] +
               MAPPING_ID_TO_AX[self._third_rot])
        return out

    @seq.setter
    def seq(self, value):
        if len(value) != 3 or set(value) != {'x', 'y', 's'}:
            raise ValueError("Sequence must be a permutation of 'x', 'y', 's'")
        self._first_rot = ROT_AX_TO_ID[value[0]]
        self._second_rot = ROT_AX_TO_ID[value[1]]
        self._third_rot = ROT_AX_TO_ID[value[2]]

    def _propagate_survey(self, v, w, backtrack):

        for ax in self.seq:
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
                        ref_rot_x_rad   = rx,
                        ref_rot_y_rad   = -ry,
                        ref_rot_s_rad   = rs,
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

env = xt.Environment()
env.elements['rot'] = Rotation(rot_s_rad=0.1, rot_x_rad=0.2, rot_y_rad=0.3, seq='yxs')

line = env.new_line(components=['rot'])

rot = line['rot']
legacy_rots = []
for ax in rot.seq:
    if ax == 'x':
        legacy_rots.append(xt.XRotation(angle=np.rad2deg(rot.rot_x_rad)))
    elif ax == 'y':
        legacy_rots.append(xt.YRotation(angle=np.rad2deg(rot.rot_y_rad)))
    elif ax == 's':
        legacy_rots.append(xt.SRotation(angle=np.rad2deg(rot.rot_s_rad)))
line_legacy = xt.Line(elements=legacy_rots)

sv = line.survey()
sv_legacy = line_legacy.survey()


