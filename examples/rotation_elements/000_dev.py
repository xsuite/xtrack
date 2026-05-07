import xtrack as xt
import xobjects as xo

env = xt.Environment()

line = env.new_line(components=[
    env.new('xrot', xt.XRotation, angle=90)
])


# xt.Rotation(rot_s_rad=0.1, rot_x_rad=0.2, rot_y_rad=0.3,
#             seq='yxs')

ROT_AX_TO_ID = {'x': 0, 'y': 1, 's': 2}
MAPPING_ID_TO_AX = {0: 'x', 1: 'y', 2: 's'}

class Rotation(xt.BeamElement):

    allow_rot_and_shift = False

    _xofields = {
        'rot_s_rad': xo.Float64,
        'rot_x_rad': xo.Float64,
        'rot_y_rad': xo.Float64,
        '_first_rot': xo.Field(xo.Int8, default=1),  # default to 'y' rotation
        '_second_rot': xo.Field(xo.Int8, default=0),  # default to 'x' rotation
        '_third_rot': xo.Field(xo.Int8, default=2),  # default to 's' rotation
    }

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
