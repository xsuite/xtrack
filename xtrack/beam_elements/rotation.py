# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

import xobjects as xo
import xtrack as xt
from ._common import (
    _ROT_AX_TO_ID,
    _ROT_ID_TO_AX,
)

class Rotation(xt.BeamElement):

    allow_rot_and_shift = False
    has_backtrack = True
    allow_loss_refinement = True

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/rotation.h"',
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

    def to_dict(self, *args, **kwargs):
        out = super().to_dict(*args, **kwargs)
        if out['seq'] == 'yxs': # default sequence, can be omitted for brevity
            out.pop('seq')
        return out

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

    def track_frame(self, frame, backtrack=False):

        seq = self.seq
        sign = 1
        if backtrack:
            seq = seq[::-1]  # reverse the sequence for backtracking
            sign = -1

        for ax in seq:
            if ax == 'x':
                frame.rotate_x(sign * self.rot_x_rad)
            elif ax == 'y':
                frame.rotate_y(sign * self.rot_y_rad)
            elif ax == 's':
                frame.rotate_s(sign * self.rot_s_rad)
            else:
                raise ValueError(f"Invalid rotation axis '{ax}' in sequence '{self.seq}'")
