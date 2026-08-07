# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

from .constants import (
    NORMAL_STRENGTHS_FROM_ATTR,
    OTHER_FIELDS_FROM_ATTR,
    OTHER_FIELDS_FROM_TABLE,
    SIGN_FLIP_FOR_ATTR_REVERSE,
    SKEW_STRENGTHS_FROM_ATTR,
)


def _reverse_strengths(out):
    ### Same convention as in MAD-X for reversing strengths
    for kk in SIGN_FLIP_FOR_ATTR_REVERSE:
        if kk in out:
            val=out[kk]#avoid passing by setitem
            np.negative(val,val)


def _add_strengths_to_twiss_res(twiss_res, line):
    tt = line.get_table(attr=True).rows[list(twiss_res.name)]
    for kk in (NORMAL_STRENGTHS_FROM_ATTR + SKEW_STRENGTHS_FROM_ATTR
                + OTHER_FIELDS_FROM_ATTR + OTHER_FIELDS_FROM_TABLE):
        twiss_res._col_names.append(kk)
        # using _data to bypass the warning on deprecated fields
        twiss_res._data[kk] = tt._data[kk].copy()
