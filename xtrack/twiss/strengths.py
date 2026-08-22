# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np


NORMAL_STRENGTHS_FROM_ATTR = ['k0l', 'k1l', 'k2l', 'k3l', 'k4l', 'k5l']
SKEW_STRENGTHS_FROM_ATTR = [
    'k0sl', 'k1sl', 'k2sl', 'k3sl', 'k4sl', 'k5sl']
OTHER_FIELDS_FROM_ATTR = [
    'angle', 'angle_rad', 'rot_s_rad', 'hkick', 'vkick', 'ks', 'bs',
    'length', '_angle_force_body']
OTHER_FIELDS_FROM_TABLE = [
    'element_type', 'isthick', 'parent_name', 'parent_type', 'prototype']
SIGN_FLIP_FOR_ATTR_REVERSE = [
    'k0l', 'k2l', 'k4l', 'k1sl', 'k3sl', 'k5sl', 'vkick', 'angle',
    'angle_rad']


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
