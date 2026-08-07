# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

DEFAULT_STEPS_R_MATRIX = {
    'dx':1e-6, 'dpx':1e-7,
    'dy':1e-6, 'dpy':1e-7,
    'dzeta':1e-5, 'ddelta':1e-6
}

DEFAULT_CO_SEARCH_TOL = [1e-11, 1e-11, 1e-11, 1e-11, 1e-5, 1e-9]

DEFAULT_MATRIX_RESPONSIVENESS_TOL = 1e-15
DEFAULT_MATRIX_STABILITY_TOL = 2e-3
DEFAULT_NUM_TURNS_SEARCH_T_REV = 10

AT_TURN_FOR_TWISS = -10 # # To avoid writing in monitors installed in the line

VARS_FOR_TWISS_INIT_GENERATION = [
    'x', 'px', 'y', 'py', 'zeta', 'delta',
    'betx', 'alfx', 'bety', 'alfy', 'bets',
    'dx', 'dpx', 'dy', 'dpy', 'dzeta',
    'mux', 'muy', 'muzeta',
    'ax_chrom', 'bx_chrom', 'ay_chrom', 'by_chrom',
    'ddx', 'ddpx', 'ddy', 'ddpy',
]

CYCLICAL_QUANTITIES = ['mux', 'muy', 'dzeta', 's']

NORMAL_STRENGTHS_FROM_ATTR=['k0l', 'k1l', 'k2l', 'k3l', 'k4l', 'k5l']
SKEW_STRENGTHS_FROM_ATTR=['k0sl', 'k1sl', 'k2sl', 'k3sl', 'k4sl', 'k5sl']
OTHER_FIELDS_FROM_ATTR=['angle', 'angle_rad', 'rot_s_rad', 'hkick', 'vkick', 'ks', 'bs', 'length', '_angle_force_body']
OTHER_FIELDS_FROM_TABLE=['element_type', 'isthick', 'parent_name', 'parent_type', 'prototype']
SIGN_FLIP_FOR_ATTR_REVERSE=['k0l', 'k2l', 'k4l', 'k1sl', 'k3sl', 'k5sl', 'vkick', 'angle', 'angle_rad']

DEFAULT_COL_ORDER = [
    'name', 'element_type', 's', 'betx', 'bety', 'alfx', 'alfy', 'dx', 'dy'
    'dpx', 'dpy', 'x', 'y', 'px', 'py', 'delta', 'zeta']
