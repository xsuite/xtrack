# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from .twiss_table import TwissTable


def _combine_loop_around_twiss_tables(tw1, tw2, init, completed_init):

    ele_name_init = init.element_name

    tw_res = TwissTable.concatenate([tw1, tw2])

    tw_res.s -= tw_res['s', ele_name_init] - init.s

    tw_res['completed_init'] = completed_init

    if 'mux' in tw_res.keys():
        tw_res.mux -= tw_res['mux', ele_name_init] - init.mux
        tw_res.muy -= tw_res['muy', ele_name_init] - init.muy
        tw_res.muzeta -= tw_res['muzeta', ele_name_init] - init.muzeta

    if 'dzeta' in tw_res.keys():
        tw_res.dzeta -= tw_res['dzeta', ele_name_init] - init.dzeta

    _remove_unsupported_phase_derivative_columns(tw_res)

    tw_res._data['loop_around'] = True

    _copy_common_metadata_from_table_pair(tw_res=tw_res, tw1=tw1, tw2=tw2)

    return tw_res


def _combine_init_inside_range_twiss_tables(tw1, tw2, init):

    ele_name_init = init.element_name

    tw_res = TwissTable.concatenate([tw1, tw2])
    tw_res['completed_init'] = tw1.completed_init

    tw_res.s -= tw_res['s', ele_name_init] - init.s
    tw_res.mux -= tw_res['mux', ele_name_init] - init.mux
    tw_res.muy -= tw_res['muy', ele_name_init] - init.muy
    tw_res.muzeta -= tw_res['muzeta', ele_name_init] - init.muzeta

    if 'dzeta' in tw_res:
        tw_res.dzeta -= tw_res['dzeta', ele_name_init] - init.dzeta

    _remove_unsupported_phase_derivative_columns(tw_res)
    _copy_common_metadata_from_table_pair(tw_res=tw_res, tw1=tw1, tw2=tw2)

    return tw_res


def _remove_unsupported_phase_derivative_columns(tw_res):

    for col_name in ['dmux', 'dmuy']:
        if col_name in tw_res.keys():
            tw_res._data.pop(col_name)
            tw_res._col_names.remove(col_name)


def _copy_common_metadata_from_table_pair(tw_res, tw1, tw2):

    for kk in ['method', 'radiation_method', 'reference_frame']:
        if tw1[kk] == tw2[kk]:
            tw_res._data[kk] = tw1[kk]
        else:
            tw_res._data[kk] = (tw1[kk], tw2[kk])
