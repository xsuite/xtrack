# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xtrack as xt  # To avoid circular imports


def _add_action_in_res(res, kwargs):
    if isinstance(res, xt.TwissInit):
        return res
    twiss_kwargs = kwargs.copy()
    action = xt.match.ActionTwiss(**twiss_kwargs)
    res._data['_action'] = action
    return res
