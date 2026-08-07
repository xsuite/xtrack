# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xtrack as xt  # To avoid circular imports


def _finalize_twiss_result(res, kwargs, zero_at=None):
    if isinstance(res, xt.TwissInit):
        return res
    if zero_at is not None:
        res.zero_at(zero_at)
    twiss_kwargs = kwargs.copy()
    action = xt.match.ActionTwiss(**twiss_kwargs)
    res._data['_action'] = action
    return res
