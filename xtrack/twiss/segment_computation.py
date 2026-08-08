# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from .base_computation import _compute_base_twiss


def _compute_twiss_segment(kwargs, **overrides):

    segment_kwargs = _kwargs_for_preflighted_twiss_segment(kwargs)
    segment_kwargs.update(overrides)

    return _compute_base_twiss(segment_kwargs)


def _kwargs_for_preflighted_twiss_segment(kwargs):

    segment_kwargs = kwargs.copy()
    segment_kwargs['disable_apertures'] = False
    segment_kwargs['freeze_longitudinal'] = False
    segment_kwargs['freeze_energy'] = False
    segment_kwargs['at_s'] = None

    return segment_kwargs


def _compute_twiss_segment_for_piece(kwargs, piece, init):

    return _compute_twiss_segment(
        kwargs, start=piece.start, end=piece.end, init=init)

