# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xtrack as xt  # To avoid circular imports


def _build_auxiliary_tracker_with_extra_markers(tracker, at_s, marker_prefix,
                                                algorithm='auto'):

    assert algorithm in ['auto', 'insert', 'regen_all_drift']
    if algorithm == 'auto':
        if len(at_s)<10:
            algorithm = 'insert'
        else:
            algorithm = 'regen_all_drifts'

    auxline = xt.Line(elements=tracker.line._element_dict.copy(),
                      element_names=list(tracker.line.element_names).copy())
    if tracker.line.particle_ref is not None:
        auxline.particle_ref = tracker.line.particle_ref.copy()

    insertions = []
    names_inserted_markers = []
    for ii, ss in enumerate(at_s):
        nn = marker_prefix + f'{ii}'
        insertions.append(auxline.env.new(nn, 'Marker', at=ss))
        names_inserted_markers.append(nn)
    auxline.insert(insertions)

    auxtracker = xt.Tracker(
        _buffer=tracker._buffer,
        io_buffer=tracker.io_buffer,
        line=auxline,
        particles_monitor_class=None,
    )
    auxtracker.line.config = tracker.line.config.copy()
    auxtracker.line._extra_config = tracker.line._extra_config.copy()

    return auxtracker, names_inserted_markers
