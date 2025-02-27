# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xobjects as xo
import xpart as xp
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_collective_tracker_indices_one_turn(test_context):
    line = xt.Line(elements=[xt.Drift(length=1) for i in range(8)])

    line.elements[2].iscollective = True
    line.elements[2].move(_context=test_context)

    line.elements[5].iscollective = True
    line.elements[5].move(_context=test_context)

    line.build_tracker(_context=test_context)
    line.reset_s_at_end_turn = False

    particles = xp.Particles(x=0, px=0, _context=test_context)

    line.track(particles)

    assert particles.s[0] == len(line.elements)
    assert particles.at_turn[0] == 1

@for_all_test_contexts
def test_get_non_collective_line(test_context):

    line = xt.Line(
        elements=[xt.Drift(length=1) for i in range(8)],
        element_names=[f'e{i}' for i in range(8)]
    )
    line['e3'].iscollective = True
    e3_buffer = line['e3']._buffer
    e3 = line.get('e3')

    try:
        line.iscollective
    except RuntimeError:
        pass
    else:
        raise ValueError('This should have failed')

    try:
        line._buffer
    except RuntimeError:
        pass
    else:
        raise ValueError('This should have failed')

    line.build_tracker(_context=test_context)

    assert line.iscollective == True
    assert line['e0']._buffer is line._buffer
    assert line['e7']._buffer is line._buffer
    assert line['e3']._buffer is not line._buffer
    assert line['e3']._buffer is e3_buffer
    assert line.get('e3') is e3
    assert line.tracker.line is line

    nc_line = line._get_non_collective_line()

    # Check that the original line is untouched
    assert line.iscollective == True
    assert line['e0']._buffer is line._buffer
    assert line['e7']._buffer is line._buffer
    assert line['e3']._buffer is not line._buffer
    assert line['e3']._buffer is e3_buffer
    assert line.get('e3') is e3
    assert line.tracker.line is line

    assert nc_line.iscollective == False
    assert nc_line._buffer is line._buffer
    assert nc_line['e0']._buffer is line._buffer
    assert nc_line['e7']._buffer is line._buffer
    assert nc_line['e3']._buffer is line._buffer
    assert nc_line.get('e3') is not e3
    assert nc_line.get('e0') is line.get('e0')
    assert nc_line.get('e7') is line.get('e7')
    assert nc_line.tracker.line is nc_line

    xo.assert_allclose(nc_line.get_s_elements(), line.get_s_elements(),
                    rtol=0, atol=1e-15)

    assert nc_line.tracker is not line.tracker
    assert nc_line.tracker._tracker_data_cache is line.tracker._tracker_data_cache
    assert line.tracker._track_kernel is nc_line.tracker._track_kernel