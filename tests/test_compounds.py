# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2023.                 #
# ######################################### #

import numpy as np
import xtrack as xt


def test_insert_in_compound():
    # Check that compound relation is correctly updated
    # when inserting a new element in the middle of the compound
    line = xt.Line(
        elements=[xt.Drift(length=1) for i in range(5)],
        element_names=['d1', 'd2', 'd3', 'd4', 'd5'],
    )
    line.define_compound('d', ['d2', 'd3', 'd4'])

    assert line.compounds == {'d': ['d2', 'd3', 'd4']}
    assert line._compound_for_element == {'d2': 'd', 'd3': 'd', 'd4': 'd'}

    line.insert_element(name='new_d3', element=xt.Drift(length=2), at_s=1.5)

    assert line.compounds == {'d': ['d2_part0', 'new_d3', 'd4_part1']}
    assert line._compound_for_element == {'d2_part0': 'd', 'new_d3': 'd', 'd4_part1': 'd'}
