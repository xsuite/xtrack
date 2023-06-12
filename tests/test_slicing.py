# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2023.                 #
# ######################################### #

from xtrack.slicing import Teapot, Uniform


def test_slicing_uniform():
    slicing_3 = Uniform(1)
    assert slicing_3.element_weights() == [1.0]
    assert slicing_3.drift_weights() == [0.5] * 2
    assert [w for w in slicing_3] == [(0.5, True), (1.0, False), (0.5, True)]

    slicing_3 = Uniform(3)
    assert slicing_3.element_weights() == [1/3] * 3
    assert slicing_3.drift_weights() == [0.25] * 4

    elem_info, drift_info = (1./3., False), (0.25, True)
    assert [w for w in slicing_3] == [
        drift_info, elem_info,
        drift_info, elem_info,
        drift_info, elem_info,
        drift_info,
    ]


def test_slicing_teapot():
    slicing_3 = Teapot(1)
    assert slicing_3.element_weights() == [1.0]
    assert slicing_3.drift_weights() == [0.5] * 2
    assert [w for w in slicing_3] == [(0.5, True), (1.0, False), (0.5, True)]

    slicing_3 = Teapot(3)
    assert slicing_3.element_weights() == [1/3] * 3
    assert slicing_3.drift_weights() == [0.125, 0.375, 0.375, 0.125]

    elem_info = (1./3., False)
    assert [w for w in slicing_3] == [
        (0.125, True), elem_info,
        (0.375, True), elem_info,
        (0.375, True), elem_info,
        (0.125, True),
    ]

