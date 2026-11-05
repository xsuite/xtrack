# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

import numpy as np
import pytest

import xtrack as xt
from xtrack import magnet_plans


def _line():
    env = xt.Environment()
    line = env.new_line(components=[
        env.new('q', xt.Quadrupole, length=1, k1=0.3, rot_s_rad=0.1),
        env.new('b', xt.Bend, length=2, angle=0.1, k0_from_h=True,
                edge_entry_model='full', edge_entry_angle=0.02),
        env.new('m', xt.Marker),
    ])
    line.set_particle_ref('proton', p0c=1e9)
    return line


def test_plan_layout_matches_c():
    # The kernel fails to compile if PLAN_DTYPE differs from MagnetPlan
    line = _line()
    line.config.XTRACK_CHECK_MAGNET_PLAN_LAYOUT = True
    line.build_tracker()
    assert magnet_plans.PLAN_DTYPE.itemsize <= magnet_plans.MAGNET_PLAN_BYTES


def test_get_magnet_plan():
    line = _line()
    line.build_tracker()

    plan = line.tracker.get_magnet_plan('q')
    np.testing.assert_array_equal(plan['field']['knorm'], [0, 0.3, 0, 0])
    assert plan['body']['length'] == 1.0
    assert plan['transform']['rot_s_rad'] == 0.1
    assert plan['is_thick'] == 1

    # The plan follows the attributes
    line['q'].k1 = 0.5
    plan = line.tracker.get_magnet_plan('q')
    np.testing.assert_array_equal(plan['field']['knorm'], [0, 0.5, 0, 0])

    # Faces in the order they are crossed
    forward = line.tracker.get_magnet_plan('b')
    assert forward['first_face']['is_exit_face'] == 0
    assert forward['first_face']['angle'] == 0.02
    backward = line.tracker.get_magnet_plan('b', backtrack=True)
    assert backward['first_face']['is_exit_face'] == 1
    assert backward['body']['length'] == -2.0

    with pytest.raises(ValueError):
        line.tracker.get_magnet_plan('m')


def test_magnet_classes_of_structs_and_elements():
    # A kernel lists xobjects structs; a prebuilt kernel's metadata lists the elements
    elements = [xt.ACDipole, xt.Bend, xt.Drift, xt.Quadrupole]
    expected = [(1, xt.Bend), (3, xt.Quadrupole)]
    assert magnet_plans.magnet_classes(elements) == expected
    structs = [cc._XoStruct for cc in elements]
    assert magnet_plans.magnet_classes(structs) == [
        (ii, cc._XoStruct) for ii, cc in expected]
