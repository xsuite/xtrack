# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

import numpy as np
import pytest

import xtrack as xt

MISALIGNMENT = dict(
    shift_x=2e-3,
    shift_y=-1e-3,
    shift_s=5e-4,
    rot_y_rad=3e-3,
    rot_x_rad=-2e-3,
    rot_s_rad_no_frame=1e-2,
)


def _make_device(length, model=None, **misalignment):
    device = xt.Device(length=length, model=model)
    for name, value in misalignment.items():
        setattr(device, name, value)
    return device


def _track_coordinates(element):
    line = xt.Line(elements=[element])
    line.particle_ref = xt.Particles(p0c=1e10)
    particles = line.build_particles(
        x=[1e-3, -2e-3, 0.], px=[1e-4, 0., -3e-4],
        y=[0., 1.5e-3, -1e-3], py=[-2e-4, 1e-4, 0.],
        zeta=[0., 1e-3, -1e-3], delta=[0., 1e-4, -1e-4])
    line.track(particles)
    return np.array([particles.x, particles.px, particles.y, particles.py,
                     particles.zeta, particles.delta])


def test_device_is_misalignable_unlike_drift():
    assert xt.Device.allow_rot_and_shift
    assert not xt.Drift.allow_rot_and_shift
    assert xt.Device.behaves_like_drift

    device = xt.Device(length=1.)
    for name in ('shift_x', 'shift_y', 'shift_s', 'rot_x_rad', 'rot_y_rad',
                 'rot_s_rad', 'rot_s_rad_no_frame', 'rot_shift_anchor'):
        assert hasattr(device, name)


@pytest.mark.parametrize(
    'misalignment',
    [dict(), dict(shift_x=2e-3, shift_y=-1e-3, shift_s=5e-4),
     dict(rot_y_rad=3e-3), dict(rot_x_rad=-2e-3),
     dict(rot_s_rad_no_frame=1e-2), MISALIGNMENT],
    ids=['none', 'shift', 'rot_y', 'rot_x', 'roll', 'combined'])
def test_misaligned_device_tracks_like_a_drift(misalignment):
    # A device does not act on the beam, so displacing it must leave the
    # particles exactly where a plain drift of the same length leaves them:
    # moving a piece of empty space does not change empty space.
    length = 2.5
    expected = _track_coordinates(xt.Drift(length=length, model='exact'))
    actual = _track_coordinates(
        _make_device(length, model='exact', **misalignment))
    np.testing.assert_allclose(actual, expected, atol=5e-15, rtol=0)


def test_misaligned_device_survey_matches_the_misalignment():
    length = 2.5
    device = _make_device(length, **MISALIGNMENT)
    line = xt.Line(elements=[device], element_names=['dev'])
    line.particle_ref = xt.Particles(p0c=1e10)
    survey = line.survey(include_element_frames=True)

    # The entrance of the element sits where the shifts put it, the reference
    # path is untouched.
    np.testing.assert_allclose(
        survey['XYZ_elem_start', 'dev'] - survey['XYZ_ref_start', 'dev'],
        [MISALIGNMENT['shift_x'], MISALIGNMENT['shift_y'],
         MISALIGNMENT['shift_s']], atol=5e-14, rtol=0)
    np.testing.assert_allclose(
        survey['XYZ_ref_end', 'dev'] - survey['XYZ_ref_start', 'dev'],
        [0., 0., length], atol=5e-14, rtol=0)

    # The element keeps its length while being displaced.
    assert np.linalg.norm(
        survey['XYZ_elem_end', 'dev']
        - survey['XYZ_elem_start', 'dev']) == pytest.approx(length, abs=5e-14)


def test_slicing_a_device_keeps_its_misalignment():
    # The thick slices of a device take their transformation from the parent.
    # Were they modelled on the drift slices, which switch transformations
    # off, slicing would silently move the device back to its nominal place.
    length = 2.5
    misaligned = _make_device(length, **MISALIGNMENT)

    line = xt.Line(elements=[misaligned], element_names=['dev'])
    line.particle_ref = xt.Particles(p0c=1e10)
    survey_thick = line.survey(include_element_frames=True)

    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Uniform(3, mode='thick'))])
    survey_sliced = line.survey(include_element_frames=True)

    # The slices themselves, skipping the entry and exit markers that slicing
    # puts around them.
    slices = [nn for nn in line.element_names
              if isinstance(line[nn], xt.ThickSliceDevice)]
    assert len(slices) == 3

    # The sliced device starts and ends where the unsliced one did.
    np.testing.assert_allclose(
        survey_sliced['XYZ_elem_start', slices[0]],
        survey_thick['XYZ_elem_start', 'dev'], atol=5e-14, rtol=0)
    np.testing.assert_allclose(
        survey_sliced['XYZ_elem_end', slices[-1]],
        survey_thick['XYZ_elem_end', 'dev'], atol=5e-14, rtol=0)

    # And the slices are laid end to end along the displaced body.
    for upstream, downstream in zip(slices[:-1], slices[1:]):
        np.testing.assert_allclose(
            survey_sliced['XYZ_elem_end', upstream],
            survey_sliced['XYZ_elem_start', downstream], atol=5e-14, rtol=0)


def test_sliced_device_tracks_like_a_drift():
    length = 2.5
    line = xt.Line(elements=[_make_device(length, model='exact',
                                          **MISALIGNMENT)],
                   element_names=['dev'])
    line.particle_ref = xt.Particles(p0c=1e10)
    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Uniform(3, mode='thick'))])

    particles = line.build_particles(
        x=[1e-3, -2e-3, 0.], px=[1e-4, 0., -3e-4],
        y=[0., 1.5e-3, -1e-3], py=[-2e-4, 1e-4, 0.],
        zeta=[0., 1e-3, -1e-3], delta=[0., 1e-4, -1e-4])
    line.track(particles)
    actual = np.array([particles.x, particles.px, particles.y, particles.py,
                       particles.zeta, particles.delta])

    expected = _track_coordinates(xt.Drift(length=length, model='exact'))
    np.testing.assert_allclose(actual, expected, atol=5e-14, rtol=0)
