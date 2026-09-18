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


@pytest.mark.parametrize('mode', [None, 'thin', 'thick'])
def test_configure_device_drift_model(mode):
    line = xt.Line(elements={
        'drift': xt.Drift(length=1.),
        'dev': xt.Device(length=2.),
        'replica': xt.Replica(parent_name='dev'),
    })
    if mode is not None:
        line.slice_thick_elements(slicing_strategies=[
            xt.Strategy(xt.Teapot(2, mode=mode), element_type=xt.Device)])
    line.build_tracker()

    # Switching the model after building the tracker must also affect device
    # replicas and slices, which read the model from their parent.
    p0 = xt.Particles(p0c=1e9, px=0.1, py=0.03, delta=0.01)
    for model in ('exact', 'expanded', 'adaptive'):
        line.configure_drift_model(model=model)
        assert line['drift'].model == model
        assert line['dev'].model == model

        particles = p0.copy()
        line.track(particles)
        assert np.all(particles.state == 1)
        denominator = (np.sqrt((1 + p0.delta)**2 - p0.px**2 - p0.py**2)
                       if model == 'exact' else 1 + p0.delta)
        np.testing.assert_allclose(
            particles.x, 5 * p0.px / denominator, atol=1e-14, rtol=0)
        np.testing.assert_allclose(
            particles.y, 5 * p0.py / denominator, atol=1e-14, rtol=0)


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


@pytest.mark.parametrize('mode', ['thin', 'thick'])
def test_slicing_a_device_keeps_its_misalignment(mode):
    # The thick slices of a device take their transformation from the parent.
    # Were they modelled on the drift slices, which switch transformations
    # off, slicing would silently move the device back to its nominal place.
    length = 2.5
    misaligned = _make_device(length, **MISALIGNMENT)

    line = xt.Line(elements=[misaligned], element_names=['dev'])
    line.particle_ref = xt.Particles(p0c=1e10)
    survey_thick = line.survey(include_element_frames=True)

    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(
            xt.Uniform(2 if mode == 'thin' else 3, mode=mode))])
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


@pytest.mark.parametrize('mode', ['thin', 'thick'])
def test_sliced_device_tracks_like_a_drift(mode):
    length = 2.5
    line = xt.Line(elements=[_make_device(length, model='exact',
                                          **MISALIGNMENT)],
                   element_names=['dev'])
    line.particle_ref = xt.Particles(p0c=1e10)
    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(
            xt.Uniform(2 if mode == 'thin' else 3, mode=mode))])

    particles = line.build_particles(
        x=[1e-3, -2e-3, 0.], px=[1e-4, 0., -3e-4],
        y=[0., 1.5e-3, -1e-3], py=[-2e-4, 1e-4, 0.],
        zeta=[0., 1e-3, -1e-3], delta=[0., 1e-4, -1e-4])
    line.track(particles)
    actual = np.array([particles.x, particles.px, particles.y, particles.py,
                       particles.zeta, particles.delta])

    expected = _track_coordinates(xt.Drift(length=length, model='exact'))
    np.testing.assert_allclose(actual, expected, atol=5e-14, rtol=0)


def test_generic_thin_slicing_with_device_aperture():
    line = xt.Line(elements={
        'aper': xt.LimitRect(min_x=-0.01, max_x=0.01,
                             min_y=-0.01, max_y=0.01),
        'dev': xt.Device(length=1.5, model='exact'),
        'quad': xt.Quadrupole(length=0.5, k1=0.2),
    })
    line.get('dev').name_associated_aperture = 'aper'
    line.slice_thick_elements([xt.Strategy(xt.Teapot(2))])

    assert sum(isinstance(ee, xt.ThickSliceDevice)
               for ee in line.elements) == 3
    assert sum(isinstance(ee, xt.ThinSliceQuadrupole)
               for ee in line.elements) == 2
    assert line.get_length() == pytest.approx(2.)

    particles = xt.Particles(p0c=1e10, x=[0., 0.], px=[0., 0.02])
    line.track(particles)
    assert np.count_nonzero(particles.state == 1) == 1
    lost = particles.state == 0
    assert np.count_nonzero(lost) == 1
    # The second particle crosses the aperture inside the device; the check
    # before the third transport segment detects it at s = 1.25 m.
    np.testing.assert_allclose(particles.s[lost], 1.25, atol=1e-14, rtol=0)


@pytest.mark.parametrize('model', ['adaptive', 'expanded', 'exact'])
@pytest.mark.parametrize('sliced', [False, True])
def test_optimize_devices_as_drifts(model, sliced):
    line = xt.Line(elements={
        'zero': _make_device(0, model=model, **MISALIGNMENT),
        'keep': xt.Marker(),
        'before': xt.Drift(length=0.5, model=model),
        'dev': _make_device(1.5, model=model, **MISALIGNMENT),
        'after': xt.Drift(length=0.75, model=model),
    }, element_names=['zero', 'keep', 'before', 'dev', 'dev', 'after'])
    line.particle_ref = xt.Particles(p0c=1e10)
    if sliced:
        line.slice_thick_elements(slicing_strategies=[
            xt.Strategy(xt.Uniform(3, mode='thick'), element_type=xt.Device)])

    particles = line.build_particles(
        x=[1e-3, -2e-3], px=[0.03, -0.02],
        y=[-1e-3, 2e-3], py=[-0.02, 0.01], delta=[0.01, -0.01])
    expected = particles.copy()
    reference = xt.Line(elements=[xt.Drift(length=4.25, model=model)])
    reference.track(expected)

    line.optimize_for_tracking(compile=False, verbose=False, keep_markers=['keep'])

    assert list(line.element_names) == ['keep', 'before']
    assert isinstance(line['before'], xt.Drift)
    assert line['before'].length == 4.25
    assert line['before'].model == model
    line.track(particles)
    for field in ('x', 'px', 'y', 'py', 'zeta', 'delta'):
        np.testing.assert_allclose(
            getattr(particles, field), getattr(expected, field),
            atol=5e-14, rtol=0)


def test_optimize_device_replicas_as_drifts():
    line = xt.Line(elements={
        'dev': _make_device(1.5, model='exact', **MISALIGNMENT),
        'replica': xt.Replica(parent_name='dev'),
    }, element_names=['dev', 'replica', 'replica'])
    line.build_tracker()
    line.optimize_for_tracking(compile=False, verbose=False)

    assert len(line.element_names) == 1
    drift = line[line.element_names[0]]
    assert isinstance(drift, xt.Drift)
    assert drift.length == 4.5
    assert drift.model == 'exact'
