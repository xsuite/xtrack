import pickle

import numpy as np
import pytest

import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

from xtrack.aperture.structures import Circle, SurveyData
from xtrack.aperture.transform import transform_matrix
from xtrack.slicing import Strategy, Uniform


def test_line_aperture_creation_uses_environment_collections():
    env = xt.Environment()
    line = env.new_line(name='line', components=[env.new('drift', xt.Drift, length=1.0)])

    profile = line.aperture.new_profile('profile', Circle, radius=0.1)
    pipe = line.aperture.new_pipe('pipe', positions=[
        profile.place(at=0.0),
        profile.place(at=1.0),
    ])

    assert profile.name == env.aperture.profiles['profile'].name
    assert env.aperture.profiles['profile'].place(at=0.5).profile_index == 0
    assert env.aperture.place_profile('profile', at=0.5).profile_index == 0
    assert env.aperture.place_profile(profile, shift_s=0.5).shift_s == 0.5
    assert pipe.name == env.aperture.pipes['pipe'].name
    assert not hasattr(env.aperture.profiles, 'place')
    assert not hasattr(env.aperture.profiles, 'new')
    assert not hasattr(env.aperture.pipes, 'new')


def test_line_with_aperture_can_be_pickled():
    env = xt.Environment()
    line = env.new_line(name='line', components=[env.new('drift', xt.Drift, length=1.0)])
    profile = env.aperture.new_profile('profile', Circle, radius=0.1)
    pipe = env.aperture.new_pipe('pipe', positions=[profile.place()])
    line.aperture.place_pipe('installed', pipe, at='drift')

    restored = pickle.loads(pickle.dumps(line))

    assert restored.aperture.line is restored
    assert restored.env.aperture.env is restored.env
    assert restored.env.aperture.profiles['profile'].name == 'profile'
    assert restored.env.aperture.pipes['pipe'].name == 'pipe'
    assert restored.aperture.pipe_positions['installed'].name == 'installed'


@for_all_test_contexts(excluding=("ContextPyopencl", "ContextCupy"))
def test_aperture_containers_build_expected_ordering(test_context):
    env = xt.Environment()
    line = env.new_line(
        name="line",
        components=[env.new("drift", xt.Drift, length=1.0)] * 2,
    )
    second = env.aperture.new_profile("second", Circle, radius=2.0)
    first = env.aperture.new_profile("first", Circle, radius=1.0)
    pipe = env.aperture.new_pipe(
        "pipe",
        curvature=0.25,
        positions=[
            first.place(at=0.7),
            second.place(at=0.1),
            second.place(at=0.4),
        ],
    )
    line.aperture.context = test_context
    line.aperture.place_pipe("installed", pipe, at="drift::0")
    line.aperture.build_model()

    assert env.aperture.profiles.keys() == ["second", "first"]
    assert env.aperture.pipes.keys() == ["pipe"]
    assert line.aperture.pipe_positions.keys() == ["installed"]
    xo.assert_allclose(
        [position.profile_index for position in pipe.raw.positions],
        [0, 0, 1],
        atol=0,
        rtol=0,
    )


def test_place_pipe_accepts_transform_or_components():
    env = xt.Environment()
    line = env.new_line(
        name="line", components=[env.new("drift", xt.Drift, length=1.0)])
    profile = env.aperture.new_profile("profile", Circle, radius=1.0)
    pipe = env.aperture.new_pipe(
        "pipe", positions=[profile.place()])
    matrix = transform_matrix(shift_x=1.0, rot_z_rad=0.2)
    placed = line.aperture.place_pipe(
        "installed", pipe, at="drift", transformation=matrix)
    xo.assert_allclose(
        placed.transformation.to_nplike(), matrix, atol=1e-14, rtol=0)

    with pytest.raises(ValueError, match="either `transformation` or transform"):
        line.aperture.place_pipe(
            "invalid", pipe, at="drift", transformation=matrix, shift_x=1.0)


def test_place_pipe_at_anchor_and_sequence_names():
    env = xt.Environment()
    line = env.new_line(
        name="line", components=[env.new("drift", xt.Drift, length=2.0)])
    profile = env.aperture.new_profile("profile", Circle, radius=1.0)
    pipe = env.aperture.new_pipe(
        "pipe", positions=[profile.place()])
    placed = line.aperture.place_pipe(
        "installed", pipe, at=["drift@start", "drift@end"])

    assert len(placed) == 2
    assert list(line.aperture._pipe_position_name_to_index) == [
        "installed.0", "installed.1"]
    assert placed[0].survey_reference_name == "drift"
    assert placed[1].survey_reference_name == "drift"
    assert placed[0].survey_index == -1
    assert placed[1].survey_index == -1
    xo.assert_allclose(placed[1].transformation[2, 3], 2.0, atol=1e-14, rtol=0)


@pytest.mark.parametrize("element_type", ["bend", "rbend", "sliced_bend"])
@pytest.mark.parametrize("anchor", ["center", "end"])
def test_pipe_anchor_uses_element_local_survey(element_type, anchor):
    env = xt.Environment()
    env.new("lead", xt.Drift, length=0.5)
    if element_type == "rbend":
        env.new(
            "bend", xt.RBend, length_straight=2.0, angle=0.2,
            rbend_model="straight-body",
        )
    else:
        env.new("bend", xt.Bend, length=2.0, angle=0.2)

    line = env.new_line(name="line", components=["lead", "bend", "bend"])
    if element_type == "sliced_bend":
        line.slice_thick_elements([
            Strategy(slicing=Uniform(2, mode="thick"), name="bend")
        ])
        reference = "bend..0"
    else:
        reference = "bend::1"

    table = line.get_table()
    survey = line.survey()
    survey_data = SurveyData.from_survey_table(survey, line)
    element_index = table.rows.get_index(reference)
    survey_index = survey.rows.get_index(reference)
    anchor_s = (table.s_center[element_index] if anchor == "center"
                else table.s_end[element_index])

    reference_pose = np.identity(4)
    reference_pose[:3, :3] = survey.E_matrix[survey_index]
    reference_pose[:3, 3] = survey.XYZ[survey_index]
    anchor_pose = survey_data.resample([float(anchor_s)]).pose.to_nplike()[0]
    expected = np.linalg.inv(reference_pose) @ anchor_pose

    profile = env.aperture.new_profile("profile", Circle, radius=1.0)
    pipe = env.aperture.new_pipe(
        "pipe", positions=[profile.place()])
    placed = line.aperture.place_pipe("installed", pipe, at=f"{reference}@{anchor}")

    xo.assert_allclose(placed.transformation.to_nplike(), expected, atol=1e-12, rtol=0)
    assert line._has_valid_tracker()
    assert list(env.lines) == ["line"]


def test_build_model_and_discard_model_freeze_and_restore_collections():
    env = xt.Environment()
    drift = env.new("drift", xt.Drift, length=1.0)
    line = env.new_line(
        name="line", components=[drift, drift])
    profile = env.aperture.new_profile("profile", Circle, radius=1.0)
    pipe = env.aperture.new_pipe(
        "pipe", positions=[profile.place()])
    assert line.aperture.empty

    line.aperture.place_pipe("installed", pipe, at="drift::0")
    assert not line.aperture.empty

    line.aperture.build_model()

    assert env.aperture.is_built
    assert line.aperture.is_built
    assert not line.aperture.empty
    with pytest.raises(RuntimeError, match="after the line aperture is built"):
        line.aperture.place_pipe("other", pipe, at="drift")
    with pytest.raises(RuntimeError, match="env.aperture.discard_model"):
        env.aperture.new_profile("added", Circle, radius=2.0)

    env.aperture.discard_model()
    assert not env.aperture.is_built
    assert not line.aperture.is_built
    assert not line.aperture.empty
    added = env.aperture.new_profile("added", Circle, radius=2.0)
    line.aperture.place_pipe("other", pipe, at="drift::1")

    line.aperture.build_model()
    assert added.name == "added"
    assert profile.name == "profile"
    assert pipe.name == "pipe"
    assert env.aperture.profiles.keys() == ["profile", "added"]
    assert line.aperture.pipe_positions.keys() == ["installed", "other"]


def test_aperture_computation_builds_model_implicitly():
    env = xt.Environment()
    line = env.new_line(
        name="line", components=[env.new("drift", xt.Drift, length=1.0)])
    profile = env.aperture.new_profile("profile", Circle, radius=1.0)
    pipe = env.aperture.new_pipe(
        "pipe",
        positions=[
            profile.place(),
            profile.place(at=1.0),
        ],
    )
    line.aperture.place_pipe("installed", pipe, at="drift")

    assert not line.aperture.is_built

    line.aperture.get_cross_sections_at_s([0.5])

    assert line.aperture.is_built


def test_views_follow_authoritative_representation_and_build_refreshes_geometry():
    env = xt.Environment()
    line = env.new_line(
        name="line", components=[env.new("drift", xt.Drift, length=1.0)])
    profile = env.aperture.new_profile("profile", Circle, radius=1.0)
    pipe = env.aperture.new_pipe(
        "pipe",
        positions=[
            profile.place(),
            profile.place(at=1.0),
        ],
    )
    line.aperture.place_pipe("installed", pipe, at="drift")
    installed = line.aperture.pipe_positions["installed"]

    assert env.aperture._struct is None
    assert env.aperture._profiles_unbuilt is not None
    assert line.aperture._model is None
    assert line.aperture._pipe_positions_unbuilt is not None
    assert installed.survey_index == -1

    line.aperture.build_model()
    assert env.aperture._struct is not None
    assert env.aperture._profiles_unbuilt is None
    assert line.aperture._model is not None
    assert line.aperture._pipe_positions_unbuilt is None
    assert installed.survey_index == 0
    assert installed.pipe.name == "pipe"

    before = line.aperture.get_cross_sections_at_s([0.5]).cross_section[0]
    profile.shape.radius = 2.0
    stale = line.aperture.get_cross_sections_at_s([0.5]).cross_section[0]
    xo.assert_allclose(stale, before, atol=1e-14, rtol=0)

    line.aperture.build_model()
    refreshed = line.aperture.get_cross_sections_at_s([0.5]).cross_section[0]
    xo.assert_allclose(
        np.max(np.linalg.norm(refreshed, axis=1)), 2.0, atol=1e-12, rtol=0)

    env.aperture.discard_model()
    assert env.aperture._struct is None
    assert env.aperture._profiles_unbuilt is not None
    assert line.aperture._model is None
    assert line.aperture._pipe_positions_unbuilt is not None
    assert installed.survey_index == -1
    assert profile.shape.radius == 2.0
    assert installed.pipe.name == "pipe"


def test_line_and_environment_aperture_json_data_roundtrip(tmp_path):
    env = xt.Environment()
    line = env.new_line(
        name="line", components=[env.new("drift", xt.Drift, length=1.0)])
    profile = env.aperture.new_profile("profile", Circle, radius=0.1)
    pipe = env.aperture.new_pipe(
        "pipe",
        positions=[
            profile.place(),
            profile.place(at=1.0),
        ],
    )
    line.aperture.place_pipe("installed", pipe, at="drift")
    line.aperture.build_model()

    assert 'aperture' not in env.to_dict()
    assert 'aperture' not in env.to_dict()['lines']['line']
    assert 'aperture' not in line.to_dict()
    assert 'aperture_environment' not in line.to_dict()

    data = env.to_dict(include_aperture=True)
    position_data = data["lines"]["line"]["aperture"]["pipe_positions"][0]
    assert position_data["pipe"] == "pipe"
    assert "pipe_index" not in position_data
    assert "survey_index" not in position_data

    other_line = env.new_line(name='other', components=['drift'])
    env_aperture_data = env.aperture.to_dict()
    assert set(env_aperture_data['lines']) == {'line', 'other'}
    assert env_aperture_data['environment'] == data['aperture']
    assert env.aperture.to_dict(include_line_models=False) == data['aperture']

    line_aperture_data = line.aperture.to_dict()
    assert line_aperture_data['aperture'] == data['lines']['line']['aperture']
    assert line_aperture_data['environment'] == data['aperture']
    assert 'lines' not in line_aperture_data

    env_file = tmp_path / 'environment.json'
    line_file = tmp_path / 'line.json'
    aperture_file = tmp_path / 'aperture.json'
    line_aperture_file = tmp_path / 'line_aperture.json'
    env.to_json(env_file)
    line.to_json(line_file)
    env.aperture.to_json(aperture_file)
    line.aperture.to_json(line_aperture_file)
    assert 'aperture' not in xt.json.load(env_file)
    assert 'aperture' not in xt.json.load(line_file)
    loaded_aperture_data = xt.json.load(aperture_file)
    assert set(loaded_aperture_data['lines']) == {'line', 'other'}
    assert loaded_aperture_data['environment']['profile_names'] == ['profile']
    assert loaded_aperture_data['lines']['line']['pipe_positions'][0]['pipe'] == 'pipe'
    env.aperture.to_json(aperture_file, include_line_models=False)
    assert xt.json.load(aperture_file) == loaded_aperture_data['environment']
    assert 'lines' not in xt.json.load(line_aperture_file)
    assert xt.json.load(line_aperture_file)['aperture']['pipe_positions'][0]['pipe'] == 'pipe'

    env.to_json(env_file, include_aperture=True)
    assert 'aperture' in xt.json.load(env_file)
    assert 'aperture' in xt.json.load(env_file)['lines']['line']
    line.to_json(line_file, include_aperture=True)
    assert 'aperture' in xt.json.load(line_file)
    assert 'aperture_environment' in xt.json.load(line_file)

    restored = xt.Environment.from_dict(data, with_progress=False)
    restored_line = restored.lines["line"]
    assert restored.aperture.profiles.keys() == ["profile"]
    assert restored.aperture.pipes.keys() == ["pipe"]
    assert restored_line.aperture.pipe_positions.keys() == ["installed"]
    xo.assert_allclose(
        restored_line.aperture.get_cross_sections_at_s([0.5]).cross_section[0, 0],
        [0.1, 0.0],
        atol=1e-12,
        rtol=0,
    )


def test_tolerances_follow_profile_and_pipe_positions_through_build_and_json():
    env = xt.Environment()
    line = env.new_line(name='line', components=[
        env.new('drift', xt.Drift, length=10.0),
    ])
    profile = env.aperture.new_profile('circle', Circle, radius=1.0)
    pipe = env.aperture.new_pipe('pipe', positions=[
        profile.place(at=0.0, tol_r=0.1, tol_x=0.2, tol_y=0.3),
        profile.place(at=10.0, tol_r=0.5, tol_x=0.6, tol_y=0.7),
    ])
    installed = line.aperture.place_pipe(
        'installed', pipe, at='drift', tol_r=0.05, tol_x=0.06, tol_y=0.07,
    )

    assert not hasattr(profile, 'tol_r')
    assert pipe[0].tol_r == 0.1
    assert installed.tol_r == 0.05

    line.aperture.build_model()
    sections = line.aperture.get_cross_sections_at_s([0.0, 2.5, 5.0, 7.5, 10.0])
    xo.assert_allclose(sections.tol_r, [0.15, 0.25, 0.35, 0.45, 0.55], atol=1e-12)
    xo.assert_allclose(sections.tol_x, [0.26, 0.36, 0.46, 0.56, 0.66], atol=1e-12)
    xo.assert_allclose(sections.tol_y, [0.37, 0.47, 0.57, 0.67, 0.77], atol=1e-12)

    restored = xt.Environment.from_dict(
        env.to_dict(include_aperture=True), with_progress=False)
    restored_pipe = restored.aperture.pipes['pipe']
    restored_installation = restored.lines['line'].aperture.pipe_positions['installed']
    assert restored_pipe[0].tol_r == 0.1
    assert restored_installation.tol_r == 0.05

    env.aperture.discard_model()
    assert pipe[0].tol_r == 0.1
    assert line.aperture.pipe_positions['installed'].tol_r == 0.05


def test_element_computations_sample_inside_thick_elements():
    env = xt.Environment()
    line = env.new_line(name='line', components=[
        env.new('drift', xt.Drift, length=1.0),
    ])
    line.particle_ref = xt.Particles(p0c=7e12)
    line.twiss_default['method'] = '4d'
    line.twiss_default['init'] = xt.TwissInit(betx=1.0, bety=1.0)

    profile = env.aperture.new_profile('circle', Circle, radius=0.04)
    pipe = env.aperture.new_pipe('pipe', positions=[
        profile.place(at=0.0),
        profile.place(at=1.0),
    ])
    line.aperture.place_pipe('installed', pipe, at='drift')
    line.aperture.build_model()

    sigmas, _ = line.aperture.get_aperture_sigmas_at_element(
        'drift', with_progress=False,
    )
    hvd, _ = line.aperture.get_hvd_aperture_sigmas_at_element(
        'drift', with_progress=False,
    )
    envelopes, _ = line.aperture.get_envelope_at_element(
        'drift', sigmas=1.0, with_progress=False,
    )
    sections = line.aperture.get_cross_sections_at_element('drift', resolution=None)

    for table in (sigmas, hvd, envelopes, sections):
        assert len(table.s) == 2
        assert np.all((0 < table.s) & (table.s < 1.0))
