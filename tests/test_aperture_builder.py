import numpy as np
import pytest

import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

from xtrack.aperture.structures import Circle, SurveyData
from xtrack.aperture.transform import transform_matrix
from xtrack.slicing import Strategy, Uniform


@for_all_test_contexts(excluding=("ContextPyopencl", "ContextCupy"))
def test_aperture_containers_build_expected_ordering(test_context):
    env = xt.Environment()
    line = env.new_line(
        name="line",
        components=[env.new("drift", xt.Drift, length=1.0)] * 2,
    )
    second = env.aperture.profiles.new("second", Circle, radius=2.0)
    first = env.aperture.profiles.new("first", Circle, radius=1.0)
    pipe = env.aperture.pipes.new(
        "pipe",
        curvature=0.25,
        positions=[
            env.aperture.profiles.place(first, at=0.7),
            env.aperture.profiles.place(second, at=0.1),
            env.aperture.profiles.place(second, at=0.4),
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
    profile = env.aperture.profiles.new("profile", Circle, radius=1.0)
    pipe = env.aperture.pipes.new(
        "pipe", positions=[env.aperture.profiles.place(profile)])
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
    profile = env.aperture.profiles.new("profile", Circle, radius=1.0)
    pipe = env.aperture.pipes.new(
        "pipe", positions=[env.aperture.profiles.place(profile)])
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

    profile = env.aperture.profiles.new("profile", Circle, radius=1.0)
    pipe = env.aperture.pipes.new(
        "pipe", positions=[env.aperture.profiles.place(profile)])
    placed = line.aperture.place_pipe("installed", pipe, at=f"{reference}@{anchor}")

    xo.assert_allclose(placed.transformation.to_nplike(), expected, atol=1e-12, rtol=0)
    assert line._has_valid_tracker()
    assert list(env.lines) == ["line"]


def test_build_model_and_discard_model_freeze_and_restore_collections():
    env = xt.Environment()
    drift = env.new("drift", xt.Drift, length=1.0)
    line = env.new_line(
        name="line", components=[drift, drift])
    profile = env.aperture.profiles.new("profile", Circle, radius=1.0)
    pipe = env.aperture.pipes.new(
        "pipe", positions=[env.aperture.profiles.place(profile)])
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
        env.aperture.profiles.new("added", Circle, radius=2.0)

    env.aperture.discard_model()
    assert not env.aperture.is_built
    assert not line.aperture.is_built
    assert not line.aperture.empty
    added = env.aperture.profiles.new("added", Circle, radius=2.0)
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
    profile = env.aperture.profiles.new("profile", Circle, radius=1.0)
    pipe = env.aperture.pipes.new(
        "pipe",
        positions=[
            env.aperture.profiles.place(profile),
            env.aperture.profiles.place(profile, at=1.0),
        ],
    )
    line.aperture.place_pipe("installed", pipe, at="drift")

    assert not line.aperture.is_built

    line.aperture.cross_sections_at_s([0.5])

    assert line.aperture.is_built


def test_views_follow_authoritative_representation_and_build_refreshes_geometry():
    env = xt.Environment()
    line = env.new_line(
        name="line", components=[env.new("drift", xt.Drift, length=1.0)])
    profile = env.aperture.profiles.new("profile", Circle, radius=1.0)
    pipe = env.aperture.pipes.new(
        "pipe",
        positions=[
            env.aperture.profiles.place(profile),
            env.aperture.profiles.place(profile, at=1.0),
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

    before = line.aperture.cross_sections_at_s([0.5]).cross_section[0]
    profile.shape.radius = 2.0
    stale = line.aperture.cross_sections_at_s([0.5]).cross_section[0]
    xo.assert_allclose(stale, before, atol=1e-14, rtol=0)

    line.aperture.build_model()
    refreshed = line.aperture.cross_sections_at_s([0.5]).cross_section[0]
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


def test_line_and_environment_aperture_json_data_roundtrip():
    env = xt.Environment()
    line = env.new_line(
        name="line", components=[env.new("drift", xt.Drift, length=1.0)])
    profile = env.aperture.profiles.new("profile", Circle, radius=0.1)
    pipe = env.aperture.pipes.new(
        "pipe",
        positions=[
            env.aperture.profiles.place(profile),
            env.aperture.profiles.place(profile, at=1.0),
        ],
    )
    line.aperture.place_pipe("installed", pipe, at="drift")
    line.aperture.build_model()

    data = env.to_dict()
    position_data = data["lines"]["line"]["aperture"]["pipe_positions"][0]
    assert position_data["pipe"] == "pipe"
    assert "pipe_index" not in position_data
    assert "survey_index" not in position_data

    restored = xt.Environment.from_dict(data, with_progress=False)
    restored_line = restored.lines["line"]
    assert restored.aperture.profiles.keys() == ["profile"]
    assert restored.aperture.pipes.keys() == ["pipe"]
    assert restored_line.aperture.pipe_positions.keys() == ["installed"]
    xo.assert_allclose(
        restored_line.aperture.cross_sections_at_s([0.5]).cross_section[0, 0],
        [0.1, 0.0],
        atol=1e-12,
        rtol=0,
    )
