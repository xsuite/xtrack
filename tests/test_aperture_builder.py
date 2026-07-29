import numpy as np

import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

from xtrack.aperture.builder import ApertureBuilder
from xtrack.aperture.plot import pipe_projection, pipe_solid
from xtrack.aperture.structures import Circle
from xtrack.aperture.transform import matrix_to_transform, transform_matrix


@for_all_test_contexts(excluding=("ContextPyopencl", "ContextCupy"))
def test_aperture_builder_builds_model_with_expected_ordering(test_context):
    env = xt.Environment()
    drift = env.new("drift", xt.Drift, length=1.0)
    line = env.new_line(name="line", components=[drift, drift])

    builder = ApertureBuilder(line)
    builder.new_profile("rect0", "Rectangle", tol_r=0.1, tol_x=0.2, tol_y=0.3, half_width=2.0, half_height=1.0)
    builder.new_profile("circ0", "Circle", radius=3.0)

    type0 = builder.new_pipe(
        "type0",
        curvature=0.5,
        positions=[
            builder.place_profile("rect0", shift_s=5.0, shift_x=1.0),
            builder.place_profile("circ0", shift_s=1.0, rot_s_rad=0.4),
        ],
    )
    type0.place_profile("rect0", shift_s=3.0, shift_y=-2.0, rot_x_rad=0.2)

    builder.place_pipe("install0", "type0", at="drift::0")
    builder.place_pipe("install1", "type0", at="drift::1", shift_x=0.3, shift_z=0.4, rot_z_rad=0.5)

    model = builder.build(context=test_context)

    assert model.profile_names == ["rect0", "circ0"]
    assert model.pipe_names == ["type0"]
    assert model.pipe_position_names == ["install0", "install1"]

    built_type = model.pipes[0]
    assert built_type.curvature == 0.5
    xo.assert_allclose([pp.shift_s for pp in built_type.positions], [1.0, 3.0, 5.0], atol=0, rtol=0)
    xo.assert_allclose([pp.profile_index for pp in built_type.positions], [1, 0, 0], atol=0, rtol=0)
    xo.assert_allclose([pp.shift_x for pp in built_type.positions], [0.0, 0.0, 1.0], atol=0, rtol=0)
    xo.assert_allclose([pp.shift_y for pp in built_type.positions], [0.0, -2.0, 0.0], atol=0, rtol=0)
    xo.assert_allclose([pp.rot_x_rad for pp in built_type.positions], [0.0, 0.2, 0.0], atol=0, rtol=0)
    xo.assert_allclose([pp.rot_s_rad for pp in built_type.positions], [0.4, 0.0, 0.0], atol=0, rtol=0)

    assert model.pipe_positions[0].survey_reference_name == "drift::0"
    assert model.pipe_positions[0].survey_index == list(line.survey().name).index("drift::0")
    assert model.pipe_positions[1].survey_reference_name == "drift::1"
    assert model.pipe_positions[1].survey_index == list(line.survey().name).index("drift::1")

    install1_transform = matrix_to_transform(model.pipe_positions[1].transformation.to_nplike())
    xo.assert_allclose(install1_transform.shift_x, 0.3, atol=1e-15, rtol=0)
    xo.assert_allclose(install1_transform.shift_y, 0.0, atol=1e-15, rtol=0)
    xo.assert_allclose(install1_transform.shift_z, 0.4, atol=1e-15, rtol=0)
    xo.assert_allclose(install1_transform.rot_y_rad, 0.0, atol=1e-15, rtol=0)
    xo.assert_allclose(install1_transform.rot_x_rad, 0.0, atol=1e-15, rtol=0)
    xo.assert_allclose(install1_transform.rot_z_rad, 0.5, atol=1e-15, rtol=0)


@for_all_test_contexts(excluding=("ContextPyopencl", "ContextCupy"))
def test_aperture_builder_place_type_accepts_explicit_matrix(test_context):
    env = xt.Environment()
    drift = env.new("drift", xt.Drift, length=1.0)
    line = env.new_line(name="line", components=[drift])

    builder = ApertureBuilder(line)
    builder.new_profile("circ0", "Circle", radius=1.0)
    builder.new_pipe("type0", positions=[builder.place_profile("circ0", shift_s=0.2)])

    matrix = transform_matrix(shift_x=1.0, shift_y=-2.0, shift_z=3.0, rot_y_rad=0.1, rot_x_rad=-0.2, rot_z_rad=0.3)
    builder.place_pipe("install0", "type0", at="drift", transformation=matrix)

    model = builder.build(context=test_context)
    built_transform = matrix_to_transform(model.pipe_positions[0].transformation.to_nplike())

    xo.assert_allclose(built_transform.shift_x, 1.0, atol=1e-15, rtol=0)
    xo.assert_allclose(built_transform.shift_y, -2.0, atol=1e-15, rtol=0)
    xo.assert_allclose(built_transform.shift_z, 3.0, atol=1e-15, rtol=0)
    xo.assert_allclose(built_transform.rot_y_rad, 0.1, atol=1e-15, rtol=0)
    xo.assert_allclose(built_transform.rot_x_rad, -0.2, atol=1e-15, rtol=0)
    xo.assert_allclose(built_transform.rot_z_rad, 0.3, atol=1e-15, rtol=0)


def test_aperture_builder_place_type_rejects_mixed_transform_inputs():
    env = xt.Environment()
    drift = env.new("drift", xt.Drift, length=1.0)
    line = env.new_line(name="line", components=[drift])

    builder = ApertureBuilder(line)
    builder.new_profile("circ0", "Circle", radius=1.0)
    builder.new_pipe("type0", positions=[builder.place_profile("circ0")])

    try:
        builder.place_pipe(
            "install0",
            "type0",
            at="drift",
            transformation=np.identity(4),
            shift_x=1.0,
        )
    except ValueError as error:
        assert "either `transformation` or transform components" in str(error)
    else:
        raise AssertionError("Expected place_pipe to reject mixed transform inputs.")


@for_all_test_contexts(excluding=("ContextPyopencl", "ContextCupy"))
def test_aperture_builder_at_anchor_keeps_survey_reference_and_encodes_offset(test_context):
    env = xt.Environment()
    drift = env.new("drift", xt.Drift, length=2.0)
    line = env.new_line(name="line", components=[drift])

    builder = ApertureBuilder(line)
    builder.new_profile("circ0", "Circle", radius=1.0)
    builder.new_pipe("type0", positions=[builder.place_profile("circ0")])

    builder.place_pipe("at_start", "type0", at="drift@start")
    builder.place_pipe("at_center", "type0", at="drift@center")
    builder.place_pipe("at_end", "type0", at="drift@end", shift_x=0.3)

    model = builder.build(context=test_context)

    for pipe_position in model.pipe_positions:
        assert pipe_position.survey_reference_name == "drift"

    start_transform = matrix_to_transform(model.pipe_positions[0].transformation.to_nplike())
    center_transform = matrix_to_transform(model.pipe_positions[1].transformation.to_nplike())
    end_transform = matrix_to_transform(model.pipe_positions[2].transformation.to_nplike())

    xo.assert_allclose(start_transform.shift_z, 0.0, atol=1e-15, rtol=0)
    xo.assert_allclose(center_transform.shift_z, 1.0, atol=1e-15, rtol=0)
    xo.assert_allclose(end_transform.shift_z, 2.0, atol=1e-15, rtol=0)
    xo.assert_allclose(end_transform.shift_x, 0.3, atol=1e-15, rtol=0)


def test_aperture_builder_exposes_blueprints():
    env = xt.Environment()
    drift = env.new("drift", xt.Drift, length=1.0)
    line = env.new_line(name="line", components=[drift])

    builder = ApertureBuilder(line)
    builder.new_profile("circ0", "Circle", radius=1.0)
    pipe = builder.new_pipe(
        "type0",
        positions=[
            builder.place_profile("circ0", shift_s=0.0),
            builder.place_profile("circ0", shift_s=1.0),
        ],
    )
    pipe_position = builder.place_pipe("install0", "type0", at="drift")

    assert builder.profiles["circ0"].shape.radius == 1.0
    assert builder.pipes["type0"] is pipe
    assert builder.pipe_positions == [pipe_position]


def test_aperture_builder_places_pipe_at_multiple_references():
    env = xt.Environment()
    drift = env.new("drift", xt.Drift, length=1.0)
    line = env.new_line(name="line", components=[drift, drift])

    builder = ApertureBuilder(line)
    builder.new_profile("circ0", "Circle", radius=1.0)
    pipe = builder.new_pipe("type0", positions=[builder.place_profile("circ0")])

    placed = pipe.place("install", at=["drift::0", "drift::1"])

    assert [pipe_position.name for pipe_position in placed] == ["install.0", "install.1"]
    assert [pipe_position.pipe_name for pipe_position in placed] == ["type0", "type0"]
    assert [pipe_position.survey_reference for pipe_position in placed] == ["drift::0", "drift::1"]
    assert builder.pipe_positions == placed


def test_pipe_projection_uses_curvature_for_z_planes_and_straight_frame_for_s_planes():
    env = xt.Environment()
    drift = env.new("drift", xt.Drift, length=1.0)
    line = env.new_line(name="line", components=[drift])

    builder = ApertureBuilder(line)
    builder.new_profile("circ0", "Circle", radius=0.1)
    pipe = builder.new_pipe(
        "type0",
        curvature=np.pi / 2,
        positions=[
            builder.place_profile("circ0", shift_s=0.0),
            builder.place_profile("circ0", shift_s=1.0),
        ],
    )
    projection_zx = pipe_projection(pipe, np.identity(4), plane="zx", max_curve_angle_rad=np.deg2rad(10))
    projection_sx = pipe_projection(pipe, np.identity(4), plane="sx", max_curve_angle_rad=np.deg2rad(10))

    assert len(projection_zx.polygons) == 1
    assert len(projection_sx.polygons) == 1
    assert np.any(np.abs(projection_zx.axis[:, 1]) > 1e-3)
    xo.assert_allclose(projection_sx.axis[:, 1], 0.0, atol=1e-14, rtol=0)
    xo.assert_allclose(projection_sx.axis[:, 0], [0.0, 1.0], atol=1e-14, rtol=0)


def test_pipe_solid_uses_curvature_for_curved_frame_and_straight_frame_for_solid_plot():
    env = xt.Environment()
    drift = env.new("drift", xt.Drift, length=1.0)
    line = env.new_line(name="line", components=[drift])

    builder = ApertureBuilder(line)
    builder.new_profile("circ0", "Circle", radius=0.1)
    pipe = builder.new_pipe(
        "type0",
        curvature=np.pi / 2,
        positions=[
            builder.place_profile("circ0", shift_s=0.0),
            builder.place_profile("circ0", shift_s=1.0),
        ],
    )
    solid_curved = pipe_solid(pipe, frame="curved", len_points=8, max_curve_angle_rad=np.deg2rad(10))
    solid_straight = pipe_solid(pipe, frame="straight", len_points=8, max_curve_angle_rad=np.deg2rad(10))

    assert len(solid_curved.faces) > len(solid_straight.faces)
    assert len(solid_curved.profile_rings) == 2
    assert len(solid_straight.profile_rings) == 2
    assert len(solid_curved.longitudinal_lines) == len(solid_curved.profile_rings[0])
    assert len(solid_straight.longitudinal_lines) == len(solid_straight.profile_rings[0])
    assert np.any(np.abs(solid_curved.axis[:, 0]) > 1e-3)
    xo.assert_allclose(solid_straight.axis[:, 0], 0.0, atol=1e-14, rtol=0)
    xo.assert_allclose(solid_straight.axis[:, 2], [0.0, 1.0], atol=1e-14, rtol=0)


@for_all_test_contexts(excluding=("ContextPyopencl", "ContextCupy"))
def test_aperture_builder_accepts_shape_class_input(test_context):
    env = xt.Environment()
    drift = env.new("drift", xt.Drift, length=1.0)
    line = env.new_line(name="line", components=[drift])

    builder = ApertureBuilder(line)
    builder.new_profile("circ0", Circle, radius=1.5)
    builder.new_pipe("type0", positions=[builder.place_profile("circ0")])
    builder.place_pipe("install0", "type0", at="drift")

    model = builder.build(context=test_context)

    assert model.profile_names == ["circ0"]
    xo.assert_allclose(model.profiles[0].shape.radius, 1.5, atol=1e-15, rtol=0)
