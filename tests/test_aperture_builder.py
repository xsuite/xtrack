import numpy as np

import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

from xtrack.aperture.builder import ApertureBuilder
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

    builder.place_pipe("install0", "type0", "drift::0")
    builder.place_pipe("install1", "type0", "drift::1", shift_x=0.3, shift_z=0.4, rot_z_rad=0.5)

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
    builder.place_pipe("install0", "type0", "drift", transformation=matrix)

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
            "drift",
            transformation=np.identity(4),
            shift_x=1.0,
        )
    except ValueError as error:
        assert "either `transformation` or transform components" in str(error)
    else:
        raise AssertionError("Expected place_pipe to reject mixed transform inputs.")


@for_all_test_contexts(excluding=("ContextPyopencl", "ContextCupy"))
def test_aperture_builder_accepts_shape_class_input(test_context):
    env = xt.Environment()
    drift = env.new("drift", xt.Drift, length=1.0)
    line = env.new_line(name="line", components=[drift])

    builder = ApertureBuilder(line)
    builder.new_profile("circ0", Circle, radius=1.5)
    builder.new_pipe("type0", positions=[builder.place_profile("circ0")])
    builder.place_pipe("install0", "type0", "drift")

    model = builder.build(context=test_context)

    assert model.profile_names == ["circ0"]
    xo.assert_allclose(model.profiles[0].shape.radius, 1.5, atol=1e-15, rtol=0)
