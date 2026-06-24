import itertools

import json
from pathlib import Path

import numpy as np
import pytest

import xobjects as xo
import xtrack as xt
from cpymad.madx import Madx

from xobjects.general import allclose_with_outliers
from xobjects.test_helpers import for_all_test_contexts, requires_context, skip_if_forbid_compile
from xtrack.aperture.aperture import Aperture, ProfilesView, _split_wrapped_s_interval
from xtrack.aperture.builder import ApertureBuilder
from xtrack.aperture.views import PipePositionsView, PipesView
from xtrack.aperture.structures import (
    ApertureModel, Pipe, Circle, FloatType, Polygon, Profile,
    ProfilePosition, Rectangle, SurveyData, PipePosition
)
from xtrack.aperture.transform import matrix_to_transform, transform_matrix
from xdeps.table import Table

TOY_RING_SEQUENCE = """
    ! Toy Ring, 4 arcs

    l_arc = 3;  ! length of the arc
    l_quad = 0.3;  ! length of the quads
    l_drift = 1;  ! length of the straight section drifts

    qf = 0.1;  ! qf strength
    qd = -0.7;  ! qd strength
    angle_arc = pi / 2;  ! arcs 90°

    mb: sbend, angle = angle_arc, l = l_arc, apertype=circle, aperture={0.1}, aper_offset={0.003, 0};
    mqf: quadrupole, k1 = qf, l = l_quad, apertype=rectangle, aperture={0.08, 0.04};
    mqd: quadrupole, k1 = qd, l = l_quad, apertype=ellipse, aperture={0.04, 0.08};
    ds: drift, l = l_drift;
    ap_ds: marker, apertype=rectellipse, aperture={0.022, 0.01715, 0.022, 0.022}, aper_tol={9e-4, 8e-4, 5e-4};
    dsa: line = (ap_ds, ds, ap_ds);

    ss_f: line = (dsa, mqf, dsa);
    ss_d: line = (dsa, mqd, dsa);

    ring: line = (ss_f, mb, ss_d, mb, ss_f, mb, ss_d, mb);

    beam, particle=proton, pc=1.2e9;
    use, period=ring;
"""


def _polygon_area(points):
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(np.dot(x[:-1], y[1:]) + x[-1] * y[0] - np.dot(y[:-1], x[1:]) - y[-1] * x[0])


def _polygon_signed_area(points):
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * (
        np.dot(x[:-1], y[1:]) + x[-1] * y[0]
        - np.dot(y[:-1], x[1:]) - y[-1] * x[0]
    )


def _best_cyclic_shift(points_ref, points_other):
    best_shift = 0
    best_cost = np.inf
    for shift in range(len(points_ref)):
        cost = np.sum((points_ref - np.roll(points_other, -shift, axis=0)) ** 2)
        if cost < best_cost:
            best_cost = cost
            best_shift = shift
    return best_shift, best_cost


def _pose_apply_polygon(points, pose):
    poly_h = np.column_stack([
        points[:, 0],
        points[:, 1],
        np.zeros(len(points)),
        np.ones(len(points)),
    ])
    return (pose @ poly_h.T).T[:, :3]


def _plane_normal(pose):
    return pose[:3, 2]


def _orientation_aware_indices(num_points, shift, reverse_orientation):
    jj = np.arange(num_points)
    if not reverse_orientation:
        return (jj + shift) % num_points
    return num_points - 1 - ((jj + shift) % num_points)


TEST_DATA_DIR = Path(__file__).resolve().parent.parent / 'test_data'


@pytest.fixture(scope='module')
def context():
    return xo.ContextCpu()


@pytest.mark.parametrize(
    'kwargs',
    [
        {},
        {'shift_x': 1.2, 'shift_y': -0.3, 'shift_z': 4.5},
        {'rot_y_rad': 0.2, 'rot_x_rad': -0.1, 'rot_z_rad': 0.3},
        {'shift_x': -1.5, 'shift_y': 2.0, 'shift_z': 0.7, 'rot_y_rad': 0.25, 'rot_x_rad': -0.15, 'rot_z_rad': 0.35},
    ],
)
def test_matrix_to_transform_roundtrip(kwargs):
    matrix = transform_matrix(**kwargs)
    out = matrix_to_transform(matrix)
    for key, value in {
        'shift_x': 0.0,
        'shift_y': 0.0,
        'shift_z': 0.0,
        'rot_y_rad': 0.0,
        'rot_x_rad': 0.0,
        'rot_z_rad': 0.0,
        **kwargs,
    }.items():
        xo.assert_allclose(getattr(out, key), value, atol=1e-14, rtol=0)


def test_rot_y_pi_projection_reverses_polygon_order_and_breaks_cyclic_matching():
    left_pose = transform_matrix()
    right_pose = transform_matrix(shift_z=1.0, rot_y_rad=-np.pi)

    poly_local = np.array([
        [1.0, 0.0],
        [0.5, 1.0],
        [-0.5, 1.0],
        [-1.0, 0.0],
        [-0.3, -0.8],
        [0.3, -0.8],
        [1.0, 0.0],
    ])

    left_world = _pose_apply_polygon(poly_local, left_pose)
    right_world = _pose_apply_polygon(poly_local, right_pose)

    left_xy = left_world[:, :2]
    right_xy = right_world[:, :2]
    cut_plane_normal = np.array([0.0, 0.0, 1.0])
    left_projection_sign = np.dot(_plane_normal(left_pose), cut_plane_normal)
    right_projection_sign = np.dot(_plane_normal(right_pose), cut_plane_normal)

    assert _polygon_signed_area(left_xy) > 0
    assert _polygon_signed_area(right_xy) < 0
    assert left_projection_sign * right_projection_sign < 0

    reverse_orientation = left_projection_sign * right_projection_sign < 0

    shift_same_order, cost_same_order = _best_cyclic_shift(left_xy[:-1], right_xy[:-1])
    shift_reversed, cost_reversed = _best_cyclic_shift(left_xy[:-1], right_xy[-2::-1])

    assert cost_same_order > 0.1
    xo.assert_allclose(cost_reversed, 0.0, atol=1e-14, rtol=0)

    mid_bad = 0.5 * (
        left_world[:-1, :2]
        + np.roll(right_world[:-1, :2], -shift_same_order, axis=0)
    )

    wrong_shift_with_reversal = right_world[
        _orientation_aware_indices(len(right_world) - 1, shift_same_order, reverse_orientation),
        :2,
    ]
    mid_wrong_shift = 0.5 * (left_world[:-1, :2] + wrong_shift_with_reversal)

    right_world_oriented = right_world[
        _orientation_aware_indices(len(right_world) - 1, shift_reversed, reverse_orientation),
        :2,
    ]
    mid_good = 0.5 * (left_world[:-1, :2] + right_world_oriented)

    xo.assert_allclose(mid_good, left_xy[:-1], atol=1e-14, rtol=0)
    assert _polygon_area(mid_bad) < 0.6 * _polygon_area(left_xy)
    assert _polygon_area(mid_wrong_shift) < 0.6 * _polygon_area(left_xy)


@pytest.fixture(scope="module")
def kernels(context):
    Profile.compile_class_kernels(context, only_if_needed=True)
    Polygon.compile_class_kernels(context, only_if_needed=True)
    SurveyData.compile_class_kernels(context, only_if_needed=True)
    ApertureModel.compile_class_kernels(context, only_if_needed=True)
    return context


def _expected_profile_bounds_from_table(table_rows, *, skip_row):
    expected = []
    for row in table_rows:
        if skip_row(row):
            continue
        expected.append((row.s_start, row.name))
        if not np.isclose(row.s_end, row.s_start):
            expected.append((row.s_end, row.name))
    return expected


def _make_pipe_table_test_ring(test_context, *, extra_pipe_positions=None):
    env = xt.Environment()

    num_bends = 8
    ring_length = 80.0
    bend_length = ring_length / num_bends
    bend_angle = 2 * np.pi / num_bends

    bend = env.new('bend', xt.Bend, length=bend_length, angle=bend_angle, k0=0)
    line = env.new_line(name='ring', components=[bend] * num_bends)
    sv = line.survey()

    profiles = [Profile(shape=Circle(radius=0.03), tol_r=0, tol_x=0, tol_y=0)]
    pipes = [
        Pipe(curvature=bend_angle / bend_length, positions=[
            ProfilePosition(profile_index=0, shift_s=0.0),
            ProfilePosition(profile_index=0, shift_s=bend_length),
        ]),
    ]
    pipe_names = ['pipe_type']
    pipe_position_names = ['pipe_regular', 'pipe_wrapped']
    pipe_positions = [
        PipePosition(
            pipe_index=0,
            survey_reference_name=sv.name[1],
            survey_index=1,
            transformation=transform_matrix(),
        ),
        PipePosition(
            pipe_index=0,
            survey_reference_name=sv.name[num_bends - 1],
            survey_index=num_bends - 1,
            transformation=transform_matrix(shift_z=5.0),
        ),
    ]

    if extra_pipe_positions is not None:
        for name, survey_reference_name, shift_z in extra_pipe_positions:
            pipe_position_names.append(name)
            pipe_positions.append(
                PipePosition(
                    pipe_index=0,
                    survey_reference_name=survey_reference_name,
                    survey_index=sv.name.tolist().index(survey_reference_name),
                    transformation=transform_matrix(shift_z=shift_z),
                ),
            )

    model = ApertureModel(
        line_name='ring',
        pipe_positions=pipe_positions,
        pipes=pipes,
        profiles=profiles,
        pipe_names=pipe_names,
        pipe_position_names=pipe_position_names,
        profile_names=['p0'],
        _context=test_context,
    )
    return line, model


def _make_transition_test_aperture(test_context):
    env = xt.Environment()
    line = env.new_line(name='line', components=[env.new('drift', xt.Drift, length=10.0)])
    sv = line.survey()

    model = ApertureModel(
        line=line,
        pipe_positions=[
            PipePosition(
                pipe_index=0,
                survey_reference_name=sv.name[0],
                survey_index=0,
                transformation=transform_matrix(),
            ),
        ],
        pipes=[Pipe(curvature=0.0, positions=[
            ProfilePosition(profile_index=0, shift_s=3.0),
            ProfilePosition(profile_index=0, shift_s=7.0),
        ])],
        profiles=[Profile(shape=Circle(radius=1.0), tol_r=0, tol_x=0, tol_y=0)],
        pipe_names=['pipe0'],
        pipe_position_names=['pipe0'],
        profile_names=['profile0'],
        _context=test_context,
    )
    return Aperture(line=line, model=model, context=test_context)


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_aperture_does_not_add_columns_to_survey(test_context):
    aperture = _make_transition_test_aperture(test_context)

    assert 'angle' not in aperture.survey._col_names
    assert 'rot_s_rad' not in aperture.survey._col_names


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_from_line_with_aperture_type_bounds(test_context):
    mad = Madx(stdout=None)
    mad.input(TOY_RING_SEQUENCE)
    ring = xt.Line.from_madx_sequence(mad.sequence.ring, enable_layout_data=True)

    aperture_model = Aperture.from_line_with_madx_metadata(ring, context=test_context)
    bounds_table = aperture_model.get_bounds_table()
    table_rows = ring.get_table().cols['s_start', 's_end', 'name', 'element_type'].rows[1:-2].rows # trim MAD-X endpoints
    expected = _expected_profile_bounds_from_table(
        table_rows,
        skip_row=lambda row: row.element_type == 'Drift',
    )

    assert len(bounds_table.name) == len(expected)
    for idx, ((expected_s, expected_name), bound_s, pipe_name) in enumerate(
        zip(expected, bounds_table.s, bounds_table.pipe_name)
    ):
        xo.assert_allclose(bound_s, expected_s, atol=1e-6)
        assert expected_name.startswith(pipe_name), f'mismatch at bound row {idx}'


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_zigzag_iterator_wrap_and_bounds(test_context):
    skip_if_forbid_compile()

    ZIGZAG_TEST_SOURCE = r"""
        #include "xtrack/aperture/headers/zigzag_iterate.h"

        void fill_zigzag_sequence(
            uint32_t start,
            uint32_t upper_bound,
            int8_t wrap,
            int32_t* out,
            uint32_t len_out
        ) {
            ZigZagIterator it = zigzag_iterator_new(start, upper_bound, wrap);

            uint32_t i_out = 0;
            if (len_out == 0) return;

            out[i_out++] = it.index;
            while (i_out < len_out && zigzag_iterator_next(&it)) {
                out[i_out++] = it.index;
            }

            while (i_out < len_out) {
                out[i_out++] = -1;
            }
        }
    """

    test_context.add_kernels(
        sources=[ZIGZAG_TEST_SOURCE],
        kernels={
            'fill_zigzag_sequence': xo.Kernel(
                c_name='fill_zigzag_sequence',
                args=[
                    xo.Arg(xo.UInt32, name='start'),
                    xo.Arg(xo.UInt32, name='upper_bound'),
                    xo.Arg(xo.Int8, name='wrap'),
                    xo.Arg(xo.Int32, pointer=True, name='out'),
                    xo.Arg(xo.UInt32, name='len_out'),
                ],
            ),
        },
    )
    zigzag_test_kernel = test_context.kernels['fill_zigzag_sequence']

    # Test odd cases
    out = np.zeros(5, dtype=np.int32)

    zigzag_test_kernel(start=4, upper_bound=5, wrap=1, out=out, len_out=len(out))
    assert all(out == [4, 0, 3, 1, 2])

    zigzag_test_kernel(start=2, upper_bound=5, wrap=0, out=out, len_out=len(out))
    assert all(out == [2, 3, 1, 4, 0])

    # Test even cases
    out = np.zeros(6, dtype=np.int32)

    zigzag_test_kernel(start=4, upper_bound=6, wrap=1, out=out, len_out=len(out))
    assert all(out == [4, 5, 3, 0, 2, 1])

    zigzag_test_kernel(start=4, upper_bound=6, wrap=0, out=out, len_out=len(out))
    assert all(out == [4, 5, 3, 2, 1, 0])


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_from_line_with_associated_apertures_type_bounds(test_context):
    env = xt.load(string=TOY_RING_SEQUENCE, format='madx', install_limits=False)
    env.set_particle_ref('proton', p0c=1.2e9)
    ring = env['ring']

    aperture_model = Aperture.from_line_with_associated_apertures(ring, context=test_context)
    bounds_table = aperture_model.get_bounds_table()
    table_rows = ring.get_table().cols['s_start', 's_end', 'name', 'element_type'].rows[:-1].rows
    expected = _expected_profile_bounds_from_table(
        table_rows,
        skip_row=lambda row: row.element_type == 'Drift',
    )

    assert len(bounds_table.name) == len(expected)
    for idx, ((expected_s, expected_name), bound_s, pipe_name) in enumerate(
        zip(expected, bounds_table.s, bounds_table.pipe_name)
    ):
        xo.assert_allclose(bound_s, expected_s, atol=1e-6)
        prototype_name, suffix = expected_name.split('::')
        _ = int(suffix)
        assert pipe_name.startswith(prototype_name), f'mismatch at bound row {idx}'


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_from_line_with_limits_type_bounds(test_context):
    env = xt.load(string=TOY_RING_SEQUENCE, format='madx', install_limits=True)
    env.set_particle_ref('proton', p0c=1.2e9)
    ring = env['ring']

    aperture_model = Aperture.from_line_with_limits(ring, context=test_context)
    bounds_table = aperture_model.get_bounds_table()

    expected = _expected_profile_bounds_from_table(
        ring.get_table().rows,
        skip_row=lambda row: not row.element_type.startswith('Limit'),
    )

    assert len(bounds_table.name) == len(expected)
    for idx, ((expected_s, expected_name), bound_s, pipe_name) in enumerate(
        zip(expected, bounds_table.s, bounds_table.pipe_name)
    ):
        xo.assert_allclose(bound_s, expected_s, atol=1e-6)
        assert expected_name.startswith(pipe_name), f'mismatch at bound row {idx}'


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_bounds_table_for_perfect_overlap_interval(test_context):
    env = xt.load(string=TOY_RING_SEQUENCE, format='madx', install_limits=False)
    env.set_particle_ref('proton', p0c=1.2e9)
    ring = env['ring']

    aperture_model = Aperture.from_line_with_associated_apertures(ring, context=test_context)

    bounds_table = aperture_model.get_bounds_table()
    mask = (
        (bounds_table.pipe_name == 'mqf_aper')
        & (bounds_table.s >= 1.0 - 1e-12)
        & (bounds_table.s <= 1.3 + 1e-12)
    )
    rows = bounds_table.rows[mask]

    assert len(rows) == 2
    xo.assert_allclose(rows.s, [1.0, 1.3], atol=1e-12)
    assert list(rows.profile_name) == ['mqf_aper', 'mqf_aper']
    assert list(rows.shape) == ['Rectangle', 'Rectangle']
    assert all(sp['half_width'] == 0.08 for sp in rows.shape_param)
    assert all(sp['half_height'] == 0.04 for sp in rows.shape_param)


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_aperture_model_views(test_context):
    model = ApertureModel(
        pipe_positions=[
            PipePosition(
                pipe_index=0,
                survey_reference_name='drift',
                survey_index=0,
                transformation=transform_matrix(),
            ),
        ],
        pipes=[Pipe(curvature=0.5, positions=[
            ProfilePosition(profile_index=0, shift_s=0.1, shift_x=0.2, shift_y=-0.3, rot_x_rad=0.4, rot_y_rad=-0.5, rot_s_rad=0.6),
            ProfilePosition(profile_index=1, shift_s=0.7),
        ])],
        profiles=[
            Profile(shape=Circle(radius=1.0), tol_r=0, tol_x=0, tol_y=0),
            Profile(shape=Rectangle(half_width=2.0, half_height=3.0), tol_r=0.1, tol_x=0.2, tol_y=0.3),
        ],
        pipe_names=['pipe0'],
        pipe_position_names=['pipe0_at_drift'],
        profile_names=['circ0', 'rect0'],
        _context=test_context,
    )

    profiles = ProfilesView(model)
    pipes = PipesView(model)
    pipe_positions = PipePositionsView(model)
    pipe0 = pipes[0]
    positions = pipe0

    assert repr(profiles) == '<ProfilesView: 2 profiles>'
    assert repr(pipes) == '<PipesView: 1 pipe>'
    assert repr(pipe_positions) == '<PipePositionsView: 1 pipe position>'
    assert profiles.keys() == ['circ0', 'rect0']
    assert pipes.keys() == ['pipe0']
    assert pipe_positions.keys() == ['pipe0_at_drift']
    assert profiles.search(r'.*0') == ['circ0', 'rect0']
    assert pipes.search(r'pipe.*') == ['pipe0']
    assert pipe_positions.search(r'pipe0_.*') == ['pipe0_at_drift']

    assert profiles[0].name == 'circ0'
    assert profiles['rect0'].name == 'rect0'
    assert list(name for name, _ in profiles.items()) == ['circ0', 'rect0']
    assert list(type(profile.raw.shape).__name__ for profile in [profiles[0], profiles[1]]) == ['Circle', 'Rectangle']
    assert [profile.name for profile in profiles.values()] == ['circ0', 'rect0']
    assert [type(profile.shape).__name__ for profile in profiles.values()] == ['Circle', 'Rectangle']

    xo.assert_allclose(profiles['rect0'].tol_r, 0.1, atol=1e-15, rtol=0)
    xo.assert_allclose(profiles['rect0'].tol_x, 0.2, atol=1e-15, rtol=0)
    xo.assert_allclose(profiles['rect0'].tol_y, 0.3, atol=1e-15, rtol=0)
    profiles['rect0'].tol_r = 0.4
    profiles['rect0'].tol_x = 0.5
    profiles['rect0'].tol_y = 0.6
    xo.assert_allclose(model.profiles[1].tol_r, 0.4, atol=1e-15, rtol=0)
    xo.assert_allclose(model.profiles[1].tol_x, 0.5, atol=1e-15, rtol=0)
    xo.assert_allclose(model.profiles[1].tol_y, 0.6, atol=1e-15, rtol=0)

    assert pipes[0].name == 'pipe0'
    assert pipes['pipe0'].name == 'pipe0'
    assert list(name for name, _ in pipes.items()) == ['pipe0']
    assert [pipe.name for pipe in pipes.values()] == ['pipe0']
    assert [pipe.curvature for pipe in pipes.values()] == [0.5]
    xo.assert_allclose(pipe0.length, 0.6, atol=1e-15, rtol=0)
    xo.assert_allclose(pipe0.angle, 0.3, atol=1e-15, rtol=0)
    pipe0.curvature = 0.25
    xo.assert_allclose(model.pipes[0].curvature, 0.25, atol=1e-15, rtol=0)

    assert pipe_positions[0].name == 'pipe0_at_drift'
    assert pipe_positions['pipe0_at_drift'].name == 'pipe0_at_drift'
    assert pipe_positions[0].pipe.name == 'pipe0'
    assert pipe_positions[0].survey_reference_name == 'drift'
    assert pipe_positions[0].survey_index == 0
    xo.assert_allclose(pipe_positions[0].shift_x, 0.0, atol=1e-15, rtol=0)
    xo.assert_allclose(pipe_positions[0].shift_y, 0.0, atol=1e-15, rtol=0)
    xo.assert_allclose(pipe_positions[0].shift_z, 0.0, atol=1e-15, rtol=0)
    xo.assert_allclose(pipe_positions[0].rot_x_rad, 0.0, atol=1e-15, rtol=0)
    xo.assert_allclose(pipe_positions[0].rot_y_rad, 0.0, atol=1e-15, rtol=0)
    xo.assert_allclose(pipe_positions[0].rot_z_rad, 0.0, atol=1e-15, rtol=0)
    assert list(name for name, _ in pipe_positions.items()) == ['pipe0_at_drift']
    assert [pipe_pos.name for pipe_pos in pipe_positions.values()] == ['pipe0_at_drift']
    assert [pipe_pos.pipe_index for pipe_pos in pipe_positions.values()] == [0]
    pipe_positions[0].survey_reference_name = 'drift_entry'
    pipe_positions[0].survey_index = 3
    pipe_positions[0].pipe_index = 0
    pipe_positions[0].shift_x = 0.1
    pipe_positions[0].shift_y = -0.2
    pipe_positions[0].shift_z = 0.3
    pipe_positions[0].rot_x_rad = 0.4
    pipe_positions[0].rot_y_rad = -0.5
    pipe_positions[0].rot_z_rad = 0.6
    assert model.pipe_positions[0].survey_reference_name == 'drift_entry'
    assert model.pipe_positions[0].survey_index == 3
    updated_transform = matrix_to_transform(model.pipe_positions[0].transformation.to_nplike())
    xo.assert_allclose(updated_transform.shift_x, 0.1, atol=1e-15, rtol=0)
    xo.assert_allclose(updated_transform.shift_y, -0.2, atol=1e-15, rtol=0)
    xo.assert_allclose(updated_transform.shift_z, 0.3, atol=1e-15, rtol=0)
    xo.assert_allclose(updated_transform.rot_x_rad, 0.4, atol=1e-15, rtol=0)
    xo.assert_allclose(updated_transform.rot_y_rad, -0.5, atol=1e-15, rtol=0)
    xo.assert_allclose(updated_transform.rot_z_rad, 0.6, atol=1e-15, rtol=0)
    pipe_positions[0].survey_reference_name = 'drift'
    pipe_positions[0].survey_index = 0

    assert positions[0].profile.name == 'circ0'
    assert positions[1].profile.name == 'rect0'
    assert [pp.profile.name for pp in positions] == ['circ0', 'rect0']
    assert [pp.profile.name for pp in positions[:]] == ['circ0', 'rect0']
    assert [pp.profile.name for pp in positions.values()] == ['circ0', 'rect0']
    xo.assert_allclose(positions[0].shift_s, 0.1, atol=1e-15, rtol=0)
    xo.assert_allclose(positions[0].shift_x, 0.2, atol=1e-15, rtol=0)
    xo.assert_allclose(positions[0].shift_y, -0.3, atol=1e-15, rtol=0)
    xo.assert_allclose(positions[0].rot_x_rad, 0.4, atol=1e-15, rtol=0)
    xo.assert_allclose(positions[0].rot_y_rad, -0.5, atol=1e-15, rtol=0)
    xo.assert_allclose(positions[0].rot_s_rad, 0.6, atol=1e-15, rtol=0)

    positions[0].profile_index = 1
    positions[0].shift_s = 0.25
    positions[0].shift_x = 0.35
    positions[0].shift_y = -0.45
    positions[0].rot_x_rad = 0.55
    positions[0].rot_y_rad = -0.65
    positions[0].rot_s_rad = 0.75

    assert positions[0].profile.name == 'rect0'
    xo.assert_allclose(model.pipes[0].positions[0].profile_index, 1, atol=0, rtol=0)
    xo.assert_allclose(model.pipes[0].positions[0].shift_s, 0.25, atol=1e-15, rtol=0)
    xo.assert_allclose(model.pipes[0].positions[0].shift_x, 0.35, atol=1e-15, rtol=0)
    xo.assert_allclose(model.pipes[0].positions[0].shift_y, -0.45, atol=1e-15, rtol=0)
    xo.assert_allclose(model.pipes[0].positions[0].rot_x_rad, 0.55, atol=1e-15, rtol=0)
    xo.assert_allclose(model.pipes[0].positions[0].rot_y_rad, -0.65, atol=1e-15, rtol=0)
    xo.assert_allclose(model.pipes[0].positions[0].rot_s_rad, 0.75, atol=1e-15, rtol=0)

    assert [pp.profile.name for pp in positions[:]] == ['rect0', 'rect0']
    with pytest.raises(AttributeError):
        positions.append(positions[0])


def test_get_limit_elements(monkeypatch):
    ap = Aperture.__new__(Aperture)
    cross_sections = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            [[2.0, 0.0], [3.0, 0.0], [2.0, 1.0], [2.0, 0.0]],
        ],
        dtype=FloatType._dtype,
    )
    table = Table(
        data={
            'index': np.array([0, 1], dtype=np.int64),
            'cross_section': cross_sections,
        },
        index='index',
    )
    monkeypatch.setattr(ap, 'cross_sections_at_s', lambda s_positions: table)

    out = ap.get_limit_elements([1.5, 3.5])

    assert list(out) == [1.5, 3.5]
    assert all(isinstance(value, xt.LimitPolygon) for value in out.values())
    xo.assert_allclose(out[1.5].x_vertices, [0.0, 1.0, 0.0], atol=0, rtol=0)
    xo.assert_allclose(out[1.5].y_vertices, [0.0, 0.0, 1.0], atol=0, rtol=0)
    xo.assert_allclose(out[3.5].x_vertices, [2.0, 3.0, 2.0], atol=0, rtol=0)
    xo.assert_allclose(out[3.5].y_vertices, [0.0, 0.0, 1.0], atol=0, rtol=0)


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_bounds_table_uses_type_position_name_in_installed_profile_name(test_context):
    env = xt.Environment()
    line = env.new_line(name='line', components=[env.new('drift', xt.Drift, length=1.0)])
    sv = line.survey()

    model = ApertureModel(
        pipe_positions=[
            PipePosition(
                pipe_index=0,
                survey_reference_name=sv.name[0],
                survey_index=0,
                transformation=transform_matrix(),
            ),
            PipePosition(
                pipe_index=0,
                survey_reference_name=sv.name[0],
                survey_index=0,
                transformation=transform_matrix(shift_z=0.5),
            ),
        ],
        pipes=[Pipe(curvature=0.0, positions=[ProfilePosition(profile_index=0)])],
        profiles=[Profile(shape=Circle(radius=1.0), tol_r=0, tol_x=0, tol_y=0)],
        pipe_names=['shared_type'],
        pipe_position_names=['entry_ap', 'middle_ap'],
        profile_names=['circ0'],
        _context=test_context,
    )

    ap = Aperture(line=line, model=model, context=test_context, _skip_validity_check=True)
    bounds_table = ap.get_bounds_table()

    assert list(bounds_table.pipe_name) == ['shared_type', 'shared_type']
    assert list(bounds_table.profile_name) == ['circ0', 'circ0']
    assert list(bounds_table.name) == ['entry_ap', 'middle_ap']


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_bounds_table_for_interval_spanning_multiple_types(test_context):
    env = xt.load(string=TOY_RING_SEQUENCE, format='madx', install_limits=False)
    env.set_particle_ref('proton', p0c=1.2e9)
    ring = env['ring']

    aperture_model = Aperture.from_line_with_associated_apertures(ring, context=test_context)

    bounds_table = aperture_model.get_bounds_table()
    mask = (bounds_table.s >= 8.0 - 1e-12) & (bounds_table.s <= 11.8 + 1e-12)
    rows = bounds_table.rows[mask]

    assert len(rows) == 4
    assert list(rows.pipe_name) == ['mb_aper', 'ap_ds_aper', 'ap_ds_aper', 'mqf_aper']
    xo.assert_allclose(rows.s, [10.6, 10.6, 11.6, 11.6], atol=1e-9)
    assert list(rows.shape) == ['Ellipse', 'RectEllipse', 'RectEllipse', 'Rectangle']

    mb_shape = rows.shape_param[0]
    assert mb_shape['half_major'] == 0.1
    assert mb_shape['half_minor'] == 0.1

    ap_ds_shape_0 = rows.shape_param[1]
    ap_ds_shape_1 = rows.shape_param[2]
    for ap_ds_shape in (ap_ds_shape_0, ap_ds_shape_1):
        assert ap_ds_shape['half_major'] == 0.022
        assert ap_ds_shape['half_minor'] == 0.022
        assert ap_ds_shape['half_width'] == 0.022
        assert ap_ds_shape['half_height'] == 0.01715

    mqf_shape = rows.shape_param[3]
    assert mqf_shape['half_width'] == 0.08
    assert mqf_shape['half_height'] == 0.04


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_get_pipe_table_handles_regular_and_wrapped_pipes(test_context):
    line, model = _make_pipe_table_test_ring(test_context)
    ap = Aperture(line, model, context=test_context, _skip_validity_check=True)

    pipe_table = ap.get_pipe_table()

    regular = pipe_table.rows['pipe_regular']
    xo.assert_allclose(regular.s_start, 10.0, atol=1e-9, rtol=0)
    xo.assert_allclose(regular.s_end, 20.0, atol=1e-9, rtol=0)
    xo.assert_allclose(regular.length, 10.0, atol=1e-9, rtol=0)
    xo.assert_allclose(regular.s_span_start, 10.0, atol=5e-4, rtol=0)
    xo.assert_allclose(regular.s_span_end, 20.0, atol=5e-4, rtol=0)
    xo.assert_allclose(regular.span, 10.0, atol=1e-9, rtol=0)

    wrapped = pipe_table.rows['pipe_wrapped']
    xo.assert_allclose(wrapped.s_start, 75.13834, atol=5e-5, rtol=0)
    xo.assert_allclose(wrapped.s_end, 3.58262, atol=5e-5, rtol=0)
    xo.assert_allclose(wrapped.length, 8.44428, atol=5e-5, rtol=0)
    xo.assert_allclose(wrapped.s_span_start, 74.7542, atol=5e-4, rtol=0)
    xo.assert_allclose(wrapped.s_span_end, 2.72965, atol=5e-4, rtol=0)
    xo.assert_allclose(wrapped.span, 7.97542, atol=5e-4, rtol=0)
    assert wrapped.s_start > wrapped.s_end
    assert wrapped.s_span_start > wrapped.s_span_end


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_pipe_overlap_validation_allows_wrapped_and_regular_non_overlapping_pipes(test_context):
    line, model = _make_pipe_table_test_ring(
        test_context,
        extra_pipe_positions=[('pipe_middle', 'bend::3', 0.0)],
    )

    ap = Aperture(line, model, context=test_context, _skip_validity_check=True)
    pipe_table = ap.get_pipe_table()
    ap._check_pipe_bounds_validity()

    middle = pipe_table.rows['pipe_middle']
    xo.assert_allclose(middle.s_start, 30.0, atol=1e-9, rtol=0)
    xo.assert_allclose(middle.s_end, 40.0, atol=1e-9, rtol=0)
    xo.assert_allclose(middle.length, 10.0, atol=1e-9, rtol=0)
    xo.assert_allclose(middle.s_span_start, 30.0, atol=5e-4, rtol=0)
    xo.assert_allclose(middle.s_span_end, 40.0, atol=5e-4, rtol=0)
    xo.assert_allclose(middle.span, 10.0, atol=1e-9, rtol=0)


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_cross_sections_do_not_interpolate_between_pipes_overlapping_within_tolerance(test_context):
    env = xt.Environment()
    line = env.new_line(
        name='line',
        components=[env.new('drift', xt.Drift, length=4.5)],
    )
    sv = line.survey()

    overlap = 5e-4
    model = ApertureModel(
        line=line,
        pipe_positions=[
            PipePosition(
                pipe_index=0,
                survey_reference_name=sv.name[0],
                survey_index=0,
                transformation=transform_matrix(),
            ),
            PipePosition(
                pipe_index=1,
                survey_reference_name=sv.name[0],
                survey_index=0,
                transformation=transform_matrix(shift_z=1.0),
            ),
            PipePosition(
                pipe_index=0,
                survey_reference_name=sv.name[0],
                survey_index=0,
                transformation=transform_matrix(shift_z=3.0),
            ),
        ],
        pipes=[
            Pipe(curvature=0.0, positions=[
                ProfilePosition(profile_index=0, shift_s=0.0),
                ProfilePosition(profile_index=0, shift_s=1.0 + overlap),
            ]),
            Pipe(curvature=0.0, positions=[
                ProfilePosition(profile_index=1, shift_s=0.0),
                ProfilePosition(profile_index=1, shift_s=1.0),
            ]),
        ],
        profiles=[
            Profile(shape=Circle(radius=0.1), tol_r=0, tol_x=0, tol_y=0),
            Profile(shape=Circle(radius=1.0), tol_r=0, tol_x=0, tol_y=0),
        ],
        pipe_names=['small_pipe', 'large_pipe'],
        pipe_position_names=['small_pipe', 'large_pipe', 'small_pipe_again'],
        profile_names=['small_circle', 'large_circle'],
    )

    aperture = Aperture(
        line=line,
        model=model,
        context=test_context,
        s_tol=1e-3,
        num_profile_points=64,
    )

    # The accepted sub-tolerance overlap does not corrupt the per-pipe table.
    pipe_table = aperture.get_pipe_table()
    small_pipe = pipe_table.rows['small_pipe']
    large_pipe = pipe_table.rows['large_pipe']
    small_pipe_again = pipe_table.rows['small_pipe_again']
    xo.assert_allclose(
        [
            small_pipe.s_start,
            small_pipe.s_end,
            small_pipe.s_span_start,
            small_pipe.s_span_end,
        ],
        [0.0, 1.0 + overlap, 0.0, 1.0 + overlap],
        atol=1e-12,
        rtol=0,
    )
    xo.assert_allclose(
        [
            large_pipe.s_start,
            large_pipe.s_end,
            large_pipe.s_span_start,
            large_pipe.s_span_end,
        ],
        [1.0, 2.0, 1.0, 2.0],
        atol=1e-12,
        rtol=0,
    )
    xo.assert_allclose(
        [small_pipe_again.s_start, small_pipe_again.s_end],
        [3.0, 4.0 + overlap],
        atol=1e-12,
        rtol=0,
    )

    # Reordering is per installed pipe position, even when a pipe type is
    # installed more than once.
    xo.assert_allclose(
        aperture._aperture_bounds.pipe_position_indices.to_nparray(),
        [0, 0, 1, 1, 2, 2],
        atol=0,
        rtol=0,
    )

    # These points are away from the overlap and lie in exactly one pipe.
    # Each cylindrical pipe must retain its constant circular cross-section.
    sections = aperture.cross_sections_at_s([0.5, 1.5, 3.5]).cross_section
    radii = np.linalg.norm(sections, axis=2)
    xo.assert_allclose(np.mean(radii, axis=1), [0.1, 1.0, 0.1], atol=1e-12, rtol=0)


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_aperture_bounds_preserve_overlap_larger_than_tolerance(test_context):
    env = xt.Environment()
    line = env.new_line(
        name='line',
        components=[env.new('drift', xt.Drift, length=2.5)],
    )
    sv = line.survey()

    overlap = 2e-3
    model = ApertureModel(
        line=line,
        pipe_positions=[
            PipePosition(
                pipe_index=0,
                survey_reference_name=sv.name[0],
                survey_index=0,
                transformation=transform_matrix(),
            ),
            PipePosition(
                pipe_index=1,
                survey_reference_name=sv.name[0],
                survey_index=0,
                transformation=transform_matrix(shift_z=1.0),
            ),
        ],
        pipes=[
            Pipe(curvature=0.0, positions=[
                ProfilePosition(profile_index=0, shift_s=0.0),
                ProfilePosition(profile_index=0, shift_s=1.0 + overlap),
            ]),
            Pipe(curvature=0.0, positions=[
                ProfilePosition(profile_index=1, shift_s=0.0),
                ProfilePosition(profile_index=1, shift_s=1.0),
            ]),
        ],
        profiles=[
            Profile(shape=Circle(radius=0.1), tol_r=0, tol_x=0, tol_y=0),
            Profile(shape=Circle(radius=1.0), tol_r=0, tol_x=0, tol_y=0),
        ],
        pipe_names=['small_pipe', 'large_pipe'],
        pipe_position_names=['small_pipe', 'large_pipe'],
        profile_names=['small_circle', 'large_circle'],
    )

    with pytest.raises(ValueError, match='overlaps pipe position'):
        Aperture(
            line=line,
            model=model,
            context=test_context,
            s_tol=1e-3,
        )

    aperture = Aperture(
        line=line,
        model=model,
        context=test_context,
        s_tol=1e-3,
        _skip_validity_check=True,
    )

    # Intentional overlaps larger than the tolerance retain geometric order.
    xo.assert_allclose(
        aperture._aperture_bounds.pipe_position_indices.to_nparray(),
        [0, 1, 0, 1],
        atol=0,
        rtol=0,
    )


def test_split_wrapped_s_interval_without_wrap():
    intervals = _split_wrapped_s_interval(3.0, 7.0, line_length=10.0, wrap=False, s_tol=1e-9)
    assert intervals == [(3.0, 7.0)]


def test_split_wrapped_s_interval_with_non_wrapped_ring_interval():
    intervals = _split_wrapped_s_interval(3.0, 7.0, line_length=10.0, wrap=True, s_tol=1e-9)
    assert intervals == [(3.0, 7.0)]


def test_split_wrapped_s_interval_with_wrapped_ring_interval():
    intervals = _split_wrapped_s_interval(8.0, 2.0, line_length=10.0, wrap=True, s_tol=1e-9)
    assert intervals == [(8.0, 10.0), (0.0, 2.0)]


def test_split_wrapped_s_interval_drops_zero_length_segment_at_ring_boundary():
    intervals = _split_wrapped_s_interval(8.0, 10.0, line_length=10.0, wrap=True, s_tol=1e-9)
    assert intervals == [(8.0, 10.0)]


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_s_around_transitions_basic_pattern_and_resolution(test_context):
    ap = _make_transition_test_aperture(test_context)

    s_positions = ap.s_around_transitions(tol=0.5)
    xo.assert_allclose(s_positions, [2.5, 3.5, 6.5, 7.5], atol=1e-12, rtol=0)

    s_positions_with_grid = ap.s_around_transitions(tol=0.5, resolution=2.0, s_range=(2.0, 8.0))
    xo.assert_allclose(s_positions_with_grid, [2.0, 2.5, 3.5, 4.0, 6.0, 6.5, 7.5, 8.0], atol=1e-12, rtol=0)


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_s_around_transitions_wrapped_ring_range(test_context):
    line, model = _make_pipe_table_test_ring(test_context)
    ap = Aperture(line, model, context=test_context, _skip_validity_check=True)

    bounds_table = ap.get_bounds_table()
    wrapped_mask = np.asarray(bounds_table.name) == 'pipe_wrapped'
    wrapped_positions = np.asarray(bounds_table.s[wrapped_mask], dtype=float)
    expected = np.unique(np.clip(
        np.concatenate([wrapped_positions - 0.1, wrapped_positions + 0.1]),
        0.0,
        line.get_length(),
    ))

    s_positions = ap.s_around_transitions(tol=0.1, s_range=(74.0, 4.0))

    xo.assert_allclose(s_positions, expected, atol=1e-12, rtol=0)
    assert np.all((s_positions >= 74.0) | (s_positions <= 4.0))


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_aperture_json_roundtrip_preserves_explicit_ring_flag(test_context, tmp_path):
    env = xt.Environment()
    drift = env.new('drift', xt.Drift, length=3.0)
    line = env.new_line(name='line', components=[drift])
    sv = line.survey()

    model = ApertureModel(
        pipe_positions=[
            PipePosition(
                pipe_index=0,
                survey_reference_name='drift',
                survey_index=sv.name.tolist().index('drift'),
                transformation=transform_matrix(),
            ),
        ],
        pipes=[Pipe(curvature=0.0, positions=[ProfilePosition(profile_index=0)])],
        profiles=[Profile(shape=Circle(radius=1.0), tol_r=0, tol_x=0, tol_y=0)],
        pipe_names=['pipe0'],
        pipe_position_names=['pipe0'],
        profile_names=['circ0'],
        _context=test_context,
    )

    ap = Aperture(line, model, context=test_context, is_ring=False)
    path = tmp_path / 'aperture.json'

    ap.to_json(path)
    loaded = Aperture.from_json(path, line=line, context=test_context)

    assert ap.is_ring is False
    assert loaded.is_ring is False


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_pipe_overlap_validation_rejects_overlap_with_wrapped_pipe(test_context):
    line, model = _make_pipe_table_test_ring(
        test_context,
        extra_pipe_positions=[('pipe_overlap', 'bend::0', 0.0)],
    )

    with pytest.raises(ValueError, match=r'pipe position pipe_overlap overlaps pipe position pipe_wrapped|pipe position pipe_wrapped overlaps pipe position pipe_overlap'):
        Aperture(line, model, context=test_context, _skip_validity_check=True)._check_pipe_bounds_validity()


def test_is_point_inside_polygon_ellipse(kernels):
    rx = 2
    ry = 3
    ellipse = [(rx * np.cos(angle), ry * np.sin(angle)) for angle in np.linspace(0, 2 * np.pi, 99)]
    ellipse.append(ellipse[0])
    ellipse = np.array(ellipse, dtype=FloatType._dtype)
    polygon = Polygon(vertices=ellipse, _context=kernels)

    @np.vectorize
    def in_ellipse(x, y):
        point = np.array([x, y], dtype=FloatType._dtype)
        return bool(polygon.is_point_inside_polygon(point=point))

    extent = np.linspace(-10, 10, 100)
    xs, ys = np.meshgrid(extent, extent)

    result = in_ellipse(xs, ys)
    expected = (xs ** 2 / rx ** 2 + ys ** 2 / ry ** 2 - 1) < 0

    assert not np.all(result) and np.any(result)  # sanity check
    assert np.all(result == expected)


def test_is_point_inside_polygon_path(kernels):
    # Define a shape that is a rectangle spanning (-1, -1) through (3, 2) minus
    # a rectangle (1, 0.5) through (2, 2)

    poly = np.array([
        (1, 2),
        (1, .5),
        (2, .5),
        (2, 2),
        (3, 2),
        (3, -1),
        (-1, -1),
        (-1, 2),
        (1, 2),
    ], dtype=FloatType._dtype)
    polygon = Polygon(vertices=poly, _context=kernels)

    @np.vectorize
    def in_poly(x, y):
        point = np.array([x, y], dtype=FloatType._dtype)
        return bool(polygon.is_point_inside_polygon(point=point))

    extent = np.linspace(-5, 5, 100)
    xs, ys = np.meshgrid(extent, extent)

    result = in_poly(xs, ys)

    in_rec1 = (-1 < xs) & (xs < 3) & (-1 < ys) & (ys < 2)
    in_rec2 = (1 < xs) & (xs < 2) & (0.5 < ys) & (ys < 2)
    expected = in_rec1 & ~in_rec2

    assert not np.all(result) and np.any(result)  # sanity check
    assert np.all(result == expected)



def test_points_inside_polygon_inscribed_circles(kernels):
    r1 = 0.11
    r2 = 1

    circ1 = [(r1 * np.cos(angle), r1 * np.sin(angle)) for angle in np.linspace(0, 2 * np.pi, 99)]
    circ1.append(circ1[0])
    circ2 = [(r2 * np.cos(angle), r2 * np.sin(angle)) for angle in np.linspace(0, 2 * np.pi, 99)]
    circ2.append(circ2[0])

    circ1 = np.array(circ1, dtype=FloatType._dtype)
    circ2 = np.array(circ2, dtype=FloatType._dtype)
    poly1 = Polygon(vertices=circ1, _context=kernels)
    poly2 = Polygon(vertices=circ2, _context=kernels)

    small_in_big = poly2.points_inside_polygon(points=circ1)

    assert bool(small_in_big)

    big_in_small = poly1.points_inside_polygon(points=circ2)

    assert not bool(big_in_small)


def test_points_inside_polygon_simple(kernels):
    poly_big = [(1, 1), (2, 3.5), (4.5, 3.5), (4.5, 1), (1, 1)]
    poly_small = [(2, 2), (3, 3), (4, 2), (3, 1.5), (2, 2)]

    poly_big = np.array(poly_big, dtype=FloatType._dtype)
    poly_small = np.array(poly_small, dtype=FloatType._dtype)
    poly_big_obj = Polygon(vertices=poly_big, _context=kernels)
    poly_small_obj = Polygon(vertices=poly_small, _context=kernels)

    small_in_big = poly_big_obj.points_inside_polygon(points=poly_small)

    assert bool(small_in_big)

    big_in_small = poly_small_obj.points_inside_polygon(points=poly_big)

    assert not bool(big_in_small)


def test_points_inside_polygon_simpler(kernels):
    poly_big = np.array([
        [1.0000000e+00, 0.0000000e+00],
        [-5.0000006e-01, 8.6602539e-01],
        [-4.9999991e-01, -8.6602545e-01],
        [1.0000000e+00, 0.0000000e+00],
    ], dtype=FloatType._dtype)
    poly_small = np.array([
        [1.1466468e-01, 0.0000000e+00],
        [-5.7332322e-02, 9.9302538e-02],
        [-5.7332378e-02, -9.9302508e-02],
        [1.1466468e-01, 0.0000000e+00],
    ], dtype=FloatType._dtype)
    poly_big_obj = Polygon(vertices=poly_big, _context=kernels)
    poly_small_obj = Polygon(vertices=poly_small, _context=kernels)

    small_in_big = poly_big_obj.points_inside_polygon(points=poly_small)

    assert bool(small_in_big)

    big_in_small = poly_small_obj.points_inside_polygon(points=poly_big)

    assert not bool(big_in_small)


@pytest.mark.parametrize('method', ['bisection', 'rays', 'exact'])
@pytest.mark.parametrize(
    'shape,aper_params,aper_tol,beam_params,halo_params,expected',
    [
        (
            'circle', (1,), (0, 0, 0),
            {'exn': 1e-3, 'eyn': 1e-3, 'gamma': np.sqrt(101), 'betx': 1, 'bety': 1, 'x': 0, 'y': 0, 'dx': 0, 'dy': 0},
            {},
            100,
        ),
        (
            'circle', (1,), (0, 0, 0),
            {'exn': 1e-3, 'eyn': 1e-3, 'gamma': np.sqrt(101), 'betx': 1, 'bety': 1, 'x': 0, 'y': 0, 'dx': 0, 'dy': 0},
            {'halo_primary': 1, 'halo_r': 2, 'halo_x': 2, 'halo_y': 2},
            50,
        ),
        (
            'rectangle', (1, 1), (0, 0, 0),
            {'exn': 1e-3, 'eyn': 1e-3, 'gamma': np.sqrt(101), 'betx': 1, 'bety': 1, 'x': 0, 'y': 0, 'dx': 0, 'dy': 0},
            {},
            100,
        ),
        (
            'rectangle', (1.1, 1.2), (0, 0, 0),
            {'exn': 1e-3, 'eyn': 1e-3, 'gamma': np.sqrt(101), 'betx': 1, 'bety': 1, 'x': -0.1, 'y': 0.2, 'dx': 0, 'dy': 0},
            {},
            100,
        ),
        (
            'racetrack', (0.28, 0.43, 0.13, 0.172), (0.002, 0.006, 0.002),
            {'exn': 4e-3, 'eyn': 4e-3, 'gamma': np.sqrt(101), 'betx': 9, 'bety': 16, 'x': 0, 'y': 0, 'dx': 0, 'dy': 0},
            {
                'tol_beta_beating': 0.8,
                'tol_disp': 1.25,
                'tol_disp_ref_beta': 4,
                'tol_disp_ref': 20,
                'halo_primary': 10,
                'halo_r': 0.7,
                'halo_x': 0.5,
                'halo_y': 0.6,
                'tol_co': 0.002,
                'delta_rms': 0.001,
            },
            100,
        ),
        (
            'racetrack', (0.3, 0.5, 0.13, 0.172), (0.002, 0.006, 0.002),
            {'exn': 4e-3, 'eyn': 4e-3, 'gamma': np.sqrt(101), 'betx': 9, 'bety': 16, 'x': -0.02, 'y': 0.07, 'dx': 0, 'dy': 0},
            {
                'tol_beta_beating': 0.8,
                'tol_disp': 1.25,
                'tol_disp_ref_beta': 4,
                'tol_disp_ref': 20,
                'halo_primary': 10,
                'halo_r': 0.7,
                'halo_x': 0.5,
                'halo_y': 0.6,
                'tol_co': 0.002,
                'delta_rms': 0.001,
            },
            100,
        ),
        (
            'racetrack', (0.32, 0.478, 0.13, 0.172), (0.002, 0.006, 0.002),
            {
                'exn': 4e-3,
                'eyn': 4e-3,
                'gamma': np.sqrt(101),
                'betx': 9,
                'bety': 16,
                'x': 0,
                'y': 0,
                'dx': 10 * np.sqrt(13),
                'dy': 10 * np.sqrt(17)
            },
            {
                'tol_beta_beating': 0.8,
                'tol_disp': 1.25,
                'tol_disp_ref_beta': 4,
                'tol_disp_ref': 20,
                'halo_primary': 10,
                'halo_r': 0.7,
                'halo_x': 0.5,
                'halo_y': 0.6,
                'tol_co': 0.002,
                'delta_rms': 0.001,
            },
            # In the following two parametrisations the limiting direction is horizontal.
            # Available horizontal clearance is 0.32 either directly, or as 0.4 - 0.08
            # once the closed-orbit offset is included.
            #
            # Halo racetrack horizontal half-size:
            #   tol_x + tol_r + tol_co + tol_dx = 0.006 + 0.002 + 0.002 + 0.03 = 0.04
            #
            # One-sigma beam horizontal half-size:
            #   (halo_x / halo_primary) * sqrt((emitx_norm / gamma) * betx) * tol_beta_beating
            #   = 0.05 * sqrt((4e-3 / 10) * 9) * 0.8 = 0.0024
            #
            # Dispersive orbit shift:
            #   dx * delta_rms = 10 * sqrt(13) * 1e-3 = sqrt(13) / 100
            #
            # Solving 0.04 + n * 0.0024 + sqrt(13) / 100 = 0.32 gives:
            (0.32 - 0.04 - np.sqrt(13) / 100) / 0.0024,
        ),
        (
            'racetrack', (0.4, 0.5, 0.13, 0.172), (0.002, 0.006, 0.002),
            {
                'exn': 4e-3,
                'eyn': 4e-3,
                'gamma': np.sqrt(101),
                'betx': 9,
                'bety': 16,
                'x': -0.08,
                'y': 0.022,
                'dx': 10 * np.sqrt(13),
                'dy': 10 * np.sqrt(17)
            },
            {
                'tol_beta_beating': 0.8,
                'tol_disp': 1.25,
                'tol_disp_ref_beta': 4,
                'tol_disp_ref': 20,
                'halo_primary': 10,
                'halo_r': 0.7,
                'halo_x': 0.5,
                'halo_y': 0.6,
                'tol_co': 0.002,
                'delta_rms': 0.001,
            },
            (0.32 - 0.04 - np.sqrt(13) / 100) / 0.0024,
        ),
    ],
    ids=[
        'circle',
        'circle-halo',
        'square',
        'square-orbit',
        'racetrack-aper_tols-halo',
        'racetrack-orbit-aper_tols-halo',
        'racetrack-dispersion-aper_tols-halo',
        'racetrack-dispersion-orbit-aper_tols-halo',
    ]
)
def test_get_aperture_sigmas_at_element_analytic(method, shape, aper_params, aper_tol, beam_params, halo_params, expected, context):
    def madx_list(values):
        return '{' + ', '.join([str(v) for v in values]) + '}'

    halo_params_for_test = {
        'emitx_norm': beam_params['exn'],
        'emity_norm': beam_params['eyn'],
        'tol_beta_beating': 1,
        'tol_disp': 0,
        'tol_disp_ref_beta': 1,
        'tol_disp_ref': 0,
        'halo_primary': 1,
        'halo_r': 1,
        'halo_x': 1,
        'halo_y': 1,
        'tol_co': 0,
        'delta_rms': 0,
    }
    halo_params_for_test.copy()
    halo_params_for_test.update(halo_params)
    halo_params = halo_params_for_test

    lattice = f"""
        m1: marker, apertype = {shape}, aperture = {madx_list(aper_params)}, aper_tol = {madx_list(aper_tol)};

        seq: sequence,l = 1;
            m1, at=0;
        endsequence;
    """

    env = xt.load(string=lattice, format='madx', install_limits=False)
    seq = env['seq']
    seq.set_particle_ref('proton', gamma0=beam_params['gamma'])
    tw = seq.twiss4d(
        betx=beam_params['betx'],
        bety=beam_params['bety'],
        x=beam_params['x'],
        y=beam_params['y'],
        dx=beam_params['dx'],
        dy=beam_params['dy'],
    )

    aperture_model = Aperture.from_line_with_associated_apertures(seq, context=context)
    aperture_model.halo_params.update(halo_params)

    # Needed as these quantities are not imported by the native madloader
    aperture_model._model.profiles[0].tol_r = aper_tol[0]
    aperture_model._model.profiles[0].tol_x = aper_tol[1]
    aperture_model._model.profiles[0].tol_y = aper_tol[2]

    # Compute n1 with Xsuite
    n1_table, tw = aperture_model.get_aperture_sigmas_at_element(
        element_name='m1',
        resolution=None,
        twiss=tw,
        envelopes_num_points=144,
        method=method,
        output_cross_sections=False,
        output_max_envelopes=False,
    )
    computed_n1 = n1_table.n1

    # There are two sources of error wrt. to the analytic solution:
    # - precision on the bisection defined in beam_aperture.h
    # - error coming from the fact that we are comparing polygons, not ideal shapes (especially a problem if x, y != 0)
    xo.assert_allclose(computed_n1, expected, atol=0.001, rtol=0.002)


def test_get_aperture_sigmas_at_element_analytic_rays(context):
    betx = 9
    bety = 16
    delta = 0.001
    gamma = np.sqrt(101)

    beam_data = {
        'emitx_norm': 4e-3,
        'emity_norm': 4e-3,
        'delta_rms': 0.001,
        'tol_co': 0.002,
        'tol_disp': 1.25,
        'tol_disp_ref': 20,
        'tol_disp_ref_beta': 4,
        'tol_beta_beating': 0.8,
        'halo_x': 0.5,
        'halo_y': 0.6,
        'halo_r': 0.7,
        'halo_primary': 10,
    }

    tol_r = 0.002
    tol_x = 0.006
    tol_y = 0.002

    expected_n1 = 100

    lattice = f"""
        m1: marker,
            apertype = racetrack,
            aperture = {{ 0.28, 0.43, 0.13, 0.172 }},
            aper_tol = {{ {tol_r}, {tol_x}, {tol_y} }};

        seq: sequence, l = 1;
            m1, at = 0;
        endsequence;
    """

    env = xt.load(string=lattice, format="madx", install_limits=False)
    seq = env["seq"]
    seq.set_particle_ref("proton", gamma0=gamma)

    tw = seq.twiss4d(betx=betx, bety=bety, delta=delta)

    aperture_model = Aperture.from_line_with_associated_apertures(seq, context=context)
    aperture_model.halo_params.update(beam_data)

    # Needed as these quantities are not imported by the native madloader
    aperture_model._model.profiles[0].tol_r = tol_r
    aperture_model._model.profiles[0].tol_x = tol_x
    aperture_model._model.profiles[0].tol_y = tol_y

    n1_table, tw = (
        aperture_model.get_aperture_sigmas_at_element(
            element_name="m1",
            resolution=None,
            twiss=tw,
            envelopes_num_points=144,
            method="rays",
            output_cross_sections=False,
            output_max_envelopes=False,
        )
    )
    computed_n1 = n1_table.n1

    # All n1-s should be the expected value, the envelope at the expected value
    # should fully cover the aperture in this case.
    xo.assert_allclose(computed_n1, expected_n1, atol=0.01, rtol=0.002)


def _build_single_marker_aperture_model(context):
    lattice = """
        m1: marker,
            apertype = racetrack,
            aperture = { 0.28, 0.43, 0.13, 0.172 },
            aper_tol = { 0.002, 0.006, 0.002 };

        seq: sequence, l = 1;
            m1, at = 0;
        endsequence;
    """

    env = xt.load(string=lattice, format="madx", install_limits=False)
    seq = env["seq"]
    seq.set_particle_ref("proton", gamma0=10)
    tw = seq.twiss4d(betx=9, bety=16)

    aperture_model = Aperture.from_line_with_associated_apertures(seq, context=context)
    aperture_model.halo_params.update({
        'emitx_norm': 4e-3,
        'emity_norm': 4e-3,
        'delta_rms': 0.001,
        'tol_co': 0.002,
        'tol_disp': 1.25,
        'tol_disp_ref': 20,
        'tol_disp_ref_beta': 4,
        'tol_beta_beating': 0.8,
        'halo_x': 0.5,
        'halo_y': 0.6,
        'halo_r': 0.7,
        'halo_primary': 10,
    })

    aperture_model._model.profiles[0].tol_r = 0.002
    aperture_model._model.profiles[0].tol_x = 0.006
    aperture_model._model.profiles[0].tol_y = 0.002

    return aperture_model, tw


@pytest.mark.parametrize('method', ['bisection', 'rays', 'exact'])
def test_get_aperture_sigmas_for_twiss_matches_at_s(method, context):
    aperture_model, tw = _build_single_marker_aperture_model(context)

    at_s_table, sliced_twiss = aperture_model.get_aperture_sigmas_at_s(
        s_positions=tw.s,
        twiss_init=tw.get_twiss_init(at_element='m1'),
        method=method,
        envelopes_num_points=144,
        output_cross_sections=True,
        output_max_envelopes=True,
    )
    from_twiss_table = aperture_model.get_aperture_sigmas_for_twiss(
        sliced_twiss=sliced_twiss,
        method=method,
        envelopes_num_points=144,
        output_cross_sections=True,
        output_max_envelopes=True,
    )

    for column in ('s', 'n1', 'cross_section', 'envelope'):
        xo.assert_allclose(
            from_twiss_table[column],
            at_s_table[column],
            atol=1e-12,
            rtol=0,
        )


@pytest.mark.parametrize('method', ['bisection', 'rays', 'exact'])
def test_get_aperture_sigmas_at_element_output_cross_sections_match_cross_sections_at_s(method, context):
    aperture_model, tw = _build_single_marker_aperture_model(context)

    n1_table, _ = aperture_model.get_aperture_sigmas_at_element(
        element_name='m1',
        resolution=None,
        twiss=tw,
        method=method,
        envelopes_num_points=144,
        output_cross_sections=True,
        output_max_envelopes=False,
    )
    sigmas = n1_table.n1
    aperture_points = n1_table.cross_section

    ref = aperture_model.cross_sections_at_element('m1', resolution=None).cross_section
    xo.assert_allclose(aperture_points, ref, atol=1e-12, rtol=0)
    assert sigmas.shape == (2,)


@pytest.mark.parametrize('method', ['bisection', 'rays'])
def test_get_aperture_sigmas_at_element_output_envelopes_match_get_envelope_at_s(method, context):
    aperture_model, tw = _build_single_marker_aperture_model(context)

    n1_table, _ = aperture_model.get_aperture_sigmas_at_element(
        element_name='m1',
        resolution=None,
        twiss=tw,
        method=method,
        envelopes_num_points=144,
        output_cross_sections=False,
        output_max_envelopes=True,
    )
    sigmas = n1_table.n1
    envelope_points = n1_table.envelope

    ref_envelopes, _ = aperture_model.get_envelope_at_element(
        element_name='m1',
        sigmas=float(sigmas[0]),
        resolution=None,
        twiss=tw,
        envelopes_num_points=144,
    )
    xo.assert_allclose(envelope_points, ref_envelopes.cross_section, atol=1e-10, rtol=0)


def test_get_envelope_at_s_can_return_extents_without_polygons(context):
    aperture_model, tw = _build_single_marker_aperture_model(context)

    envelope_table, _ = aperture_model.get_envelope_at_element(
        element_name='m1',
        sigmas=3,
        resolution=None,
        twiss=tw,
        envelopes_num_points=144,
        extents=True,
    )
    extents_table, _ = aperture_model.get_envelope_at_element(
        element_name='m1',
        sigmas=3,
        resolution=None,
        twiss=tw,
        envelopes_num_points=144,
        polygons=False,
        extents=True,
    )

    polygons = envelope_table.cross_section
    xo.assert_allclose(extents_table.min_x, np.min(polygons[:, :, 0], axis=1), atol=1e-12, rtol=0)
    xo.assert_allclose(extents_table.max_x, np.max(polygons[:, :, 0], axis=1), atol=1e-12, rtol=0)
    xo.assert_allclose(extents_table.min_y, np.min(polygons[:, :, 1], axis=1), atol=1e-12, rtol=0)
    xo.assert_allclose(extents_table.max_y, np.max(polygons[:, :, 1], axis=1), atol=1e-12, rtol=0)
    assert 'cross_section' not in extents_table._col_names


def test_get_aperture_sigmas_at_element_output_envelopes_exact_is_contained_in_full_envelope(context):
    aperture_model, tw = _build_single_marker_aperture_model(context)

    n1_table, _ = aperture_model.get_aperture_sigmas_at_element(
        element_name='m1',
        resolution=None,
        twiss=tw,
        method='exact',
        envelopes_num_points=144,
        output_cross_sections=False,
        output_max_envelopes=True,
    )
    sigmas = n1_table.n1
    envelope_points = n1_table.envelope

    ref_envelopes, _ = aperture_model.get_envelope_at_element(
        element_name='m1',
        sigmas=float(sigmas[0]),
        resolution=None,
        twiss=tw,
        envelopes_num_points=2048,
    )

    for exact_env, full_env in zip(envelope_points, ref_envelopes.cross_section):
        assert exact_env[:, 0].min() >= full_env[:, 0].min()
        assert exact_env[:, 0].max() <= full_env[:, 0].max()
        assert exact_env[:, 1].min() >= full_env[:, 1].min()
        assert exact_env[:, 1].max() <= full_env[:, 1].max()
        assert _polygon_area(exact_env) < _polygon_area(full_env)


@pytest.mark.parametrize('method', ['bisection', 'rays', 'exact'])
def test_get_aperture_sigmas_at_element_can_skip_optional_outputs(method, context):
    aperture_model, tw = _build_single_marker_aperture_model(context)

    table_with_outputs, _ = aperture_model.get_aperture_sigmas_at_element(
        element_name='m1',
        resolution=None,
        twiss=tw,
        method=method,
        envelopes_num_points=144,
        output_cross_sections=True,
        output_max_envelopes=True,
    )
    table_without_outputs, _ = aperture_model.get_aperture_sigmas_at_element(
        element_name='m1',
        resolution=None,
        twiss=tw,
        method=method,
        envelopes_num_points=144,
        output_cross_sections=False,
        output_max_envelopes=False,
    )
    sigmas_with_outputs = table_with_outputs.n1
    sigmas_without_outputs = table_without_outputs.n1
    aperture_points = table_with_outputs.cross_section
    envelope_points = table_with_outputs.envelope
    aperture_points_none = table_without_outputs._data.get('cross_section')
    envelope_points_none = table_without_outputs._data.get('envelope')

    assert aperture_points is not None
    assert envelope_points is not None
    assert aperture_points_none is None
    assert envelope_points_none is None
    xo.assert_allclose(sigmas_with_outputs, sigmas_without_outputs, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    'shape,aper_params,aper_tol,exn,eyn,gamma,betx,bety,x,y,halo_params',
    [
        ('ellipse', (1, 1.3), (0.01, 0.015, 0.01), 1e-3, 1e-3, 10, 1, 1, 0, 0, {}),
        ('circle', (1,), (0.01, 0.01, 0.02), 1e-3, 1e-3, 10, 1, 1, 0, 0,
            {'halo_primary': 1, 'halo_r': 2, 'halo_x': 2, 'halo_y': 2}),
        ('rectangle', (1, 1), (0.01, 0.02, 0.03), 1e-3, 1e-3, 10, 1, 1, 0, 0, {}),
        ('rectangle', (1.1, 1.2), (0.04, 0.02, 0.02), 1e-3, 1e-3, 10, 1, 1, -0.1, 0.2,
            {'halo_primary': 1, 'halo_r': 2, 'halo_x': 2, 'halo_y': 2}),
    ],
    ids=['ellipse-tols', 'circle-halo-tols', 'square-tols', 'square-orbit-halo-tols'],
)
def test_get_aperture_sigmas_at_element_vs_madx(
        shape,
        aper_params,
        aper_tol,
        exn,
        eyn,
        gamma,
        betx,
        bety,
        x,
        y,
        halo_params,
        context,
        sandbox_cwd,
):
    """Test the computation of sigmas vs MAD-X

    MAD-X uses a different approach to computing N1 when dispersion is present, hence we only test the cases without.
    """
    def madx_list(values):
        return '{' + ', '.join([str(v) for v in values]) + '}'

    halo_params_for_test = {
        'emitx_norm': exn,
        'emity_norm': eyn,
        'tol_beta_beating': 1,
        'tol_disp': 0,
        'tol_disp_ref_beta': 1,
        'tol_disp_ref': 0,
        'halo_primary': 1,
        'halo_r': 1,
        'halo_x': 1,
        'halo_y': 1,
        'tol_co': 0,
        'delta_rms': 0,
    }
    halo_params_for_test.copy()
    halo_params_for_test.update(halo_params)
    halo_params = halo_params_for_test

    mad = Madx(stdout=False)
    mad.input(f"""
        m1: marker, apertype = {shape}, aperture = {madx_list(aper_params)}, aper_tol = {madx_list(aper_tol)};

        seq: sequence,l = 1;
            m1, at=0;
        endsequence;

        beam, particle = proton, exn = {exn}, eyn = {eyn}, gamma = {gamma};
        use, sequence = seq;
        twiss, betx = {betx}, bety = {bety}, x = {x}, y = {y};

        aperture,
            dqf = {halo_params['tol_disp_ref']},
            betaqfx = {halo_params['tol_disp_ref_beta']},
            dp = {halo_params['delta_rms']},  ! called `twiss_deltap` in the table
            dparx = {halo_params['tol_disp']},
            dpary = {halo_params['tol_disp']},
            cor = {halo_params['tol_co']},
            bbeat = {halo_params['tol_beta_beating']},
            halo = {madx_list([halo_params[param] for param in ('halo_primary', 'halo_r', 'halo_x', 'halo_y')])};

        write, table = aperture;
    """)

    madx_n1 = mad.table['aperture'].n1[1]

    seq = xt.Line.from_madx_sequence(mad.sequence.seq, enable_layout_data=True)
    seq.set_particle_ref('proton', gamma0=mad.beam.gamma)
    tw = seq.twiss4d(betx=betx, bety=bety, x=x, y=y)

    xo.assert_allclose(mad.beam.gamma, seq.particle_ref.gamma0, atol=1e-10)
    xo.assert_allclose(mad.beam.beta, seq.particle_ref.beta0, atol=1e-10)

    aperture_model = Aperture.from_line_with_madx_metadata(seq, context=context)
    aperture_model.halo_params.update(halo_params)

    # Sanity checks
    aper_summ = mad.table.aperture.summary
    xo.assert_allclose(aperture_model.halo_params['tol_disp_ref'], aper_summ.dqf, atol=1e-8, rtol=0)
    xo.assert_allclose(aperture_model.halo_params['tol_disp_ref_beta'], aper_summ.betaqfx, atol=1e-8, rtol=0)
    xo.assert_allclose(aperture_model.halo_params['delta_rms'], aper_summ.dp_bucket_size, atol=1e-8, rtol=0)
    xo.assert_allclose(aperture_model.halo_params['tol_disp'], aper_summ.paras_dx, atol=1e-8, rtol=0)
    xo.assert_allclose(aperture_model.halo_params['tol_co'], aper_summ.co_radius, atol=1e-8, rtol=0)
    xo.assert_allclose(aperture_model.halo_params['tol_beta_beating'], aper_summ.beta_beating, atol=1e-8, rtol=0)
    xo.assert_allclose(aperture_model.halo_params['halo_primary'], aper_summ.halo_prim, atol=1e-8, rtol=0)
    xo.assert_allclose(aperture_model.halo_params['halo_r'], aper_summ.halo_r, atol=3e-6, rtol=0)
    xo.assert_allclose(aperture_model.halo_params['halo_x'], aper_summ.halo_h, atol=1e-8, rtol=0)
    xo.assert_allclose(aperture_model.halo_params['halo_y'], aper_summ.halo_v, atol=1e-8, rtol=0)
    xo.assert_allclose(aperture_model.halo_params['emitx_norm'], mad.beam.exn, atol=1e-8, rtol=0)
    xo.assert_allclose(aperture_model.halo_params['emity_norm'], mad.beam.eyn, atol=1e-8, rtol=0)

    # Compute n1 with Xsuite
    n1_table, tw = aperture_model.get_aperture_sigmas_at_element(
        element_name='m1',
        resolution=None,
        twiss=tw,
        envelopes_num_points=144,
        method='bisection',
        output_cross_sections=False,
        output_max_envelopes=False,
    )
    computed_n1 = n1_table.n1

    xo.assert_allclose(madx_n1, computed_n1, rtol=0.01)


def test_survey_resample_out_of_range_returns_nans_with_precision_tolerance(context):
    eps = 1e-6
    env = xt.Environment()
    bend = env.new(
        'bend', xt.Bend, length=1.0, angle=0.1, rot_s_rad=0.2)
    line = env.new_line(name='line', components=[bend])
    survey_table = line.survey()

    survey = SurveyData.from_survey_table(
        survey_table, line=line, context=context)

    assert 'angle' not in survey_table._col_names
    assert 'rot_s_rad' not in survey_table._col_names
    xo.assert_allclose(survey.angle, [0.1, 0.0], atol=0, rtol=0)
    xo.assert_allclose(survey.tilt, [0.2, 0.0], atol=0, rtol=0)

    s_query = np.array([
        -2 * eps,
        -0.5 * eps,
        0.0,
        1.0,
        1.0 + 0.5 * eps,
        1.0 + 2 * eps,
    ], dtype=float)
    resampled = survey.resample(s_query)

    assert np.isnan(resampled.s[0])
    assert np.isnan(resampled.pose[0, 0, 0])
    xo.assert_allclose(resampled.s[1], 0.0, atol=0, rtol=0)
    xo.assert_allclose(resampled.s[2], 0.0, atol=0, rtol=0)
    xo.assert_allclose(resampled.s[3], 1.0, atol=0, rtol=0)
    xo.assert_allclose(resampled.s[4], 1.0, atol=0, rtol=0)
    assert np.isnan(resampled.s[5])
    assert np.isnan(resampled.pose[5, 0, 0])


@pytest.mark.parametrize(
    "rbend_model,angle_diff,default_entry_shift",
    [
        ('straight-body', 'angle', False),
        ('straight-body', 'zero', False),
        ('curved-body', 'angle', False),
        ('straight-body', 'zero', True),
    ],
)
@pytest.mark.parametrize(
    "rot_s_rad,shift_y",
    [
        (0, 0),
        (0.3, 0.4),
    ],
)
@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_survey_resample_rbend_matches_sliced_survey(
    test_context,
    rbend_model,
    angle_diff,
    default_entry_shift,
    rot_s_rad,
    shift_y,
):
    angle = np.pi / 6
    env = xt.Environment()
    drift = env.new('drift', xt.Drift, length=10)
    env.new(
        'rbend',
        xt.RBend,
        length_straight=10,
        angle=angle,
        rbend_angle_diff=angle if angle_diff == 'angle' else 0,
        rbend_model=rbend_model,
        rot_s_rad=rot_s_rad,
        shift_y=shift_y,
    )
    rbend = env['rbend']

    if not default_entry_shift:
        rbend.rbend_compensate_sagitta = False
        # Ensure that the survey inside the bend is continuous at the entry
        rbend.rbend_shift = (
            np.cos(rbend._angle_in)
            - np.sqrt(1 - (np.sin(rbend._angle_in) - 0.5 * rbend.h * rbend.length_straight ) ** 2)
        ) / rbend.h

    line = env.new_line(components=[drift, 'rbend', drift])
    thick_survey = line.survey()
    survey_data = SurveyData.from_survey_table(
        thick_survey, line=line, context=test_context)
    survey_poses = survey_data.pose.to_nparray()

    # RBend metadata is auxiliary: converting the survey must not alter any
    # position or orientation already stored in the original table.
    xo.assert_allclose(
        survey_poses[:, :3, :3],
        thick_survey.E_matrix,
        atol=0,
        rtol=0,
    )
    xo.assert_allclose(
        survey_poses[:, :3, 3],
        thick_survey.XYZ,
        atol=0,
        rtol=0,
    )

    rbend_start = 10.0
    cuts = np.linspace(rbend_start + 0.1, rbend_start + rbend.length - 0.1, 17)

    # An explicitly sliced line provides the reference poses inside the RBend,
    # including any transverse shift or roll applied to the element.
    sliced_line = line.copy()
    sliced_line.cut_at_s(cuts)
    sliced_survey = sliced_line.survey()

    sliced_xyz = []
    sliced_orientations = []
    for s_cut in cuts:
        matches = np.flatnonzero(np.isclose(sliced_survey.s, s_cut, atol=1e-12, rtol=0))
        assert len(matches) == 1
        sliced_xyz.append(sliced_survey.XYZ[matches[0]])
        sliced_orientations.append(sliced_survey.E_matrix[matches[0]])

    resampled = survey_data.resample(cuts)
    resampled_poses = resampled.pose.to_nparray()
    xo.assert_allclose(
        resampled_poses[:, :3, 3],
        sliced_xyz,
        atol=2e-13,
        rtol=0,
    )
    xo.assert_allclose(
        resampled_poses[:, :3, :3],
        sliced_orientations,
        atol=2e-13,
        rtol=0,
    )


@pytest.mark.parametrize(
    'rot_x_rad,rot_y_rad,dx,dy,ds1,ds2,ds_bounds1,ds_bounds2',
    [
        (0, 0, 0, 0, 0, 0, 0, 0),
        (np.deg2rad(45), np.deg2rad(30), np.sqrt(3), 1, 1, 1, 1 / np.sqrt(2), 0.5),
    ]
)
@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_aperture_bounds_straight_survey(rot_x_rad, rot_y_rad, dx, dy, ds1, ds2, ds_bounds1, ds_bounds2, test_context):
    env = xt.Environment()
    drift = env.new('drift', xt.Drift, length=1)
    line = env.new_line(name='line', components=10 * [drift])
    sv = line.survey()

    circle = Circle(radius=1)
    rectangle = Rectangle(half_width=0.6, half_height=0.4)

    profiles = [
        Profile(shape=circle, tol_r=0, tol_x=0, tol_y=0),
        Profile(shape=rectangle, tol_r=0, tol_x=0, tol_y=0),
    ]

    profile_positions = [
        ProfilePosition(profile_index=0, shift_s=-1.5),
        ProfilePosition(profile_index=0, shift_s=0.5, rot_x_rad=rot_x_rad),
        ProfilePosition(profile_index=0, shift_s=2.5, rot_y_rad=rot_y_rad),
        ProfilePosition(profile_index=0, shift_s=8.5),
    ]

    pipes = [
        Pipe(curvature=0., positions=profile_positions),
    ]

    pipe_positions = [
        PipePosition(
            pipe_index=0,
            survey_reference_name='drift::0',
            survey_index=sv.name.tolist().index('drift::0'),
            transformation=transform_matrix(
                shift_z=1.5,
                shift_x=dx,
                shift_y=dy,
            ),
        ),
    ]

    model = ApertureModel(
        line=line,
        pipe_positions=pipe_positions,
        pipes=pipes,
        profiles=profiles,
        pipe_names=['pipe0'],
        pipe_position_names=['pipe0'],
        profile_names=['circle', 'rectangle'],
    )

    # Skip validity check as in this case some profiles are outside the survey
    ap = Aperture(line=line, model=model, context=test_context, _skip_validity_check=True)

    xo.assert_allclose(ap._aperture_bounds.s_positions[0], 0, atol=1e-6, rtol=1e-8)
    xo.assert_allclose(ap._aperture_bounds.s_start[0], 0, atol=1e-6, rtol=1e-8)
    xo.assert_allclose(ap._aperture_bounds.s_end[0], 0, atol=1e-6, rtol=1e-8)

    xo.assert_allclose(ap._aperture_bounds.s_positions[1], 2 + ds1, atol=1e-6, rtol=1e-8)
    xo.assert_allclose(ap._aperture_bounds.s_start[1], 2 - ds_bounds1, atol=1e-4, rtol=1e-8)
    xo.assert_allclose(ap._aperture_bounds.s_end[1], 2 + ds_bounds1, atol=1e-4, rtol=1e-8)

    xo.assert_allclose(ap._aperture_bounds.s_positions[2], 4 + ds2, atol=1e-6, rtol=1e-8)
    xo.assert_allclose(ap._aperture_bounds.s_start[2], 4 - ds_bounds2, atol=2e-4, rtol=1e-8)  # atol < 1mm but quite high
    xo.assert_allclose(ap._aperture_bounds.s_end[2], 4 + ds_bounds2, atol=2e-4, rtol=1e-8)  # ditto

    xo.assert_allclose(ap._aperture_bounds.s_positions[3], 10, atol=1e-6, rtol=1e-8)
    xo.assert_allclose(ap._aperture_bounds.s_start[3], 10, atol=1e-6, rtol=1e-8)
    xo.assert_allclose(ap._aperture_bounds.s_end[3], 10, atol=1e-6, rtol=1e-8)


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_aperture_bounds_and_cross_sections_curved_survey_follows_pipe(test_context):
    env = xt.Environment()
    angle = np.deg2rad(35.0)
    length = 3.2
    radius = 0.6

    bend = env.new('bend', xt.Bend, length=length, angle=angle, k0=0)
    drift = env.new('drift', xt.Drift, length=length)
    anti_bend = env.new('anti_bend', xt.Bend, length=length, angle=-angle, k0=0)
    line = env.new_line(name='line', components=[bend, drift, anti_bend])
    sv = line.survey()

    shape = Circle(radius=radius)
    profiles = [
        Profile(shape=shape, tol_r=0, tol_x=0, tol_y=0),
    ]
    profile_positions = [
        ProfilePosition(profile_index=0, shift_s=0.0),
        ProfilePosition(profile_index=0, shift_s=length),
    ]

    model = ApertureModel(
        line=line,
        pipe_positions=[
            PipePosition(
                pipe_index=0,
                survey_reference_name=sv.name[0],
                survey_index=0,
                transformation=transform_matrix(),
            ),
            PipePosition(
                pipe_index=1,
                survey_reference_name=sv.name[1],
                survey_index=1,
                transformation=transform_matrix(),
            ),
            PipePosition(
                pipe_index=2,
                survey_reference_name=sv.name[2],
                survey_index=2,
                transformation=transform_matrix(),
            ),
        ],
        pipes=[
            Pipe(curvature=angle / length, positions=profile_positions),
            Pipe(curvature=0, positions=profile_positions),
            Pipe(curvature=-angle / length, positions=profile_positions),
        ],
        profiles=profiles,
        pipe_names=['pipe0', 'type1', 'type2'],
        pipe_position_names=['pipe0', 'type1', 'type2'],
        profile_names=['circ0'],
    )

    ap = Aperture(line=line, model=model, num_profile_points=256, context=test_context)

    bounds_table = ap.get_bounds_table()
    bounds_s = [0, length, length, 2 * length, 2 * length, 3 * length]
    xo.assert_allclose(bounds_table.s, bounds_s, atol=1e-6, rtol=1e-6)
    xo.assert_allclose(bounds_table.s_start, bounds_s, atol=1e-6, rtol=1e-6)
    xo.assert_allclose(bounds_table.s_end, bounds_s, atol=1e-6, rtol=1e-6)
    assert all(bounds_table.pipe_name == ['pipe0', 'pipe0', 'type1', 'type1', 'type2', 'type2'])
    assert all(bounds_table.profile_name == ['circ0'])

    s_samples = np.linspace(0, 3 * length, 51, dtype=FloatType._dtype)
    sections_table = ap.cross_sections_at_s(s_samples)
    sections = sections_table.cross_section

    for ii in range(1, len(sections)):
        xo.assert_allclose(np.linalg.norm(sections[ii], axis=1), radius, atol=1e-6, rtol=0)


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_aperture_bounds_follow_straight_body_rbend(test_context):
    env = xt.Environment()
    angle = np.deg2rad(35.0)
    env.new(
        'rbend',
        xt.RBend,
        length_straight=3.2,
        angle=angle,
        rbend_angle_diff=angle,
        rbend_model='straight-body',
    )
    rbend = env['rbend']
    line = env.new_line(name='line', components=['rbend'])

    builder = ApertureBuilder(line)
    builder.new_profile('circle', Circle, radius=0.1)
    builder.new_pipe(
        'pipe',
        positions=[
            builder.place_profile(
                'circle',
                shift_s=0.25 * rbend.length_straight,
            ),
            builder.place_profile(
                'circle',
                shift_s=0.75 * rbend.length_straight,
            ),
        ],
    )
    builder.place_pipe('pipe', 'pipe', 'rbend')
    model = builder.build(context=test_context)

    aperture = Aperture(
        line=line,
        model=model,
        context=test_context,
        num_profile_points=64,
    )
    bounds = aperture.get_bounds_table()

    # Profiles installed along the straight body still use the RBend's survey
    # length as their longitudinal coordinate.
    expected_s = rbend.length * np.array([0.25, 0.75])
    xo.assert_allclose(
        bounds.s,
        expected_s,
        atol=1e-12,
        rtol=0,
    )
    xo.assert_allclose(bounds.s_start, expected_s, atol=1e-12, rtol=0)
    xo.assert_allclose(bounds.s_end, expected_s, atol=1e-12, rtol=0)


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_aperture_bounds_ignore_translation_discontinuity(
    test_context,
):
    env = xt.Environment()
    env.new('drift_before', xt.Drift, length=1)
    env.new('translation', xt.Translation, shift_x=10)
    env.new('drift_after', xt.Drift, length=1)
    line = env.new_line(
        name='line',
        components=['drift_before', 'translation', 'drift_after'],
    )

    builder = ApertureBuilder(line)
    builder.new_profile('circle', Circle, radius=0.01)
    builder.new_pipe(
        'pipe',
        positions=[builder.place_profile('circle')],
    )
    builder.place_pipe(
        'pipe',
        'pipe',
        'translation',
        # The profile plane intersects the translated drift at s=1.5.
        shift_x=10,
        shift_z=0.5,
        rot_y_rad=3 * np.pi / 4,
    )
    model = builder.build(context=test_context)

    aperture = Aperture(
        line=line,
        model=model,
        context=test_context,
        num_profile_points=64,
    )
    bounds = aperture.get_bounds_table()

    # The zero-length Translation is a discontinuity, not a segment joining
    # its unshifted pose to the shifted pose of the following drift.
    xo.assert_allclose(bounds.s, [1.5], atol=1e-12, rtol=0)
    assert bounds.s_start[0] < bounds.s[0] < bounds.s_end[0]


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_aperture_bounds_and_cross_sections_large_curved_ring_follows_pipe(test_context):
    env = xt.Environment()

    num_bends = 720
    bend_angle = np.deg2rad(0.5)
    ring_length = 30_000.0
    bend_length = ring_length / num_bends
    aperture_radius = 0.03

    bend = env.new('bend', xt.Bend, length=bend_length, angle=bend_angle, k0=0)
    line = env.new_line(name='line', components=[bend] * num_bends)
    sv = line.survey()

    shape = Circle(radius=aperture_radius)
    profiles = [Profile(shape=shape, tol_r=0, tol_x=0, tol_y=0)]
    profile_positions = [
        ProfilePosition(profile_index=0, shift_s=0.0),
        ProfilePosition(profile_index=0, shift_s=bend_length),
    ]

    pipe_positions = [
        PipePosition(
            pipe_index=ii,
            survey_reference_name=sv.name[ii],
            survey_index=ii,
            transformation=transform_matrix(),
        )
        for ii in range(num_bends)
    ]
    pipes = [Pipe(curvature=bend_angle / bend_length, positions=profile_positions)] * num_bends

    model = ApertureModel(
        line=line,
        pipe_positions=pipe_positions,
        pipes=pipes,
        profiles=profiles,
        pipe_names=[f'type{ii}' for ii in range(num_bends)],
        pipe_position_names=[f'type{ii}' for ii in range(num_bends)],
        profile_names=['circ0'],
    )

    ap = Aperture(
        line=line,
        model=model,
        num_profile_points=64,
        context=test_context,
        _skip_validity_check=True,
    )

    bounds_table = ap.get_bounds_table()
    expected_s = np.repeat(np.arange(1, num_bends + 1, dtype=FloatType._dtype) * bend_length, 2)
    expected_s[::2] -= bend_length

    xo.assert_allclose(bounds_table.s, expected_s, atol=1e-6, rtol=0)
    xo.assert_allclose(bounds_table.s_start, expected_s, atol=1e-6, rtol=0)
    xo.assert_allclose(bounds_table.s_end, expected_s, atol=1e-6, rtol=0)

    assert np.all(np.diff(bounds_table.s) >= -1e-6)
    assert np.all(np.isfinite(bounds_table.s))
    assert np.all(np.isfinite(bounds_table.s_start))
    assert np.all(np.isfinite(bounds_table.s_end))

    s_samples = np.linspace(0, ring_length, 101, dtype=FloatType._dtype)
    sections = ap.cross_sections_at_s(s_samples).cross_section
    radii = np.linalg.norm(sections, axis=2)
    xo.assert_allclose(radii, aperture_radius, atol=1e-6, rtol=0)


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_aperture_bounds_large_curved_ring_with_shifted_survey_references(test_context):
    env = xt.Environment()

    num_bends = 720
    bend_angle = np.deg2rad(0.5)
    ring_length = 30_000.0
    bend_length = ring_length / num_bends
    aperture_radius = 0.03

    bend = env.new('bend', xt.Bend, length=bend_length, angle=bend_angle, k0=0)
    line = env.new_line(name='line', components=[bend] * num_bends)
    sv = line.survey()

    shape = Circle(radius=aperture_radius)
    profiles = [Profile(shape=shape, tol_r=0, tol_x=0, tol_y=0)]

    # Same physical profile planes, expressed in three different reference frames.
    pipes = [
        Pipe(
            curvature=bend_angle / bend_length,
            positions=[
                ProfilePosition(profile_index=0, shift_s=0.0),
                ProfilePosition(profile_index=0, shift_s=bend_length),
            ],
        ),
        Pipe(
            curvature=bend_angle / bend_length,
            positions=[
                ProfilePosition(profile_index=0, shift_s=-bend_length),
                ProfilePosition(profile_index=0, shift_s=0.0),
            ],
        ),
        Pipe(
            curvature=bend_angle / bend_length,
            positions=[
                ProfilePosition(profile_index=0, shift_s=-2 * bend_length),
                ProfilePosition(profile_index=0, shift_s=-bend_length),
            ],
        ),
    ]

    pipe_positions = []
    for ii in range(num_bends):
        # Cycle references where possible; near the end stay in-range.
        if ii <= num_bends - 3:
            shift = ii % 3
        else:
            shift = 0

        pipe_positions.append(
            PipePosition(
                pipe_index=shift,
                survey_reference_name=sv.name[ii + shift],
                survey_index=ii + shift,
                transformation=transform_matrix(),
            )
        )

    model = ApertureModel(
        line=line,
        pipe_positions=pipe_positions,
        pipes=pipes,
        profiles=profiles,
        pipe_names=['type_ref0', 'type_ref1', 'type_ref2'],
        pipe_position_names=[['type_ref0', 'type_ref1', 'type_ref2'][tp.pipe_index] for tp in pipe_positions],
        profile_names=['circ0'],
    )

    ap = Aperture(
        line=line,
        model=model,
        num_profile_points=64,
        context=test_context,
        _skip_validity_check=True,
    )

    bounds_table = ap.get_bounds_table()
    expected_s = np.repeat(np.arange(1, num_bends + 1, dtype=FloatType._dtype) * bend_length, 2)
    expected_s[::2] -= bend_length

    xo.assert_allclose(bounds_table.s, expected_s, atol=1e-6, rtol=0)
    xo.assert_allclose(bounds_table.s_start, expected_s, atol=1e-6, rtol=0)
    xo.assert_allclose(bounds_table.s_end, expected_s, atol=1e-6, rtol=0)

    assert np.all(np.diff(bounds_table.s) >= -1e-6)
    assert np.all(np.isfinite(bounds_table.s))
    assert np.all(np.isfinite(bounds_table.s_start))
    assert np.all(np.isfinite(bounds_table.s_end))

    s_samples = np.linspace(0, ring_length, 101, dtype=FloatType._dtype)
    sections = ap.cross_sections_at_s(s_samples).cross_section
    radii = np.linalg.norm(sections, axis=2)
    xo.assert_allclose(radii, aperture_radius, atol=1e-6, rtol=0)


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_aperture_bounds_large_curved_ring_single_type_wraparound_regression(test_context):
    env = xt.Environment()

    num_bends = 720
    bend_angle = np.deg2rad(0.5)
    ring_length = 30_000.0
    bend_length = ring_length / num_bends
    aperture_radius = 0.03

    bend = env.new('bend', xt.Bend, length=bend_length, angle=bend_angle, k0=0)
    line = env.new_line(name='line', components=[bend] * num_bends)
    sv = line.survey()

    model = ApertureModel(
        line=line,
        pipe_positions=[
            PipePosition(
                pipe_index=0,
                survey_reference_name=sv.name[num_bends - 1],
                survey_index=num_bends - 1,
                # Shift the type forward so the installed profiles should wrap to small s.
                transformation=transform_matrix(shift_z=2 * bend_length),
            )
        ],
        pipes=[
            Pipe(
                curvature=bend_angle / bend_length,
                positions=[
                    ProfilePosition(profile_index=0, shift_s=0.0),
                    ProfilePosition(profile_index=0, shift_s=bend_length),
                ],
            )
        ],
        profiles=[Profile(shape=Circle(radius=aperture_radius), tol_r=0, tol_x=0, tol_y=0)],
        pipe_names=['wrapped_type'],
        pipe_position_names=['wrapped_type'],
        profile_names=['circ0'],
    )

    ap = Aperture(
        line=line,
        model=model,
        num_profile_points=64,
        context=test_context,
        _skip_validity_check=True,
    )

    bounds_table = ap.get_bounds_table()
    expected_s = np.array([bend_length, 2 * bend_length], dtype=FloatType._dtype)

    assert np.all(np.isfinite(bounds_table.s))
    xo.assert_allclose(bounds_table.s, expected_s, atol=5e-3, rtol=0)
    xo.assert_allclose(bounds_table.s_start, expected_s, atol=3e-2, rtol=0)
    xo.assert_allclose(bounds_table.s_end, expected_s, atol=3e-2, rtol=0)


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_cross_sections_at_s_interpolate_circles_to_cone(test_context):
    env = xt.Environment()
    length = 1.0
    angle = np.deg2rad(30.0)
    l_straight = 1.0 / np.sin(angle / 2)
    rho = 0.5 * l_straight / np.sin(angle / 2)
    l_curv = rho * angle

    drift = env.new('drift', xt.Drift, length=length)
    bend_plus = env.new('bend_plus', xt.Bend, length=l_curv, angle=angle, k0=0)
    bend_minus = env.new('bend_minus', xt.Bend, length=l_curv, angle=-angle, k0=0)
    line = env.new_line(name='line', components=[drift, bend_plus, drift, drift, bend_minus, drift])
    sv = line.survey()

    s0, s1 = 0.0, 11.0
    r0, r1 = 0.8, 2.0

    profiles = [
        Profile(shape=Circle(radius=r0), tol_r=0, tol_x=0, tol_y=0),
        Profile(shape=Circle(radius=r1), tol_r=0, tol_x=0, tol_y=0),
    ]
    profile_positions = [
        ProfilePosition(profile_index=0, shift_s=s0),
        ProfilePosition(profile_index=1, shift_s=s1),
    ]

    model = ApertureModel(
        line=line,
        pipe_positions=[
            PipePosition(
                pipe_index=0,
                survey_reference_name=sv.name[0],
                survey_index=0,
                transformation=transform_matrix(shift_x=-1.5),
            ),
        ],
        pipes=[Pipe(curvature=0.0, positions=profile_positions)],
        profiles=profiles,
        pipe_names=['pipe0'],
        pipe_position_names=['pipe0'],
        profile_names=['circle0', 'circle1'],
    )

    ap = Aperture(line=line, model=model, context=test_context)

    s_samples = np.linspace(1.0, 11.0, 21, dtype=FloatType._dtype)
    sections_table = ap.cross_sections_at_s(s_samples)
    sections = sections_table.cross_section
    poses = sections_table.pose

    # Transform all cross-section points to the (fixed) type frame.
    # In this frame, two circle profiles at z=s0/s1 define a cone:
    # sqrt(x^2 + y^2) == r0 + (r1-r0) * (z-s0)/(s1-s0)
    sv_ref = sv.rows[0]
    sv_ref_mat = np.identity(4)
    sv_ref_mat[:3, 0] = sv_ref.ex
    sv_ref_mat[:3, 1] = sv_ref.ey
    sv_ref_mat[:3, 2] = sv_ref.ez
    sv_ref_mat[:3, 3] = np.array([sv_ref.X[0], sv_ref.Y[0], sv_ref.Z[0]])
    world_from_type = sv_ref_mat @ model.pipe_positions[0].transformation.to_nparray()
    pipe_from_world = np.linalg.inv(world_from_type)

    for ii in range(len(s_samples)):
        sec_xy = sections[ii]
        assert not np.isnan(sec_xy).any()

        sec_hom = np.column_stack([
            sec_xy,
            np.zeros(len(sec_xy), dtype=FloatType._dtype),
            np.ones(len(sec_xy), dtype=FloatType._dtype),
        ])
        sec_world = (poses[ii] @ sec_hom.T).T
        sec_type = (pipe_from_world @ sec_world.T).T

        rr = np.linalg.norm(sec_type[:, :2], axis=1)
        z = sec_type[:, 2]
        expected_r = r0 + (r1 - r0) * (z - s0) / (s1 - s0)

        xo.assert_allclose(rr, expected_r, atol=1e-6, rtol=0)


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_cross_sections_at_s_interpolates_tolerances(test_context):
    env = xt.Environment()
    line = env.new_line(name='line', components=[env.new('drift', xt.Drift, length=10.0)])
    sv = line.survey()

    profiles = [
        Profile(shape=Circle(radius=1.0), tol_r=0.1, tol_x=0.2, tol_y=0.3),
        Profile(shape=Circle(radius=1.0), tol_r=0.5, tol_x=0.6, tol_y=0.7),
    ]
    profile_positions = [
        ProfilePosition(profile_index=0, shift_s=0.0),
        ProfilePosition(profile_index=1, shift_s=10.0),
    ]

    model = ApertureModel(
        line=line,
        pipe_positions=[
            PipePosition(
                pipe_index=0,
                survey_reference_name=sv.name[0],
                survey_index=0,
                transformation=transform_matrix(),
            ),
        ],
        pipes=[Pipe(curvature=0.0, positions=profile_positions)],
        profiles=profiles,
        pipe_names=['pipe0'],
        pipe_position_names=['pipe0'],
        profile_names=['profile0', 'profile1'],
    )

    ap = Aperture(line=line, model=model, context=test_context)

    s_samples = np.array([0.0, 2.5, 5.0, 7.5, 10.0], dtype=FloatType._dtype)
    sections_table = ap.cross_sections_at_s(s_samples)

    xo.assert_allclose(sections_table.tol_r, [0.1, 0.2, 0.3, 0.4, 0.5], atol=1e-12, rtol=0)
    xo.assert_allclose(sections_table.tol_x, [0.2, 0.3, 0.4, 0.5, 0.6], atol=1e-12, rtol=0)
    xo.assert_allclose(sections_table.tol_y, [0.3, 0.4, 0.5, 0.6, 0.7], atol=1e-12, rtol=0)


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_cross_sections_at_s_curved_type_preserves_profile_shape(test_context):
    env = xt.Environment()
    angle = np.deg2rad(35.0)
    length = 3.2
    radius = 1.4

    bend_name = env.new('bend', xt.Bend, length=length, angle=angle, k0=0)
    line = env.new_line(name='line', components=[bend_name])
    sv = line.survey()

    shape = Circle(radius=radius)
    profiles = [
        Profile(shape=shape, tol_r=0, tol_x=0, tol_y=0),
    ]
    profile_positions = [
        ProfilePosition(profile_index=0, shift_s=0.0),
        ProfilePosition(profile_index=0, shift_s=length),
    ]

    model = ApertureModel(
        line=line,
        pipe_positions=[
            PipePosition(
                pipe_index=0,
                survey_reference_name=sv.name[0],
                survey_index=0,
                transformation=transform_matrix(),
            ),
        ],
        pipes=[Pipe(curvature=angle / length, positions=profile_positions)],
        profiles=profiles,
        pipe_names=['pipe0'],
        pipe_position_names=['pipe0'],
        profile_names=['circ0'],
    )

    ap = Aperture(line=line, model=model, context=test_context, num_profile_points=256)

    s_samples = np.linspace(0.0, length, 33, dtype=FloatType._dtype)
    sections = ap.cross_sections_at_s(s_samples).cross_section

    for ii in range(1, len(sections)):
        xo.assert_allclose(np.linalg.norm(sections[ii], axis=1), radius, atol=1e-6, rtol=0)


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_cross_sections_at_s_returns_axis_extents(test_context):
    env = xt.Environment()
    line = env.new_line(name='line', components=[env.new('drift', xt.Drift, length=1.0)])
    sv = line.survey()

    model = ApertureModel(
        line=line,
        pipe_positions=[
            PipePosition(
                pipe_index=0,
                survey_reference_name=sv.name[0],
                survey_index=0,
                transformation=transform_matrix(),
            ),
        ],
        pipes=[
            Pipe(
                curvature=0.0,
                positions=[
                    ProfilePosition(profile_index=0, shift_s=0.0),
                    ProfilePosition(profile_index=0, shift_s=1.0),
                ],
            ),
        ],
        profiles=[Profile(shape=Rectangle(half_width=2.0, half_height=1.5), tol_r=0, tol_x=0, tol_y=0)],
        pipe_names=['pipe0'],
        pipe_position_names=['pipe0'],
        profile_names=['profile0'],
    )

    ap = Aperture(line=line, model=model, context=test_context)
    sections_table = ap.cross_sections_at_s([0.0, 0.5], extents=True, polygons=False)
    sections_with_polygons = ap.cross_sections_at_s([0.0, 0.5], extents=True)

    xo.assert_allclose(sections_table.min_x, [-2.0, -2.0], atol=1e-12, rtol=0)
    xo.assert_allclose(sections_table.max_x, [2.0, 2.0], atol=1e-12, rtol=0)
    xo.assert_allclose(sections_table.min_y, [-1.5, -1.5], atol=1e-12, rtol=0)
    xo.assert_allclose(sections_table.max_y, [1.5, 1.5], atol=1e-12, rtol=0)
    for column in ('min_x', 'max_x', 'min_y', 'max_y'):
        xo.assert_allclose(
            sections_table[column],
            sections_with_polygons[column],
            atol=1e-12,
            rtol=0,
        )
    assert 'cross_section' not in sections_table._col_names
    assert 'cross_section' in sections_with_polygons._col_names


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_cross_sections_at_s_invalid_section_has_nan_polygon_and_extents(test_context):
    env = xt.Environment()
    line = env.new_line(name='line', components=[env.new('drift', xt.Drift, length=3.0)])
    sv = line.survey()

    model = ApertureModel(
        line=line,
        pipe_positions=[
            PipePosition(
                pipe_index=0,
                survey_reference_name=sv.name[0],
                survey_index=0,
                transformation=transform_matrix(),
            ),
        ],
        pipes=[
            Pipe(
                curvature=0.0,
                positions=[
                    ProfilePosition(profile_index=0, shift_s=1.0),
                    ProfilePosition(profile_index=0, shift_s=2.0),
                ],
            ),
        ],
        profiles=[Profile(shape=Circle(radius=0.1), tol_r=0, tol_x=0, tol_y=0)],
        pipe_names=['pipe0'],
        pipe_position_names=['pipe0'],
        profile_names=['profile0'],
    )

    ap = Aperture(line=line, model=model, context=test_context)
    sections = ap.cross_sections_at_s([0.0], extents=True)

    assert np.all(np.isnan(sections.cross_section))
    assert np.isnan(sections.min_x[0])
    assert np.isnan(sections.max_x[0])
    assert np.isnan(sections.min_y[0])
    assert np.isnan(sections.max_y[0])


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_cross_sections_at_s_wraps_profile_neighbours_on_ring(test_context):
    env = xt.Environment()
    bend = env.new('bend', xt.Bend, length=1.0, angle=np.pi / 2, k0=0)
    line = env.new_line(name='line', components=[bend] * 4)
    sv = line.survey()

    model = ApertureModel(
        line=line,
        pipe_positions=[
            PipePosition(
                pipe_index=0,
                survey_reference_name=sv.name[0],
                survey_index=0,
                transformation=transform_matrix(),
            ),
            PipePosition(
                pipe_index=0,
                survey_reference_name=sv.name[3],
                survey_index=3,
                transformation=transform_matrix(),
            ),
        ],
        pipes=[
            Pipe(
                curvature=np.pi / 2,
                positions=[ProfilePosition(profile_index=0, shift_s=0.5)],
            ),
        ],
        profiles=[Profile(shape=Circle(radius=0.01), tol_r=0, tol_x=0, tol_y=0)],
        pipe_names=['pipe0'],
        pipe_position_names=['after_seam', 'before_seam'],
        profile_names=['profile0'],
    )

    aperture = Aperture(line=line, model=model, context=test_context, is_ring=True)
    bounds = aperture.get_bounds_table()
    xo.assert_allclose(bounds.s, [0.5, 3.5], atol=1e-12, rtol=0)

    sections = aperture.cross_sections_at_s([0.1])

    assert np.all(np.isfinite(sections.cross_section))
    # TODO: This only asserts that there is a cross section. We should assert on the shape, but since this is
    #  interpolated across pipes, it falls back to a straight line interpolation, so the shape is not a circle.
    #  Perhaps we should still do the curved interpolation if the curvature is constant between the interpolands.


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_cross_sections_at_s_compare_straight_curved(test_context):
    env = xt.Environment()
    angle = np.deg2rad(35.0)
    length = 3.2

    drift = env.new('drift', xt.Drift, length=length)
    bend = env.new('bend', xt.Bend, length=length, angle=angle)
    line = env.new_line(name='line', components=[drift, bend])
    sv = line.survey()

    circle = Circle(radius=1.4)
    rectangle = Rectangle(half_width=0.4, half_height=1.9)
    profiles = [
        Profile(shape=rectangle, tol_r=0, tol_x=0, tol_y=0),
        Profile(shape=circle, tol_r=0, tol_x=0, tol_y=0),
    ]
    profile_positions = [
        ProfilePosition(profile_index=0, shift_s=0.0),
        ProfilePosition(profile_index=1, shift_s=length),
    ]

    model = ApertureModel(
        line=line,
        pipes=[
            Pipe(curvature=0, positions=profile_positions),
            Pipe(curvature=angle / length, positions=profile_positions),
        ],
        pipe_positions=[
            PipePosition(
                pipe_index=0,
                survey_reference_name='drift',
                survey_index=list(sv.name).index('drift'),
                transformation=np.identity(4),
            ),
            PipePosition(
                pipe_index=1,
                survey_reference_name='bend',
                survey_index=list(sv.name).index('bend'),
                transformation=np.identity(4),
            ),
        ],
        profiles=profiles,
        pipe_names=['type_straight', 'type_curv'],
        pipe_position_names=['type_straight', 'type_curv'],
        profile_names=['rect0', 'circ0'],
    )

    ap = Aperture(line=line, model=model, context=test_context, num_profile_points=256)

    s_samples0 = np.linspace(0.1, length - 0.1, 33, dtype=FloatType._dtype)
    s_samples1 = s_samples0 + length

    sections_straight = ap.cross_sections_at_s(s_samples0).cross_section
    sections_curv = ap.cross_sections_at_s(s_samples1).cross_section

    # Compare up to cyclic polygon indexing: point ordering can differ by a
    # rotation along the closed contour while representing the same shape.
    for ii in range(sections_straight.shape[0]):
        sec_ref = sections_straight[ii]
        sec_cur = sections_curv[ii]

        best_shift = 0
        best_cost = np.inf
        for shift in range(sec_cur.shape[0]):
            sec_shifted = np.roll(sec_cur, -shift, axis=0)
            cost = np.sum((sec_ref - sec_shifted) ** 2)
            if cost < best_cost:
                best_cost = cost
                best_shift = shift

        xo.assert_allclose(sec_ref, np.roll(sec_cur, -best_shift, axis=0), atol=1e-6, rtol=0)


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_cross_sections_at_s_interpolated_sections_stay_closed(test_context):
    env = xt.Environment()
    angle = np.deg2rad(35.0)
    length = 3.2

    bend = env.new('bend', xt.Bend, length=length, angle=angle)
    line = env.new_line(name='line', components=[bend])
    sv = line.survey()

    rectangle = Rectangle(half_width=0.4, half_height=1.9)
    circle = Circle(radius=1.4)
    profiles = [
        Profile(shape=rectangle, tol_r=0, tol_x=0, tol_y=0),
        Profile(shape=circle, tol_r=0, tol_x=0, tol_y=0),
    ]
    profile_positions = [
        ProfilePosition(profile_index=0, shift_s=0.0),
        ProfilePosition(profile_index=1, shift_s=length),
    ]

    model = ApertureModel(
        line=line,
        pipe_positions=[
            PipePosition(
                pipe_index=0,
                survey_reference_name='bend',
                survey_index=list(sv.name).index('bend'),
                transformation=np.identity(4),
            ),
        ],
        pipes=[Pipe(curvature=angle / length, positions=profile_positions)],
        profiles=profiles,
        pipe_names=['type_curv'],
        pipe_position_names=['type_curv'],
        profile_names=['rect0', 'circ0'],
    )

    ap = Aperture(line=line, model=model, context=test_context, num_profile_points=256)

    s_samples = np.linspace(0.1, length - 0.1, 33, dtype=FloatType._dtype)
    sections = ap.cross_sections_at_s(s_samples).cross_section

    xo.assert_allclose(sections[:, 0, :], sections[:, -1, :], atol=1e-12, rtol=0)


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_open_line_aperture_bounds_do_not_wrap_search(test_context):
    env = xt.Environment()

    length = 1.0
    dx = 1.0
    angle = np.deg2rad(30.0)
    l_straight = dx / np.sin(angle / 2)
    rho = 0.5 * l_straight / np.sin(angle / 2)
    l_curv = rho * angle

    drift = env.new('drift', xt.Drift, length=length)
    rot_plus = env.new('rot_plus', xt.Bend, length=l_curv, angle=angle, k0=0)
    rot_minus = env.new('rot_minus', xt.Bend, length=l_curv, angle=-angle, k0=0)
    line = env.new_line(name='line', components=[drift, rot_plus, drift, drift, rot_minus, drift])
    sv = line.survey()

    circle = Circle(radius=2.0)
    rectangle = Rectangle(half_width=2.0, half_height=1.0)
    profiles = [
        Profile(shape=circle, tol_r=0, tol_x=0, tol_y=0),
        Profile(shape=rectangle, tol_r=0, tol_x=0, tol_y=0),
    ]
    pipes = [
        Pipe(curvature=0.0, positions=[
            ProfilePosition(profile_index=1, shift_s=0.0, rot_s_rad=np.deg2rad(15.0)),
            ProfilePosition(profile_index=1, shift_s=5.5, rot_s_rad=np.deg2rad(90.0)),
            ProfilePosition(profile_index=0, shift_s=11.0, rot_x_rad=np.deg2rad(10.0)),
        ]),
    ]
    pipe_positions = [
        PipePosition(
            pipe_index=0,
            survey_reference_name='drift::0',
            survey_index=sv.name.tolist().index('drift::0'),
            transformation=transform_matrix(shift_x=-1.5),
        ),
    ]
    model = ApertureModel(
        line_name='line',
        pipe_positions=pipe_positions,
        pipes=pipes,
        profiles=profiles,
        pipe_names=['pipe0'],
        pipe_position_names=['pipe0'],
        profile_names=['circle', 'rectangle'],
    )

    ap = Aperture(line, model, context=test_context)
    bounds_table = ap.get_bounds_table()

    assert np.all(np.isfinite(bounds_table.s))
    assert np.all(np.isfinite(bounds_table.s_start))
    assert np.all(np.isfinite(bounds_table.s_end))
    assert np.all(bounds_table.s_start <= bounds_table.s)
    assert np.all(bounds_table.s <= bounds_table.s_end)


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_aperture_bounds_do_not_depend_on_profile_position_order(test_context):
    env = xt.Environment()
    angle = np.deg2rad(35.0)
    rot_x_rad = np.deg2rad(20.0)
    length = 3.2
    radius = 0.6

    bend = env.new('bend', xt.Bend, length=length, angle=angle, k0=0)
    drift = env.new('drift', xt.Drift, length=length)
    anti_bend = env.new('anti_bend', xt.Bend, length=length, angle=-angle, k0=0)
    line = env.new_line(name='line', components=[bend, drift, anti_bend])
    sv = line.survey()

    profiles = [Profile(shape=Circle(radius=radius), tol_r=0, tol_x=0, tol_y=0)]

    def make_position(shift_s):
        return ProfilePosition(
            profile_index=0,
            shift_s=shift_s,
            shift_y=-shift_s * np.tan(rot_x_rad),
        )

    sorted_positions = [
        make_position(0.0),
        make_position(0.5 * length),
        make_position(length),
    ]
    scrambled_positions = [
        make_position(length),
        make_position(0.0),
        make_position(0.5 * length),
    ]

    def build_model(positions, curvature):
        return ApertureModel(
            line=line,
            pipe_positions=[
                PipePosition(
                    pipe_index=0,
                    survey_reference_name=sv.name[0],
                    survey_index=0,
                    transformation=transform_matrix(rot_x_rad=rot_x_rad),
                ),
            ],
            pipes=[Pipe(curvature=curvature, positions=positions)],
            profiles=profiles,
            pipe_names=['pipe0'],
            pipe_position_names=['pipe0'],
            profile_names=['circ0'],
        )

    ap_forward = Aperture(
        line=line,
        model=build_model(sorted_positions, curvature=angle / length),
        context=test_context,
    )
    ap_reversed = Aperture(
        line=line,
        model=build_model(scrambled_positions, curvature=angle / length),
        context=test_context,
    )

    table_forward = ap_forward.get_bounds_table()
    table_reversed = ap_reversed.get_bounds_table()

    assert list(table_forward.name) == list(table_reversed.name)
    assert list(table_forward.profile_name) == list(table_reversed.profile_name)
    xo.assert_allclose(table_forward.s, table_reversed.s, atol=1e-9, rtol=0)
    xo.assert_allclose(table_forward.s_start, table_reversed.s_start, atol=1e-9, rtol=0)
    xo.assert_allclose(table_forward.s_end, table_reversed.s_end, atol=1e-9, rtol=0)


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_aperture_bounds_rot_x_minus_pi_matches_reversed_profiles_with_shift(test_context):
    """Compare equivalent pipe installations related by a pi flip about x.

    For a symmetric profile such as a circle, rotating the installed pipe by ``pi`` about the x-axis
    should not change the aperture bounds, if the local profile positions and offsets are mirrored.
    """
    env = xt.Environment()
    rot_x_rad = np.deg2rad(20.0)
    length = 3.2
    radius = 0.6

    drift = env.new('drift', xt.Drift, length=length)
    line = env.new_line(name='line', components=10 * [drift])
    sv = line.survey()

    profiles = [Profile(shape=Circle(radius=radius), tol_r=0, tol_x=0, tol_y=0)]

    positions = [
        ProfilePosition(profile_index=0, shift_s=0.2 * length, shift_y=-0.2 * length * np.tan(rot_x_rad)),
        ProfilePosition(profile_index=0, shift_s=0.5 * length, shift_y=-0.5 * length * np.tan(rot_x_rad)),
        ProfilePosition(profile_index=0, shift_s=0.8 * length, shift_y=-0.8 * length * np.tan(rot_x_rad)),
    ]
    reversed_positions = [
        ProfilePosition(
            profile_index=0,
            shift_s=-pos.shift_s,
            shift_y=pos.shift_s * np.tan(rot_x_rad),
        )
        for pos in positions
    ]

    def build_model(positions, transformation):
        return ApertureModel(
            line=line,
            pipe_positions=[
                PipePosition(
                    pipe_index=0,
                    survey_reference_name=sv.name[0],
                    survey_index=0,
                    transformation=transformation,
                ),
            ],
            pipes=[Pipe(curvature=0.0, positions=positions)],
            profiles=profiles,
            pipe_names=['pipe0'],
            pipe_position_names=['pipe0'],
            profile_names=['circ0'],
        )

    forward_transform = transform_matrix(rot_x_rad=rot_x_rad)
    reversed_transform = transform_matrix(rot_x_rad=rot_x_rad - np.pi)

    ap_forward = Aperture(
        line=line,
        model=build_model(positions, forward_transform),
        context=test_context,
    )
    ap_reversed = Aperture(
        line=line,
        model=build_model(reversed_positions, reversed_transform),
        context=test_context,
    )

    table_forward = ap_forward.get_bounds_table()
    table_reversed = ap_reversed.get_bounds_table()

    xo.assert_allclose(table_forward.s, table_reversed.s, atol=1e-9, rtol=0)
    xo.assert_allclose(table_forward.s_start, table_reversed.s_start, atol=1e-9, rtol=0)
    xo.assert_allclose(table_forward.s_end, table_reversed.s_end, atol=1e-9, rtol=0)


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_aperture_bounds_upstream_of_reference_across_marker(test_context):
    """Ensure bounds search works when the profile lies upstream of the survey reference.

    The installed pipe is referenced at a marker, but its actual profile centre is shifted
    back across the marker. The bound computation must still find the correct intersection
    point and produce finite aperture limits.
    """
    env = xt.Environment()

    d0 = env.new('d0', xt.Drift, length=1.0)
    marker = env.new('marker', xt.Marker)
    d1 = env.new('d1', xt.Drift, length=1.0)
    ref = env.new('ref', xt.Marker)
    tail = env.new('tail', xt.Drift, length=1.0)
    line = env.new_line(name='line', components=[d0, marker, d1, ref, tail])
    sv = line.survey()

    model = ApertureModel(
        line_name='line',
        pipe_positions=[
            PipePosition(
                pipe_index=0,
                survey_reference_name='ref',
                survey_index=sv.name.tolist().index('ref'),
                transformation=transform_matrix(shift_z=-1.5),
            ),
        ],
        pipes=[Pipe(curvature=0.0, positions=[ProfilePosition(profile_index=0)])],
        profiles=[Profile(shape=Circle(radius=0.01), tol_r=0, tol_x=0, tol_y=0)],
        pipe_names=['pipe0'],
        pipe_position_names=['pipe0'],
        profile_names=['circle'],
    )

    ap = Aperture(line, model, context=test_context)
    bounds_table = ap.get_bounds_table()

    xo.assert_allclose(bounds_table.s[0], 0.5, atol=1e-12, rtol=0)
    assert np.isfinite(bounds_table.s_start[0])
    assert np.isfinite(bounds_table.s_end[0])


@pytest.fixture(scope='module')
def hllhc19_end_to_end_model(tmp_path_factory):
    local_context = xo.ContextCpu()
    tmp_path = tmp_path_factory.mktemp('hllhc19_end_to_end_model')

    lhc = xt.load(TEST_DATA_DIR / 'hllhc19_apertures/lhc_aperture.json')
    out = {}
    for beam in ('b1', 'b2'):
        line = getattr(lhc, beam).copy(_context=local_context)
        aperture = Aperture.from_line_with_madx_metadata(
            line,
            num_profile_points=100,
            include_offsets=True,
            context=local_context,
        )

        aperture_path = tmp_path / f'aperture_model_{beam}.json'
        aperture.to_json(aperture_path)
        aperture = Aperture.from_json(aperture_path, line)

        out[beam] = (line, aperture)

    return out


@requires_context('ContextCpu')
@pytest.mark.parametrize('ir', [f'ir{ir_idx}b{b_idx}' for ir_idx, b_idx in itertools.product(range(1, 9), (1, 2))])
@pytest.mark.parametrize('method', ['rays', 'exact'])
def test_hllhc19_end_to_end(ir, method, hllhc19_end_to_end_model):
    # See `test_data/hllhc19_apertures` for more info on the file generation
    # In particular these are sanitised by clamping to nan values that are too large
    # or spurious (a lot of sequences nan, single value, nan, single value, nan, ...)
    ref_file = TEST_DATA_DIR / f'hllhc19_apertures/{ir}.json'

    reference = json.loads(ref_file.read_text())
    line, aperture = hllhc19_end_to_end_model[reference['beam']]
    aperture.halo_params.update(reference['halo_params'])

    s_positions = np.asarray(reference['s_positions'], dtype=float)
    n1_madx = np.asarray(reference['n1_madx'], dtype=float)
    order = np.argsort(s_positions)
    undo_order = np.empty_like(order)
    undo_order[order] = np.arange(len(order))

    n1_table, _ = aperture.get_aperture_sigmas_at_s(
        s_positions=s_positions[order],
        method=method,
    )
    n1_xt = np.asarray(n1_table.n1[undo_order], dtype=float)

    valid_mask = np.isfinite(n1_madx)

    # Assert that 99% of the data points are within 1% of MAD-X
    assert allclose_with_outliers(
        n1_madx[valid_mask],
        n1_xt[valid_mask],
        rtol=0.01,
        max_outliers=int(len(n1_madx) / 100),
    ), ref_file.name
