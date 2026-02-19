from itertools import zip_longest

import numpy as np
import pytest
import xobjects as xo
from cpymad.madx import Madx
from xobjects.test_helpers import for_all_test_contexts

import xtrack as xt
from xtrack.aperture.aperture import Aperture
from xtrack.aperture.kernels import build_aperture_kernels
from xtrack.aperture.structures import Ellipse, Rectangle, RectEllipse

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


@pytest.fixture(scope='module')
def context():
    return xo.ContextCpu()


@pytest.fixture(scope="module")
def kernels(context):
    build_aperture_kernels(context)
    return context.kernels


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_aperture_from_line_with_aperture_type_bounds(test_context):
    pass
    mad = Madx(stdout=None)
    mad.input(TOY_RING_SEQUENCE)
    env = xt.Environment.from_madx(madx=mad, enable_layout_data=True)
    ring = env['ring']

    aperture_model = Aperture.from_line_with_madx_metadata(ring, line_name='ring', context=test_context)
    type_bounds = aperture_model._type_bounds()
    type_name_bounds = [(a, b, aperture_model.model.type_name_for_position(c) if c else None) for a, b, c in type_bounds]
    table_rows = ring.get_table().cols['s_start', 's_end', 'name', 'element_type'].rows[:-1].rows
    #
    for type_bound, table_row in zip_longest(type_name_bounds, table_rows):
        type_start = type_bound[0]
        type_end = type_bound[1]
        type_name = type_bound[2]

        element_start = table_row.s_start
        element_end = table_row.s_end
        element_name = table_row.name

        xo.assert_allclose(element_start, type_start, atol=1e-6)
        xo.assert_allclose(element_end, type_end, atol=1e-6)

        if table_row.element_type == 'Drift':  # MAD-X won't allow apertures on drifts, so these shouldn't have bounds
            assert type_name is None
            continue

        assert element_name.startswith(type_name)


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_aperture_from_line_with_associated_apertures_type_bounds(test_context):
    env = xt.load(string=TOY_RING_SEQUENCE, format='madx', install_limits=False)
    env.set_particle_ref('proton', p0c=1.2e9)
    ring = env['ring']

    aperture_model = Aperture.from_line_with_associated_apertures(ring, line_name='ring', context=test_context)
    type_bounds = aperture_model._type_bounds()
    type_name_bounds = [(a, b, aperture_model.model.type_name_for_position(c) if c else None) for a, b, c in type_bounds]
    table_rows = ring.get_table().cols['s_start', 's_end', 'name', 'element_type'].rows[:-1].rows

    for type_bound, table_row in zip_longest(type_name_bounds, table_rows):
        type_start = type_bound[0]
        type_end = type_bound[1]
        type_name = type_bound[2]

        element_start = table_row.s_start
        element_end = table_row.s_end
        element_name = table_row.name

        xo.assert_allclose(element_start, type_start, atol=1e-6)
        xo.assert_allclose(element_end, type_end, atol=1e-6)

        if table_row.element_type == 'Drift':  # MAD-X won't allow apertures on drifts, so these shouldn't have bounds
            assert type_name is None
            continue

        # element names in survey have ::N at the end, we make the check disregarding the suffix:
        prototype_name, suffix = element_name.split('::')
        _ = int(suffix)
        assert type_name.startswith(prototype_name)


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_aperture_from_line_with_limits_type_bounds(test_context):
    env = xt.load(string=TOY_RING_SEQUENCE, format='madx', install_limits=True)
    env.set_particle_ref('proton', p0c=1.2e9)
    ring = env['ring']

    aperture_model = Aperture.from_line_with_limits(ring, line_name='ring', context=test_context)
    type_bounds = aperture_model._type_bounds()
    type_name_bounds_only_limits = [(a, b, aperture_model.model.type_name_for_position(c)) for a, b, c in type_bounds if c]

    bounds_from_table = []
    for row in ring.get_table().rows:
        if row.element_type.startswith('Limit'):
            bounds_from_table.append((row.s_start, row.s_end, row.name))

    for type_bound, table_bound in zip_longest(type_name_bounds_only_limits, bounds_from_table):
        type_start = type_bound[0]
        type_end = type_bound[1]
        type_name = type_bound[2]

        element_start = table_bound[0]
        element_end = table_bound[1]
        element_name = table_bound[2]

        xo.assert_allclose(element_start, type_start, atol=1e-6)
        xo.assert_allclose(element_end, type_end, atol=1e-6)
        assert element_name.startswith(type_name)


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_aperture_find_type_positions_perfect_overlap(test_context):
    env = xt.load(string=TOY_RING_SEQUENCE, format='madx', install_limits=False)
    env.set_particle_ref('proton', p0c=1.2e9)
    ring = env['ring']

    aperture_model = Aperture.from_line_with_associated_apertures(ring, line_name='ring', context=test_context)

    mqf0, = aperture_model._find_type_positions(1, 1.3, 'ring')
    assert mqf0.survey_reference_name == 'mqf::0'

    mqf0_name = aperture_model.model.type_name_for_position(mqf0)
    assert mqf0_name == 'mqf_aper'

    mqf0_type = aperture_model.model.type_for_position(mqf0)
    mqf0_profile_names = [aperture_model.model.profile_name_for_position(pos) for pos in mqf0_type.positions]
    assert mqf0_profile_names == ['mqf_aper', 'mqf_aper']

    mqf0_prof_pos0, mqf0_prof_pos1 = mqf0_type.positions
    assert mqf0_prof_pos0.s_position == 0.
    assert mqf0_prof_pos1.s_position == 0.3
    assert mqf0_prof_pos0.shift_x == mqf0_prof_pos0.shift_y == mqf0_prof_pos1.shift_x == mqf0_prof_pos1.shift_y == 0.

    mqf0_profile_start, mqf0_profile_end = [aperture_model.model.profile_for_position(pos) for pos in mqf0_type.positions]
    assert isinstance(mqf0_profile_start.shape, Rectangle)
    assert mqf0_profile_start.shape.half_width == mqf0_profile_end.shape.half_width == 0.08
    assert mqf0_profile_start.shape.half_height == mqf0_profile_end.shape.half_height == 0.04


@for_all_test_contexts(excluding=('ContextPyopencl', 'ContextCupy'))
def test_aperture_find_type_positions_partially_spanning_multiple_types(test_context):
    env = xt.load(string=TOY_RING_SEQUENCE, format='madx', install_limits=False)
    env.set_particle_ref('proton', p0c=1.2e9)
    ring = env['ring']

    aperture_model = Aperture.from_line_with_associated_apertures(ring, line_name='ring', context=test_context)

    overlapping = aperture_model._find_type_positions(8, 11.8, 'ring')
    mb1, ap_ds8, ap_ds9, mqf1 = overlapping

    # Check the bend
    assert mb1.survey_reference_name == 'mb::1'

    mb1_name = aperture_model.model.type_name_for_position(mb1)
    assert mb1_name == 'mb_aper'

    mb1_type = aperture_model.model.type_for_position(mb1)
    xo.assert_allclose(mb1_type.curvature, ring['mb'].h, atol=1e-6)
    mb1_profile_names = [aperture_model.model.profile_name_for_position(pos) for pos in mb1_type.positions]
    assert mb1_profile_names == ['mb_aper', 'mb_aper']

    mb1_prof_pos0, mb1_prof_pos1 = mb1_type.positions
    assert mb1_prof_pos0.s_position == 0.
    assert mb1_prof_pos1.s_position == 3.
    assert mb1_prof_pos0.shift_x == mb1_prof_pos0.shift_y == mb1_prof_pos1.shift_x == mb1_prof_pos1.shift_y == 0.

    mb1_profile_start, mb1_profile_end = [aperture_model.model.profile_for_position(pos) for pos in mb1_type.positions]
    assert isinstance(mb1_profile_start.shape, Ellipse)
    assert mb1_profile_start.shape.half_major == mb1_profile_end.shape.half_major == 0.1
    assert mb1_profile_start.shape.half_minor == mb1_profile_end.shape.half_minor == 0.1

    # Check the mqf
    mqf1_name = aperture_model.model.type_name_for_position(mqf1)
    assert mqf1_name == 'mqf_aper'

    mqf1_type = aperture_model.model.type_for_position(mqf1)
    mqf1_profile_names = [aperture_model.model.profile_name_for_position(pos) for pos in mqf1_type.positions]
    assert mqf1_profile_names == ['mqf_aper', 'mqf_aper']

    mqf1_prof_pos0, mqf1_prof_pos1 = mqf1_type.positions
    assert mqf1_prof_pos0.s_position == 0.
    assert mqf1_prof_pos1.s_position == 0.3
    assert mqf1_prof_pos0.shift_x == mqf1_prof_pos0.shift_y == mqf1_prof_pos1.shift_x == mqf1_prof_pos1.shift_y == 0.

    mqf1_profile_start, mqf1_profile_end = [aperture_model.model.profile_for_position(pos) for pos in mqf1_type.positions]
    assert isinstance(mqf1_profile_start.shape, Rectangle)
    assert mqf1_profile_start.shape.half_width == mqf1_profile_end.shape.half_width == 0.08
    assert mqf1_profile_start.shape.half_height == mqf1_profile_end.shape.half_height == 0.04

    # Check the ap_ds
    ap_ds8_name = aperture_model.model.type_name_for_position(ap_ds8)
    ap_ds9_name = aperture_model.model.type_name_for_position(ap_ds8)
    assert ap_ds8_name == ap_ds9_name == 'ap_ds_aper'

    ap_ds8_type = aperture_model.model.type_for_position(ap_ds8)
    assert ap_ds8.type_index == ap_ds9.type_index

    ap_ds8_profile_names = [aperture_model.model.profile_name_for_position(pos) for pos in ap_ds8_type.positions]
    assert ap_ds8_profile_names == ['ap_ds_aper']

    ap_ds8_prof_pos0, = ap_ds8_type.positions
    assert ap_ds8_prof_pos0.s_position == 0.
    assert ap_ds8_prof_pos0.shift_x == ap_ds8_prof_pos0.shift_y

    ap_ds8_profile_start = aperture_model.model.profile_for_position(ap_ds8_prof_pos0)
    assert isinstance(ap_ds8_profile_start.shape, RectEllipse)
    assert ap_ds8_profile_start.shape.half_major == 0.022
    assert ap_ds8_profile_start.shape.half_minor == 0.022
    assert ap_ds8_profile_start.shape.half_width == 0.022
    assert ap_ds8_profile_start.shape.half_height == 0.01715


def test_is_point_inside_polygon_ellipse(kernels):
    rx = 2
    ry = 3
    ellipse = [(rx * np.cos(angle), ry * np.sin(angle)) for angle in np.linspace(0, 2 * np.pi, 99)]
    ellipse.append(ellipse[0])
    ellipse = np.array(ellipse, dtype=np.float32)

    @np.vectorize
    def in_ellipse(x, y):
        point = np.array([x, y], dtype=np.float32)
        return bool(kernels['_is_point_inside_polygon'](point=point, points=ellipse, len_points=ellipse.shape[0]))

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
    ], dtype=np.float32)

    @np.vectorize
    def in_poly(x, y):
        point = np.array([x, y], dtype=np.float32)
        return bool(kernels['_is_point_inside_polygon'](point=point, points=poly, len_points=poly.shape[0]))

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

    circ1 = np.array(circ1, dtype=np.float32)
    circ2 = np.array(circ2, dtype=np.float32)

    small_in_big = kernels["_points_inside_polygon"](
        points=circ1,
        poly_points=circ2,
        len_points=circ1.shape[0],
        len_poly_points=circ2.shape[0],
    )

    assert bool(small_in_big)

    big_in_small = kernels["_points_inside_polygon"](
        points=circ2,
        poly_points=circ1,
        len_points=circ2.shape[0],
        len_poly_points=circ1.shape[0],
    )

    assert not bool(big_in_small)


def test_points_inside_polygon_simple(kernels):
    poly_big = [(1, 1), (2, 3.5), (4.5, 3.5), (4.5, 1), (1, 1)]
    poly_small = [(2, 2), (3, 3), (4, 2), (3, 1.5), (2, 2)]

    poly_big = np.array(poly_big, dtype=np.float32)
    poly_small = np.array(poly_small, dtype=np.float32)

    small_in_big = kernels["_points_inside_polygon"](
        points=poly_small,
        poly_points=poly_big,
        len_points=poly_small.shape[0],
        len_poly_points=poly_big.shape[0],
    )

    assert bool(small_in_big)

    big_in_small = kernels["_points_inside_polygon"](
        points=poly_big,
        poly_points=poly_small,
        len_points=poly_big.shape[0],
        len_poly_points=poly_small.shape[0],
    )

    assert not bool(big_in_small)


def test_points_inside_polygon_simpler(kernels):
    poly_big = np.array([
        [1.0000000e+00, 0.0000000e+00],
        [-5.0000006e-01, 8.6602539e-01],
        [-4.9999991e-01, -8.6602545e-01],
        [1.0000000e+00, 0.0000000e+00],
    ], dtype=np.float32)
    poly_small = np.array([
        [1.1466468e-01, 0.0000000e+00],
        [-5.7332322e-02, 9.9302538e-02],
        [-5.7332378e-02, -9.9302508e-02],
        [1.1466468e-01, 0.0000000e+00],
    ], dtype=np.float32)

    small_in_big = kernels["_points_inside_polygon"](
        points=poly_small,
        poly_points=poly_big,
        len_points=poly_small.shape[0],
        len_poly_points=poly_big.shape[0],
    )

    assert bool(small_in_big)

    big_in_small = kernels["_points_inside_polygon"](
        points=poly_big,
        poly_points=poly_small,
        len_points=poly_big.shape[0],
        len_poly_points=poly_small.shape[0],
    )

    assert not bool(big_in_small)


@pytest.mark.parametrize('method', ['bisection', 'rays'])
@pytest.mark.parametrize(
    'shape,aper_params,aper_tol,beam_params,halo_params,expected',
    [
        (
            'circle', (1,), (0, 0, 0),
            {'exn': 1e-3, 'eyn': 1e-3, 'gamma': 10, 'betx': 1, 'bety': 1, 'x': 0, 'y': 0, 'dx': 0, 'dy': 0},
            {},
            100,
        ),
        (
            'circle', (1,), (0, 0, 0),
            {'exn': 1e-3, 'eyn': 1e-3, 'gamma': 10, 'betx': 1, 'bety': 1, 'x': 0, 'y': 0, 'dx': 0, 'dy': 0},
            {'halo_primary': 1, 'halo_r': 2, 'halo_x': 2, 'halo_y': 2},
            50,
        ),
        (
            'rectangle', (1, 1), (0, 0, 0),
            {'exn': 1e-3, 'eyn': 1e-3, 'gamma': 10, 'betx': 1, 'bety': 1, 'x': 0, 'y': 0, 'dx': 0, 'dy': 0},
            {},
            100,
        ),
        (
            'rectangle', (1.1, 1.2), (0, 0, 0),
            {'exn': 1e-3, 'eyn': 1e-3, 'gamma': 10, 'betx': 1, 'bety': 1, 'x': -0.1, 'y': 0.2, 'dx': 0, 'dy': 0},
            {},
            100,
        ),
        (
            'racetrack', (0.28, 0.43, 0.13, 0.172), (0.002, 0.006, 0.002),
            {'exn': 4e-3, 'eyn': 4e-3, 'gamma': 10, 'betx': 9, 'bety': 16, 'x': 0, 'y': 0, 'dx': 0, 'dy': 0},
            {
                'tol_beta_beating': 0.8,
                'tol_disp': 1.25,
                'tol_disp_ref_beta': 4,
                'tol_disp_ref_dx': 20,
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
            {'exn': 4e-3, 'eyn': 4e-3, 'gamma': 10, 'betx': 9, 'bety': 16, 'x': -0.02, 'y': 0.07, 'dx': 0, 'dy': 0},
            {
                'tol_beta_beating': 0.8,
                'tol_disp': 1.25,
                'tol_disp_ref_beta': 4,
                'tol_disp_ref_dx': 20,
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
                'gamma': 10,
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
                'tol_disp_ref_dx': 20,
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
            'racetrack', (0.4, 0.5, 0.13, 0.172), (0.002, 0.006, 0.002),
            {
                'exn': 4e-3,
                'eyn': 4e-3,
                'gamma': 10,
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
                'tol_disp_ref_dx': 20,
                'halo_primary': 10,
                'halo_r': 0.7,
                'halo_x': 0.5,
                'halo_y': 0.6,
                'tol_co': 0.002,
                'delta_rms': 0.001,
            },
            100,
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
    def madx_list(l):
        return '{' + ', '.join([str(v) for v in l]) + '}'

    halo_params_for_test = {
        'emitx_norm': beam_params['exn'],
        'emity_norm': beam_params['eyn'],
        'tol_beta_beating': 1,
        'tol_disp': 0,
        'tol_disp_ref_beta': 1,
        'tol_disp_ref_dx': 0,
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

    aperture_model = Aperture.from_line_with_associated_apertures(seq, line_name='seq', context=context)
    aperture_model.halo_params.update(halo_params)

    # Needed as these quantities are not imported by the native madloader
    aperture_model.model.profiles[0].tol_r = aper_tol[0]
    aperture_model.model.profiles[0].tol_x = aper_tol[1]
    aperture_model.model.profiles[0].tol_y = aper_tol[2]

    # Compute n1 with Xsuite
    computed_n1, tw, apertures_points, envelope_points = aperture_model.get_aperture_sigmas_at_element(
        line_name='seq',
        element_name='m1',
        resolution=None,
        twiss=tw,
        cross_sections_num_points=144,
        envelopes_num_points=144,
        method=method,
    )

    if method == 'rays':
        computed_n1 = np.min(computed_n1, axis=1)

    # There are two sources of error wrt. to the analytic solution:
    # - precision of 0.01 on the bisection defined in beam_aperture.h
    # - error coming from the fact that we are comparing polygons, not ideal shapes (especially a problem if x, y != 0)
    xo.assert_allclose(computed_n1, expected, atol=0.01, rtol=0.002)


def test_get_aperture_sigmas_at_element_analytic_rays(context):
    betx = 9
    bety = 16
    delta = 0.001
    gamma = 10

    beam_data = {
        'emitx_norm': 4e-3,
        'emity_norm': 4e-3,
        'delta_rms': 0.001,
        'tol_co': 0.002,
        'tol_disp': 1.25,
        'tol_disp_ref_dx': 20,
        'tol_disp_ref_beta': 4,
        'tol_energy': 0.001,
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

    aperture_model = Aperture.from_line_with_associated_apertures(
        seq, line_name="seq", context=context
    )
    aperture_model.halo_params.update(beam_data)

    # Needed as these quantities are not imported by the native madloader
    aperture_model.model.profiles[0].tol_r = tol_r
    aperture_model.model.profiles[0].tol_x = tol_x
    aperture_model.model.profiles[0].tol_y = tol_y

    computed_n1, tw, apertures_points, envelope_points = (
        aperture_model.get_aperture_sigmas_at_element(
            line_name="seq",
            element_name="m1",
            resolution=None,
            twiss=tw,
            cross_sections_num_points=144,
            envelopes_num_points=144,
            method="rays",
        )
    )

    # All n1-s should be the expected value, the envelope at the expected value
    # should fully cover the aperture in this case.
    xo.assert_allclose(computed_n1, expected_n1, atol=0.01, rtol=0.002)


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
        context
):
    """Test the computation of sigmas vs MAD-X

    MAD-X uses a different approach to computing N1 when dispersion is present, hence we only test the cases without.
    """
    def madx_list(l):
        return '{' + ', '.join([str(v) for v in l]) + '}'

    halo_params_for_test = {
        'emitx_norm': exn,
        'emity_norm': eyn,
        'tol_beta_beating': 1,
        'tol_disp': 0,
        'tol_disp_ref_beta': 1,
        'tol_disp_ref_dx': 0,
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
            dqf = {halo_params['tol_disp_ref_dx']},
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

    env = xt.Environment.from_madx(madx=mad, enable_layout_data=True)
    seq = env['seq']
    seq.set_particle_ref('proton', gamma0=mad.beam.gamma)
    tw = seq.twiss4d(betx=betx, bety=bety, x=x, y=y)

    xo.assert_allclose(mad.beam.gamma, seq.particle_ref.gamma0, atol=1e-10)
    xo.assert_allclose(mad.beam.beta, seq.particle_ref.beta0, atol=1e-10)

    aperture_model = Aperture.from_line_with_madx_metadata(seq, line_name='seq', context=context)
    aperture_model.halo_params.update(halo_params)

    # Sanity checks
    aper_summ = mad.table.aperture.summary
    xo.assert_allclose(aperture_model.halo_params['tol_disp_ref_dx'], aper_summ.dqf, atol=1e-8, rtol=0)
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
    computed_n1, tw, apertures_points, envelope_points = aperture_model.get_aperture_sigmas_at_element(
        line_name='seq',
        element_name='m1',
        resolution=None,
        twiss=tw,
        cross_sections_num_points=144,
        envelopes_num_points=144,
    )

    xo.assert_allclose(madx_n1, computed_n1, rtol=0.01)
