import xtrack as xt
from xtrack.mad_writer import to_madng_sequence


def test_madng_relative_multipoles_are_scaled_by_main_strength():
    bend = xt.Bend(
        length=14.3,
        angle=5.1e-3,
        k0=5.1e-3 / 14.3,
        knl=[0.0, 0.0, 0.0],
        ksl=[0.0, 0.0, 0.0],
        knl_rel=[0.0, 0.25, -0.5],
        ksl_rel=[0.0, -0.125, 0.5],
    )
    line = xt.Line(elements=[bend], element_names=['b'])

    madng = to_madng_sequence(line, name='seq')
    main_strength = bend.main_strength

    assert f'dknl := {{0.0,{0.25 * main_strength},{-0.5 * main_strength}}}' in madng
    assert f'dksl := {{0.0,{-0.125 * main_strength},{0.5 * main_strength}}}' in madng


def test_madng_straight_body_rbend_sets_true_rbend_and_transforms_edges():
    rbend = xt.RBend(
        length_straight=3.0,
        angle=0.01,
        k0_from_h=True,
        rbend_model='straight-body',
        rbend_angle_diff=0.004,
    )
    rbend.edge_entry_angle = 0.03
    rbend.edge_exit_angle = -0.02
    line = xt.Line(elements=[rbend], element_names=['rb'])

    madng = to_madng_sequence(line, name='seq')

    assert 'true_rbend = true' in madng
    assert 'e1 = 0.027999999999999997' in madng
    assert 'e2 = -0.018000000000000002' in madng
