import numpy as np
import pytest
import xtrack as xt
import xpart as xp


def _optimize_after_thin_slicing(element):
    line = xt.Line(elements=[element], element_names=['e'])
    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(slicing=xt.Teapot(1, mode='thin'))])
    line.build_tracker()
    line.optimize_for_tracking(compile=False, verbose=False)
    return line


def _bend_with_relative_strengths():
    return xt.Bend(
        length=2, angle=0.2, k1=0.03,
        knl=[0.01, 0.02, 0.03, 0.04],
        ksl=[0.05, 0.06, 0.07, 0.08],
        knl_rel=[0.1, 0.2, 0.3, 0.4],
        ksl_rel=[0.5, 0.6, 0.7, 0.8],
        edge_entry_active=False, edge_exit_active=False)


def _rbend_with_relative_strengths():
    return xt.RBend(
        length_straight=2, angle=0.2, k1=0.03,
        knl=[0.01, 0.02, 0.03, 0.04],
        ksl=[0.05, 0.06, 0.07, 0.08],
        knl_rel=[0.1, 0.2, 0.3, 0.4],
        ksl_rel=[0.5, 0.6, 0.7, 0.8],
        edge_entry_active=False, edge_exit_active=False,
        rbend_model='curved-body')


def _quadrupole_with_relative_strengths():
    return xt.Quadrupole(
        length=2, k1=0.3, k1s=0.2,
        knl=[0.01, 0.02, 0.03, 0.04],
        ksl=[0.05, 0.06, 0.07, 0.08],
        knl_rel=[0.1, 0.2, 0.3, 0.4],
        ksl_rel=[0.5, 0.6, 0.7, 0.8])


def _sextupole_with_relative_strengths():
    return xt.Sextupole(
        length=2, k2=0.3, k2s=0.2,
        knl=[0.01, 0.02, 0.03, 0.04],
        ksl=[0.05, 0.06, 0.07, 0.08],
        knl_rel=[0.1, 0.2, 0.3, 0.4],
        ksl_rel=[0.5, 0.6, 0.7, 0.8])


def _octupole_with_relative_strengths():
    return xt.Octupole(
        length=2, k3=0.3, k3s=0.2,
        knl=[0.01, 0.02, 0.03, 0.04],
        ksl=[0.05, 0.06, 0.07, 0.08],
        knl_rel=[0.1, 0.2, 0.3, 0.4],
        ksl_rel=[0.5, 0.6, 0.7, 0.8])


def _multipole_with_relative_strengths():
    return xt.Multipole(
        length=2, isthick=True, main_order=2,
        knl=[0.01, 0.02, 0.03, 0.04],
        ksl=[0.05, 0.06, 0.07, 0.08],
        knl_rel=[0.1, 0.2, 0.3, 0.4],
        ksl_rel=[0.5, 0.6, 0.7, 0.8])


@pytest.mark.parametrize('element_factory', [
    _bend_with_relative_strengths,
    _rbend_with_relative_strengths,
    _quadrupole_with_relative_strengths,
    _sextupole_with_relative_strengths,
    _octupole_with_relative_strengths,
    _multipole_with_relative_strengths,
])
def test_optimize_thin_slices_use_total_knl_ksl(element_factory):
    element = element_factory()
    line = _optimize_after_thin_slicing(element)

    multipoles = [ee for ee in line.elements if isinstance(ee, xt.Multipole)]
    assert len(multipoles) == 1

    knl, ksl = element.get_total_knl_ksl()
    np.testing.assert_allclose(multipoles[0].knl, knl)
    np.testing.assert_allclose(multipoles[0].ksl, ksl)


def test_optimize_edges_use_total_knl_ksl():
    quad = xt.Quadrupole(
        length=2, k1=0.3, k1s=0.2,
        edge_entry_active=True, edge_exit_active=True,
        knl=[0.01, 0.02, 0.03],
        ksl=[0.04, 0.05, 0.06],
        knl_rel=[0.1, 0.2, 0.3],
        ksl_rel=[0.4, 0.5, 0.6])
    line = _optimize_after_thin_slicing(quad)

    kn, ks = quad.get_total_knl_ksl()
    kn = kn / quad.length
    ks = ks / quad.length
    kn[0] = 0
    ks[0] = 0

    edges = [ee for ee in line.elements if isinstance(ee, xt.MultipoleEdge)]
    assert len(edges) == 2
    for edge in edges:
        np.testing.assert_allclose(edge.kn, kn)
        np.testing.assert_allclose(edge.ks, ks)

    bend = xt.Bend(
        length=2, angle=0.2,
        edge_entry_active=True, edge_exit_active=True,
        knl=[0.01, 0.02],
        ksl=[0.03, 0.04],
        knl_rel=[0.1, 0.2],
        ksl_rel=[0.3, 0.4])
    line = _optimize_after_thin_slicing(bend)

    knl, _ = bend.get_total_knl_ksl()
    expected_k = knl[0] / bend.length

    edges = [ee for ee in line.elements if isinstance(ee, xt.DipoleEdge)]
    assert len(edges) == 2
    for edge in edges:
        np.testing.assert_allclose(edge.k, expected_k)


def test_optimize_raw_multipoles_use_total_knl_ksl():
    line = xt.Line(
        elements=[
            xt.Multipole(knl=[1], knl_rel=[-1]),
            xt.Multipole(knl=[0, 2], knl_rel=[0, 0.1], main_order=1),
            xt.Multipole(knl=[0, 3], knl_rel=[0, 0.2], main_order=1),
        ],
        element_names=['inactive', 'm1', 'm2'])

    line.build_tracker()
    line.optimize_for_tracking(compile=False, verbose=False)

    assert 'inactive' not in line.element_names
    assert len(line.element_names) == 1
    assert isinstance(line[line.element_names[0]], xt.SimpleThinQuadrupole)
    np.testing.assert_allclose(line[line.element_names[0]].knl, [0, 5.8])


def test_optimize_keeps_raw_multipoles_with_transverse_rotation():
    line = xt.Line(
        elements=[
            xt.Multipole(knl=[0, 2], rot_x_rad=1e-6),
            xt.Multipole(knl=[0, 3]),
        ],
        element_names=['rotated', 'plain'])

    line.build_tracker()
    line.optimize_for_tracking(compile=False, verbose=False)

    assert isinstance(line['rotated'], xt.Multipole)
    assert not isinstance(line['rotated'], xt.SimpleThinQuadrupole)
    assert len(line.element_names) == 2


def test_optimize_sliced_parent_transverse_rotation_raises():
    quad = xt.Quadrupole(length=1, k1=0.2, rot_y_rad=1e-6)
    line = xt.Line(elements=[quad], element_names=['q'])
    line.slice_thick_elements(
        slicing_strategies=[xt.Strategy(slicing=xt.Teapot(1, mode='thin'))])
    line.build_tracker()

    with pytest.raises(ValueError, match='rot_y_rad'):
        line.optimize_for_tracking(compile=False, verbose=False)


def test_optimize_with_radiation():

    env = xt.Environment()

    env.new('mb', xt.Bend, length=1, k0=1, angle=1)
    env.new('mq', xt.Quadrupole, length=1, k1=1)
    env.new('ms', xt.Sextupole, length=1, k2=1)
    env.new('mo', xt.Octupole, length=1, k3=1)

    element_list = ['mb', 'mq', 'ms', 'mo']

    for element in element_list:
        line = env.new_line(components=[element])

        line.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Teapot(1))])

        line.build_tracker()
        line.configure_radiation('mean')

        part = xp.Particles(p0c=1e15, x=0.11)
        line.track(part)

        line.optimize_for_tracking()
        part_opt = xp.Particles(p0c=1e15, x=0.11)
        line.track(part_opt)

        assert part.x == part_opt.x
        assert part.y == part_opt.y
        assert part.px == part_opt.px
        assert part.py == part_opt.py
        assert part.zeta == part_opt.zeta
        assert part.pzeta == part_opt.pzeta

def test_optimize_with_delta_taper():

    env = xt.Environment()

    env.new('mb', xt.Bend, length=1, k0=1, angle=1)
    env.new('mq', xt.Quadrupole, length=1, k1=1)
    env.new('ms', xt.Sextupole, length=1, k2=1)
    env.new('mo', xt.Octupole, length=1, k3=1)

    thin_classes = [xt.ThinSliceBendEntry, xt.ThinSliceBendExit, xt.ThinSliceBend,
                    xt.ThinSliceQuadrupole, xt.ThinSliceSextupole, xt.ThinSliceOctupole]

    element_list = ['mb', 'mq', 'ms', 'mo']

    for element in element_list:
        line = env.new_line(components=[element])

        line.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Teapot(1))])

        for el in line.elements:
            if type(el) in thin_classes:
                el.delta_taper = 0.1
                print(el)

        line.build_tracker()
        line.configure_radiation('mean')

        part = xp.Particles(p0c=1e15, x=0.11)
        line.track(part)

        line.optimize_for_tracking()
        part_opt = xp.Particles(p0c=1e15, x=0.11)
        line.track(part_opt)

        assert part.x == part_opt.x
        assert part.y == part_opt.y
        assert part.px == part_opt.px
        assert part.py == part_opt.py
        assert part.zeta == part_opt.zeta
        assert part.pzeta == part_opt.pzeta
