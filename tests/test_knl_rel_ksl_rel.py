import xtrack as xt
import xobjects as xo
import numpy as np
import pytest


def _assert_spin_tracking_matches(line_test, line_ref):
    for line in (line_test, line_ref):
        line.set_particle_ref(
            'positron', p0c=1e9, anomalous_magnetic_moment=0.00115965218128)

    tw_test = line_test.twiss(
        spin=True, x=1e-2, y=2e-2, spin_y=1, betx=1, bety=1)
    tw_ref = line_ref.twiss(
        spin=True, x=1e-2, y=2e-2, spin_y=1, betx=1, bety=1)

    xo.assert_allclose(tw_test.spin_x, tw_ref.spin_x, rtol=0, atol=1e-10)
    xo.assert_allclose(tw_test.spin_y, tw_ref.spin_y, rtol=0, atol=1e-10)
    xo.assert_allclose(tw_test.spin_z, tw_ref.spin_z, rtol=0, atol=1e-10)


@pytest.mark.parametrize("main_is_skew", [True, False])
def test_knl_rel_ksl_rel_quadrupole(main_is_skew):

    knl = np.array([0.001, 1e-3, 2e-2])
    ksl = np.array([0.002, 2e-3, 3e-2])
    knl_rel = np.array([1, 20, 30, 400, 50000, 6000000])
    ksl_rel = np.array([2, 40, 60, 800, 100000])
    k1 = 0.001
    k1s = 0.002

    el_test = xt.Quadrupole(k1=k1, k1s=k1s, length=0.2,
                        knl=knl,
                        ksl=ksl,
                        knl_rel=knl_rel,
                        ksl_rel=ksl_rel,
                        main_is_skew=main_is_skew)

    knl = np.pad(knl, (0, 10 - len(knl)))
    ksl = np.pad(ksl, (0, 10 - len(ksl)))
    knl_rel = np.pad(knl_rel, (0, 10 - len(knl_rel)))
    ksl_rel = np.pad(ksl_rel, (0, 10 - len(ksl_rel)))

    if main_is_skew:
        xo.assert_allclose(el_test.main_strength, 0.002 * 0.2, rtol=0, atol=1e-12)
    else:
        xo.assert_allclose(el_test.main_strength, 0.001 * 0.2, rtol=0, atol=1e-12)

    expected_knl = knl + knl_rel * el_test.main_strength
    expected_ksl = ksl + ksl_rel * el_test.main_strength

    expected_knl[1] += k1 * el_test.length
    expected_ksl[1] += k1s * el_test.length

    knl_tot, ksl_tot = el_test.get_total_knl_ksl()

    knl_tot = np.pad(knl_tot, (0, 10 - len(knl_tot)))
    ksl_tot = np.pad(ksl_tot, (0, 10 - len(ksl_tot)))

    xo.assert_allclose(knl_tot, expected_knl, rtol=0, atol=1e-12)
    xo.assert_allclose(ksl_tot, expected_ksl, rtol=0, atol=1e-12)

    el_ref = xt.Quadrupole(k1=k1, k1s=k1s, length=0.2,
                        knl=expected_knl, ksl=expected_ksl)
    el_ref.knl[1] -= k1 * el_ref.length  # Not to double count the main quadrupole contribution
    el_ref.ksl[1] -= k1s * el_ref.length

    p0 = xt.Particles(p0c=1e9, x=1e-2, y=2e-2)

    p_test = p0.copy()
    el_test.track(p_test)

    p_ref = p0.copy()
    el_ref.track(p_ref)

    xo.assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-13)

    line_test = xt.Line(elements=[el_test])
    line_ref = xt.Line(elements=[el_ref])

    p_test = p0.copy()
    line_test.track(p_test)
    line_test.track(p_test, backtrack=True)

    xo.assert_allclose(p_test.x, p0.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p0.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p0.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p0.py, rtol=0, atol=1e-13)

    # Check thick slicing
    line_test_slice_thick = line_test.copy(shallow=True)
    line_ref_slice_thick = line_ref.copy(shallow=True)

    line_test_slice_thick.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thick'))])
    line_ref_slice_thick.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thick'))])

    p_test = p0.copy()
    line_test_slice_thick.track(p_test)
    line_test_slice_thick.track(p_test, backtrack=True)

    xo.assert_allclose(p_test.x, p0.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p0.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p0.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p0.py, rtol=0, atol=1e-13)

    p_test = p0.copy()
    line_test_slice_thick.track(p_test)
    p_ref = p0.copy()
    line_ref_slice_thick.track(p_ref)

    xo.assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-13)

    # Check thin slicing
    line_test_slice_thin = line_test.copy(shallow=True)
    line_ref_slice_thin = line_ref.copy(shallow=True)
    line_test_slice_thin.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thin'))])
    line_ref_slice_thin.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thin'))])

    p_test = p0.copy()
    line_test_slice_thin.track(p_test)
    line_test_slice_thin.track(p_test, backtrack=True)

    xo.assert_allclose(p_test.x, p0.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p0.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p0.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p0.py, rtol=0, atol=1e-13)

    p_test = p0.copy()
    line_test_slice_thin.track(p_test)
    p_ref = p0.copy()
    line_ref_slice_thin.track(p_ref)

    xo.assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-13)

    _assert_spin_tracking_matches(line_test, line_ref)
    _assert_spin_tracking_matches(line_test_slice_thick, line_ref_slice_thick)
    _assert_spin_tracking_matches(line_test_slice_thin, line_ref_slice_thin)

@pytest.mark.parametrize("main_is_skew", [True, False])
def test_knl_rel_ksl_rel_sextupole(main_is_skew):

    knl = np.array([0.001, 1e-3, 2e-2])
    ksl = np.array([0.002, 2e-3, 3e-2])
    knl_rel = np.array([1, 20, 30, 400, 50000, 6000000])
    ksl_rel = np.array([2, 40, 60, 800, 100000])
    k2 = 0.001
    k2s = 0.002

    el_test = xt.Sextupole(k2=k2, k2s=k2s, length=0.2,
                        knl=knl,
                        ksl=ksl,
                        knl_rel=knl_rel,
                        ksl_rel=ksl_rel,
                        main_is_skew=main_is_skew)

    knl = np.pad(knl, (0, 10 - len(knl)))
    ksl = np.pad(ksl, (0, 10 - len(ksl)))
    knl_rel = np.pad(knl_rel, (0, 10 - len(knl_rel)))
    ksl_rel = np.pad(ksl_rel, (0, 10 - len(ksl_rel)))

    if main_is_skew:
        xo.assert_allclose(el_test.main_strength, 0.002 * 0.2, rtol=0, atol=1e-12)
    else:
        xo.assert_allclose(el_test.main_strength, 0.001 * 0.2, rtol=0, atol=1e-12)

    expected_knl = knl + knl_rel * el_test.main_strength
    expected_ksl = ksl + ksl_rel * el_test.main_strength

    expected_knl[2] += k2 * el_test.length
    expected_ksl[2] += k2s * el_test.length

    knl_tot, ksl_tot = el_test.get_total_knl_ksl()

    knl_tot = np.pad(knl_tot, (0, 10 - len(knl_tot)))
    ksl_tot = np.pad(ksl_tot, (0, 10 - len(ksl_tot)))

    xo.assert_allclose(knl_tot, expected_knl, rtol=0, atol=1e-12)
    xo.assert_allclose(ksl_tot, expected_ksl, rtol=0, atol=1e-12)

    el_ref = xt.Sextupole(k2=k2, k2s=k2s, length=0.2,
                        knl=expected_knl, ksl=expected_ksl)
    el_ref.knl[2] -= k2 * el_ref.length  # Not to double count the main sextupole contribution
    el_ref.ksl[2] -= k2s * el_ref.length

    p0 = xt.Particles(p0c=1e9, x=1e-2, y=2e-2)

    p_test = p0.copy()
    el_test.track(p_test)

    p_ref = p0.copy()
    el_ref.track(p_ref)

    xo.assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-13)

    line_test = xt.Line(elements=[el_test])
    line_ref = xt.Line(elements=[el_ref])

    p_test = p0.copy()
    line_test.track(p_test)
    line_test.track(p_test, backtrack=True)

    xo.assert_allclose(p_test.x, p0.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p0.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p0.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p0.py, rtol=0, atol=1e-13)

    # Check thick slicing
    line_test_slice_thick = line_test.copy(shallow=True)
    line_ref_slice_thick = line_ref.copy(shallow=True)

    line_test_slice_thick.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thick'))])
    line_ref_slice_thick.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thick'))])

    p_test = p0.copy()
    line_test_slice_thick.track(p_test)
    line_test_slice_thick.track(p_test, backtrack=True)

    xo.assert_allclose(p_test.x, p0.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p0.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p0.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p0.py, rtol=0, atol=1e-13)

    p_test = p0.copy()
    line_test_slice_thick.track(p_test)
    p_ref = p0.copy()
    line_ref_slice_thick.track(p_ref)

    xo.assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-13)

    # Check thin slicing
    line_test_slice_thin = line_test.copy(shallow=True)
    line_ref_slice_thin = line_ref.copy(shallow=True)
    line_test_slice_thin.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thin'))])
    line_ref_slice_thin.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thin'))])

    p_test = p0.copy()
    line_test_slice_thin.track(p_test)
    line_test_slice_thin.track(p_test, backtrack=True)

    xo.assert_allclose(p_test.x, p0.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p0.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p0.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p0.py, rtol=0, atol=1e-13)

    p_test = p0.copy()
    line_test_slice_thin.track(p_test)
    p_ref = p0.copy()
    line_ref_slice_thin.track(p_ref)

    xo.assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-13)

    _assert_spin_tracking_matches(line_test, line_ref)
    _assert_spin_tracking_matches(line_test_slice_thick, line_ref_slice_thick)
    _assert_spin_tracking_matches(line_test_slice_thin, line_ref_slice_thin)

@pytest.mark.parametrize("main_is_skew", [True, False])
def test_knl_rel_ksl_rel_octupole(main_is_skew):

    knl = np.array([0.001, 1e-3, 2e-2])
    ksl = np.array([0.002, 2e-3, 3e-2])
    knl_rel = np.array([1, 20, 30, 400, 50000, 6000000])
    ksl_rel = np.array([2, 40, 60, 800, 100000])
    k3 = 0.001
    k3s = 0.002

    el_test = xt.Octupole(k3=k3, k3s=k3s, length=0.2,
                        knl=knl,
                        ksl=ksl,
                        knl_rel=knl_rel,
                        ksl_rel=ksl_rel,
                        main_is_skew=main_is_skew)

    knl = np.pad(knl, (0, 10 - len(knl)))
    ksl = np.pad(ksl, (0, 10 - len(ksl)))
    knl_rel = np.pad(knl_rel, (0, 10 - len(knl_rel)))
    ksl_rel = np.pad(ksl_rel, (0, 10 - len(ksl_rel)))

    if main_is_skew:
        xo.assert_allclose(el_test.main_strength, 0.002 * 0.2, rtol=0, atol=1e-12)
    else:
        xo.assert_allclose(el_test.main_strength, 0.001 * 0.2, rtol=0, atol=1e-12)

    expected_knl = knl + knl_rel * el_test.main_strength
    expected_ksl = ksl + ksl_rel * el_test.main_strength

    expected_knl[3] += k3 * el_test.length
    expected_ksl[3] += k3s * el_test.length

    knl_tot, ksl_tot = el_test.get_total_knl_ksl()

    knl_tot = np.pad(knl_tot, (0, 10 - len(knl_tot)))
    ksl_tot = np.pad(ksl_tot, (0, 10 - len(ksl_tot)))

    xo.assert_allclose(knl_tot, expected_knl, rtol=0, atol=1e-12)
    xo.assert_allclose(ksl_tot, expected_ksl, rtol=0, atol=1e-12)

    el_ref = xt.Octupole(k3=k3, k3s=k3s, length=0.2,
                        knl=expected_knl, ksl=expected_ksl)
    el_ref.knl[3] -= k3 * el_ref.length  # Not to double count the main octupole contribution
    el_ref.ksl[3] -= k3s * el_ref.length

    p0 = xt.Particles(p0c=1e9, x=1e-2, y=2e-2)

    p_test = p0.copy()
    el_test.track(p_test)

    p_ref = p0.copy()
    el_ref.track(p_ref)

    xo.assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-13)

    line_test = xt.Line(elements=[el_test])
    line_ref = xt.Line(elements=[el_ref])

    p_test = p0.copy()
    line_test.track(p_test)
    line_test.track(p_test, backtrack=True)

    xo.assert_allclose(p_test.x, p0.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p0.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p0.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p0.py, rtol=0, atol=1e-13)

    # Check thick slicing
    line_test_slice_thick = line_test.copy(shallow=True)
    line_ref_slice_thick = line_ref.copy(shallow=True)

    line_test_slice_thick.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thick'))])
    line_ref_slice_thick.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thick'))])

    p_test = p0.copy()
    line_test_slice_thick.track(p_test)
    line_test_slice_thick.track(p_test, backtrack=True)

    xo.assert_allclose(p_test.x, p0.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p0.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p0.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p0.py, rtol=0, atol=1e-13)

    p_test = p0.copy()
    line_test_slice_thick.track(p_test)
    p_ref = p0.copy()
    line_ref_slice_thick.track(p_ref)

    xo.assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-13)

    # Check thin slicing
    line_test_slice_thin = line_test.copy(shallow=True)
    line_ref_slice_thin = line_ref.copy(shallow=True)
    line_test_slice_thin.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thin'))])
    line_ref_slice_thin.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thin'))])

    p_test = p0.copy()
    line_test_slice_thin.track(p_test)
    line_test_slice_thin.track(p_test, backtrack=True)

    xo.assert_allclose(p_test.x, p0.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p0.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p0.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p0.py, rtol=0, atol=1e-13)

    p_test = p0.copy()
    line_test_slice_thin.track(p_test)
    p_ref = p0.copy()
    line_ref_slice_thin.track(p_ref)

    xo.assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-13)

    _assert_spin_tracking_matches(line_test, line_ref)
    _assert_spin_tracking_matches(line_test_slice_thick, line_ref_slice_thick)
    _assert_spin_tracking_matches(line_test_slice_thin, line_ref_slice_thin)

@pytest.mark.parametrize("main_is_skew", [True, False])
def test_knl_rel_ksl_rel_multipole(main_is_skew):

    knl = np.array([0.001, 1e-3, 2e-2])
    ksl = np.array([0.002, 2e-3, 3e-2])
    knl_rel = np.array([1, 20, 30, 400, 50000, 6000000])
    ksl_rel = np.array([2, 40, 60, 800, 100000])

    el_test = xt.Multipole(length=0.2,
                        knl=knl,
                        ksl=ksl,
                        knl_rel=knl_rel,
                        ksl_rel=ksl_rel,
                        main_order=2,
                        main_is_skew=main_is_skew)

    knl = np.pad(knl, (0, 10 - len(knl)))
    ksl = np.pad(ksl, (0, 10 - len(ksl)))
    knl_rel = np.pad(knl_rel, (0, 10 - len(knl_rel)))
    ksl_rel = np.pad(ksl_rel, (0, 10 - len(ksl_rel)))

    if main_is_skew:
        xo.assert_allclose(el_test.main_strength, 3e-2, rtol=0, atol=1e-12)
    else:
        xo.assert_allclose(el_test.main_strength, 2e-2, rtol=0, atol=1e-12)

    expected_knl = knl + knl_rel * el_test.main_strength
    expected_ksl = ksl + ksl_rel * el_test.main_strength

    knl_tot, ksl_tot = el_test.get_total_knl_ksl()

    knl_tot = np.pad(knl_tot, (0, 10 - len(knl_tot)))
    ksl_tot = np.pad(ksl_tot, (0, 10 - len(ksl_tot)))

    xo.assert_allclose(knl_tot, expected_knl, rtol=0, atol=1e-12)
    xo.assert_allclose(ksl_tot, expected_ksl, rtol=0, atol=1e-12)

    el_ref = xt.Multipole(length=0.2, knl=expected_knl, ksl=expected_ksl)

    p0 = xt.Particles(p0c=1e9, x=1e-2, y=2e-2)

    p_test = p0.copy()
    el_test.track(p_test)

    p_ref = p0.copy()
    el_ref.track(p_ref)

    xo.assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-13)

    line_test = xt.Line(elements=[el_test])
    line_ref = xt.Line(elements=[el_ref])

    p_test = p0.copy()
    line_test.track(p_test)
    line_test.track(p_test, backtrack=True)

    xo.assert_allclose(p_test.x, p0.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p0.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p0.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p0.py, rtol=0, atol=1e-13)

    # Check thick slicing
    line_test_slice_thick = line_test.copy(shallow=True)
    line_ref_slice_thick = line_ref.copy(shallow=True)

    line_test_slice_thick.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thick'))])
    line_ref_slice_thick.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thick'))])

    p_test = p0.copy()
    line_test_slice_thick.track(p_test)
    line_test_slice_thick.track(p_test, backtrack=True)

    xo.assert_allclose(p_test.x, p0.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p0.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p0.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p0.py, rtol=0, atol=1e-13)

    p_test = p0.copy()
    line_test_slice_thick.track(p_test)
    p_ref = p0.copy()
    line_ref_slice_thick.track(p_ref)

    xo.assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-13)

    # Check thin slicing
    line_test_slice_thin = line_test.copy(shallow=True)
    line_ref_slice_thin = line_ref.copy(shallow=True)
    line_test_slice_thin.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thin'))])
    line_ref_slice_thin.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thin'))])

    p_test = p0.copy()
    line_test_slice_thin.track(p_test)
    line_test_slice_thin.track(p_test, backtrack=True)

    xo.assert_allclose(p_test.x, p0.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p0.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p0.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p0.py, rtol=0, atol=1e-13)

    p_test = p0.copy()
    line_test_slice_thin.track(p_test)
    p_ref = p0.copy()
    line_ref_slice_thin.track(p_ref)

    xo.assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-13)

    _assert_spin_tracking_matches(line_test, line_ref)
    _assert_spin_tracking_matches(line_test_slice_thick, line_ref_slice_thick)
    _assert_spin_tracking_matches(line_test_slice_thin, line_ref_slice_thin)

def test_knl_rel_ksl_rel_bend():

    knl = np.array([0.001, 1e-3, 2e-2])
    ksl = np.array([0.002, 2e-3, 3e-2])
    knl_rel = np.array([1, 20, 30, 400, 50000, 6000000])
    ksl_rel = np.array([2, 40, 60, 800, 100000])

    el_test = xt.Bend(length=0.2,
                    angle=0.01,
                    knl=knl,
                    ksl=ksl,
                    knl_rel=knl_rel,
                    ksl_rel=ksl_rel)

    knl = np.pad(knl, (0, 10 - len(knl)))
    ksl = np.pad(ksl, (0, 10 - len(ksl)))
    knl_rel = np.pad(knl_rel, (0, 10 - len(knl_rel)))
    ksl_rel = np.pad(ksl_rel, (0, 10 - len(ksl_rel)))

    xo.assert_allclose(el_test.main_strength, 0.01, rtol=0, atol=1e-12)

    expected_knl = knl + knl_rel * el_test.main_strength
    expected_ksl = ksl + ksl_rel * el_test.main_strength

    expected_knl[0] += el_test._k0 * el_test.length

    knl_tot, ksl_tot = el_test.get_total_knl_ksl()

    knl_tot = np.pad(knl_tot, (0, 10 - len(knl_tot)))
    ksl_tot = np.pad(ksl_tot, (0, 10 - len(ksl_tot)))

    xo.assert_allclose(knl_tot, expected_knl, rtol=0, atol=1e-12)
    xo.assert_allclose(ksl_tot, expected_ksl, rtol=0, atol=1e-12)

    expected_knl[0] -= el_test._k0 * el_test.length

    el_ref = xt.Bend(length=0.2, knl=expected_knl, ksl=expected_ksl, angle=0.01)

    p0 = xt.Particles(p0c=1e9, x=1e-2, y=2e-2)

    p_test = p0.copy()
    el_test.track(p_test)

    p_ref = p0.copy()
    el_ref.track(p_ref)

    xo.assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-13)

    line_test = xt.Line(elements=[el_test])
    line_ref = xt.Line(elements=[el_ref])

    p_test = p0.copy()
    line_test.track(p_test)
    line_test.track(p_test, backtrack=True)

    xo.assert_allclose(p_test.x, p0.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p0.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p0.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p0.py, rtol=0, atol=1e-13)

    # Check thick slicing
    line_test_slice_thick = line_test.copy(shallow=True)
    line_ref_slice_thick = line_ref.copy(shallow=True)

    line_test_slice_thick.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thick'))])
    line_ref_slice_thick.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thick'))])

    p_test = p0.copy()
    line_test_slice_thick.track(p_test)
    line_test_slice_thick.track(p_test, backtrack=True)

    xo.assert_allclose(p_test.x, p0.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p0.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p0.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p0.py, rtol=0, atol=1e-13)

    p_test = p0.copy()
    line_test_slice_thick.track(p_test)
    p_ref = p0.copy()
    line_ref_slice_thick.track(p_ref)

    xo.assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-13)

    # Check thin slicing
    line_test_slice_thin = line_test.copy(shallow=True)
    line_ref_slice_thin = line_ref.copy(shallow=True)
    line_test_slice_thin.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thin'))])
    line_ref_slice_thin.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thin'))])

    p_test = p0.copy()
    line_test_slice_thin.track(p_test)
    line_test_slice_thin.track(p_test, backtrack=True)

    xo.assert_allclose(p_test.x, p0.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p0.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p0.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p0.py, rtol=0, atol=1e-13)

    p_test = p0.copy()
    line_test_slice_thin.track(p_test)
    p_ref = p0.copy()
    line_ref_slice_thin.track(p_ref)

    xo.assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-13)

    _assert_spin_tracking_matches(line_test, line_ref)
    _assert_spin_tracking_matches(line_test_slice_thick, line_ref_slice_thick)
    _assert_spin_tracking_matches(line_test_slice_thin, line_ref_slice_thin)

def test_knl_rel_ksl_rel_rbend():

    knl = np.array([0.001, 1e-3, 2e-2])
    ksl = np.array([0.002, 2e-3, 3e-2])
    knl_rel = np.array([1, 20, 30, 400, 50000, 6000000])
    ksl_rel = np.array([2, 40, 60, 800, 100000])

    el_test = xt.RBend(length_straight=0.2,
                    angle=0.01,
                    knl=knl,
                    ksl=ksl,
                    knl_rel=knl_rel,
                    ksl_rel=ksl_rel)

    knl = np.pad(knl, (0, 10 - len(knl)))
    ksl = np.pad(ksl, (0, 10 - len(ksl)))
    knl_rel = np.pad(knl_rel, (0, 10 - len(knl_rel)))
    ksl_rel = np.pad(ksl_rel, (0, 10 - len(ksl_rel)))

    xo.assert_allclose(el_test.main_strength, 0.01, rtol=0, atol=1e-12)

    expected_knl = knl + knl_rel * el_test.main_strength
    expected_ksl = ksl + ksl_rel * el_test.main_strength

    expected_knl[0] += el_test._k0 * el_test.length

    knl_tot, ksl_tot = el_test.get_total_knl_ksl()

    knl_tot = np.pad(knl_tot, (0, 10 - len(knl_tot)))
    ksl_tot = np.pad(ksl_tot, (0, 10 - len(ksl_tot)))

    xo.assert_allclose(knl_tot, expected_knl, rtol=0, atol=1e-12)
    xo.assert_allclose(ksl_tot, expected_ksl, rtol=0, atol=1e-12)

    expected_knl[0] -= el_test._k0 * el_test.length

    el_ref = xt.RBend(length_straight=0.2, knl=expected_knl, ksl=expected_ksl, angle=0.01)

    p0 = xt.Particles(p0c=1e9, x=1e-2, y=2e-2)

    p_test = p0.copy()
    el_test.track(p_test)

    p_ref = p0.copy()
    el_ref.track(p_ref)

    xo.assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-13)

    line_test = xt.Line(elements=[el_test])
    line_ref = xt.Line(elements=[el_ref])

    p_test = p0.copy()
    line_test.track(p_test)
    line_test.track(p_test, backtrack=True)

    xo.assert_allclose(p_test.x, p0.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p0.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p0.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p0.py, rtol=0, atol=1e-13)

    # Check thick slicing
    line_test_slice_thick = line_test.copy(shallow=True)
    line_ref_slice_thick = line_ref.copy(shallow=True)

    line_test_slice_thick.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thick'))])
    line_ref_slice_thick.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thick'))])

    p_test = p0.copy()
    line_test_slice_thick.track(p_test)
    line_test_slice_thick.track(p_test, backtrack=True)

    xo.assert_allclose(p_test.x, p0.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p0.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p0.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p0.py, rtol=0, atol=1e-13)

    p_test = p0.copy()
    line_test_slice_thick.track(p_test)
    p_ref = p0.copy()
    line_ref_slice_thick.track(p_ref)

    xo.assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-13)

    # Check thin slicing
    line_test_slice_thin = line_test.copy(shallow=True)
    line_ref_slice_thin = line_ref.copy(shallow=True)
    line_test_slice_thin.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thin'))])
    line_ref_slice_thin.slice_thick_elements(
            slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thin'))])

    p_test = p0.copy()
    line_test_slice_thin.track(p_test)
    line_test_slice_thin.track(p_test, backtrack=True)

    xo.assert_allclose(p_test.x, p0.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p0.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p0.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p0.py, rtol=0, atol=1e-13)

    p_test = p0.copy()
    line_test_slice_thin.track(p_test)
    p_ref = p0.copy()
    line_ref_slice_thin.track(p_ref)

    xo.assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-13)

    _assert_spin_tracking_matches(line_test, line_ref)
    _assert_spin_tracking_matches(line_test_slice_thick, line_ref_slice_thick)
    _assert_spin_tracking_matches(line_test_slice_thin, line_ref_slice_thin)
