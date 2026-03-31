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

    xo.assert_allclose(tw_test.spin_x, tw_ref.spin_x, rtol=0, atol=1e-14)
    xo.assert_allclose(tw_test.spin_y, tw_ref.spin_y, rtol=0, atol=1e-14)
    xo.assert_allclose(tw_test.spin_z, tw_ref.spin_z, rtol=0, atol=1e-14)

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

def test_knl_rel_ksl_rel_in_line_table():

    env = xt.Environment()

    line1 = env.new_line(components=[
        env.new('bend', xt.Bend, length=0.1, angle=0.01, at=0.3,
                knl=[0.002, 0.03, 0.4, 5, 6],
                ksl=[0.003, 0.04, 0.5, 6, 7],
                knl_rel=[0.1, 0.2, 0.3, 0.4, 0.5],
                ksl_rel=[0.1, 0.2, 0.3, 0.4, 0.5]),
        env.new('rbend', xt.RBend, length_straight=0.1, angle=0.01, at=0.6,
                knl=[0.003, 0.04, 0.5, 6, 7],
                ksl=[0.004, 0.05, 0.6, 7, 8],
                knl_rel=[0.2, 0.3, 0.4, 0.5, 0.6],
                ksl_rel=[0.7, 0.6, 0.5, 0.4, 0.3]),
        env.new('quad', xt.Quadrupole, length=0.1, k1=2, at=1,
                knl=[0.001, 0.02, 0.3, 4, 5],
                ksl=[0.002, 0.03, 0.4, 5, 6],
                knl_rel=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                ksl_rel=[0.8, 0.7, 0.6, 0.5, 0.4, 0.3]),
        env.new('skew_quad', xt.Quadrupole, length=0.1, k1s=2, main_is_skew=True, at=2,
                knl=[0.001, 0.02, 0.3, 4, 5],
                ksl=[0.002, 0.03, 0.4, 5, 6],
                knl_rel=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                ksl_rel=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
        env.new('sext', xt.Sextupole, length=0.1, k2=3, at=3,
                knl=[0.0005, 0.01, 0.2, 3, 4],
                ksl=[0.001, 0.02, 0.3, 4, 5],
                knl_rel=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                ksl_rel=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
        env.new('skew_sext', xt.Sextupole, length=0.1, k2s=3, main_is_skew=True, at=4,
                knl=[0.0005, 0.01, 0.2, 3, 4],
                ksl=[0.001, 0.02, 0.3, 4, 5],
                knl_rel=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                ksl_rel=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
        env.new('oct', xt.Octupole, length=0.1, k3=4, at=5),
        env.new('skew_oct', xt.Octupole, length=0.1, k3s=4, main_is_skew=True, at=6,
                knl=[0.0001, 0.001, 0.01, 2, 3],
                ksl=[0.0002, 0.002, 0.02, 3, 4],
                knl_rel=[0.5, 0.6, 0.7, 0.8, 0.9, 1],
                ksl_rel=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
        env.new('multipole', xt.Multipole, length=0.1, knl=[0,0,0,0,2], isthick=True,
                main_order=4, at=7,
                ksl=[1,2,3,4,5],
                knl_rel=[0.1, 0.2, 0.3, 0.4, 0.5],
                ksl_rel=[0.5, 0.4, 0.3, 0.2, 0.1],
                ),
        env.new('skew_multipole', xt.Multipole, length=0.1, ksl=[0,0,0,0,3], isthick=True,
                main_is_skew=True, main_order=4, at=8,
                knl=[5,4,3,2,1],
                knl_rel=[0.5, 0.4, 0.3, 0.2, 0.1],
                ksl_rel=[0.1, 0.2, 0.3, 0.4, 0.5],
                ),
    ])
    line2 = line1.copy(shallow=True)
    line2.slice_thick_elements(slicing_strategies=[
        xt.Strategy(slicing=xt.Teapot(2, mode='thick'))])

    line3 = line1.copy(shallow=True)
    line3.slice_thick_elements(slicing_strategies=[
        xt.Strategy(slicing=xt.Teapot(2, mode='thin'))])

    line = line1 + line2 + line3

    tt = line.get_table(attr=True)

    # Check _main_strength, k0l, k0sl, k2l, k2sl, k3l, k3sl, k4l, k4sl, k5l, k5sl
    for nn in tt.name:

        if nn == '_end_point':
            continue

        ee = line[nn]

        if isinstance(ee, (xt.Bend, xt.RBend, xt.Quadrupole, xt.Sextupole, xt.Octupole, xt.Multipole)):
            xo.assert_allclose(ee.main_strength, tt['_main_strength', nn], rtol=0, atol=1e-14)
            knl, ksl = ee.get_total_knl_ksl()
            for ii in range(6):
                if ii >= len(knl):
                    assert tt[f'k{ii}l', nn] == 0
                    assert tt[f'k{ii}sl', nn] == 0
                else:
                    xo.assert_allclose(knl[ii], tt[f'k{ii}l', nn], rtol=0, atol=1e-14)
                    xo.assert_allclose(ksl[ii], tt[f'k{ii}sl', nn], rtol=0, atol=1e-14)
        elif (ee.__class__.__name__.startswith('ThickSlice')
            or ee.__class__.__name__.startswith('ThinSlice')
            or ee.__class__.__name__.startswith('DriftSlice')):
            xo.assert_allclose(ee._parent.main_strength*ee.weight*ee._inherit_strengths, tt['_main_strength', nn], rtol=0, atol=1e-14)
            knl_parent, ksl_parent = ee._parent.get_total_knl_ksl()
            for ii in range(6):
                if ii >= len(knl_parent):
                    assert tt[f'k{ii}l', nn] == 0
                    assert tt[f'k{ii}sl', nn] == 0
                else:
                    xo.assert_allclose(knl_parent[ii]*ee.weight*ee._inherit_strengths, tt[f'k{ii}l', nn], rtol=0, atol=1e-14)
                    xo.assert_allclose(ksl_parent[ii]*ee.weight*ee._inherit_strengths, tt[f'k{ii}sl', nn], rtol=0, atol=1e-14)
        else:
            assert isinstance(ee, (xt.Drift, xt.Marker))
            assert tt['_main_strength', nn] == 0

    assert np.all(tt.name == np.array([
        '||drift_1::0', 'bend', '||drift_2::0', 'rbend', '||drift_3::0',
        'quad', '||drift_4::0', 'skew_quad', '||drift_5::0', 'sext',
        '||drift_4::1', 'skew_sext', '||drift_5::1', 'oct', '||drift_5::2',
        'skew_oct', '||drift_5::3', 'multipole', '||drift_5::4',
        'skew_multipole', '||drift_1::1', 'bend_entry::0',
        'bend..entry_map', 'bend..0', 'bend..1', 'bend..exit_map',
        'bend_exit::0', '||drift_2::1', 'rbend_entry::0',
        'rbend..entry_map', 'rbend..0', 'rbend..1', 'rbend..exit_map',
        'rbend_exit::0', '||drift_3::1', 'quad_entry::0',
        'quad..entry_map', 'quad..0', 'quad..1', 'quad..exit_map',
        'quad_exit::0', '||drift_4::2', 'skew_quad_entry::0',
        'skew_quad..entry_map', 'skew_quad..0', 'skew_quad..1',
        'skew_quad..exit_map', 'skew_quad_exit::0', '||drift_5::5',
        'sext_entry::0', 'sext..entry_map', 'sext..0', 'sext..1',
        'sext..exit_map', 'sext_exit::0', '||drift_4::3',
        'skew_sext_entry::0', 'skew_sext..entry_map', 'skew_sext..0',
        'skew_sext..1', 'skew_sext..exit_map', 'skew_sext_exit::0',
        '||drift_5::6', 'oct_entry::0', 'oct..entry_map', 'oct..0',
        'oct..1', 'oct..exit_map', 'oct_exit::0', '||drift_5::7',
        'skew_oct_entry::0', 'skew_oct..entry_map', 'skew_oct..0',
        'skew_oct..1', 'skew_oct..exit_map', 'skew_oct_exit::0',
        '||drift_5::8', 'multipole_entry::0', 'multipole..0',
        'multipole..1', 'multipole_exit::0', '||drift_5::9',
        'skew_multipole_entry::0', 'skew_multipole..0',
        'skew_multipole..1', 'skew_multipole_exit::0', '||drift_1::2',
        'bend_entry::1', 'bend..entry_map_0', 'drift_bend..0', 'bend..2',
        'drift_bend..1', 'bend..3', 'drift_bend..2', 'bend..exit_map_0',
        'bend_exit::1', '||drift_2::2', 'rbend_entry::1',
        'rbend..entry_map_0', 'drift_rbend..0', 'rbend..2',
        'drift_rbend..1', 'rbend..3', 'drift_rbend..2',
        'rbend..exit_map_0', 'rbend_exit::1', '||drift_3::2',
        'quad_entry::1', 'quad..entry_map_0', 'drift_quad..0', 'quad..2',
        'drift_quad..1', 'quad..3', 'drift_quad..2', 'quad..exit_map_0',
        'quad_exit::1', '||drift_4::4', 'skew_quad_entry::1',
        'skew_quad..entry_map_0', 'drift_skew_quad..0', 'skew_quad..2',
        'drift_skew_quad..1', 'skew_quad..3', 'drift_skew_quad..2',
        'skew_quad..exit_map_0', 'skew_quad_exit::1', '||drift_5::10',
        'sext_entry::1', 'sext..entry_map_0', 'drift_sext..0', 'sext..2',
        'drift_sext..1', 'sext..3', 'drift_sext..2', 'sext..exit_map_0',
        'sext_exit::1', '||drift_4::5', 'skew_sext_entry::1',
        'skew_sext..entry_map_0', 'drift_skew_sext..0', 'skew_sext..2',
        'drift_skew_sext..1', 'skew_sext..3', 'drift_skew_sext..2',
        'skew_sext..exit_map_0', 'skew_sext_exit::1', '||drift_5::11',
        'oct_entry::1', 'oct..entry_map_0', 'drift_oct..0', 'oct..2',
        'drift_oct..1', 'oct..3', 'drift_oct..2', 'oct..exit_map_0',
        'oct_exit::1', '||drift_5::12', 'skew_oct_entry::1',
        'skew_oct..entry_map_0', 'drift_skew_oct..0', 'skew_oct..2',
        'drift_skew_oct..1', 'skew_oct..3', 'drift_skew_oct..2',
        'skew_oct..exit_map_0', 'skew_oct_exit::1', '||drift_5::13',
        'multipole_entry::1', 'drift_multipole..0', 'multipole..2',
        'drift_multipole..1', 'multipole..3', 'drift_multipole..2',
        'multipole_exit::1', '||drift_5::14', 'skew_multipole_entry::1',
        'drift_skew_multipole..0', 'skew_multipole..2',
        'drift_skew_multipole..1', 'skew_multipole..3',
        'drift_skew_multipole..2', 'skew_multipole_exit::1', '_end_point']))

    assert np.all(tt.element_type == np.array(
        ['Drift', 'Bend', 'Drift', 'RBend', 'Drift', 'Quadrupole', 'Drift',
        'Quadrupole', 'Drift', 'Sextupole', 'Drift', 'Sextupole', 'Drift',
        'Octupole', 'Drift', 'Octupole', 'Drift', 'Multipole', 'Drift',
        'Multipole', 'Drift', 'Marker', 'ThinSliceBendEntry',
        'ThickSliceBend', 'ThickSliceBend', 'ThinSliceBendExit', 'Marker',
        'Drift', 'Marker', 'ThinSliceRBendEntry', 'ThickSliceRBend',
        'ThickSliceRBend', 'ThinSliceRBendExit', 'Marker', 'Drift',
        'Marker', 'ThinSliceQuadrupoleEntry', 'ThickSliceQuadrupole',
        'ThickSliceQuadrupole', 'ThinSliceQuadrupoleExit', 'Marker',
        'Drift', 'Marker', 'ThinSliceQuadrupoleEntry',
        'ThickSliceQuadrupole', 'ThickSliceQuadrupole',
        'ThinSliceQuadrupoleExit', 'Marker', 'Drift', 'Marker',
        'ThinSliceSextupoleEntry', 'ThickSliceSextupole',
        'ThickSliceSextupole', 'ThinSliceSextupoleExit', 'Marker', 'Drift',
        'Marker', 'ThinSliceSextupoleEntry', 'ThickSliceSextupole',
        'ThickSliceSextupole', 'ThinSliceSextupoleExit', 'Marker', 'Drift',
        'Marker', 'ThinSliceOctupoleEntry', 'ThickSliceOctupole',
        'ThickSliceOctupole', 'ThinSliceOctupoleExit', 'Marker', 'Drift',
        'Marker', 'ThinSliceOctupoleEntry', 'ThickSliceOctupole',
        'ThickSliceOctupole', 'ThinSliceOctupoleExit', 'Marker', 'Drift',
        'Marker', 'ThickSliceMultipole', 'ThickSliceMultipole', 'Marker',
        'Drift', 'Marker', 'ThickSliceMultipole', 'ThickSliceMultipole',
        'Marker', 'Drift', 'Marker', 'ThinSliceBendEntry',
        'DriftSliceBend', 'ThinSliceBend', 'DriftSliceBend',
        'ThinSliceBend', 'DriftSliceBend', 'ThinSliceBendExit', 'Marker',
        'Drift', 'Marker', 'ThinSliceRBendEntry', 'DriftSliceRBend',
        'ThinSliceRBend', 'DriftSliceRBend', 'ThinSliceRBend',
        'DriftSliceRBend', 'ThinSliceRBendExit', 'Marker', 'Drift',
        'Marker', 'ThinSliceQuadrupoleEntry', 'DriftSliceQuadrupole',
        'ThinSliceQuadrupole', 'DriftSliceQuadrupole',
        'ThinSliceQuadrupole', 'DriftSliceQuadrupole',
        'ThinSliceQuadrupoleExit', 'Marker', 'Drift', 'Marker',
        'ThinSliceQuadrupoleEntry', 'DriftSliceQuadrupole',
        'ThinSliceQuadrupole', 'DriftSliceQuadrupole',
        'ThinSliceQuadrupole', 'DriftSliceQuadrupole',
        'ThinSliceQuadrupoleExit', 'Marker', 'Drift', 'Marker',
        'ThinSliceSextupoleEntry', 'DriftSliceSextupole',
        'ThinSliceSextupole', 'DriftSliceSextupole', 'ThinSliceSextupole',
        'DriftSliceSextupole', 'ThinSliceSextupoleExit', 'Marker', 'Drift',
        'Marker', 'ThinSliceSextupoleEntry', 'DriftSliceSextupole',
        'ThinSliceSextupole', 'DriftSliceSextupole', 'ThinSliceSextupole',
        'DriftSliceSextupole', 'ThinSliceSextupoleExit', 'Marker', 'Drift',
        'Marker', 'ThinSliceOctupoleEntry', 'DriftSliceOctupole',
        'ThinSliceOctupole', 'DriftSliceOctupole', 'ThinSliceOctupole',
        'DriftSliceOctupole', 'ThinSliceOctupoleExit', 'Marker', 'Drift',
        'Marker', 'ThinSliceOctupoleEntry', 'DriftSliceOctupole',
        'ThinSliceOctupole', 'DriftSliceOctupole', 'ThinSliceOctupole',
        'DriftSliceOctupole', 'ThinSliceOctupoleExit', 'Marker', 'Drift',
        'Marker', 'DriftSliceMultipole', 'ThinSliceMultipole',
        'DriftSliceMultipole', 'ThinSliceMultipole', 'DriftSliceMultipole',
        'Marker', 'Drift', 'Marker', 'DriftSliceMultipole',
        'ThinSliceMultipole', 'DriftSliceMultipole', 'ThinSliceMultipole',
        'DriftSliceMultipole', 'Marker', '']))

    assert np.all(tt.parent_type == np.array(
        [None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
        'Bend', 'Bend', 'Bend', 'Bend', None, None, None, 'RBend', 'RBend',
        'RBend', 'RBend', None, None, None, 'Quadrupole', 'Quadrupole',
        'Quadrupole', 'Quadrupole', None, None, None, 'Quadrupole',
        'Quadrupole', 'Quadrupole', 'Quadrupole', None, None, None,
        'Sextupole', 'Sextupole', 'Sextupole', 'Sextupole', None, None,
        None, 'Sextupole', 'Sextupole', 'Sextupole', 'Sextupole', None,
        None, None, 'Octupole', 'Octupole', 'Octupole', 'Octupole', None,
        None, None, 'Octupole', 'Octupole', 'Octupole', 'Octupole', None,
        None, None, 'Multipole', 'Multipole', None, None, None,
        'Multipole', 'Multipole', None, None, None, 'Bend', 'Bend', 'Bend',
        'Bend', 'Bend', 'Bend', 'Bend', None, None, None, 'RBend', 'RBend',
        'RBend', 'RBend', 'RBend', 'RBend', 'RBend', None, None, None,
        'Quadrupole', 'Quadrupole', 'Quadrupole', 'Quadrupole',
        'Quadrupole', 'Quadrupole', 'Quadrupole', None, None, None,
        'Quadrupole', 'Quadrupole', 'Quadrupole', 'Quadrupole',
        'Quadrupole', 'Quadrupole', 'Quadrupole', None, None, None,
        'Sextupole', 'Sextupole', 'Sextupole', 'Sextupole', 'Sextupole',
        'Sextupole', 'Sextupole', None, None, None, 'Sextupole',
        'Sextupole', 'Sextupole', 'Sextupole', 'Sextupole', 'Sextupole',
        'Sextupole', None, None, None, 'Octupole', 'Octupole', 'Octupole',
        'Octupole', 'Octupole', 'Octupole', 'Octupole', None, None, None,
        'Octupole', 'Octupole', 'Octupole', 'Octupole', 'Octupole',
        'Octupole', 'Octupole', None, None, None, 'Multipole', 'Multipole',
        'Multipole', 'Multipole', 'Multipole', None, None, None,
        'Multipole', 'Multipole', 'Multipole', 'Multipole', 'Multipole',
        None, None]))
