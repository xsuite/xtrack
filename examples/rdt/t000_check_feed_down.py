import xtrack as xt
import feed_down as fd
import numpy as np
import xobjects as xo
def test_feed_down_rotation_dipole():
    kn = np.array([1, 0])
    ks = np.array([0, 0])
    psi = np.pi / 2 # 90 degrees tilt

    kn_eff, kskew_eff = fd.feed_down(
        kn=kn[None, :],
        kskew=ks[None, :],
        shift_x=0.,
        shift_y=0.,
        psi=psi,
        x0=0.,
        y0=0.,
        max_output_order=None,
    )

    assert kn_eff.shape == (1, 2)
    assert kskew_eff.shape == (1, 2)

    xo.assert_allclose(kn_eff, np.array([[0, 0]]), atol=1e-15)
    xo.assert_allclose(kskew_eff, np.array([[-1, 0]]), atol=1e-15)

    m = xt.Multipole(knl=kn, ksl=ks, rot_s_rad=psi)
    m_fdown = xt.Multipole(knl=kn_eff[0, :], ksl=kskew_eff[0, :], rot_s_rad=0.0)

    p = xt.Particles()
    p_fdown = p.copy()
    m.track(p)
    m_fdown.track(p_fdown)

    xo.assert_allclose(p.px, p_fdown.px, atol=1e-15)
    xo.assert_allclose(p.py, p_fdown.py, atol=1e-15)

def test_feed_down_rotation_quadrupole():
    kn = np.array([0, 1])
    ks = np.array([0, 0])
    psi = np.pi / 4 # 45 degrees tilt

    kn_eff, kskew_eff = fd.feed_down(
        kn=kn[None, :],
        kskew=ks[None, :],
        shift_x=0.,
        shift_y=0.,
        psi=psi,
        x0=0.,
        y0=0.,
        max_output_order=None,
    )

    m = xt.Multipole(knl=kn, ksl=ks, rot_s_rad=psi)
    m_fdown = xt.Multipole(knl=kn_eff[0, :], ksl=kskew_eff[0, :], rot_s_rad=0.0)

    p = xt.Particles()
    p_fdown = p.copy()
    m.track(p)
    m_fdown.track(p_fdown)

    xo.assert_allclose(p.px, p_fdown.px, atol=1e-15)
    xo.assert_allclose(p.py, p_fdown.py, atol=1e-15)


def test_feed_down_rotation_higher_order():
    kn = np.array([0, 1, 2, 10])
    ks = np.array([0, 3, 2, 20])
    psi = np.pi / 4 # 45 degrees tilt
    shift_x = 0.001
    shift_y = -0.002

    kn_eff, kskew_eff = fd.feed_down(
        kn=kn[None, :],
        kskew=ks[None, :],
        shift_x=shift_x,
        shift_y=shift_y,
        psi=psi,
        x0=0.,
        y0=0.,
        max_output_order=None,
    )

    m = xt.Multipole(knl=kn, ksl=ks, rot_s_rad=psi, shift_x=shift_x, shift_y=shift_y)
    m_fdown = xt.Multipole(knl=kn_eff[0, :], ksl=kskew_eff[0, :])

    p = xt.Particles(x=4e-3, y=5e-3)
    p_fdown = p.copy()
    m.track(p)
    m_fdown.track(p_fdown)

    xo.assert_allclose(p.px, p_fdown.px, atol=1e-15)
    xo.assert_allclose(p.py, p_fdown.py, atol=1e-15)


if __name__ == '__main__':
    test_feed_down_rotation_dipole()
    test_feed_down_rotation_quadrupole()
    test_feed_down_rotation_higher_order()