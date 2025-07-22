import xtrack as xt
import numpy as np

def test_quadrupole_wedge():
    """
    Hardcoded test for quadrupole wedge with hard edge fringe.
    """
    
    angle = 0.1
    b2 = 5
    b1 = 0

    quadrupole = xt.Bend(length=0, k0=b1, k1=b2, edge_entry_angle=angle, edge_entry_model='1')  # 1 with quadrupole fringe
    line= xt.Line(elements=[quadrupole])

    x=np.linspace(-1e-2, 1e-2, 5)
    px=np.linspace(-5e-2, 5e-2, 5)
    y=np.linspace(-2e-2, 2e-2, 5)
    py=np.linspace(-3e-2, 3e-2, 5)

    p0 = xt.Particles(x=x,px=px,y=y)

    line.discard_tracker()
    line.build_tracker(use_prebuilt_kernels=False)
    line.track(p0)

    x_expval = np.array([-0.01000545, -0.00500069, 0., 0.00500069, 0.01000558])
    px_expval = np.array([-5.00811572e-02, -2.50242752e-02, 1.38777878e-17, 2.49676638e-02, 4.98542802e-02])
    y_expval = np.array([-0.01999418, -0.00999927, 0., 0.00999927, 0.01999413])
    py_expval = np.array([-1.25035798e-04, -3.44174278e-05, 0.00000000e+00, -4.08274331e-05, -1.76367653e-04])

    assert np.allclose(p0.x, x_expval)
    assert np.allclose(p0.px, px_expval)
    assert np.allclose(p0.y, y_expval)
    assert np.allclose(p0.py, py_expval)



# # The combined effect of Lee-Whiting quadrupole fringe with a wedge is in small angle approximation 
# # a sextupole with strength -3/2 b2 theta. The Lee-Whiting map is an approximation of the hard edge in XSuite.
#
# quadrupole = xt.Bend(length=0, k0=b1, k1=b2, edge_entry_angle=angle, edge_entry_model='2')  # 2 only dipole fringe
# sextupole = xt.Multipole(knl = [0, 0, -3/2 * b2 * angle])
# line_sext = xt.Line(elements=[sextupole, quadrupole])
#
# p1 = xt.Particles(x=x,px=px,y=y)
#
# line_sext.discard_tracker()
# line_sext.build_tracker(use_prebuilt_kernels=False)
# line_sext.track(p1)
#
# print(p1.x, p1.px, p1.y, p1.py)
