import xtrack as xt
import numpy as np
from cpymad.madx import Madx

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

    p0 = xt.Particles(x=x,px=px,y=y,py=py)  # To do: update values and add py

    line.discard_tracker()
    line.build_tracker(use_prebuilt_kernels=False)
    line.track(p0)

    x_expval = np.array([-0.01000546, -0.00500069, 0., 0.00500069, 0.01000559])
    px_expval = np.array([-5.00959400e-02, -2.50261314e-02, 1.38777878e-17, 2.49695392e-02, 4.98693687e-02])
    y_expval = np.array([-0.01999424, -0.00999928, 0., 0.00999926, 0.01999407])
    py_expval = np.array([-0.03014372, -0.01503676, 0., 0.01496153, 0.0298426 ])

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


def test_quadrupole_wedge_ptc():
    angle = 2
    b2 = 5
    b1 = 0
    length=0.000001

    x0 = 0.01
    px0 = 0.05
    y0 = 0.01
    py0 = 0.0
    t0 = 0.0  # Convert to xsuite if nonzero
    pt0 = 0.0  # Convert to xsuite if nonzero

    # XSuite
    quadrupole = xt.Bend(length=length, k0=b1, k1=b2, edge_entry_angle=angle, edge_entry_model='1', edge_exit_model='1')  # 1 with quadrupole fringe
    line = xt.Line(elements=[quadrupole])
    
    p0 = xt.Particles(x=x0,px=px0,y=y0,py=py0)

    line.discard_tracker()
    line.build_tracker(use_prebuilt_kernels=False)
    line.track(p0)
    print(p0.x, p0.px, p0.y, p0.py)
    
    
    # PTC tracking 
    madx = Madx(stdout=True)
    
    madx_sequence = line.to_madx_sequence('quadrupole_with_wedge')
    madx.beam()
    madx.input(madx_sequence)
    madx.use('quadrupole_with_wedge')
    
    
    # exact=True to include edge angles, 
    # fringe=True to include hard edge fringe
    madx.input(f"""
               ptc_create_universe;
               ptc_create_layout, EXACT=true;
               ptc_setswitch, fringe=true;
               PTC_START, X={x0}, PX={px0}, Y={y0}, PY={py0}, T={t0}, PT={pt0};
               PTC_TRACK, TURNS=1;
               PTC_END;
    """)
    
    df = madx.table.tracksumm.dframe()
    
    print(p0.x, p0.px, p0.y, p0.py)
    print(df.x[1], df.px[1], df.y[1], df.py[1])
    
    