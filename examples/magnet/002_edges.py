import numpy as np

import xtrack as xt
import xobjects as xo
import numpy as np

from xtrack.beam_elements.magnets import Magnet, MagnetEdge

def compare(e_test, e_ref):
    p0 = xt.Particles(kinetic_energy0=50e6,
                      x=1e-3, y=2e-3, zeta=1e-2, px=10e-3, py=20e-3, delta=1e-2)

    e_test.compile_kernels()

    # Expanded drift
    p_test = p0.copy()
    p_ref = p0.copy()

    e_test.track(p_test)

    if isinstance(e_ref, list):
        mini_line = xt.Line(elements=e_ref)
        mini_line.build_tracker()
        mini_line.track(p_ref)
    else:
        e_ref.track(p_ref)

    xo.assert_allclose(p_test.x, p_ref.x, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test.y, p_ref.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test.zeta, p_ref.zeta, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test.px, p_ref.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test.py, p_ref.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p_test.delta, p_ref.delta, atol=1e-15, rtol=0)


# Supressed
e_test = MagnetEdge(model='suppressed', kn=[0], ks=[0])
e_ref = xt.DipoleEdge(model='suppressed', k=0)
compare(e_test, e_ref)

# Linear but does nothing
e_test = MagnetEdge(model='linear', kn=[0], ks=[0])
e_ref = xt.DipoleEdge(model='linear', k=0)
compare(e_test, e_ref)

# Full but does nothing
e_test = MagnetEdge(model='full', kn=[0], ks=[0])
e_ref = xt.DipoleEdge(model='full', k=0)
compare(e_test, e_ref)

# Only linear
e_test = MagnetEdge(model='linear', kn=[3], face_angle=0.1, face_angle_feed_down=0.2, fringe_integral=0.3, half_gap=0.4)
e_ref = xt.DipoleEdge(model='linear', k=3, e1=0.1, e1_fd=0.2, fint=0.3, hgap=0.4)
compare(e_test, e_ref)

# Full but with only the dipole component
e_test = MagnetEdge(model='full', kn=[3], face_angle=0.1, face_angle_feed_down=0.2, fringe_integral=0.3, half_gap=0.4)
e_ref = xt.DipoleEdge(model='full', k=3, e1=0.1, e1_fd=0.2, fint=0.3, hgap=0.4)
compare(e_test, e_ref)

# Multipole fringe, without the dipole component
e_test = MagnetEdge(model='full', kn=[0, 2, 3], order=2)
e_ref = xt.MultipoleEdge(kn=[0, 2, 3], order=2)
compare(e_test, e_ref)

# Test the full model with a dipole component, no angle
e_test = MagnetEdge(model='full', kn=[3, 4, 5], fringe_integral=0.3, half_gap=0.4, order=2)
sandwich = [
    xt.DipoleEdge(model='full', k=3, fint=0.3, hgap=0.4),
    xt.MultipoleEdge(kn=[0, 4, 5], order=2),
]
compare(e_test, sandwich)

# Test the full model with a dipole component and angle
e_test = MagnetEdge(model='full', kn=[3, 4, 5], face_angle=0.2, face_angle_feed_down=0.0, fringe_integral=0.3, half_gap=0.4, order=2)
sandwich = [
    xt.YRotation(angle=np.rad2deg(-0.2)),  # The rotation is also the other way than in the underlying map :'(
    xt.DipoleEdge(model='full', k=3, fint=0.3, hgap=0.4),
    xt.MultipoleEdge(kn=[0, 4, 5], order=2),
    xt.Wedge(angle=-0.2, k=3),
]
compare(e_test, sandwich)
