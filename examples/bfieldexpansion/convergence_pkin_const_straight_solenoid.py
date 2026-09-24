"""Degree-six round solenoid: pkin_const convergence and symplecticity.

Run from the repository root:
    python -m examples.bfieldexpansion.convergence_pkin_const_straight_solenoid

The on-axis Bs/(B rho) is 0.30+0.10*64*u^3*(1-u)^3, u=s/L, L=1 m.
Include all radial fringe terms of its exact, finite axisymmetric expansion.
The cubic approximations fit ksol and each transverse derivative ksc[i] separately;
the cubic pieces need not individually retain exact axisymmetry.

The study includes smooth-field RK4 convergence, cubic-segment refinement,
both pkin_const settings, and canonical 6D symplectic defects versus y and N.
--no-plot and --save-plot /tmp/straight_solenoid.png support batch runs.
"""

from math import factorial

import numpy as np
from numpy.polynomial import Polynomial

from ._convergence_cases import FieldCase, round_solenoid_seeds, run_case


def make_case():
    profile = Polynomial([0.30, 0, 0, 6.4, -19.2, 19.2, -6.4])
    ksc, knc, ksol = round_solenoid_seeds(profile)

    def check_field(element):
        x, y, s = np.meshgrid(np.linspace(-0.06, 0.06, 5),
                              np.linspace(-0.06, 0.06, 5),
                              np.linspace(0, 1, 9), indexing='ij')
        r2 = x*x+y*y
        radial = sum((-1)**(k+1)*r2**k*profile.deriv(2*k+1)(s)
                     / (2**(2*k+1)*factorial(k)*factorial(k+1)) for k in range(3))
        longitudinal = sum((-1)**k*r2**k*profile.deriv(2*k)(s)
                           / (2**(2*k)*factorial(k)**2) for k in range(4))
        field = element.get_field(x, y, s)
        for name, expected in [('Bx', x*radial), ('By', y*radial), ('Bs', longitudinal)]:
            np.testing.assert_allclose(field[name], expected, rtol=0, atol=2e-13)

    return FieldCase('Straight round solenoid', length=1., h=0., num_phi=9,
                     ksc=ksc, knc=knc, ksol=ksol, profile=profile, profile_label='On-axis Bs/(B rho) [1/m]',
                     field_check=check_field)


if __name__ == '__main__':
    run_case(make_case(), __doc__)
