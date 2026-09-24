"""Curved dipole: pkin_const convergence and canonical symplecticity.

Run from the repository root:
    python -m examples.bfieldexpansion.convergence_pkin_const_curved_dipole

Use h=0.3 1/m, L=1 m and knc[0](s)=h+0.10*64*u^3*(1-u)^3 [1/m], u=s/L.
The constant part matches the design curvature; the sixth-order bump adds
a smooth field variation. All skew coefficients and the on-axis ksol vanish.
The curved Maxwell recurrence generates the transverse and fringe fields.

Cubic Hermite segments match knc[0] and knc[0]' at their boundaries. Use even
ny=10 so the highest odd scalar-potential term is included in the vector
potential. Check ny+2 and integration refinement before interpreting a floor.
The reference Lorentz equations include the curved metric and h*p_s term.
Both pkin_const modes have trajectory and canonical symplecticity plots.
--no-plot and --save-plot /tmp/curved_dipole.png support batch runs.
"""

from numpy.polynomial import Polynomial

from ._convergence_cases import FieldCase, run_case


def make_case():
    curvature = 0.3
    profile = Polynomial([curvature, 0, 0, 6.4, -19.2, 19.2, -6.4])
    return FieldCase('Curved dipole', length=1., h=curvature, ny=10,
                     ksc=[Polynomial([0.])], knc=[profile], ksol=Polynomial([0.]),
                     profile=profile, profile_label='On-axis By/(B rho) [1/m]')


if __name__ == '__main__':
    run_case(make_case(), __doc__)
