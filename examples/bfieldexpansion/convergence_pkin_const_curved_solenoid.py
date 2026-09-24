"""Curved solenoid: pkin_const convergence and canonical symplecticity.

Run from the repository root:
    python -m examples.bfieldexpansion.convergence_pkin_const_curved_solenoid

Use the round-solenoid mid-plane seed coefficients with Bs/(B rho) of
degree six, in a constant-curvature frame h=0.3 1/m over L=1 m. The curved
Maxwell recurrence defines the off-axis field; this is not a rigidly rotated
straight solenoid or a specified winding geometry. At y=0 its longitudinal
field contains the geometric factor 1/(1+h*x). There is no guide dipole:
the reference coordinate curve is not assumed to be a particle trajectory.

Fit all nonzero seed profiles independently with cubic Hermite pieces.
Use odd ny=9 so the highest even scalar-potential term is included in Ax.
The shared runner checks ny+2, reference integration, and RK4 refinement,
and plots trajectory and native-Jacobian symplectic errors versus y and N.
The curved multipole evaluation can show a floating-point floor; consult
the printed reference and finite-difference refinement checks before
interpreting small defects as physical effects.
--no-plot and --save-plot /tmp/curved_solenoid.png support batch runs.
"""

from numpy.polynomial import Polynomial

from ._convergence_cases import FieldCase, round_solenoid_seeds, run_case


def make_case():
    profile = Polynomial([0.30, 0, 0, 6.4, -19.2, 19.2, -6.4])
    ksc, knc, ksol = round_solenoid_seeds(profile)
    return FieldCase('Curved solenoid', length=1., h=0.3, ny=9,
                     ksc=ksc, knc=knc, ksol=ksol, profile=profile, profile_label='On-axis Bs/(B rho) [1/m]')


if __name__ == '__main__':
    run_case(make_case(), __doc__)
