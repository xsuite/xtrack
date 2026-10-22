"""Single thick octupole: TPSA, knobs, evaluation, derivatives, Hessians."""
import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
from xtrack.tpsa import ParticlesTpsa, COORDS
from madng_tpsa import Descriptor


# Octupole parameters and initial conditions
P0C = 7e12
LENGTH = 0.3
K3 = 3000.0  # m^-4, strongly nonlinear at millimetre amplitudes
ORDER = 4  # x^3 * dk3 requires total order >= 4


# Some phase space points
origin = np.zeros(6)
initial = np.array([1.2e-3, 3e-5, 0.8e-3, -2e-5, 0., 0.])


# A. The unperturbed thick-element map
# ====================================

oct0 = xt.Octupole(length=LENGTH, k3=K3)
m0 = ParticlesTpsa(truncation_order=ORDER, p0c=P0C, mass0=xt.PROTON_MASS_EV)
oct0.track(m0)

print('No-parameter octupole map:', m0.show(max_terms=2))
print()

print('Map at initial point:', m0.evaluate(initial))
print('Jacobian at initial point:\n', m0.evaluate_jacobian(initial))
print('Hessian shape:', m0.hessian().shape)  # (6, 6, 6)

# Direct tracking through an equivalent physical thick octupole.
particle = xt.Particles(p0c=P0C, mass0=xt.PROTON_MASS_EV,
                        x=initial[0], px=initial[1], y=initial[2],
                        py=initial[3], zeta=initial[4], pzeta=initial[5])
oct0.track(particle)
direct = np.array([float(getattr(particle, c)[0]) for c in COORDS])
print('Direct Xtrack:', direct)
print('Map - direct:', m0.evaluate(initial) - direct)


# B. A TPSA parameter in the octupole strength: k3 = K3 + dk3
# ===========================================================

# Zero parameter input means nominal K3.
d = Descriptor(variables=COORDS, order=ORDER,
               parameters=['dk3'], param_order=1)
octk = xt.Octupole(length=LENGTH, k3=K3)
octk.k3 = d.param('dk3', K3)
mk = ParticlesTpsa(truncation_order=ORDER, descriptor=d, p0c=P0C, mass0=xt.PROTON_MASS_EV)
octk.track(mk)
print('Parameter derivative d(px_out)/d(dk3) at input:',
      mk.partial_derivative('dk3').evaluate(initial, parameters=[0.])[1])
print('Parameter Jacobian shape:', mk.jacobian(wrt='parameters').shape)

print('Mixed derivative d^2 px_out / dx_in d(dk3):',
      mk.partial_derivative(('x', 'dk3')).evaluate(initial, parameters=[0.])[1])
print('Full mixed Hessian shape:',
      mk.hessian(wrt='all').shape)

# Sweep the strength WITHOUT re-tracking a TPSA map; compare native Xtrack
# scalar tracking at a few sample strengths. Small |dk3| is important.
dks = np.linspace(-0.12 * K3, +0.12 * K3, 61)
px_map = np.array([mk.evaluate(initial, parameters=[dk])[1] for dk in dks])
px_native = []
for dk in dks:
    part = xt.Particles(p0c=P0C, mass0=xt.PROTON_MASS_EV,
                        x=initial[0], px=initial[1], y=initial[2],
                        py=initial[3], zeta=initial[4], pzeta=initial[5])
    xt.Octupole(length=LENGTH, k3=K3 + dk).track(part)
    px_native.append(float(part.px[0]))
px_native = np.array(px_native)

# First-order Taylor prediction in dk3 at dk3=0.
px_base = mk.evaluate(initial, parameters=[0.])[1]
slope = mk.partial_derivative('dk3').evaluate(initial, parameters=[0.])[1]
plt.figure(figsize=(8, 5))
plt.plot((K3 + dks), px_native, 'k-', label='Native Xtrack')
plt.plot((K3 + dks), px_map, '--', label=f'TPSA order {ORDER}')
plt.plot((K3 + dks), px_base + slope * dks, ':', label='Linear response in strength')
plt.xlabel(r'Octupole $k_3$ [$\mathrm{m}^{-4}$]')
plt.ylabel(r'Exit $p_x$')
plt.title('Thick octupole: tracking versus parametric Taylor map')
plt.legend()
plt.tight_layout()
plt.savefig('octupole_parameter_scan.png', dpi=170)

# The derivative is a polynomial in the input phase-space coordinates.
xs = np.linspace(-2e-3, 2e-3, 91)
sens = np.array([
    mk.partial_derivative('dk3').evaluate([x, 0., 0., 0., 0., 0.], parameters=[0.])[1]
    for x in xs
])
plt.figure(figsize=(7, 4.5))
plt.plot(xs * 1e3, sens)
plt.xlabel('Entrance x [mm]')
plt.ylabel(r'$\partial p_{x,\mathrm{out}}/\partial k_3$')
plt.title('Third-order octupole response')
plt.tight_layout()
plt.savefig('octupole_derivative_vs_amplitude.png', dpi=170)
print('Figures written: octupole_parameter_scan.png, octupole_derivative_vs_amplitude.png')



# C. Symplecticity checks
# =======================

# C.1 At the origin
# -----------------

S2 = np.array([[0, 1], [-1, 0]])
S = np.kron(np.eye(3), S2)

J0 = m0.evaluate_jacobian()
residual = J0.T @ S @ J0 - S
print("Maximum symplecticity error:", np.max(np.abs(residual)))
print("Symplectic:", np.allclose(residual, 0, atol=1e-10))

Jk = mk.evaluate_jacobian()
residual = Jk.T @ S @ Jk - S
print("Maximum symplecticity error:", np.max(np.abs(residual)))
print("Symplectic:", np.allclose(residual, 0, atol=1e-10))


# C.2 At an arbitrary point
# -------------------------

z = np.array([1e-3, -2.1e-5, -7.3e-4, -1.9e-5, 0, 0])  # Displacement from the expansion point
J0_tpsa = m0.evaluate_jacobian(z)
residual = J0_tpsa.T @ S @ J0_tpsa - S
print("Maximum symplecticity error at arbitrary point:", np.max(np.abs(residual)))
print("Symplectic at arbitrary point:", np.allclose(residual, 0, atol=1e-10))

Jk_tpsa = mk.evaluate_jacobian(z, parameters=[0.01])
residual = Jk_tpsa.T @ S @ Jk_tpsa - S
print("Maximum symplecticity error at arbitrary point:", np.max(np.abs(residual)))
print("Symplectic at arbitrary point:", np.allclose(residual, 0, atol=1e-10))


# Symbolically
J0 = m0.jacobian()
residual = J0.T @ S @ J0 - S
for i in range(6):
    for j in range(6):
        print(i, j, residual[i, j].monomial_coeffs())

Jk = mk.jacobian()
residual = Jk.T @ S @ Jk - S
for i in range(6):
    for j in range(6):
        print(i, j, residual[i, j].monomial_coeffs())

