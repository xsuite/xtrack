"""Compact periodic demonstration: 2 parametric strengths, iterate, compose,
phase portraits, symplecticity and a 10,000-turn composition timing comparison.
"""
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
from xtrack.tpsa import ParticlesTpsa, COORDS
from madng_tpsa import Descriptor, TpsaMap


P0C = 7e12
ORDER = 3  # mixed coordinate-parameter response fits into order 3


line = xt.Line(elements=[
    xt.Drift(length=2.0), xt.Quadrupole(length=0.5, k1=0.13),
    xt.Drift(length=2.0), xt.Quadrupole(length=0.5, k1=-0.13),
], element_names=['d1', 'qf', 'd2', 'qd'])
line.particle_ref = xt.Particles(p0c=P0C, mass0=xt.PROTON_MASS_EV)
line.build_tracker(compile=False)


# A. Map without knob parameters
# ==============================

# Both single-particle and map tracking use
# the *same* line model. This is a cell map, not a realistic LHC ring.
m0 = ParticlesTpsa(truncation_order=ORDER, p0c=P0C, mass0=xt.PROTON_MASS_EV)
line.track(m0)
print('Cell map Jacobian at origin:\n', m0.evaluate_jacobian())

# Evaluation
point = np.array([2e-3, 0, 1e-3, 0, 0, 0])
particle = xt.Particles(
    p0c=P0C, mass0=xt.PROTON_MASS_EV,
    **dict(zip(COORDS, point, strict=True)),
)
line.track(particle)
direct = np.array([float(getattr(particle, c)[0]) for c in COORDS])
print('Direct Xtrack:', direct)
print('Map - direct:', m0.evaluate(point) - direct)


# B. Parameterised map with focusing and defocusing quadrupoles
# =============================================================

d = Descriptor(variables=COORDS, parameters=['dkf', 'dkd'], order=ORDER, param_order=1)
line['qf'].k1 = d.param('dkf', 0.13)
line['qd'].k1 = d.param('dkd', -0.13)
mp = ParticlesTpsa(truncation_order=ORDER, descriptor=d, p0c=P0C, mass0=xt.PROTON_MASS_EV)
line.track(mp)
print('Parameter gradients at an off-axis point:\n',
      mp.evaluate_jacobian(np.array([2e-3, 0, 1e-3, 0, 0, 0]), parameters=[0, 0], wrt='parameters'))


# C. Repeated evaluation versus self-composition
# ==============================================

# A cell's longitudinal pzeta is frozen here at 0; only transverse motion matters.
initial = np.array([1.2e-3, 0.0, 0.7e-3, 0.0, 0., 0.])
steps = 2500
eval_path = [initial]
start = perf_counter()
for turn in range(steps):
    eval_path.append(mp.evaluate(eval_path[-1], parameters=[0., 0.]))
print('Evaluation time for TPSA map evaluation:', perf_counter() - start)
eval_path = np.asarray(eval_path)

# Independent tracking reference: rebuild the identical FODO cell with
# *scalar* strengths, because native scalar tracking must not be mixed with
# TPSA-valued element fields in the same line.
scalar_line = xt.Line(elements=[
    xt.Drift(length=2.0), xt.Quadrupole(length=0.5, k1=0.13),
    xt.Drift(length=2.0), xt.Quadrupole(length=0.5, k1=-0.13),
], element_names=['d1', 'qf', 'd2', 'qd'])
scalar_line.particle_ref = xt.Particles(p0c=P0C, mass0=xt.PROTON_MASS_EV)
scalar_line.build_tracker(compile=False)
scalar_particle = xt.Particles(
    p0c=P0C, mass0=xt.PROTON_MASS_EV,
    **dict(zip(COORDS, initial, strict=True)),
)
track_path = [initial]
start = perf_counter()
for turn in range(steps):
    scalar_line.track(scalar_particle)
    track_path.append(np.array([
        float(getattr(scalar_particle, name)[0]) for name in COORDS
    ]))
print('Tracking time for native tracking:', perf_counter() - start)
track_path = np.asarray(track_path)
print('Maximum absolute difference: evaluated TPSA map versus native tracking:',
      np.max(np.abs(eval_path - track_path)))

# Self-Composition of the map for multiple turns
temp = [mp.copy()]
start = perf_counter()
for turn in range(steps - 1):
    temp.append(mp << temp[-1])
print('Stepwise self-composition time for TPSA map:', perf_counter() - start)
start = perf_counter()
full_map = mp ^ steps  # Much quicker; scales as O(log n) instead of O(n) for stepwise composition
print('Direct self-composition time for TPSA map:', perf_counter() - start)
compose_path = np.array([initial] + [mm.evaluate(initial, parameters=[0, 0]) for mm in temp])
print('Maximum absolute difference: composed TPSA map versus native tracking:',
      np.max(np.abs(compose_path - track_path)))
print('Maximum absolute difference: composed TPSA map versus evaluated TPSA map:',
      np.max(np.abs(compose_path - eval_path)))

# Compare errors
fig, ax = plt.subplots(1, 3, figsize=(14, 8))
ax[0].plot(np.abs(compose_path - eval_path) / np.abs(eval_path))
ax[0].set_title('Composed vs Evaluated')
ax[1].plot(np.abs(compose_path - track_path) / np.abs(track_path))
ax[1].set_title('Composed vs Native')
ax[2].plot(np.abs(eval_path - track_path) / np.abs(track_path))
ax[2].set_title('Evaluated vs Native')
ax[0].set_yscale('log')
ax[1].set_yscale('log')
ax[2].set_yscale('log')
plt.tight_layout()
plt.savefig('fodo_errors.png', dpi=170)
plt.show()


# D. Parameter-dependent phase-space trajectories
# ===============================================

fig, ax = plt.subplots(figsize=(7, 5))
for value in np.linspace(0., 0.025, 6):
    z = initial.copy()
    trajectory = [z]
    for turn in range(80):
        z = mp.evaluate(z, parameters=[value, -value])
        trajectory.append(z)
    trajectory = np.asarray(trajectory)
    ax.plot(trajectory[:, 0] * 1e3, trajectory[:, 1] * 1e3,
             '.-', ms=2, lw=.8, label=f'kf = {value}')
ax.set_xlabel('x [mm]')
ax.set_ylabel('px [mrad]')
ax.set_title('Parameter-dependent phase-space trajectory, FODO cell')
ax.legend()
plt.tight_layout()
plt.savefig('fodo_parameter_phase_space.png', dpi=170)
plt.show()


# E. Test symplecticity of the map
# ================================

# Numerical test of symplecticity at selected points.
S = np.zeros((6, 6))
for i in range(0, 6, 2):
    S[i, i + 1] = 1.
    S[i + 1, i] = -1.

for name, z, pars in [
    ('origin', np.zeros(6), [0., 0.]),
    ('off axis', initial, [0., 0.]),
    ('with parameters', initial, [0.005, -0.005]),
]:
    J = mp.evaluate_jacobian(z, parameters=pars)
    defect = J.T @ S @ J - S
    print(f'Max numerical symplectic defect ({name}): {np.max(np.abs(defect)):.3e}')

# Symbolic matrix of TPSAs. This tests the polynomial identity, not just
# one evaluation point. Ignore coefficients above the mathematically guaranteed
# order (map Taylor order minus one).
JO = mp.jacobian()
symplectic_polynomial = JO.T @ S @ JO - S
low_order_defects = []
for item in symplectic_polynomial.flat:
    for mono, coefficient in item.monomial_coeffs(tol=0).items():
        if sum(mono) <= ORDER - 1:
            low_order_defects.append(abs(coefficient))
print('Symbolic symplectic defect through degree', ORDER - 1,
      ':', max(low_order_defects, default=0.))


# F. Long-term tracking by self-composition
# =========================================

# Composition by squaring: O(log N) native MAD-NG compositions.
# We use a linearised cell for 100,000 turns, because high-order long-term
# polynomial truncation is a separate (important!) source of error.

def manual_power(map_, N):
    power = map_.copy()
    result = TpsaMap(map_.descriptor)
    count = 0
    while N:
        if N & 1:
            result = result << power
            count += 1
        N //= 2
        if N:
            power = power << power
            count += 1
    return result, count


lin = m0.truncate(1)
N = 100_000

start = perf_counter()
lin_power, count = manual_power(lin, N)
elapsed = perf_counter() - start

start = perf_counter()
lin_power_built_in = lin^N
elapsed_built_in = perf_counter() - start

assert lin_power == lin_power_built_in

print(f'100,000-turn linear map built with {count} compositions in {elapsed:.4g} s')
print(f'This took {elapsed_built_in:.4g} s with the built-in function')
print(f'Warning: map coefficient errors can still accumulate despite only ~{count} calls.')
