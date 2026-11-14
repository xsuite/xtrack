"""Actual LHC one-turn map from an Xtrack example model."""
import json
import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt
from xtrack.tpsa import ParticlesTpsa, COORDS
from madng_tpsa import Descriptor


ORDER = 6
p0c = 7e12

env = xt.load('../../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
line = env.lhcb1
line.particle_ref = xt.Particles(p0c=p0c, mass0=xt.PROTON_MASS_EV)
line.build_tracker(compile=False)

# The origin is an *input* origin; this need not be the actual closed orbit.
# For a nonlinear normal form recenter on a computed closed orbit instead.
z = np.array([2e-5, 1e-7, -2e-5, -1e-7, 0., 0.])


# A. Create the one-turn map and compare to native tracking
# =========================================================

m = ParticlesTpsa(truncation_order=ORDER, p0c=p0c, mass0=xt.PROTON_MASS_EV)
print(f'Tracking a {ORDER}-order identity map through {len(line.element_names)} elements')
line.track(m)
print('One-turn constant term:', m.const_part)
print('One-turn Jacobian:\n', m.evaluate_jacobian())
print('TPSA prediction:', m.evaluate(z))

part = xt.Particles(p0c=p0c, mass0=xt.PROTON_MASS_EV, x=z[0], px=z[1], y=z[2], py=z[3],
                    zeta=z[4], pzeta=z[5])
line.track(part)
tracked = np.array([float(getattr(part, name)[0]) for name in COORDS])
print('Actual one-turn trajectory:', tracked)
print('Map minus particle:', m.evaluate(z) - tracked)


# B. Check symplecticity of the one-turn map
# ==========================================

S = np.zeros((6, 6))
for i in range(0, 6, 2):
    S[i, i+1] = 1
    S[i+1, i] = -1
for point in (np.zeros(6), z):
    J = m.evaluate_jacobian(point)
    print(f'Max symplectic defect at {point} = {np.abs(J.T @ S @ J - S).max()}')


# C. Parametric one-turn map for two selected quadrupoles
# =======================================================

# The illustrative two-parameter LHC map below varies ONE focusing and ONE
# defocusing THICK QUADRUPOLE. These represent two independently labelled
# magnet strengths, NOT a full optics-matched QF/QD family knob.
quad_pos = []
quad_neg = []
for name in line.element_names:
    element = line[name]
    if isinstance(element, xt.Quadrupole):
        k1 = float(element.k1)
        if k1 > 0:
            quad_pos.append((name, k1))
        elif k1 < 0:
            quad_neg.append((name, k1))

name_f, base_f = quad_pos[0]
name_d, base_d = quad_neg[0]
print('Choosing quadrupoles:', name_f, base_f, name_d, base_d)
d = Descriptor(variables=COORDS, parameters=['relative_QF', 'relative_QD'],
               order=max(ORDER, 2), param_order=1)

# Two small dimensionless relative strength changes about zero.
line[name_f].k1 = base_f * (1 + d.param('relative_QF'))
line[name_d].k1 = base_d * (1 + d.param('relative_QD'))
mp = ParticlesTpsa(truncation_order=d.order, descriptor=d, p0c=p0c, mass0=xt.PROTON_MASS_EV)

line.track(mp)
print('Parametric one-turn Jacobian:', mp.evaluate_jacobian(z, parameters=[0, 0]))
print('Response d z_out/d(relative_QF, relative_QD):\n',
        mp.evaluate_jacobian(z, parameters=[0, 0], wrt='parameters'))

strengths = np.linspace(-0.005, 0.005, 31)
values = np.asarray([mp.evaluate(z, parameters=[value, 0])[0] for value in strengths])
slope = mp.partial_derivative('relative_QF').evaluate(z, parameters=[0, 0])[0]
nominal = mp.evaluate(z, parameters=[0, 0])[0]

plt.figure(figsize=(7, 4.5))
plt.plot(strengths * 100, values * 1e6, label='TPSA evaluation')
plt.plot(strengths * 100, (nominal + strengths * slope) * 1e6,
            '--', label='Linear sensitivity')
plt.xlabel('Relative change in one QF strength [%]')
plt.ylabel('One-turn x [micrometres]')
plt.title('LHC one-turn-map sensitivity to a quadrupole strength')
plt.legend()
plt.tight_layout()
plt.savefig('lhc_quadrupole_response.png', dpi=170)
plt.show()
