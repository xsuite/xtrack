"""Exact SymPy verification of the documented dipole-fringe map.

Run with Python and SymPy installed. No Xtrack kernel compilation is needed.

We use canonical coordinates q=(x,y,zeta), p=(px,py,pzeta), where
pzeta=ptau/beta0. Keeping psi as an arbitrary smooth function proves the
identity for the documented tangent coefficient as well as other smooth
choices, provided ALL derivatives in the map come from that same psi.
The cubic coefficient depends only on pzeta (equivalently delta).

This verifies the analytical map, not its floating-point implementation.
The identity holds on the differentiable, invertible domain of the map;
it does not assert that the map exists at its singularities.
"""

import sympy as sp


def assert_zero(matrix):
    reduced = matrix.applyfunc(sp.simplify)
    assert reduced == sp.zeros(*matrix.shape), reduced
    return reduced


px, py, pzeta = sp.symbols('px py pzeta', real=True)
xf, yf, zetaf = sp.symbols('x_f y_f zeta_f', real=True)
momenta = sp.Matrix([px, py, pzeta])
outgoing_positions = sp.Matrix([xf, yf, zetaf])
psi = sp.Function('psi')(px, py, pzeta)
c3 = sp.Function('c3')(pzeta)

# Mixed generating function in CANONICAL longitudinal coordinates.
# In particular psi_pzeta = beta0*psi_ptau, and similarly for c3.
G = (momenta.dot(outgoing_positions)
     - psi*yf**2/2 - c3*yf**4)

# The documented map written implicitly in incoming momenta and outgoing
# positions. Writing it this way avoids unnecessary square-root algebra.
incoming_positions = sp.Matrix([
    xf - sp.diff(psi, px)*yf**2/2,
    yf - sp.diff(psi, py)*yf**2/2,
    zetaf - sp.diff(psi, pzeta)*yf**2/2 - sp.diff(c3, pzeta)*yf**4,
])
outgoing_momenta = sp.Matrix([px, py - psi*yf - 4*c3*yf**3, pzeta])

# First verify every map relation, including the cubic longitudinal term.
assert_zero(incoming_positions - sp.Matrix([sp.diff(G, p) for p in momenta]))
assert_zero(outgoing_momenta
            - sp.Matrix([sp.diff(G, q) for q in outgoing_positions]))
print('All six map relations match the generating function exactly.')

# Differentiate the map itself. If q = q(Q,p), then
# dq = C dQ + A dp, so dQ = C^{-1} dq - C^{-1} A dp.
# Also dP = B dQ + D dp. These yield the full forward Jacobian.
C = incoming_positions.jacobian(outgoing_positions)
A = incoming_positions.jacobian(momenta)
B = outgoing_momenta.jacobian(outgoing_positions)
D = outgoing_momenta.jacobian(momenta)
det_C = sp.factor(C.det())
assert sp.simplify(det_C - (1 - yf*sp.diff(psi, py))) == 0
C_inv = C.inv()
M = (C_inv.row_join(-C_inv*A)
     .col_join((B*C_inv).row_join(D - B*C_inv*A)))

# Block coordinate ordering: (x,y,zeta,px,py,pzeta).
zero, identity = sp.zeros(3), sp.eye(3)
J = zero.row_join(identity).col_join((-identity).row_join(zero))
residual = assert_zero(M.T*J*M - J)
print('M.T * J * M - J is exactly:')
sp.pprint(residual)
print('Implicit-map invertibility requires det(C) != 0, with det(C) =')
sp.pprint(det_C)

# Verify the explicit branch used in tracking. Let a = psi_py, which is
# independent of position. The implicit relation is y = yf - a*yf**2/2.
y, a = sp.symbols('y a', real=True)
yf_explicit = 2*y/(1 + sp.sqrt(1 - 2*a*y))
assert sp.simplify(yf_explicit - a*yf_explicit**2/2 - y) == 0
assert sp.simplify(1 - a*yf_explicit - sp.sqrt(1 - 2*a*y)) == 0
print('The explicit square-root branch satisfies the implicit map exactly.')
print('Its real differentiable domain requires 1 - 2*psi_py*y > 0.')

# Connect the generic functions above to the documented coefficients.
K0, g, F = sp.symbols('K0 g F', real=True, nonzero=True)
beta0 = sp.symbols('beta0', positive=True)
ptau = beta0*pzeta
p = sp.sqrt(1 + 2*pzeta + beta0**2*pzeta**2)  # p = 1 + delta
ps = sp.sqrt(p**2 - px**2 - py**2)
xp, yp = px/ps, py/ps
u = xp/(1 + yp**2)
angle = sp.atan(u) - g*K0*F*(1 + xp**2*(2 + yp**2))/(ps*(1 + u**2))
psi_documented = K0*sp.tan(angle)
c3_documented = K0**2/(36*g*F*p)
T = 1/beta0 + ptau
assert sp.simplify(sp.diff(c3_documented, pzeta)
                   + beta0*c3_documented*T/p**2) == 0
print('Documented cubic derivative: dc3/dpzeta = -beta0*c3*T/p**2.')
print('Thus the longitudinal cubic update is -beta0*c3*T*yf**4/p**2.')
print('The generic identity applies to psi_documented wherever it is smooth:')
print('p > 0, ps > 0, cos(angle) != 0, and the invertible branch above.')
print('The c3=0 option is also covered by the same symbolic identity.')
