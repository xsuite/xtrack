from math import factorial
import numpy as np
import xtrack as xt

from tilted_solenoid import TiltedSolenoid


def finite_difference_coefficients(offsets, derivative_order):
    offsets = np.asarray(offsets, dtype=float)
    matrix = np.vstack([offsets**ii for ii in range(len(offsets))])
    rhs = np.zeros(len(offsets))
    rhs[derivative_order] = factorial(derivative_order)
    return np.linalg.solve(matrix, rhs)


def compute_pure_field_derivatives(
        field, s, direction, step, component, max_order=4, min_order=1):
    offsets = np.arange(-4, 5)
    zero = np.zeros_like(s)
    component_index = {'x': 0, 'y': 1, 'z': 2}[component]
    field_at_offsets = []

    for offset in offsets:
        if direction == 'x':
            x = zero + offset * step
            y = zero
        elif direction == 'y':
            x = zero
            y = zero + offset * step
        else:
            raise ValueError("direction must be 'x' or 'y'")

        field_at_offsets.append(field.get_field(x, y, s)[component_index])

    field_at_offsets = np.array(field_at_offsets)

    derivatives = {}
    for order in range(min_order, max_order + 1):
        coefficients = finite_difference_coefficients(offsets, order)
        derivatives[order] = (
            np.tensordot(coefficients, field_at_offsets, axes=(0, 0))
            / step**order
        )

    return derivatives


def compute_pure_by_derivatives(field, s, direction, step, max_order=4):
    return compute_pure_field_derivatives(
        field=field, s=s, direction=direction, step=step,
        component='y', max_order=max_order, min_order=1)


theta = -0.015
sf = TiltedSolenoid(L=1.23*2, a=0.13, B0=2., theta=theta)

dx = 1e-4
dy = 1e-4

s = np.linspace(-2.5, 2.5, 201)

bx, by, bz = sf.get_field(0*s, 0*s, s)

by_derivatives_x = compute_pure_field_derivatives(
    field=sf, s=s, direction='x', step=dx, component='y',
    max_order=4, min_order=0)
by_derivatives_y = compute_pure_by_derivatives(
    field=sf, s=s, direction='y', step=dy, max_order=4)
bx_derivatives_x = compute_pure_field_derivatives(
    field=sf, s=s, direction='x', step=dx, component='x',
    max_order=4, min_order=0)

dby_dx = by_derivatives_x[1]
d2by_dx2 = by_derivatives_x[2]
d3by_dx3 = by_derivatives_x[3]
d4by_dx4 = by_derivatives_x[4]

dby_dy = by_derivatives_y[1]
d2by_dy2 = by_derivatives_y[2]
d3by_dy3 = by_derivatives_y[3]
d4by_dy4 = by_derivatives_y[4]

env = xt.Environment()
env.new_particle('ref_part', 'positron', energy0=45.6e9)
rigidity0 = env['ref_part'].rigidity0[0]

Bn = by_derivatives_x
An = bx_derivatives_x

Kn = {nn: Bn[nn] / rigidity0 for nn in range(5)}
Ksn = {nn: An[nn] / rigidity0 for nn in range(5)}

bx_plus = sf.get_field(dx + 0*s, 0*s, s)[0]
bx_minus = sf.get_field(-dx + 0*s, 0*s, s)[0]
k1s_central = (bx_plus - bx_minus) / (2 * dx) / rigidity0

np.testing.assert_allclose(k1s_central, Ksn[1], rtol=0, atol=1e-8)
