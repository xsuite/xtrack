from math import factorial

import numpy as np

from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField


class TiltedSolenoid:

    def __init__(self, L, a, B0, theta):

        self._sf = SolenoidField(L=L, a=a, B0=B0, z0=0)
        self.theta = theta
        self._break = False

    def get_field(self, x, y, z):

        if self._break:
            breakpoint()

        theta = self.theta
        stheta = np.sin(theta)
        ctheta = np.cos(theta)
        x_sol = x * ctheta + z * stheta
        z_sol = -x * stheta + z * ctheta
        y_sol = y

        bx_sol, by_sol, bz_sol = self._sf.get_field(x_sol, y_sol, z_sol)

        bx = bx_sol * np.cos(theta) - bz_sol * np.sin(theta)
        bz = bx_sol * np.sin(theta) + bz_sol * np.cos(theta)
        by = by_sol

        return bx, by, bz

    @staticmethod
    def finite_difference_coefficients(offsets, derivative_order):
        offsets = np.asarray(offsets, dtype=float)
        matrix = np.vstack([offsets**ii for ii in range(len(offsets))])
        rhs = np.zeros(len(offsets))
        rhs[derivative_order] = factorial(derivative_order)
        return np.linalg.solve(matrix, rhs)

    def compute_pure_field_derivatives(
            self, s, direction, step, component, max_order=4, min_order=1):
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

            field_at_offsets.append(self.get_field(x, y, s)[component_index])

        field_at_offsets = np.array(field_at_offsets)

        derivatives = {}
        for order in range(min_order, max_order + 1):
            coefficients = self.finite_difference_coefficients(offsets, order)
            derivatives[order] = (
                np.tensordot(coefficients, field_at_offsets, axes=(0, 0))
                / step**order
            )

        return derivatives

    def compute_pure_by_derivatives(self, s, direction, step, max_order=4):
        return self.compute_pure_field_derivatives(
            s=s, direction=direction, step=step,
            component='y', max_order=max_order, min_order=1)
