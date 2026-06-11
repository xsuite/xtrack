from math import factorial

import numpy as np
import scipy
# Hampton et al., Closed-form expressions for the magnetic fields of rectangular
# and circular finite-length solenoids and current loops
# https://pubs.aip.org/aip/adv/article/10/6/065320/997382/

def ellipp(n, m):
    """
    Elliptic integral of the third kind
    """

from scipy.special import elliprf, elliprj

def ellipp(n, m):
    assert (m <= 1).all()
    y = 1 - m
    rf = elliprf(0, y, 1)
    rj = elliprj(0, y, 1, 1 - n)
    return rf + rj * n / 3

class SolenoidField:

    def __init__(self, L, a, B0, z0):
        self.L = L
        self.a = a
        self.B0 = B0
        self.z0 = z0

    def get_field(self, x, y, z):

        a = self.a
        B0 = self.B0
        z0 = self.z0
        L = self.L

        r = np.sqrt(x**2 + y**2)

        u = 4 * a * r / (a + r)**2
        zeta_plus = z - z0 + L / 2
        zeta_minus = z -z0 - L / 2
        m_plus = 4 * a * r / ((a + r)**2 + zeta_plus**2)
        m_minus = 4 * a * r / ((a + r)**2 + zeta_minus**2)

        # sqrt_plus = sqrt(m_plus / (a * r))
        sqrt_plus = np.sqrt(4 / ((a + r)**2 + zeta_plus**2))

        # sqrt_minus = sqrt(m_minus / (a * r))
        sqrt_minus = np.sqrt(4 / ((a + r)**2 + zeta_minus**2))

        KK = scipy.special.ellipk
        EE = scipy.special.ellipe
        PP = ellipp

        Bz_plus = B0 * zeta_plus / (4 * np.pi) * sqrt_plus * (
                        (KK(m_plus) + (a - r)/(a + r) * PP(u, m_plus)))

        Bz_minus = B0 * zeta_minus / (4 * np.pi) * sqrt_minus * (
                        (KK(m_minus) + (a - r)/(a + r) * PP(u, m_minus)))

        Bz = Bz_plus - Bz_minus

        sqrt_r_plus = 0 * sqrt_plus
        sqrt_r_minus = 0 * sqrt_minus

        mask_r_nonzero = r > 1e-11
        sqrt_r_plus[mask_r_nonzero] = np.sqrt(a / (r[mask_r_nonzero] * m_plus[mask_r_nonzero]))
        sqrt_r_minus[mask_r_nonzero] = np.sqrt(a / (r[mask_r_nonzero] * m_minus[mask_r_nonzero]))

        Br_plus = B0 / np.pi * sqrt_r_plus * (EE(m_plus) - (1 - m_plus / 2) * KK(m_plus))
        Br_minus = B0 / np.pi * sqrt_r_minus * (EE(m_minus) - (1 - m_minus / 2) * KK(m_minus))

        Br = Br_plus - Br_minus

        Bx = 0 * z
        By = 0 * z

        Bx[mask_r_nonzero] = Br[mask_r_nonzero] * x[mask_r_nonzero] / r[mask_r_nonzero]
        By[mask_r_nonzero] = Br[mask_r_nonzero] * y[mask_r_nonzero] / r[mask_r_nonzero]

        return Bx, By, Bz

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
            coefficients = SolenoidField.finite_difference_coefficients(
                offsets, order)
            derivatives[order] = (
                np.tensordot(coefficients, field_at_offsets, axes=(0, 0))
                / step**order
            )

        return derivatives

    def compute_pure_by_derivatives(self, s, direction, step, max_order=4):
        return self.compute_pure_field_derivatives(
            s=s, direction=direction, step=step,
            component='y', max_order=max_order, min_order=1)

class Multifield:

    def __init__(self, fields):
        self.fields = fields

    def get_field(self, x, y, z):
        Bx = 0 * x
        By = 0 * x
        Bz = 0 * x
        for field in self.fields:
            Bx_, By_, Bz_ = field.get_field(x, y, z)
            Bx += Bx_
            By += By_
            Bz += Bz_
        return Bx, By, Bz
