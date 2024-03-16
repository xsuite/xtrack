import numpy as np
import scipy

# https://par.nsf.gov/servlets/purl/10220882

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