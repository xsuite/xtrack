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

    def get_vector_potential(self, x, y, z, r_eps=1e-12):
        """
        Analytic azimuthal-gauge vector potential for the thin-wall finite solenoid.
        Uses closed forms with complete elliptic integrals (K, E) only; no quadrature.

        Returns (Ax, Ay, Az) in Cartesian coordinates (Az=0 in this gauge).
        Matches the normalization used in your get_field (B0).
        """
        a   = self.a
        B0  = self.B0
        z0  = self.z0
        L   = self.L

        KK = scipy.special.ellipk
        EE = scipy.special.ellipe

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
        out_shape = np.broadcast(x, y, z).shape

        xf = np.broadcast_to(x, out_shape).ravel()
        yf = np.broadcast_to(y, out_shape).ravel()
        zf = np.broadcast_to(z, out_shape).ravel()
        N  = xf.size

        Ax = np.zeros(N, dtype=float)
        Ay = np.zeros(N, dtype=float)
        Az = np.zeros(N, dtype=float)  # identically zero in this gauge

        def Aphi_loop(r, zeta):
            # Guard axis to avoid division by zero in sqrt(a/(r*m))
            r_safe = np.maximum(r, r_eps)
            m = 4.0 * a * r_safe / ((a + r_safe)**2 + zeta**2)  # k^2
            # When r=0 -> m=0, the bracket tends to 0 and Aphi ~ r*Bz(0)/2
            pref = - (B0 / np.pi) * np.sqrt(a / (r_safe * np.maximum(m, 1e-300)))
            bracket = EE(m) - (1.0 - 0.5*m) * KK(m)
            Aphi = pref * bracket
            # Enforce regularity on axis by the small-r series (optional but nice):
            on_axis = (r < r_eps)
            if np.any(on_axis):
                # Aphi(r,z) ~ r * Bz(0,z) / 2
                # Compute Bz(0,z) from your analytic expression (r->0 limit):
                # At r=0, m->0 and Π(u,m) term vanishes; the known on-axis Bz is:
                denom_plus  = a*a + zeta**2
                # This is the loop on-axis; for the full solenoid we use end-difference,
                # so here we just keep the loop piece. We'll not use this path for general points.
                Aphi[on_axis] = 0.0  # caller uses end-difference; axis handled after difference
            return Aphi

        for i in range(N):
            xi, yi, zi = xf[i], yf[i], zf[i]
            ri = np.hypot(xi, yi)

            zeta_plus  = zi - z0 + L/2.0
            zeta_minus = zi - z0 - L/2.0

            Aphi_p = Aphi_loop(ri, zeta_plus)
            Aphi_m = Aphi_loop(ri, zeta_minus)
            Aphi = Aphi_p - Aphi_m

            # Axis regularization (Aphi -> 0 as r -> 0)
            if ri < r_eps:
                Aphi = 0.0

            # Convert to Cartesian: e_phi = (-sinφ, cosφ, 0)
            if ri >= r_eps:
                Ax[i] = -Aphi * (yi / ri)
                Ay[i] =  Aphi * (xi / ri)
            else:
                Ax[i] = 0.0
                Ay[i] = 0.0

        return Ax.reshape(out_shape), Ay.reshape(out_shape), Az.reshape(out_shape)


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