import numpy as np

from scipy.constants import c as clight
from scipy.constants import e as qe

class BorisSpatialIntegrator:

    isthick = True

    def __init__(self, fieldmap_callable, s_start, s_end, n_steps, verbose=False):
        self.fieldmap_callable = fieldmap_callable
        self.s_start = s_start
        self.s_end = s_end
        self.ds = (s_end - s_start) / n_steps
        self.n_steps = n_steps
        self.verbose = verbose
        self.length = s_end - s_start

    def track(self, p):

        mask_alive = p.state > 0

        x_log = []
        y_log = []
        z_log = []
        s_in = p.s[mask_alive].copy()
        p.s[mask_alive] = self.s_start

        for ii in range(self.n_steps):

            if self.verbose:
                print(f's_in = {s_in[0]:.3f} s_in_map = {p.s[0]:.3f}', end='\r', flush=True)

            x = p.x[mask_alive].copy()
            y = p.y[mask_alive].copy()
            z = p.s[mask_alive].copy()
            px = p.px[mask_alive].copy()
            py = p.py[mask_alive].copy()
            delta = p.delta[mask_alive].copy()
            energy = p.energy[mask_alive].copy()
            p0c = p.p0c[mask_alive].copy()
            beta0 = p.beta0[mask_alive].copy()
            charge = p.charge[mask_alive].copy()
            mass = p.mass[mask_alive].copy()

            charge_coulomb = charge * qe
            mass_kg = mass * qe / clight**2
            gamma = energy / mass

            P0 = p0c * qe / clight # in kg m/s
            P = P0 * (1 + delta)

            Px = px * P0
            Py = py * P0

            w = np.zeros((Px.shape[0], 3), dtype=Px.dtype)
            w[:, 0] = Px
            w[:, 1] = Py
            w[:, 2] = P

            x_new, y_new, z_new, w_new, dt = step_spatial_boris_B(x, y, z, w,
                charge_coulomb, self.ds,
                field_fn=self.fieldmap_callable, m_kg=mass_kg, gamma=gamma)
            p.x[mask_alive] = x_new
            p.y[mask_alive] = y_new
            p.s[mask_alive] = z_new
            p.px[mask_alive] = w_new[:, 0] / P0
            p.py[mask_alive] = w_new[:, 1] / P0
            p.zeta[mask_alive] += (self.ds - dt * clight * beta0)

            x_log.append(p.x.copy())
            y_log.append(p.y.copy())
            z_log.append(p.s.copy())
        p.s[mask_alive] = s_in + self.length
        self.x_log = np.array(x_log)
        self.y_log = np.array(y_log)
        self.z_log = np.array(z_log)


import numpy as np
c = 299_792_458.0  # m/s

def step_spatial_boris_B(x, y, z, w, q, dz, field_fn, m_kg, gamma):
    """
    Spatial Boris step for magnetic fields only (E = 0),
    using constant total momentum magnitude P = w[:,2].

    Parameters
    ----------
    x, y, z : (N,) arrays
        Particle positions [m].
    w : (N, 3) array
        Momentum-like state: (px, py, P), where
        P = |p| = sqrt(px^2 + py^2 + pz^2) is constant.
    q : float
        Particle charge [C].
    dz : float
        Spatial step along z [m].
    field_fn : callable(x, y, z) -> (Bx, By, Bz)
        Function returning magnetic-field components [T].

    Returns
    -------
    x1, y1, z1 : (N,) arrays
        Updated positions [m].
    w1 : (N, 3) array
        Updated momentum state (px, py, P).
    """

    dt = 0.

    # --- Unpack and ensure arrays
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    w = np.asarray(w)
    px, py, P = w[:, 0], w[:, 1], w[:, 2]

    # --- Compute longitudinal momentum from constant magnitude
    pz = np.sqrt(np.maximum(P**2 - px**2 - py**2, 0.0))

    # --- Half drift in (x, y)
    xh = x + (px / pz) * (dz * 0.5)
    yh = y + (py / pz) * (dz * 0.5)
    zh = z + dz * 0.5
    dt += (dz * 0.5) / pz * gamma * m_kg #  s

    # --- Evaluate magnetic field at mid-step
    Bx, By, Bz = field_fn(xh, yh, zh)

    # ============================================
    # (1) FIRST HALF-KICK from (Bx, By)
    # ============================================
    pxm = px - 0.5 * q * dz * By
    pym = py + 0.5 * q * dz * Bx

    # --- Recompute pz to maintain |p| = P
    pz = np.sqrt(np.maximum(P**2 - pxm**2 - pym**2, 0.0))

    # ============================================
    # (2) ROTATION due to Bz
    # ============================================
    t = 0.5 * q * Bz * dz / pz
    t2 = t * t
    s = 2.0 * t / (1.0 + t2)
    c0 = (1.0 - t2) / (1.0 + t2)

    # pxp = c0 * pxm - s * pym
    # pyp = s * pxm + c0 * pym
    pxp = c0 * pxm + s * pym
    pyp = -s * pxm + c0 * pym

    # ============================================
    # (3) SECOND HALF-KICK from (Bx, By)
    # ============================================
    px1 = pxp - 0.5 * q * dz * By
    py1 = pyp + 0.5 * q * dz * Bx

    # --- Recompute pz after full step
    pz1 = np.sqrt(np.maximum(P**2 - px1**2 - py1**2, 0.0))

    # ============================================
    # (4) SECOND HALF-DRIFT (x, y)
    # ============================================
    x1 = xh + (px1 / pz1) * (dz * 0.5)
    y1 = yh + (py1 / pz1) * (dz * 0.5)
    z1 = z + dz
    dt += (dz * 0.5) / pz1 * gamma * m_kg #  s

    # --- Pack updated (px, py, P)
    w1 = np.stack([px1, py1, P], axis=1)

    return x1, y1, z1, w1, dt
