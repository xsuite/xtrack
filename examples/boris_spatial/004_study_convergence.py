import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe

import xobjects as xo
import xtrack as xt
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField

p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                energy0=45.6e9/1000,
                x=[-1e-3,],
                px=0,
                y=1e-3,
                delta=0)

p = p0.copy()


class LinearFringeSolenoid:

    def __init__(self, B0, s1, s2, s3, s4):
        self.B0 = B0
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.s4 = s4

    def get_field(self, x, y, s):

        mask_entry_taper = (s >= self.s1) & (s < self.s2)
        mask_constant = (s >= self.s2) & (s < self.s3)
        mask_exit_taper = (s >= self.s3) & (s < self.s4)

        Bx = np.zeros_like(x)
        By = np.zeros_like(y)
        Bz = np.zeros_like(s)

        Bz[mask_entry_taper] = self.B0 * (s[mask_entry_taper] - self.s1) / (self.s2 - self.s1)
        Bz[mask_constant] = self.B0
        Bz[mask_exit_taper] = self.B0 * (self.s4 - s[mask_exit_taper]) / (self.s4 - self.s3)

        # Bx = -x dBz/dz /2
        # By = -y dBz/dz /2

        dBz_ds_entry = self.B0 / (self.s2 - self.s1)
        dBz_ds_exit = -self.B0 / (self.s4 - self.s3)

        Bx[mask_entry_taper] = -x[mask_entry_taper] * dBz_ds_entry / 2
        Bx[mask_exit_taper] = -x[mask_exit_taper] * dBz_ds_exit / 2
        By[mask_entry_taper] = -y[mask_entry_taper] * dBz_ds_entry / 2
        By[mask_exit_taper] = -y[mask_exit_taper] * dBz_ds_exit / 2

        return Bx, By, Bz

sf = LinearFringeSolenoid(B0=1.0, s1=0.0, s2=3, s3=7, s4=10)

n_steps_vect = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
# n_steps_vect = [5000, 10000, 20000, 50000, 100000] # 200000]
sympl_error = []
S = xt.linear_normal_form.S
x = []
for n_steps in n_steps_vect:
    print('n_steps=', n_steps)
    integrator = xt.BorisSpatialIntegrator(fieldmap_callable=sf.get_field,
                                            s_start=0,
                                            s_end=sf.s4,
                                            n_steps=n_steps)
    line = xt.Line(elements=[integrator])
    R_obj = line.compute_one_turn_matrix_finite_differences(particle_on_co=p0.copy(),
                                                        include_collective=True
                                                    )

    RR = R_obj["R_matrix"]

    se = np.linalg.norm(RR.T @ S @ RR - S, ord=2)
    sympl_error.append(se)

    p_boris = p.copy()
    integrator.track(p_boris)
    x.append(p_boris.x[0])

err = np.abs(np.array(x)-x[-1])

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.loglog(n_steps_vect, err, '-o')
plt.loglog(n_steps_vect, err[0]*n_steps_vect[0]**2 * 1/np.array(n_steps_vect)**(2), '--', label='~1/Nsteps^2')
plt.xlabel('Number of steps')
plt.ylabel('Error in x (m)')
plt.legend()

plt.figure(2)
plt.loglog(n_steps_vect, np.abs(sympl_error), '-o')
plt.loglog(n_steps_vect, np.abs(sympl_error[0])*n_steps_vect[0]**2 * 1/np.array(n_steps_vect)**(2), '--', label='~1/Nsteps^2')
plt.xlabel('Number of steps')
plt.ylabel('Symplecticity error')
plt.legend()

plt.show()