import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe

import xobjects as xo
import xtrack as xt
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField


p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                energy0=45.6e9/1000,
                x=[-1e-3,],
                px=-1e-3,
                y=1e-3,
                delta=0)

p = p0.copy()

sf = SolenoidField(L=4, a=0.3, B0=1.5, z0=20)

n_steps_vect = [1000, 5000, 10000, 20000, 50000, 100000, 200000]
sympl_error = []
S = xt.linear_normal_form.S
x = []
for n_steps in n_steps_vect:
    print('n_steps=', n_steps)
    integrator = xt.BorisSpatialIntegrator(fieldmap_callable=sf.get_field,
                                            s_start=0,
                                            s_end=30,
                                            n_steps=n_steps)
    line = xt.Line(elements=[integrator])
    line.build_tracker()
    R_obj = line.compute_one_turn_matrix_finite_differences(particle_on_co=p0.copy(),
                                                        include_collective=True
                                                    )

    RR = R_obj["R_matrix"]

    se = np.linalg.norm(RR.T @ S @ RR - S, ord=2) - 1
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