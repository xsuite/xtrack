import numpy as np
import xtrack as xt

from linear_fringe_solenoid import LinearFringeSolenoid

p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                energy0=50e6,
                x=[-5e-3,],
                px=0,
                y=2e-3,
                delta=0)



sf = LinearFringeSolenoid(B0=1.0, s1=0.0, s2=3, s3=7, s4=10)

n_steps_vect = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

sympl_error = []
det_R = []
S = xt.linear_normal_form.S
x = []
y = []
for n_steps in n_steps_vect:
    print('n_steps=', n_steps)
    integrator = xt.BorisSpatialIntegrator(fieldmap_callable=sf.get_field,
                                            s_start=0,
                                            s_end=sf.s4,
                                            n_steps=n_steps)
    line = xt.Line(elements=[integrator])
    R_obj = line.get_R_matrix(particle_on_co=p0.copy(),
                                  include_collective=True
                                  )

    RR = R_obj["R_matrix"]

    se = np.linalg.norm(RR.T @ S @ RR - S, ord=2)
    sympl_error.append(se)

    p_boris = p0.copy()
    integrator.track(p_boris)
    x.append(p_boris.x[0])
    y.append(p_boris.y[0])
    det_R.append(np.linalg.det(RR))
err = np.sqrt((np.array(x) - x[-1])**2 + (np.array(y) - y[-1])**2)

import matplotlib.pyplot as plt
plt.close('all')
fig1 = plt.figure(1, figsize=(6.4, 4.8))
plt.loglog(n_steps_vect[:-1], err[:-1], '-o', label='Simulation')
plt.loglog(n_steps_vect[:-1], err[0]*n_steps_vect[0]**2 * 1/np.array(n_steps_vect[:-1])**(2), '--', label=r'~ 1/$N_\text{steps}^2$')
plt.xlabel('Number of steps')
plt.ylabel('Error on exit position (m)')
plt.xlim(n_steps_vect[0]/2, n_steps_vect[-1])
plt.legend()

fig2 = plt.figure(2, figsize=(6.4, 4.8))
plt.loglog(n_steps_vect[:-1], np.abs(sympl_error[:-1]), '-o', label='Simulation')
plt.loglog(n_steps_vect[:-1], np.abs(sympl_error[0])*n_steps_vect[0]**2 * 1/np.array(n_steps_vect[:-1])**(2), '--', label=r'~ 1/$N_\text{steps}^2$')
plt.xlabel('Number of steps')
plt.ylabel('Symplectic deviation')
plt.xlim(n_steps_vect[0]/2, n_steps_vect[-1])
plt.legend()

fig3 = plt.figure(3, figsize=(6.4, 4.8))
plt.loglog(n_steps_vect, np.abs(np.abs(det_R)- 1), '-o', label='Simulation')
plt.xlabel('Number of steps')
plt.ylabel('Deviation from unity on the determinant of R')
plt.legend()

plt.show()