import xtrack as xt

import numpy as np

magnet = xt.Magnet(k0=0.002, h=0.002, k1=0.02, length=2)

m_exact = magnet.copy()
m_exact.model = 'bend-kick-bend'
m_exact.integrator='yoshida4'
m_exact.num_multipole_kicks = 1000

m_expanded = magnet.copy()
m_expanded.model = 'mat-kick-mat'
m_expanded.integrator='yoshida4'
m_expanded.num_multipole_kicks = 1000

p0 = xt.Particles(x=1e-3, y=2e-3)
px_list = [1e-7, 2e-7, 5e-7, 1e-6, 2e-6, 5e-6,
           1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
px_in = np.array(px_list)

x_out_exact = []
px_out_exact = []
y_out_exact = []
py_out_exact = []
zeta_out_exact = []

x_out_expanded = []
px_out_expanded = []
y_out_expanded = []
py_out_expanded = []
zeta_out_expanded = []

for dd in px_list:
    p_exact = p0.copy()
    p_exact.px = dd
    m_exact.track(p_exact)

    p_expanded = p0.copy()
    p_expanded.px = dd
    m_expanded.track(p_expanded)

    x_out_exact.append(p_exact.x[0])
    px_out_exact.append(p_exact.px[0])
    y_out_exact.append(p_exact.y[0])
    py_out_exact.append(p_exact.py[0])
    zeta_out_exact.append(p_exact.zeta[0])

    x_out_expanded.append(p_expanded.x[0])
    px_out_expanded.append(p_expanded.px[0])
    y_out_expanded.append(p_expanded.y[0])
    py_out_expanded.append(p_expanded.py[0])
    zeta_out_expanded.append(p_expanded.zeta[0])


# Cast to numpy arrays
import numpy as np
x_out_exact = np.array(x_out_exact)
px_out_exact = np.array(px_out_exact)
y_out_exact = np.array(y_out_exact)
py_out_exact = np.array(py_out_exact)
zeta_out_exact = np.array(zeta_out_exact)

x_out_expanded = np.array(x_out_expanded)
px_out_expanded = np.array(px_out_expanded)
y_out_expanded = np.array(y_out_expanded)
py_out_expanded = np.array(py_out_expanded)
zeta_out_expanded = np.array(zeta_out_expanded)


import matplotlib.pyplot as plt

plt.close('all')
fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)

# Plot for px
axs[0, 0].loglog(px_list, np.abs(px_out_exact - px_out_expanded), '.-')
axs[0, 0].set_ylabel('Error in px out')
axs[0, 0].set_title('Difference between exact and expanded models')

# Plot for x
axs[0, 1].loglog(px_list, np.abs(x_out_exact - x_out_expanded), '.-')
axs[0, 1].set_ylabel('Error in x out')
axs[0, 1].set_title('')

# Plot for py
axs[0, 2].loglog(px_list, np.abs(py_out_exact - py_out_expanded), '.-')
axs[0, 2].set_ylabel('Error in py out')
axs[0, 2].set_title('')

# Plot for y
axs[1, 0].loglog(px_list, np.abs(y_out_exact - y_out_expanded), '.-')
axs[1, 0].set_ylabel('Error in y out')
axs[1, 0].set_xlabel('px in')

# Plot for zeta
axs[1, 1].loglog(px_list, np.abs(zeta_out_exact - zeta_out_expanded), '.-')
axs[1, 1].set_ylabel('Error in zeta out')
axs[1, 1].set_xlabel('px in')

# Hide the last unused subplot
axs[1, 2].axis('off')


plt.show()





