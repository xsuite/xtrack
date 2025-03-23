import xtrack as xt
import numpy as np

magnet = xt.Magnet(k0=0.02, h=0., k1=0.01, length=2.,
                   k2=0.005, k3=0.03,
                   k1s=0.01, k2s=0.005, k3s=0.05,
                   knl=[0.003, 0.001, 0.01, 0.02, 4., 6e2, 7e6],
                   ksl=[-0.005, 0.002, -0.02, 0.03, -2, 700., 4e6])
magnet.integrator = 'yoshida4'
magnet.num_multipole_kicks = 100

p0 = xt.Particles(x=1e-2, y=2e-2, py=1e-3, delta=3e-2)

model_to_test = 'drift-kick-drift-exact'

m_ref = magnet.copy()
m_ref.model = 'bend-kick-bend'
p_ref = p0.copy()
m_ref.track(p_ref)

m_uniform = magnet.copy()
# m_uniform.knl[-6] = 0
m_uniform.model = model_to_test
m_uniform.integrator='uniform'

m_teapot = magnet.copy()
m_teapot.model = model_to_test
m_teapot.integrator='teapot'

m_yoshida = magnet.copy()
m_yoshida.model = model_to_test
m_yoshida.integrator='yoshida4'

num_kicks = [1, 2, 5, 10, 20, 50, 100, 200, 500,
             1000, 2000, 5000, 10000, 20000, 50000, 100000]
log_data = {'num_kicks': [], 'x_teapot': [], 'px_teapot': [], 'y_teapot': [], 'py_teapot': [], 'zeta_teapot': [], 'delta_teapot': [],
            'x_uniform': [], 'px_uniform': [], 'y_uniform': [], 'py_uniform': [], 'zeta_uniform': [], 'delta_uniform': [],
            'x_yoshida': [], 'px_yoshida': [], 'y_yoshida': [], 'py_yoshida': [], 'zeta_yoshida': [], 'delta_yoshida': []}

for nn in num_kicks:
    m_teapot.num_multipole_kicks = nn
    m_uniform.num_multipole_kicks = nn
    m_yoshida.num_multipole_kicks = nn

    p_teapot = p0.copy()
    m_teapot.track(p_teapot)

    p_uniform = p0.copy()
    m_uniform.track(p_uniform)

    p_yoshida = p0.copy()
    m_yoshida.track(p_yoshida)

    # Log data for teapot integrator
    log_data['num_kicks'].append(nn)
    log_data['x_teapot'].append(p_teapot.x[0])
    log_data['px_teapot'].append(p_teapot.px[0])
    log_data['y_teapot'].append(p_teapot.y[0])
    log_data['py_teapot'].append(p_teapot.py[0])
    log_data['zeta_teapot'].append(p_teapot.zeta[0])
    log_data['delta_teapot'].append(p_teapot.delta[0])

    # Log data for uniform integrator
    log_data['x_uniform'].append(p_uniform.x[0])
    log_data['px_uniform'].append(p_uniform.px[0])
    log_data['y_uniform'].append(p_uniform.y[0])
    log_data['py_uniform'].append(p_uniform.py[0])
    log_data['zeta_uniform'].append(p_uniform.zeta[0])
    log_data['delta_uniform'].append(p_uniform.delta[0])

    # Log data for yoshida integrator
    log_data['x_yoshida'].append(p_yoshida.x[0])
    log_data['px_yoshida'].append(p_yoshida.px[0])
    log_data['y_yoshida'].append(p_yoshida.y[0])
    log_data['py_yoshida'].append(p_yoshida.py[0])
    log_data['zeta_yoshida'].append(p_yoshida.zeta[0])
    log_data['delta_yoshida'].append(p_yoshida.delta[0])

for key in log_data.keys():
    log_data[key] = np.array(log_data[key])

import matplotlib.pyplot as plt


def myplot(x, y, *args, **kwargs):
    if np.abs(y).max() <=0:
        plt.seimilogx(x, y, *args, **kwargs)
    else:
        plt.loglog(x, y, *args, **kwargs)

plt.close('all')
plt.figure(1, figsize=(12, 8))
ax1 = plt.subplot(2, 3, 1)
myplot(log_data['num_kicks'], np.abs(log_data['x_uniform'] - p_ref.x[0]), '.-', label='uniform')
myplot(log_data['num_kicks'], np.abs(log_data['x_teapot'] - p_ref.x[0]), '.-', label='teapot')
myplot(log_data['num_kicks'], np.abs(log_data['x_yoshida'] - p_ref.x[0]), '.-', label='yoshida')
plt.xlabel('num_kicks')
plt.ylabel('Error in x')
plt.legend()

ax2 = plt.subplot(2, 3, 2, sharex=ax1)
myplot(log_data['num_kicks'], np.abs(log_data['y_uniform'] - p_ref.y[0]), '.-', label='uniform')
myplot(log_data['num_kicks'], np.abs(log_data['y_teapot'] - p_ref.y[0]), '.-', label='teapot')
myplot(log_data['num_kicks'], np.abs(log_data['y_yoshida'] - p_ref.y[0]), '.-', label='yoshida')
plt.xlabel('num_kicks')
plt.ylabel('Error in y')

ax3 = plt.subplot(2, 3, 3, sharex=ax1)
myplot(log_data['num_kicks'], np.abs(log_data['zeta_uniform'] - p_ref.zeta[0]), '.-', label='uniform')
myplot(log_data['num_kicks'], np.abs(log_data['zeta_teapot'] - p_ref.zeta[0]), '.-', label='teapot')
myplot(log_data['num_kicks'], np.abs(log_data['zeta_yoshida'] - p_ref.zeta[0]), '.-', label='yoshida')
plt.xlabel('num_kicks')
plt.ylabel('Error in zeta')

ax4 = plt.subplot(2,3,4, sharex=ax1)
myplot(log_data['num_kicks'], np.abs(log_data['px_uniform'] - p_ref.px[0]), '.-', label='uniform')
myplot(log_data['num_kicks'], np.abs(log_data['px_teapot'] - p_ref.px[0]), '.-', label='teapot')
myplot(log_data['num_kicks'], np.abs(log_data['px_yoshida'] - p_ref.px[0]), '.-', label='yoshida')
plt.xlabel('num_kicks')
plt.ylabel('Error in px')

ax5 = plt.subplot(2,3, 5, sharex=ax1)
myplot(log_data['num_kicks'], np.abs(log_data['py_uniform'] - p_ref.py[0]), '.-', label='uniform')
myplot(log_data['num_kicks'], np.abs(log_data['py_teapot'] - p_ref.py[0]), '.-', label='teapot')
myplot(log_data['num_kicks'], np.abs(log_data['py_yoshida'] - p_ref.py[0]), '.-', label='yoshida')
plt.xlabel('num_kicks')
plt.ylabel('Error in py')


plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.show()
