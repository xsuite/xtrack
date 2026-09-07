"""mat-kick-mat-exact vs plain mat-kick-mat: convergence in num_slices.
"""
import matplotlib.pyplot as plt

import xtrack as xt

N_KICKS_YOSHIDA = 7  # track_magnet.h: n_kicks_yoshida for the yoshida4 integrator

magnet = xt.Magnet(k0=0.05, k1=0.37, length=1.2, angle=0.0)

p0 = xt.Particles(p0c=7e12, x=2e-2, px=-3e-3, y=-1.5e-2, py=2e-3, delta=0.1)

m_ref = magnet.copy()
m_ref.model = 'drift-kick-drift-exact'
m_ref.integrator = 'yoshida4'
m_ref.num_multipole_kicks = 15 * 400
p_ref = p0.copy()
m_ref.track(p_ref)

m_plain = magnet.copy()
m_plain.model = 'mat-kick-mat'
m_plain.integrator = 'yoshida4'

m_exact = magnet.copy()
m_exact.model = 'mat-kick-mat-exact'
m_exact.integrator = 'yoshida4'

num_slices = [1, 2, 3, 5, 10, 20, 30, 50, 80]

coords = ('x', 'px', 'y', 'py', 'zeta')
err_plain = {cc: [] for cc in coords}
err_exact = {cc: [] for cc in coords}

for nslice in num_slices:
    m_plain.num_multipole_kicks = nslice * N_KICKS_YOSHIDA
    m_exact.num_multipole_kicks = nslice * N_KICKS_YOSHIDA

    p_plain = p0.copy()
    m_plain.track(p_plain)

    p_exact = p0.copy()
    m_exact.track(p_exact)

    for cc in coords:
        err_plain[cc].append(abs(getattr(p_plain, cc)[0] - getattr(p_ref, cc)[0]))
        err_exact[cc].append(abs(getattr(p_exact, cc)[0] - getattr(p_ref, cc)[0]))

for cc in coords:
    print(f'{cc}: exact error at nslice={num_slices[-1]} = {err_exact[cc][-1]:.3e}')
    assert err_exact[cc][-1] < 1e-14, f'{cc} did not converge below 1e-14'

plt.close('all')
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

for ax, cc in zip(axes.flat, coords):
    ax.loglog(num_slices, err_plain[cc], 'o-', label='mat-kick-mat')
    ax.loglog(num_slices, err_exact[cc], 's-', label='mat-kick-mat-exact')
    ax.set_xlabel('num_slices')
    ax.set_ylabel(f'error in {cc}')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
axes.flat[-1].set_axis_off()

fig.suptitle('Convergence to drift-kick-drift-exact reference (h=0)')
fig.tight_layout()
plt.show()
