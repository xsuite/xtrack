"""mat-kick-mat-exact vs mat-kick-mat vs drift-kick-drift-exact:
max error (over all coordinates) vs tracking time, labelled by num_slices.
mat-kick-mat is a single uniform slice; the others go through uniform,
yoshida-4, yoshida-6 (1 slice each), then yoshida-6 with more slices.
"""
import time

import matplotlib.pyplot as plt
import numpy as np

import xtrack as xt

# kicks per slice for each integrator
n_kicks = {'uniform': 1, 'yoshida-4': 3, 'yoshida-6': 7}
models = ['mat-kick-mat', 'mat-kick-mat-exact', 'drift-kick-drift-exact']
mkme_slices = [2, 3, 5, 10, 20]
dkd_slice = mkme_slices + [50]
# (integrator, num_slices, label) for each model
scan = [('uniform', 1, 'u'), ('yoshida-4', 1, 'y4'), ('yoshida-6', 1, 'y6')]
configs = {
    'mat-kick-mat': [('uniform', 1, '1')],
    'mat-kick-mat-exact': scan + [('yoshida-6', n, f'y6 x{n}') for n in mkme_slices],
    'drift-kick-drift-exact': scan + [('yoshida-6', n, f'y6 x{n}') for n in dkd_slice],
}
coords = ('x', 'px', 'y', 'py', 'zeta')
n_repeat = 5

magnet = xt.Quadrupole(k1=0.37, length=1.2)

rng = np.random.default_rng(1)
n_part = 10_000
p0 = xt.Particles(
    p0c=7e12,
    x=rng.uniform(-1, 1, n_part) * 1e-2,
    px=rng.uniform(-1, 1, n_part) * 1e-3,
    y=rng.uniform(-1, 1, n_part) * 1e-2,
    py=rng.uniform(-1, 1, n_part) * 1e-3,
    delta=rng.uniform(-1, 1, n_part) * 0.05,
)

# Reference: heavily sliced drift-kick-drift-exact
m_ref = magnet.copy()
m_ref.model = 'drift-kick-drift-exact'
m_ref.integrator = 'yoshida-6'
m_ref.num_multipole_kicks = 200 * n_kicks['yoshida-6']
p_ref = p0.copy()
m_ref.track(p_ref)

max_err = {mod: [] for mod in models}  # max abs error over all coords
time_ms = {mod: [] for mod in models}

print(
    f'{"model":24s} {"integrator":>10s} {"nslice":>6s} {"max error":>10s} {"time [ms]":>10s}'
)
for model in models:
    for integrator, nslice, _ in configs[model]:
        m = magnet.copy()
        m.model = model
        m.integrator = integrator
        m.num_multipole_kicks = nslice * n_kicks[integrator]

        p = p0.copy()
        m.track(p)  # warm-up (kernel load)

        times = []
        for _ in range(n_repeat):
            p = p0.copy()
            t0 = time.perf_counter()
            m.track(p)
            times.append(time.perf_counter() - t0)
        time_ms[model].append(min(times) * 1e3)

        max_err[model].append(
            max(np.abs(getattr(p, cc) - getattr(p_ref, cc)).max() for cc in coords)
        )
        print(
            f'{model:24s} {integrator:>10s} {nslice:6d} {max_err[model][-1]:10.2e} '
            f'{time_ms[model][-1]:10.1f}'
        )

style = {
    'mat-kick-mat': {'marker': 'o', 'color': 'C0'},
    'mat-kick-mat-exact': {'marker': 's', 'color': 'C1'},
    'drift-kick-drift-exact': {'marker': '^', 'color': 'C2'},
}

# Everything relative to the single-slice mat-kick-mat
t_ref = time_ms['mat-kick-mat'][0]
e_ref = max_err['mat-kick-mat'][0]

plt.close('all')
fig, ax = plt.subplots(figsize=(8, 6))
for mod in models:
    rel_t = np.array(time_ms[mod]) / t_ref
    rel_e = np.array(max_err[mod]) / e_ref
    ax.loglog(rel_t, rel_e, '-', label=mod, **style[mod])
    for (_, _, n), t, e in zip(configs[mod], rel_t, rel_e):
        ax.annotate(
            str(n),
            (t, e),
            textcoords='offset points',
            xytext=(0, 7),
            ha='center',
            fontsize=8,
            color=style[mod]['color'],
        )
ax.set_xlabel('track time / mat-kick-mat track time')
ax.set_ylabel('max error / mat-kick-mat max error')
ax.set_title(
    'Accuracy vs cost relative to mat-kick-mat; labels = integrator / num_slices'
)
ax.grid(True, which='both', alpha=0.3)
ax.legend()
fig.tight_layout()

plt.show()
