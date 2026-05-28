from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt


HERE = Path(__file__).parent
JSON_FILE = HERE / 'temp_fcc_ee_lcc_non_local_boris_solenoid.json'
SAMPLES_PER_SLICE = 10


def sample_splineboris_field(line, prefix):
    tt = line.get_table()

    s_out = []
    bx_out = []
    by_out = []
    bs_out = []

    for nn in line.element_names:
        if not nn.startswith(prefix):
            continue
        if tt['element_type', nn] != 'SplineBoris':
            continue

        ee = line[nn]
        s_local = np.linspace(0, ee.length, SAMPLES_PER_SLICE + 1)
        bx, by, bs = ee.get_field(
            np.zeros_like(s_local),
            np.zeros_like(s_local),
            s_local,
        )

        s_out.append(tt['s', nn] + s_local)
        bx_out.append(bx)
        by_out.append(by)
        bs_out.append(bs)

    if not s_out:
        raise ValueError(f'No SplineBoris elements found with prefix {prefix!r}')

    s_out = np.concatenate(s_out)
    bx_out = np.concatenate(bx_out)
    by_out = np.concatenate(by_out)
    bs_out = np.concatenate(bs_out)

    order = np.argsort(s_out)
    return s_out[order], bx_out[order], by_out[order], bs_out[order]


env = xt.load(JSON_FILE)
line = env.fccee_p_ring

s_sol, bx_sol, by_sol, bs_sol = sample_splineboris_field(line, 'sol_slice_')
s_comp, bx_comp, by_comp, bs_comp = sample_splineboris_field(
    line, 'comp_sol_slice_')

plt.close('all')
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8))

components = [
    ('Bx [T]', bx_sol, bx_comp),
    ('By [T]', by_sol, by_comp),
    ('Bs [T]', bs_sol, bs_comp),
]

for ax, (ylabel, sol_values, comp_values) in zip(axes, components):
    ax.plot(s_sol, sol_values, '-', ms=1.5, label='solenoid')
    ax.plot(s_comp, comp_values, '-', ms=1.5, label='antisolenoid')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

axes[0].legend(loc='best')
axes[-1].set_xlabel('s [m]')
fig.suptitle('SplineBoris field sampled at 10 points per slice')
fig.tight_layout()
fig.savefig(HERE / '004c_non_local_boris_field.png', dpi=200)

plt.show()
