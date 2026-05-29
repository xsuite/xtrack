from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt


HERE = Path(__file__).parent
INPUT_LATTICE_JSON = HERE / 'temp_fcc_ee_lcc_non_local_boris_solenoid.json'

IP_NAMES = ['ipa', 'ipd', 'ipg', 'ipj']
SAMPLES_PER_ELEMENT = 10

# The 005d output can be saved with the solenoid knobs off. For this inspection
# plot, switch them on before extracting B_s from the SplineBoris elements.
SET_SOLENOID_KNOBS_FOR_PLOT = True
ON_SOL_VALUE_FOR_PLOT = 1.0
ON_COMP_SOL_VALUE_FOR_PLOT = 1.0


# Load the FCC ring with the solenoid insertions prepared by 005d.
env = xt.load(INPUT_LATTICE_JSON)
line = env.fccee_p_ring

if SET_SOLENOID_KNOBS_FOR_PLOT:
    for ip_name in IP_NAMES:
        env[f'on_sol_{ip_name}'] = ON_SOL_VALUE_FOR_PLOT
        env[f'on_comp_sol_{ip_name}'] = ON_COMP_SOL_VALUE_FOR_PLOT


# Find all SplineBoris elements in ring order and sample their longitudinal
# field in their local coordinate. The sampled local coordinate is shifted by
# the table s_start to obtain the global ring s coordinate.
tt = line.get_table()
mask_splineboris = tt.element_type == 'SplineBoris'
idx_splineboris = np.where(mask_splineboris)[0]

s_bs_plot_chunks = []
bs_plot_chunks = []
sampled_segments = []
bs_by_element = {}
bs_integral_by_element = {}
bs_integral_main_solenoids = 0.0
bs_integral_compensation_solenoids = 0.0

for ii in idx_splineboris:
    name = tt.name[ii]
    env_name = tt.env_name[ii]
    element = env.get(env_name)

    s_local = np.linspace(0.0, element.length, SAMPLES_PER_ELEMENT + 1)
    _, _, bs_local = element.get_field(
        np.zeros_like(s_local),
        np.zeros_like(s_local),
        s_local,
    )

    s_ring = tt.s_start[ii] + s_local
    bs_integral = np.trapezoid(bs_local, s_ring)

    bs_by_element[name] = {
        's_ring': s_ring,
        'bs': bs_local,
        'env_name': env_name,
        'scale_b': element.scale_b,
        'integral': bs_integral,
    }
    bs_integral_by_element[name] = bs_integral

    if name.startswith('sol_slice_'):
        bs_integral_main_solenoids += bs_integral
    elif name.startswith('comp_sol_slice_'):
        bs_integral_compensation_solenoids += bs_integral

    sampled_segments.append((s_ring, bs_local))

bs_integral_all_solenoids = (
    bs_integral_main_solenoids + bs_integral_compensation_solenoids)

current_s = []
current_bs = []
previous_s_end = None
for s_ring, bs_local in sampled_segments:
    starts_new_region = (
        previous_s_end is None
        or abs(s_ring[0] - previous_s_end) > 1e-9
    )

    if starts_new_region and current_s:
        s_region = np.concatenate(current_s)
        bs_region = np.concatenate(current_bs)
        s_bs_plot_chunks.append(np.r_[s_region[0], s_region, s_region[-1]])
        bs_plot_chunks.append(np.r_[0.0, bs_region, 0.0])
        current_s = []
        current_bs = []

    if current_s and abs(s_ring[0] - previous_s_end) <= 1e-9:
        current_s.append(s_ring[1:])
        current_bs.append(bs_local[1:])
    else:
        current_s.append(s_ring)
        current_bs.append(bs_local)

    previous_s_end = s_ring[-1]

if current_s:
    s_region = np.concatenate(current_s)
    bs_region = np.concatenate(current_bs)
    s_bs_plot_chunks.append(np.r_[s_region[0], s_region, s_region[-1]])
    bs_plot_chunks.append(np.r_[0.0, bs_region, 0.0])

s_bs = np.concatenate(s_bs_plot_chunks)
bs = np.concatenate(bs_plot_chunks)


# Plot B_s from the SplineBoris elements against the global ring coordinate.
plt.close('all')

fig_bs_ring, ax_bs_ring = plt.subplots(figsize=(12, 5))
ax_bs_ring.plot(s_bs, bs, '-', color='C0', label='SplineBoris $B_s$')
ax_bs_ring.axhline(0.0, color='0.4', linewidth=0.8)
ax_bs_ring.set_xlabel('ring s [m]')
ax_bs_ring.set_ylabel('$B_s$ [T]')
ax_bs_ring.set_title(
    'Longitudinal field from SplineBoris elements in the FCC ring\n'
    r'$\int B_s ds$ main='
    f'{bs_integral_main_solenoids:.6g} T m, compensation='
    f'{bs_integral_compensation_solenoids:.6g} T m, total='
    f'{bs_integral_all_solenoids:.6g} T m')
ax_bs_ring.grid(True, alpha=0.3)
ax_bs_ring.legend(loc='best')
fig_bs_ring.tight_layout()

print(f'Loaded {INPUT_LATTICE_JSON}')
print(f'Found {len(idx_splineboris)} SplineBoris elements')
print(f'Extracted {len(s_bs)} B_s plot samples')
print(
    f'Integral main solenoids: '
    f'{bs_integral_main_solenoids:.12g} T m')
print(
    f'Integral compensation solenoids: '
    f'{bs_integral_compensation_solenoids:.12g} T m')
print(f'Integral all solenoids: {bs_integral_all_solenoids:.12g} T m')

plt.show()
