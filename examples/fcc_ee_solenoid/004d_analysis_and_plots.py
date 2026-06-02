from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt


HERE = Path(__file__).parent
INPUT_LATTICE_JSON = HERE / 'fccee_z_lcc_splineboris_solenoids_coupling_corrected.json'

IP_NAMES = ['ipa', 'ipd', 'ipg', 'ipj']
IP_PLOT = 'ipg'
SAMPLES_PER_SPLINEBORIS_ELEMENT = 4


################################
# Load and prepare the lattice #
################################

env = xt.load(INPUT_LATTICE_JSON)

# Work on a copy, so extra cuts used only for plotting do not alter the
# environment loaded from JSON.
line = env.fccee_p_ring.copy(shallow=True)
line.particle_ref.anomalous_magnetic_moment = 0.00115965218128

line.cycle(f'end_ds_start_straight_{IP_NAMES[0]}')
table_before_cuts = line.get_table()
for ip_name in IP_NAMES:
    s_cut_right = np.arange(
        table_before_cuts['s', ip_name] + 2.4,
        table_before_cuts['s', ip_name] + 11.0,
        0.2,
    )
    line.cut_at_s(s_cut_right)

    s_cut_left = np.arange(
        table_before_cuts['s', ip_name] - 11.0,
        table_before_cuts['s', ip_name] - 2.4,
        0.2,
    )
    line.cut_at_s(s_cut_left)


#####################################
# Twiss with solenoids/corrections off #
#####################################

for ip_name in IP_NAMES:
    line[f'on_sol_{ip_name}'] = 0
    line[f'on_sol_corr_{ip_name}'] = 0

tw_off = line.twiss6d(strengths=True)


####################################
# Twiss with solenoids/corrections on #
####################################

for ip_name in IP_NAMES:
    line[f'on_sol_{ip_name}'] = 1
    line[f'on_sol_corr_{ip_name}'] = 1

tw4d = line.twiss4d(
    strengths=True,
    polarization_analysis=True,
    radiation_integrals=True,
)
tw = line.twiss6d(strengths=True, polarization_analysis=True)
two = line.twiss(betx=tw_off.betx[0], bety=tw_off.bety[0])


################################################
# Extract B_s directly from SplineBoris elements #
################################################

table = line.get_table()
idx_splineboris = np.where(table.element_type == 'SplineBoris')[0]

sampled_segments = []
for ii in idx_splineboris:
    element = env.get(table.env_name[ii])
    s_local = np.linspace(
        0.0, element.length, SAMPLES_PER_SPLINEBORIS_ELEMENT + 1)
    _, _, bs_local = element.get_field(
        np.zeros_like(s_local),
        np.zeros_like(s_local),
        s_local,
    )
    bs_local = np.nan_to_num(bs_local, nan=0.0)
    sampled_segments.append((table.s_start[ii] + s_local, bs_local))

s_plot_chunks = []
bs_plot_chunks = []
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
        s_plot_chunks.append(np.r_[s_region[0], s_region, s_region[-1]])
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
    s_plot_chunks.append(np.r_[s_region[0], s_region, s_region[-1]])
    bs_plot_chunks.append(np.r_[0.0, bs_region, 0.0])

s_bs = np.concatenate(s_plot_chunks)
bs = np.concatenate(bs_plot_chunks)


######################
# Radiation analysis #
######################

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()
tw_rad = line.twiss6d(strengths=True, radiation_analysis=True)

energy_eV = (
    tw_rad.ptau * line.particle_ref.p0c[0]
    + line.particle_ref.energy0[0]
)
dE_eV = -np.diff(energy_eV, append=energy_eV[-1])
length = tw.length
mask_len = length > 0
dE_ds_eV_per_m = np.zeros_like(dE_eV)
dE_ds_eV_per_m[mask_len] = dE_eV[mask_len] / length[mask_len]
dE_ds_eV_per_m[dE_ds_eV_per_m < 0] = 0.0

tw_off.zero_at(IP_PLOT)
tw.zero_at(IP_PLOT)
s_bs = s_bs - tw['s', IP_PLOT]


#########
# Plots #
#########

plt.close('all')

fig1 = plt.figure(figsize=(6.4, 4.8 * 1.8))
ax1 = fig1.add_subplot(5, 1, 1)
tw_off.plot(ax=ax1)

ax2 = fig1.add_subplot(5, 1, 2, sharex=ax1)
ax2.plot(s_bs, bs)
ax2.set_ylabel(r'$B_s$ [T]')
ax2.grid(True)

ax3 = fig1.add_subplot(5, 1, 3, sharex=ax1)
ax3.plot(tw.s, tw.y * 1e3)
ax3.set_ylabel('y [mm]')
ax3.set_ylim(-0.2, 0.2)
ax3.grid(True)

ax4 = fig1.add_subplot(5, 1, 4, sharex=ax1)
ax4.plot(tw.s, tw.dy * 1e3)
ax4.set_ylabel(r'$D_y$ [mm]')
ax4.set_ylim(-0.2, 0.2)
ax4.grid(True)

ax5 = fig1.add_subplot(5, 1, 5, sharex=ax1)
ax5.plot(tw.s, tw.betx2, label=r'$\beta_{x2}$')
ax5.plot(tw.s, tw.bety1, label=r'$\beta_{y1}$')
ax5.set_ylabel(r'$\beta_{x2,y1}$')
ax5.legend(loc='best')
ax5.grid(True)

ax1.set_xlabel('')
ax5.set_xlabel('s [m]')
fig1.subplots_adjust(hspace=0.25, top=0.95, bottom=0.06, left=0.14)
ax5.set_xlim(-20, 20)

fig2 = plt.figure(figsize=(6.4, 4.8 * 1.8))
ax1 = fig2.add_subplot(5, 1, 1)
tw_off.plot(ax=ax1)

ax2 = fig2.add_subplot(5, 1, 2, sharex=ax1)
ax2.plot(s_bs, bs)
ax2.set_ylabel(r'$B_s$ [T]')
ax2.grid(True)

ax3 = fig2.add_subplot(5, 1, 3, sharex=ax1)
ax3.plot(tw.s, dE_ds_eV_per_m / 1e6)
ax3.set_ylabel(r'dE/ds [MeV/m]')
ax3.grid(True)

ax4 = fig2.add_subplot(5, 1, 4, sharex=ax1)
ax4.plot(tw.s, tw.spin_y)
ax4.set_ylabel(r'spin y')
ax4.grid(True)

ax5 = fig2.add_subplot(5, 1, 5, sharex=ax1)
ax5.plot(tw.s, tw.spin_x, label='spin x')
ax5.plot(tw.s, tw.spin_z, label='spin z')
ax5.set_ylabel(r'spin x, z')
ax5.legend(loc='best')
ax5.grid(True)

ax1.set_xlabel('')
ax5.set_xlabel('s [m]')
fig2.subplots_adjust(hspace=0.25, top=0.95, bottom=0.06, left=0.14)
ax5.set_xlim(-20, 20)

print(f'Loaded {INPUT_LATTICE_JSON}')
print(f'tw4d qx = {tw4d.qx:.12g}, qy = {tw4d.qy:.12g}')
print(f'tw6d qx = {tw.qx:.12g}, qy = {tw.qy:.12g}, qs = {tw.qs:.12g}')

plt.show()
