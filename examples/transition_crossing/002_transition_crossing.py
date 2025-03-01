import xtrack as xt
import xpart as xp
from xpart.longitudinal.rf_bucket import RFBucket

import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

gamma0 = 3. # defines the energy of the beam
gamma_transition = 4.
momentum_compaction_factor = 1 / gamma_transition**2
compensate_phase = True

particle_ref = xt.Particles(gamma0=gamma0,
                            mass0=xt.PROTON_MASS_EV)

circumference = 1000.
t_rev = circumference / (particle_ref.beta0[0] * clight)
f_rev = 1 / t_rev

energy_ref_increment =  50e3

eta = momentum_compaction_factor - 1 / particle_ref.gamma0[0]**2

h_rf = 40

f_rf = h_rf * f_rev
v_rf = 100e3
lag_rf = 180. if eta > 0. else 0.

# Compute momentum increment using auxiliary particle
dp0c_eV = energy_ref_increment / particle_ref.beta0[0]

if compensate_phase:
    phi_below = np.arcsin(dp0c_eV * particle_ref.beta0[0] / v_rf)
    phi_above = np.pi - phi_below
    lag_rf_above = np.rad2deg(phi_above)
    lag_rf_below = np.rad2deg(phi_below)
    if eta > 0:
        lag_rf = lag_rf_above
    else:
        lag_rf = lag_rf_below

otm = xt.LineSegmentMap(
    betx=1., bety=1,
    qx=6.3, qy=6.4,
    momentum_compaction_factor=momentum_compaction_factor,
    longitudinal_mode="nonlinear",
    voltage_rf=v_rf,
    frequency_rf=f_rf,
    lag_rf=lag_rf,
    length=circumference,
    energy_ref_increment=energy_ref_increment
)

line = xt.Line(elements={'otm': otm}, particle_ref=particle_ref)

tw = line.twiss()

p, matcher = xp.generate_matched_gaussian_bunch(
    line=line,
    num_particles=10_000,
    nemitt_x=2.5e-6,
    nemitt_y=2.5e-6,
    sigma_z=5,
    return_matcher=True)


# Logger (log every ten turns)
num_turns = 50_000
log_every = 20
n_log = num_turns // log_every
mon = xt.ParticlesMonitor(
    start_at_turn=0,
    stop_at_turn=1,
    n_repetitions=n_log,
    repetition_period=log_every,
    num_particles=len(p.x))

#########
# Track #
#########

jumped = False
i_jumped = None
while p.at_turn[0] < num_turns:
    print(f'Turn {p.at_turn[0]}/{num_turns}            ', end='\r', flush=True)

    # Phase jump
    if p.gamma0[0] > gamma_transition and not jumped:
        print(f'Jumped at turn: {p.at_turn[0]}')
        line['otm'].lag_rf = lag_rf_above
        i_jumped = p.at_turn[0]
        jumped = True

    # Track
    line.track(p, num_turns=100, turn_by_turn_monitor=mon)
    p.reorganize() # (put lost particles at the end)

############
# Plotting #
############

# Generate separatrix for video
i_separatrix = []
z_separatrix = []
delta_separatrix = []
for i_sep in np.arange(0, num_turns, 100):
    print(f'Separatrix {i_sep}/{num_turns}            ', end='\r', flush=True)
    line.particle_ref.gamma0 = mon.gamma0[i_sep//log_every, 0, 0]
    line['otm'].lag_rf = lag_rf_above if i_sep >= i_jumped else lag_rf_below
    try:
        rfb = line._get_bucket()
        i_separatrix.append(i_sep)
        z_separatrix.append(np.linspace(rfb.z_left, rfb.z_right, 1000))
        delta_separatrix.append(rfb.separatrix(z_separatrix[-1]))
    except:
        i_separatrix.append(i_sep)
        z_separatrix.append(np.zeros(1000))
        delta_separatrix.append(np.zeros(1000))
        z_separatrix[-1][:] = -1e20
        delta_separatrix[-1][:] = 0

sigma_z_rms = np.squeeze(mon.zeta.std(axis=1))
z_separatrix = np.array(z_separatrix)
delta_separatrix = np.array(delta_separatrix)
i_separatrix = np.array(i_separatrix)

plt.close('all')
fig1 = plt.figure(1)
plt.plot(mon.at_turn[:, 0, 0], sigma_z_rms)
plt.xlabel('Turn')
plt.ylabel('Bunch length [m]')
plt.show()

f_sep_z = interp1d(i_separatrix, z_separatrix, axis=0,
                   bounds_error=False, fill_value='extrapolate')
f_sep_delta = interp1d(i_separatrix, delta_separatrix, axis=0,
                       bounds_error=False, fill_value='extrapolate')

# Make movie (needed `conda install -c conda-forge ffmpeg``)
def update_plot(i_log, fig):
    i_turn = mon.at_turn[i_log, 0, 0]
    z_sep = f_sep_z(i_turn)
    delta_sep = f_sep_delta(i_turn)

    plot_separatrix = True
    if np.abs(i_turn - i_jumped) < 100:
        plot_separatrix = False

    phi_rf_deg = np.rad2deg(phi_above) if i_turn >= i_jumped else np.rad2deg(phi_below)
    plt.clf()
    plt.plot(mon.zeta[i_log, :], mon.delta[i_log, :], '.', markersize=1)
    if plot_separatrix:
        plt.plot(z_sep, delta_sep, color='C1', linewidth=2)
        plt.plot(z_sep, -delta_sep, color='C1', linewidth=2)
    plt.xlim(-10, 10)
    plt.ylim(-10e-3, 10e-3)
    plt.xlabel('z [m]')
    plt.ylabel(r'$\Delta p / p_0$')
    plt.title(f'Turn {i_turn} '
              r'$\sigma_\zeta = $' f'{sigma_z_rms[i_log]:.2f}\n'
              r'$\gamma_0 = $' f'{mon.gamma0[i_log, 0, 0]:.2f} '
              r'$\gamma_t = $' f'{gamma_transition:.2f} '
              r'$\phi_{\mathrm{rf}} = $' f'{phi_rf_deg:.2f}')
    plt.subplots_adjust(left=0.2, top=0.82)
    plt.grid(alpha=0.5)

fig = plt.figure()
from matplotlib.animation import FFMpegFileWriter
moviewriter = FFMpegFileWriter(fps=15)
with moviewriter.saving(fig, 'transition.mp4', dpi=100):
    for j in range(0, len(mon.zeta[:, 0, 0]), 1):
        print(f'Frame {j}/{len(mon.zeta[:, 0, 0])}            ', end='\r', flush=True)
        update_plot(j, fig)
        moviewriter.grab_frame()
