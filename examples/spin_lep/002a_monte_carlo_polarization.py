import xtrack as xt
import xpart as xp
import xobjects as xo
import numpy as np

num_turns = 200

line = xt.Line.from_json('lep_sol.json')
line.particle_ref.anomalous_magnetic_moment=0.00115965218128
spin_tune = line.particle_ref.anomalous_magnetic_moment[0]*line.particle_ref.gamma0[0]

# Match tunes
opt = line.match(
    method='4d',
    solve=False,
    vary=xt.VaryList(['kqf', 'kqd'], step=1e-4),
    targets=xt.TargetSet(qx=65.10, qy=71.20, tol=1e-4)
)
opt.solve()

tw = line.twiss4d(spin=True, radiation_integrals=True, polarization=True)

line['on_solenoids'] = 1
line['on_spin_bumps'] = 1
line['on_coupling_corrections'] = 1

# RF
line['vrfc231'] = 12.65 # qs=0.6

tw = line.twiss(spin=True, radiation_integrals=True, polarization=True)

line.configure_radiation('mean')
tw_rad = line.twiss(spin=True, eneloss_and_damping=True, polarization=True)

particles = xp.generate_matched_gaussian_bunch(
    line=line,
    nemitt_x=tw_rad.eq_nemitt_x,
    nemitt_y=tw_rad.eq_nemitt_y,
    sigma_z=np.sqrt(tw_rad.eq_gemitt_zeta * tw_rad.bets0),
    num_particles=300,
    engine='linear')
particles.zeta += tw_rad.zeta[0]
particles.delta += tw_rad.delta[0]
particles.spin_x = tw_rad.spin_x[0]
particles.spin_y = tw_rad.spin_y[0]
particles.spin_z = tw_rad.spin_z[0]

line.configure_radiation(model='quantum')
line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads=10))
line.track(particles, num_turns=num_turns, turn_by_turn_monitor=True,
           with_progress=10)
mon = line.record_last_track
mask_alive = mon.state > 0
pol_x = mon.spin_x.sum(axis=0)/mask_alive.sum(axis=0)
pol_y = mon.spin_y.sum(axis=0)/mask_alive.sum(axis=0)
pol_z = mon.spin_z.sum(axis=0)/mask_alive.sum(axis=0)
pol = np.sqrt(pol_x**2 + pol_y**2 + pol_z**2)

# Fit depolarization time (linear fit)
from scipy.stats import linregress
def fit_depolarization_time(turns, pol):
    # Remove NaN values
    mask = ~np.isnan(pol)
    turns = turns[mask]
    pol = pol[mask]

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(turns, pol)

    # Calculate depolarization time
    depolarization_time = -1 / slope

    return depolarization_time, intercept

i_start = 3

pol_to_fit = pol[i_start:]/pol[i_start]

t_dep_turns, intercept = fit_depolarization_time(np.arange(len(pol_to_fit)), pol_to_fit)
tw._data['pol'] = tw['spin_polarization_inf_no_depol'] * (1 / (1 + tw['spin_t_pol_component_s']/tw.T_rev0 / t_dep_turns))


import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(pol_to_fit, label=r'$P_{eq}$ = ' f'{tw["pol"]:.2f}')

i_turn = np.arange(num_turns)
plt.plot(intercept*np.exp(-i_turn/t_dep_turns), label='exp(-t/t_dep)')


plt.xlabel('turns')
plt.ylabel('polarization')
plt.suptitle(f'qx = {tw.qx:.2f}, qy = {tw.qy:.2f}, qs = {tw.qs:.2f}')
plt.legend()

plt.show()
