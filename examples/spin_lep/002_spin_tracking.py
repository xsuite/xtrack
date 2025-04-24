import xtrack as xt
import xpart as xp
import xobjects as xo
import numpy as np

num_turns = 500

line = xt.Line.from_json('lep_sol.json')
line.particle_ref.anomalous_magnetic_moment=0.00115965218128
spin_tune = line.particle_ref.anomalous_magnetic_moment[0]*line.particle_ref.gamma0[0]

# Match tunes
opt = line.match(
    method='4d',
    solve=False,
    vary=xt.VaryList(['kqf', 'kqd'], step=1e-4),
    targets=xt.TargetSet(qx=65.28, qy=71.1, tol=1e-4)
)
opt.solve()

tw = line.twiss4d(spin=True, radiation_integrals=True)

# All off
line['on_sol.2'] = 0
line['on_sol.4'] = 0
line['on_sol.6'] = 0
line['on_sol.8'] = 0
line['on_spin_bump.2'] = 0
line['on_spin_bump.4'] = 0
line['on_spin_bump.6'] = 0
line['on_spin_bump.8'] = 0
line['on_coupl_sol.2'] = 0
line['on_coupl_sol.4'] = 0
line['on_coupl_sol.6'] = 0
line['on_coupl_sol.8'] = 0
line['on_coupl_sol_bump.2'] = 0
line['on_coupl_sol_bump.4'] = 0
line['on_coupl_sol_bump.6'] = 0
line['on_coupl_sol_bump.8'] = 0

# RF
line['vrfc231'] = 12.65 # qs=0.6

# Bare machine
tw_bare = line.twiss(spin=True, radiation_integrals=True)

line.configure_radiation('mean')
tw_bare_rad = line.twiss(spin=True, eneloss_and_damping=True)

p_bare = xp.generate_matched_gaussian_bunch(
    line=line,
    nemitt_x=tw_bare_rad.eq_nemitt_x,
    nemitt_y=tw_bare_rad.eq_nemitt_y,
    sigma_z=np.sqrt(tw_bare_rad.eq_gemitt_zeta * tw_bare_rad.bets0),
    num_particles=100,
    engine='linear')
p_bare.zeta += tw_bare_rad.zeta[0]
p_bare.delta += tw_bare_rad.delta[0]
p_bare.spin_x = tw_bare_rad.spin_x[0]
p_bare.spin_y = tw_bare_rad.spin_y[0]
p_bare.spin_z = tw_bare_rad.spin_z[0]

line.configure_radiation(model='quantum')
line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads=10))
line.track(p_bare, num_turns=num_turns, turn_by_turn_monitor=True,
           with_progress=10)
mon_bare = line.record_last_track

pol_x_bare = mon_bare.spin_x.sum(axis=0)/mon_bare.state.sum(axis=0)
pol_y_bare = mon_bare.spin_y.sum(axis=0)/mon_bare.state.sum(axis=0)
pol_z_bare = mon_bare.spin_z.sum(axis=0)/mon_bare.state.sum(axis=0)
pol_bare = np.sqrt(pol_x_bare**2 + pol_y_bare**2 + pol_z_bare**2)

line.configure_radiation(model=None)
line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))

# Solenoids without bumps
line['on_sol.2'] = 1
line['on_sol.4'] = 1
line['on_sol.6'] = 1
line['on_sol.8'] = 1
line['on_coupl_sol.2'] = 1
line['on_coupl_sol.4'] = 1
line['on_coupl_sol.6'] = 1
line['on_coupl_sol.8'] = 1

tw_sol = line.twiss(spin=True, radiation_integrals=True)
line.configure_radiation(model='mean')
tw_sol_rad = line.twiss(spin=True, eneloss_and_damping=True)
p_sol = xp.generate_matched_gaussian_bunch(
    line=line,
    nemitt_x=tw_sol_rad.eq_nemitt_x,
    nemitt_y=tw_sol_rad.eq_nemitt_y,
    sigma_z=np.sqrt(tw_sol_rad.eq_gemitt_zeta * tw_sol_rad.bets0),
    num_particles=100,
    engine='linear')
p_sol.zeta += tw_sol_rad.zeta[0]
p_sol.delta += tw_sol_rad.delta[0]
p_sol.spin_x = tw_sol_rad.spin_x[0]
p_sol.spin_y = tw_sol_rad.spin_y[0]
p_sol.spin_z = tw_sol_rad.spin_z[0]

line.configure_radiation(model='quantum')
line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads=10))
line.track(p_sol, num_turns=num_turns, turn_by_turn_monitor=True,
           with_progress=10)
mon_sol = line.record_last_track
pol_x_sol = mon_sol.spin_x.sum(axis=0)/mon_sol.state.sum(axis=0)
pol_y_sol = mon_sol.spin_y.sum(axis=0)/mon_sol.state.sum(axis=0)
pol_z_sol = mon_sol.spin_z.sum(axis=0)/mon_sol.state.sum(axis=0)
pol_sol = np.sqrt(pol_x_sol**2 + pol_y_sol**2 + pol_z_sol**2)

line.configure_radiation(model=None)
line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))

# Solenoids with bumps
line['on_spin_bump.2'] = 1
line['on_spin_bump.4'] = 1
line['on_spin_bump.6'] = 1
line['on_spin_bump.8'] = 1
line['on_coupl_sol_bump.2'] = 1
line['on_coupl_sol_bump.4'] = 1
line['on_coupl_sol_bump.6'] = 1
line['on_coupl_sol_bump.8'] = 1

tw_sol_bump = line.twiss(spin=True, radiation_integrals=True)
line.configure_radiation(model='mean')
tw_sol_bump_rad = line.twiss(spin=True, eneloss_and_damping=True)
p_sol_bump = xp.generate_matched_gaussian_bunch(
    line=line,
    nemitt_x=tw_sol_bump_rad.eq_nemitt_x,
    nemitt_y=tw_sol_bump_rad.eq_nemitt_y,
    sigma_z=np.sqrt(tw_sol_bump_rad.eq_gemitt_zeta * tw_sol_bump_rad.bets0),
    num_particles=100,
    engine='linear')
p_sol_bump.zeta += tw_sol_bump_rad.zeta[0]
p_sol_bump.delta += tw_sol_bump_rad.delta[0]
p_sol_bump.spin_x = tw_sol_bump_rad.spin_x[0]
p_sol_bump.spin_y = tw_sol_bump_rad.spin_y[0]
p_sol_bump.spin_z = tw_sol_bump_rad.spin_z[0]
line.configure_radiation(model='quantum')
line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads=10))
line.track(p_sol_bump, num_turns=num_turns, turn_by_turn_monitor=True,
           with_progress=10)
mon_sol_bump = line.record_last_track
pol_x_sol_bump = mon_sol_bump.spin_x.sum(axis=0)/mon_sol_bump.state.sum(axis=0)
pol_y_sol_bump = mon_sol_bump.spin_y.sum(axis=0)/mon_sol_bump.state.sum(axis=0)
pol_z_sol_bump = mon_sol_bump.spin_z.sum(axis=0)/mon_sol_bump.state.sum(axis=0)
pol_sol_bump = np.sqrt(pol_x_sol_bump**2 + pol_y_sol_bump**2 + pol_z_sol_bump**2)
line.configure_radiation(model=None)
line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads=0))

from scipy.constants import e as qe
from scipy.constants import c as clight
from scipy.constants import hbar

for ttww in [tw_bare, tw_sol, tw_sol_bump]:

    kappa_x = ttww.rad_int_kappa_x
    kappa_y = ttww.rad_int_kappa_y
    kappa = ttww.rad_int_kappa
    iv_x = ttww.rad_int_iv_x
    iv_y = ttww.rad_int_iv_y
    iv_z = ttww.rad_int_iv_z

    n0_iv = ttww.spin_x * iv_x + ttww.spin_y * iv_y + ttww.spin_z * iv_z
    r0 = ttww.particle_on_co.get_classical_particle_radius0()
    m0_J = ttww.particle_on_co.mass0 * qe
    m0_kg = m0_J / clight**2

    # reference https://lib-extopc.kek.jp/preprints/PDF/1980/8011/8011060.pdf

    alpha_plus_co = 1. / ttww.circumference * np.sum(
        kappa**3 * (1 - 2./9. * n0_iv**2) * ttww.length)

    tp_inv = 5 * np.sqrt(3) / 8 * r0 * hbar * ttww.gamma0**5 / m0_kg * alpha_plus_co
    tp_s = 1 / tp_inv
    tp_turn = tp_s / ttww.T_rev0

    brho_ref = ttww.particle_on_co.p0c[0] / clight / ttww.particle_on_co.q0
    brho_part = (brho_ref * ttww.particle_on_co.rvv[0] * ttww.particle_on_co.energy[0]
                / ttww.particle_on_co.energy0[0])

    By = kappa_x * brho_part
    Bx = -kappa_y * brho_part
    Bz = ttww.ks * brho_ref
    B_mod = np.sqrt(Bx**2 + By**2 + Bz**2)
    B_mod[B_mod == 0] = 999. # avoid division by zero

    ib_x = Bx / B_mod
    ib_y = By / B_mod
    ib_z = Bz / B_mod

    n0_ib = ttww.spin_x * ib_x + ttww.spin_y * ib_y + ttww.spin_z * ib_z

    alpha_minus_co = 1. / ttww.circumference * np.sum(kappa**3 * n0_ib *  ttww.length)

    pol_inf = 8 / 5 / np.sqrt(3) * alpha_minus_co / alpha_plus_co

    ttww._data['alpha_plus_co'] = alpha_plus_co
    ttww._data['alpha_minus_co'] = alpha_minus_co
    ttww._data['pol_inf'] = pol_inf
    ttww['n0_ib'] = n0_ib
    ttww['t_pol_turn'] = tp_turn

# Fit depolarization time (linear fit)
from scipy.stats import linregress
def fit_depolarization_time(turns, pol):
    # Remove NaN values
    mask = ~np.isnan(pol)
    turns = turns[mask]
    pol = pol[mask]

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(turns, np.log(pol))

    # Calculate depolarization time
    depolarization_time = -1 / slope

    return depolarization_time

tw_bare._data['t_dep_turn'] = fit_depolarization_time(np.arange(num_turns), pol_bare)
tw_sol._data['t_dep_turn'] = fit_depolarization_time(np.arange(num_turns), pol_sol)
tw_sol_bump._data['t_dep_turn']= fit_depolarization_time(np.arange(num_turns), pol_sol_bump)

for ttww in [tw_bare, tw_sol, tw_sol_bump]:
    ttww._data['pol'] = ttww['pol_inf'] * (1 / (1 + ttww['t_pol_turn'] / ttww['t_dep_turn']))

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(pol_bare, label=r'bare - $P_{eq}$ = ' f'{tw_bare["pol"]:.2f}')
plt.plot(pol_sol, label=r'sol - $P_{eq}$ = ' f'{tw_sol["pol"]:.2f}')
plt.plot(pol_sol_bump, label=r'sol & bump - $P_{eq}$ = ' f'{tw_sol_bump["pol"]:.2f}')

plt.xlabel('turns')
plt.ylabel('polarization')
plt.suptitle(f'qx = {tw_sol_bump_rad.qx:.2f}, qy = {tw_sol_bump_rad.qy:.2f}, qs = {tw_sol_bump_rad.qs:.2f}')
plt.legend()

plt.show()
