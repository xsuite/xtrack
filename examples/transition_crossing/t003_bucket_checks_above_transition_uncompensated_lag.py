import xtrack as xt
import xpart as xp
import xobjects as xo
from xpart.longitudinal.rf_bucket import RFBucket

import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe
import matplotlib.pyplot as plt

gamma0 = 10. # defines the energy of the beam
gamma_transition = 4.
momentum_compaction_factor = 1 / gamma_transition**2

particle_ref = xt.Particles(gamma0=gamma0,
                            mass0=xt.PROTON_MASS_EV)

circumference = 1000.
t_rev = circumference / (particle_ref.beta0[0] * clight)
f_rev = 1 / t_rev

energy_ref_increment =  50e3 # eV

eta = momentum_compaction_factor - 1 / particle_ref.gamma0[0]**2
assert eta > 0

h_rf = 40

f_rf = h_rf * f_rev
v_rf = 100e3
lag_rf = 180. if eta > 0. else 0.

# Compute momentum increment using auxiliary particle
dp0c_eV = energy_ref_increment / particle_ref.beta0[0]

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
xo.assert_allclose(tw.slip_factor, eta, atol=1e-3, rtol=0)
xo.assert_allclose(tw.qs, 0.00176525, atol=1e-7, rtol=0)

rfb = line._get_bucket()

# Mostly checking that they do not change
xo.assert_allclose(line['otm'].lag_rf[0], 180., # degrees
                   atol=1e-4, rtol=0)

xo.assert_allclose(np.rad2deg(rfb.dphi[0]), 180.,
                   atol=1e-4, rtol=0)

z_shift = 2.08333333
xo.assert_allclose(rfb.z_sfp, z_shift, atol=1e-4, rtol=0)
xo.assert_allclose(rfb.z_ufp, 8.33333 + z_shift, atol=1e-4, rtol=0)
xo.assert_allclose(rfb.z_left, -4.76990 + z_shift, atol=1e-4, rtol=0)
xo.assert_allclose(rfb.z_right, rfb.z_ufp, atol=1e-4, rtol=0)

xo.assert_allclose(rfb.h_sfp(), 8.75048, atol=1e-3, rtol=0)
xo.assert_allclose(rfb.h_sfp(), rfb.hamiltonian(rfb.z_sfp, 0), atol=1e-3, rtol=0)
xo.assert_allclose(rfb.h_sfp(make_convex=True), 8.75048, atol=1e-3, rtol=0)
xo.assert_allclose(rfb.h_sfp(make_convex=True),
                   rfb.hamiltonian(rfb.z_sfp, 0, make_convex=True),
                   atol=1e-3, rtol=0)

# Build separatrix
z_separatrix_up = np.linspace(rfb.z_left, rfb.z_right, 1000)
delta_separatrix_up = rfb.separatrix(z_separatrix_up)

z_separatrix = np.array(
    list(z_separatrix_up) + list(z_separatrix_up[::-1]))
delta_separatrix = np.array(
    list(delta_separatrix_up) + list(-delta_separatrix_up[::-1]))

# Hamiltonian is defined to be zero on the separatrix
xo.assert_allclose(rfb.hamiltonian(z_separatrix, delta_separatrix), 0,
                   atol=1e-3, rtol=0)
xo.assert_allclose(rfb.hamiltonian(z_separatrix, delta_separatrix, make_convex=True), 0,
                   atol=1e-3, rtol=0)

# Check that the separatrix behaves as such in tracking
p = line.build_particles(delta=delta_separatrix[::10]*0.98, zeta=z_separatrix[::10]*0.98)
line.track(p, turn_by_turn_monitor=True, num_turns=3000)
mon = line.record_last_track
assert np.all(mon.zeta < rfb.z_right)
assert np.all(mon.zeta > rfb.z_left)

p = line.build_particles(delta=delta_separatrix[::10]*1.01, zeta=z_separatrix[::10]*1.01)
line.track(p, turn_by_turn_monitor=True, num_turns=3000)
mon = line.record_last_track
assert not np.all(mon.zeta < rfb.z_right)
assert not np.all(mon.zeta > rfb.z_left)

# Check the stable fixed point against tracking
p = line.build_particles(delta=0, zeta=rfb.z_sfp)
line.track(p, turn_by_turn_monitor=True, num_turns=3000)
mon = line.record_last_track
xo.assert_allclose(mon.zeta, rfb.z_sfp, atol=2e-3*(rfb.z_right - rfb.z_left),
                   rtol=0)
bucket_height = rfb.separatrix(rfb.z_sfp)[0]
xo.assert_allclose(mon.delta, 0, atol=2e-2*bucket_height, rtol=0)

# Fix numpy random seed
np.random.seed(0)

# Match a bunch
sigma_z = 2.
p, matcher = xp.generate_matched_gaussian_bunch(
    line=line,
    num_particles=100_000,
    nemitt_x=2.5e-6,
    nemitt_y=2.5e-6,
    sigma_z=sigma_z,
    return_matcher=True)
sigma_delta = p.delta.std()

assert np.all(p.zeta < rfb.z_right)
assert np.all(p.zeta > rfb.z_left)
assert np.all(p.delta < bucket_height)
assert np.all(p.delta > -bucket_height)

xo.assert_allclose(p.delta.max(), bucket_height, atol=0, rtol=0.03)
xo.assert_allclose(p.delta.min(), -bucket_height, atol=0, rtol=0.03)
xo.assert_allclose(p.zeta.max(), rfb.z_right, atol=0, rtol=0.7) # this part of the bucket is poorly populated
xo.assert_allclose(p.zeta.min(), rfb.z_left, atol=0, rtol=0.04)
xo.assert_allclose(p.zeta.std(), sigma_z, atol=0, rtol=0.002)

# Check that the distribution stays roughly stable over one synchrotron period
p, matcher = xp.generate_matched_gaussian_bunch(
    line=line,
    num_particles=10_000,
    nemitt_x=2.5e-6,
    nemitt_y=2.5e-6,
    sigma_z=sigma_z,
    return_matcher=True)

num_turns = int(np.round(1/tw.qs))
log_every = 3
n_log = num_turns // log_every
mon = xt.ParticlesMonitor(
    start_at_turn=0,
    stop_at_turn=1,
    n_repetitions=n_log,
    repetition_period=log_every,
    num_particles=len(p.x))

line.track(p, num_turns=num_turns, turn_by_turn_monitor=mon,
           with_progress=10)

z_mean = np.squeeze(np.mean(mon.zeta, axis=1))
z_std = np.squeeze(np.std(mon.zeta, axis=1))
delta_mean = np.squeeze(np.mean(mon.delta, axis=1))
delta_std = np.squeeze(np.std(mon.delta, axis=1))

xo.assert_allclose(z_mean, np.mean(z_mean), atol=0.02*sigma_z)
xo.assert_allclose(z_std, np.mean(z_std), atol=0.02*sigma_z)
xo.assert_allclose(delta_mean, np.mean(delta_mean), atol=0.02*sigma_delta)
xo.assert_allclose(delta_std, np.mean(delta_std), atol=0.02*sigma_delta)
