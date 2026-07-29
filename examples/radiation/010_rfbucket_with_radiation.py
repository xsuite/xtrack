import xtrack as xt
import xobjects as xo
import xpart as xp
import numpy as np

# TODO:
# - Review docstring of particles generation method


env = xt.load('../../test_data/fcc_ee/fccee_h.seq')
pc_gev = 120.

line = env.fccee_p_ring
line.set_particle_ref('positron', p0c=pc_gev*1e9)

tw_no_rad = line.twiss6d()
rfb_no_rad = line._get_bucket()

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()

tw_rad = line.twiss6d(radiation_analysis=True)
rfb_rad = line._get_bucket()

# Check that the effect of the radiation is visible on qs
assert tw_no_rad.qs > 0.045
assert tw_rad.qs < 0.035

# Check consistency of qs and bets0 between twiss and rfb
xo.assert_allclose(rfb_no_rad.Q_s, tw_no_rad.qs, rtol=0.01)
xo.assert_allclose(rfb_rad.Q_s, tw_rad.qs, rtol=0.01)
xo.assert_allclose(rfb_no_rad.beta_z, tw_no_rad.bets0, rtol=0.015)
xo.assert_allclose(rfb_rad.beta_z, tw_rad.bets0, rtol=0.015)
xo.assert_allclose(rfb_no_rad.z_sfp, tw_no_rad.zeta[0], rtol=0, atol=5e-7)
xo.assert_allclose(rfb_rad.z_sfp, tw_rad.zeta[0], rtol=0, atol=5e-7)


particles = xp.generate_matched_gaussian_bunch(
    num_particles=10000,
    nemitt_x=tw_rad.eq_gemitt_x,
    nemitt_y=tw_rad.eq_gemitt_y,
    sigma_z=np.sqrt(tw_rad.bets0*tw_rad.eq_gemitt_zeta),
    total_intensity_particles=1e11,
    line=line
)