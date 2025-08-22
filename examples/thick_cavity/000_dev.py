import xtrack as xt

# TODO:
# - Write tests for imported hkicker, vkicker, tkicker, kicker, rfcavity, crabcavity
# - Tapering
# - Exception tapering for sliced cavities (to be implemented at a later stage)
# - Disable lag_taper when there is no radiation
# - optimize_for_tracking
# - Survey for thick multipoles
# - Is curvature handled correctly in radiation integrals?
# - And in spin calculation?
# - What does the slicing do when length=0?

rf = xt.TempRF(frequency=1e9, voltage=1e6, lag=30, length=2)
cav = xt.Cavity(frequency=1e9, voltage=1e6, lag=30)

rf.integrator = 'yoshida4'
rf.num_kicks = 30

rf.compile_kernels()

p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, p0c=1e9,
                  x=1e-3, y=2e-4, zeta=0.1)

p_rf = p0.copy()
p_cav = p0.copy()

rf.track(p_rf)
cav.track(p_cav)

rf_with_mult = xt.TempRF(frequency=1e9, length=0.000001,
                         knl=[1], ksl=[4])
rfm = xt.RFMultipole(
    frequency=1e9,
    knl=[1], ksl=[4]
)
rf_with_mult.integrator = 'uniform'
rf_with_mult.num_kicks = 10

p_rf_mult = p0.copy()
p_cav_mult = p0.copy()

rf_with_mult.track(p_rf_mult)
rfm.track(p_cav_mult)
