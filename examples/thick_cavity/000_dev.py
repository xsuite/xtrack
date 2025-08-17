import xtrack as xt

# TODO:
# - Propagate integrator/model to C code
# - Backtrack
# - Tapering
# - Absolute time
# - Slicing


rf = xt.TempRF(frequency=1e9, voltage=1e6, lag=30, length=2)
cav = xt.Cavity(frequency=1e9, voltage=1e6, lag=30)

rf.integrator = 'yoshida4'
rf.num_kicks = 30

rf.compile_kernels()

p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, p0c=1e9)

p_rf = p0.copy()
p_cav = p0.copy()

rf.track(p_rf)
cav.track(p_cav)

rf_with_mult = xt.TempRF(frequency=1e9,
                         knl=[1, 2, 3], ksl=[4, 5, 6])
rfm = xt.RFMultipole(
    frequency=1e9,
    knl=[1, 2, 3], ksl=[4, 5, 6]
)