import xtrack as xt

rf = xt.TempRF(frequency=1e9, voltage=1e6, lag=90, length=2)
cav = xt.Cavity(frequency=1e9, voltage=1e6, lag=90)

rf.compile_kernels()

p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, p0c=1e9)

p_rf = p0.copy()
p_cav = p0.copy()

rf.track(p_rf)
cav.track(p_cav)
