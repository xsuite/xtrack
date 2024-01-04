# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import xtrack as xt

# Load a line and build tracker
line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_and_particle.json')
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, energy0=7e12)
line.build_tracker()

# Twiss
tw = line.twiss()

# Transverse normalized emittances
nemitt_x = 2.5e-6
nemitt_y = 2.5e-6

# Longitudinal emittance from energy spread
sigma_pzeta = 2e-4
gemitt_zeta = sigma_pzeta**2 * tw.bets0
# similarly, if the bunch length is known, the emittance can be computed as
# gemitt_zeta = sigma_zeta**2 / tw.bets0

# Compute beam sizes
beam_sizes = tw.get_beam_covariance(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                                    gemitt_zeta=gemitt_zeta)

# Inspect beam sizes (table can be accessed similarly to twiss tables)
beam_sizes.rows['ip.?'].show()
# prints
#
# name                       s     sigma_x     sigma_y sigma_zeta    sigma_px ...
# ip3                        0 0.000226516 0.000270642    0.19694 4.35287e-06
# ip4                  3332.28 0.000281326 0.000320321   0.196941 1.30435e-06
# ip5                  6664.57  7.0898e-06 7.08975e-06    0.19694  4.7265e-05
# ip6                  9997.01 0.000314392 0.000248136   0.196939 1.61401e-06
# ip7                  13329.4 0.000205156 0.000223772   0.196939 2.70123e-06
# ip8                  16650.7 2.24199e-05 2.24198e-05   0.196939 1.49465e-05
# ip1                  19994.2 7.08975e-06 7.08979e-06   0.196939 4.72651e-05
# ip2                  23326.6 5.78877e-05 5.78878e-05   0.196939 5.78879e-06

# All covariances are computed including those from linear coupling
beam_sizes.keys()
# is:
#
# ['s', 'name', 'sigma_x', 'sigma_y', 'sigma_zeta', 'sigma_px', 'sigma_py',
# 'sigma_pzeta', 'Sigma', 'Sigma11', 'Sigma12', 'Sigma13', 'Sigma14', 'Sigma15',
# 'Sigma16', 'Sigma21', 'Sigma22', 'Sigma23', 'Sigma24', 'Sigma25', 'Sigma26',
# 'Sigma31', 'Sigma32', 'Sigma33', 'Sigma34', 'Sigma41', 'Sigma42', 'Sigma43',
# 'Sigma44', 'Sigma51', 'Sigma52'])

# Plot
import matplotlib.pyplot as plt
plt.close('all')

fig1 = plt.figure(1, figsize=(6.4, 4.8*1.5))
spbet = plt.subplot(3,1,1)
spdisp = plt.subplot(3,1,2, sharex=spbet)
spbsz = plt.subplot(3,1,3, sharex=spbet)

spbet.plot(tw.s, tw.betx)
spbet.plot(tw.s, tw.bety)
spbet.set_ylabel(r'$\beta_{x,y}$ [m]')

spdisp.plot(tw.s, tw.dx)
spdisp.plot(tw.s, tw.dy)
spdisp.set_ylabel(r'$D_{x,y}$ [m]')

spbsz.plot(beam_sizes.s, beam_sizes.sigma_x)
spbsz.plot(beam_sizes.s, beam_sizes.sigma_y)
spbsz.set_ylabel(r'$\sigma_{x,y}$ [m]')
spbsz.set_xlabel('s [m]')

spbet.set_xlim(tw['s', 'ip5'] - 2000, tw['s', 'ip5'] + 2000)

fig1.subplots_adjust(left=.15, right=.92, hspace=.27)
plt.show()

#!end-doc-part
assert np.allclose(beam_sizes.sigma_pzeta, 2e-4, atol=0, rtol=2e-5)
assert np.allclose(
    beam_sizes.sigma_zeta / beam_sizes.sigma_pzeta, tw.bets0, atol=0, rtol=5e-5)
