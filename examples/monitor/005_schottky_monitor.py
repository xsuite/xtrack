# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

"""Schottky monitor example.
This script builds a simple LHC-like ring using a LineSegmentMap, inserts
a Schottky monitor, tracks a Gaussian bunch, and produces Schottky spectra.
"""

import xtrack as xt
import xpart as xp
import matplotlib.pyplot as plt

# Build simple LHC-like lattice
length_lhc = 26658.8831999989
lmap = xt.LineSegmentMap(length=length_lhc, qx=0.27, qy=0.295, dqx=15, dqy=15,
                         longitudinal_mode='nonlinear', voltage_rf=4e6,
                         frequency_rf=400e6, lag_rf=180,
                         momentum_compaction_factor=3.225e-04,
                         betx=1, bety=1)
line = xt.Line(elements=[lmap])
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, energy0=450e9)

# Twiss to get revolution frequency
tw = line.twiss()

# Create and insert Schottky monitor
schottky_monitor = xt.monitors.SchottkyMonitor(f_rev=1 / tw.T_rev0,
                                               schottky_harmonic=427_725,
                                               n_taylor=4)
line.discard_tracker()
line.append_element(element=schottky_monitor, name='schottky_monitor')
line.build_tracker()

# Generate a matched Gaussian bunch
num_particles = 10_000
bunch = xp.generate_matched_gaussian_bunch(num_particles=num_particles,
                                           nemitt_x=1.5e-6, nemitt_y=1.5e-6,
                                           line=line,
                                           total_intensity_particles=1e11,
                                           sigma_z=7e-2)

# Track for 10k turns and process a first spectrum
line.track(bunch, num_turns=10_000, with_progress=True)
schottky_monitor.process_spectrum(inst_spectrum_len=10_000,
                                  delta_q=5e-5, band_width=0.3,
                                  qx=0.27, qy=0.295,
                                  x=True, y=False, z=True)

schottky_monitor.plot()
# Or plot specific regions in log scale
# schottky_monitor.plot(regions=['lowerH','center','upperH'], log=True)


# (Optional) accumulate additional statistics: uncomment for more averaging
# line.track(bunch, num_turns=200_000, with_progress=True)
# schottky_monitor.process_spectrum(inst_spectrum_len=10_000,
#                                   delta_q=5e-5, band_width=0.3,
#                                   qx=0.27, qy=0.295,
#                                   x=True, y=False, z=True)
# schottky_monitor.plot(regions=['lowerH','center','upperH'], log=True)

# Reprocess with different processing parameters (without tracking again)
schottky_monitor.clear_spectrum()
schottky_monitor.process_spectrum(inst_spectrum_len=5000, delta_q=5e-5,
                                  band_width=0.1, qx=0.27, qy=0.295,
                                  x=True, y=False, z=True)
schottky_monitor.plot(regions=['lowerH','center','upperH'])

plt.show()
