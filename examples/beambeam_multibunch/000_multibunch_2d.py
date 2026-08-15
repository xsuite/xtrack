# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
Multi-bunch coherent (2D) beam-beam example.

Two counter-rotating beams are each represented by several bunches, with one
macroparticle per bunch (the macroparticle holds the bunch centroid, its
longitudinal position `zeta` and its population through `weight`).

The element `BeamBeamBiGaussianMultibunch2D` applies, to every bunch of one
beam, the transverse (dipole) beam-beam kick produced by the opposing bunch it
meets: a particle (bunch) at `zeta` interacts with the opposing bunch located
at `zeta + zeta_offset`. With `zeta_offset = 0` and both beams sharing the same
longitudinal grid, bunch `i` of beam 1 collides with bunch `i` of beam 2.

Each turn we:
  1) snapshot both beams into the opposing element (`update_from_other_beam`)
     BEFORE applying any kick, so both beams are kicked using the bunch
     positions at the same turn (strong-strong simultaneity);
  2) apply the beam-beam kicks;
  3) transport both beams with a simple linear one-turn map.
"""

import numpy as np
import matplotlib.pyplot as plt

import xpart as xp
import xtrack as xt
import xfields as xf

# ----------------------------------------------------------------------------
# Beam / machine parameters
# ----------------------------------------------------------------------------
p0c = 7e12                      # 7 TeV protons
mass0 = xp.PROTON_MASS_EV

n_bunches = 4                   # bunches per beam
bunch_spacing_zeta = 0.75       # m, spacing between bunches in zeta
bunch_intensity = 1.2e11        # real charges per bunch

# Per-bunch normalized emittances and beta functions at the IP. Here we give
# each bunch a slightly different horizontal emittance to show the per-bunch
# treatment (bunch 0 the smallest, bunch 3 the largest).
nemitt_x = np.array([1.8e-6, 2.0e-6, 2.2e-6, 2.5e-6])   # m rad
nemitt_y = 2.0e-6 * np.ones(n_bunches)                   # m rad
betx_ip = 1.0 * np.ones(n_bunches)                       # m
bety_ip = 1.0 * np.ones(n_bunches)                       # m

gamma0 = p0c / mass0            # ultrarelativistic: beta*gamma ~ gamma
beta0_rel = np.sqrt(1 - 1 / gamma0**2)

qx, qy = 0.31, 0.32             # transverse tunes of the one-turn map
beta_x, beta_y = 1.0, 1.0       # m, beta functions of the one-turn map

# Reference horizontal size (bunch 0), only used to scale the plots / offset
sigma_x = np.sqrt(nemitt_x[0] / (beta0_rel * gamma0) * betx_ip[0])

n_turns = 2048

# ----------------------------------------------------------------------------
# Build the two multi-bunch beams (1 macroparticle = 1 bunch)
# ----------------------------------------------------------------------------
zeta_bunches = np.arange(n_bunches) * bunch_spacing_zeta

def make_beam(x0, y0):
    p = xp.Particles(
        p0c=p0c, mass0=mass0, q0=1.0,
        x=x0 * np.ones(n_bunches),
        y=y0 * np.ones(n_bunches),
        zeta=zeta_bunches.copy(),
        weight=bunch_intensity,   # bunch population
    )
    return p

# Give the two beams a small initial transverse offset so that the coherent
# beam-beam force drives the bunch centroids.
beam1 = make_beam(x0=0.5 * sigma_x, y0=0.0)
beam2 = make_beam(x0=-0.5 * sigma_x, y0=0.0)

# ----------------------------------------------------------------------------
# Beam-beam elements (one per beam: it kicks "its" beam using the other one)
# ----------------------------------------------------------------------------
common_bb_kwargs = dict(
    zeta_offset=0.0,            # head-on: bunch i meets bunch i
    zeta_match_tol=1e-6,
    other_beam_q0=1.0,
    other_beam_beta0=beta0_rel,
    # per-bunch transverse sizes (sigma = sqrt(beta * nemitt / gamma0))
    other_beam_sigma_x=np.sqrt(betx_ip * nemitt_x / gamma0),
    other_beam_sigma_y=np.sqrt(bety_ip * nemitt_y / gamma0),
)

# The opposing beam is passed as a Particles object (each active macroparticle
# is one bunch); `num_bunches` and the initial bunch centroids are taken from it.
bb_on_beam1 = xf.BeamBeamBiGaussianMultibunch2D(
    other_particles=beam2, **common_bb_kwargs)  # other = beam2
bb_on_beam2 = xf.BeamBeamBiGaussianMultibunch2D(
    other_particles=beam1, **common_bb_kwargs)  # other = beam1

# Report the per-bunch transverse sizes derived from the emittances and betas.
print('opposing bunch sigma_x [um]:', bb_on_beam1.other_beam_sigma_x * 1e6)
print('opposing bunch sigma_y [um]:', bb_on_beam1.other_beam_sigma_y * 1e6)

# ----------------------------------------------------------------------------
# Simple linear one-turn maps (transverse rotation by the tune)
# ----------------------------------------------------------------------------
otm1 = xt.LineSegmentMap(qx=qx, qy=qy, betx=beta_x, bety=beta_y)
otm2 = xt.LineSegmentMap(qx=qx, qy=qy, betx=beta_x, bety=beta_y)

# ----------------------------------------------------------------------------
# Tracking loop
# ----------------------------------------------------------------------------
rec_x1 = np.zeros((n_turns, n_bunches))
rec_x2 = np.zeros((n_turns, n_bunches))

for turn in range(n_turns):
    # 1) snapshot both beams BEFORE kicking either of them
    bb_on_beam1.update_from_other_beam(beam2)
    bb_on_beam2.update_from_other_beam(beam1)

    # 2) beam-beam kicks
    bb_on_beam1.track(beam1)
    bb_on_beam2.track(beam2)

    # 3) linear transport
    otm1.track(beam1)
    otm2.track(beam2)

    # record bunch centroids
    rec_x1[turn] = beam1.x
    rec_x2[turn] = beam2.x

# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for ib in range(n_bunches):
    axs[0].plot(rec_x1[:, ib] / sigma_x, label=f'bunch {ib}')
    axs[1].plot(rec_x2[:, ib] / sigma_x, label=f'bunch {ib}')
axs[0].set_ylabel('beam 1  x / $\\sigma_x$')
axs[1].set_ylabel('beam 2  x / $\\sigma_x$')
axs[1].set_xlabel('turn')
axs[0].legend(ncol=n_bunches, fontsize=8)
axs[0].set_title('Coherent multi-bunch beam-beam (2D)')

# Coherent beam-beam tune spectra of all bunches of beam 1
fig2, ax2 = plt.subplots(figsize=(8, 4))
freqs = np.fft.rfftfreq(n_turns)
for ib in range(n_bunches):
    spectrum = np.abs(np.fft.rfft(rec_x1[:, ib] - rec_x1[:, ib].mean()))
    ax2.plot(freqs, spectrum, label=f'bunch {ib}')
ax2.axvline(qx % 1, color='k', ls='--', label='unperturbed $q_x$')
ax2.set_xlabel('tune')
ax2.set_ylabel('amplitude')
ax2.set_title('Spectra of beam 1 bunches (x)')
ax2.legend()
ax2.set_xlim(0.28, 0.34)

plt.tight_layout()
plt.show()
