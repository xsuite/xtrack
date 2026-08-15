# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
Per-bunch closed solution of a multi-bunch beam with `twiss_multibunch`.

Each bunch of a multi-bunch beam sits at a distinct longitudinal position
`zeta` and, through the `BeamBeamBiGaussianMultibunch2D` element, sees a
different opposing bunch. Its closed orbit and optics (in particular the tunes)
therefore differ from bunch to bunch. `line.twiss_multibunch(...)` fixes `zeta`
to each bunch position in turn and returns the corresponding periodic solution.

The example has two parts:
  A) per-bunch tune footprint of a beam colliding head-on with an opposing beam
     whose bunches have (slightly) different emittances;
  B) self-consistent per-bunch closed orbit of two colliding beams that are
     transversely separated, obtained by iterating `twiss_multibunch` on each
     beam while updating the opposing-beam centroids.
"""

import numpy as np
import matplotlib.pyplot as plt

import xpart as xp
import xtrack as xt
import xfields as xf

# ----------------------------------------------------------------------------
# Beam / machine parameters
# ----------------------------------------------------------------------------
p0c = 7e12
mass0 = xp.PROTON_MASS_EV
gamma0 = p0c / mass0
beta0_rel = np.sqrt(1 - 1 / gamma0**2)

n_bunches = 8
bunch_spacing_zeta = 0.75
bunch_intensity = 2.0e11

# Per-bunch normalized emittances (a small bunch-by-bunch spread)
nemitt_x = np.linspace(1.8e-6, 2.6e-6, n_bunches)
nemitt_y = 2.0e-6 * np.ones(n_bunches)
betx_ip = 1.0 * np.ones(n_bunches)
bety_ip = 1.0 * np.ones(n_bunches)

qx0, qy0 = 0.31, 0.32

zeta_bunches = np.arange(n_bunches) * bunch_spacing_zeta

# ============================================================================
# Part A: per-bunch tune footprint (head-on, opposing beam on axis)
# ============================================================================
opposing = xp.Particles(
    p0c=p0c, mass0=mass0, q0=1.0,
    x=np.zeros(n_bunches), y=np.zeros(n_bunches),
    zeta=zeta_bunches.copy(), weight=bunch_intensity)

bb = xf.BeamBeamBiGaussianMultibunch2D(
    other_particles=opposing, zeta_offset=0.0, zeta_match_tol=1e-6,
    other_beam_q0=1.0, other_beam_beta0=beta0_rel,
    other_beam_sigma_x=np.sqrt(betx_ip * nemitt_x / gamma0),
    other_beam_sigma_y=np.sqrt(bety_ip * nemitt_y / gamma0))

otm = xt.LineSegmentMap(qx=qx0, qy=qy0, betx=1.0, bety=1.0,
                        longitudinal_mode='frozen')
line = xt.Line(elements=[otm, bb], element_names=['otm', 'bb'])
line.particle_ref = xp.Particles(p0c=p0c, mass0=mass0, q0=1.0)

mbtw = line.twiss_multibunch(zeta_bunches=zeta_bunches)

print('Per-bunch tunes:')
for ib in range(n_bunches):
    print(f'  bunch {ib}: qx={mbtw.qx[ib]:.6f}  qy={mbtw.qy[ib]:.6f}  '
          f'(beam-beam dQx={mbtw.qx[ib]-qx0:+.2e})')

# ============================================================================
# Part B: self-consistent per-bunch closed orbit of two separated beams
# ============================================================================
# The two beams are horizontally separated at the IP (e.g. a long-range-like
# offset), so every bunch receives a dipole beam-beam kick and develops a
# closed-orbit distortion. The orbit of one beam depends on the orbit of the
# other, so we iterate to self-consistency.
# Separation of a few beam sigmas, so the (per-bunch) beam-beam dipole kick is
# significant and bunch-dependent (bunches differ in size through nemitt_x).
sigma_x_ref = np.sqrt(betx_ip[0] * nemitt_x.mean() / gamma0)
separation = 4.0 * sigma_x_ref   # nominal horizontal separation [m]


def make_line(x_ref):
    # A one-turn map whose reference (closed) orbit is offset by `x_ref`, plus
    # the beam-beam element (opposing beam filled in during the iteration).
    otm = xt.LineSegmentMap(qx=qx0, qy=qy0, betx=1.0, bety=1.0,
                            x_ref=x_ref, longitudinal_mode='frozen')
    bb = xf.BeamBeamBiGaussianMultibunch2D(
        num_bunches=n_bunches, zeta_offset=0.0, zeta_match_tol=1e-6,
        other_beam_q0=1.0, other_beam_beta0=beta0_rel,
        other_beam_sigma_x=np.sqrt(betx_ip * nemitt_x / gamma0),
        other_beam_sigma_y=np.sqrt(bety_ip * nemitt_y / gamma0))
    ln = xt.Line(elements=[otm, bb], element_names=['otm', 'bb'])
    ln.particle_ref = xp.Particles(p0c=p0c, mass0=mass0, q0=1.0)
    return ln, bb


line1, bb1 = make_line(x_ref=+separation / 2)
line2, bb2 = make_line(x_ref=-separation / 2)


def co_particles(mbtw, weight):
    # Build a Particles object with the per-bunch closed orbit at the bb element
    # (each active macroparticle = one bunch), to feed the opposing element.
    x = np.array([tw['x', 'bb'] for tw in mbtw])
    y = np.array([tw['y', 'bb'] for tw in mbtw])
    return xp.Particles(p0c=p0c, mass0=mass0, q0=1.0,
                        x=x, y=y, zeta=zeta_bunches.copy(), weight=weight)


# Initial guess: both beams on their bare reference orbit (+/- separation/2)
x1 = np.full(n_bunches, +separation / 2)
x2 = np.full(n_bunches, -separation / 2)

n_iter = 8
history = np.zeros((n_iter, n_bunches))
for it in range(n_iter):
    # opposing centroids from the other beam's current closed orbit
    bb1.update_from_other_beam(xp.Particles(
        p0c=p0c, mass0=mass0, q0=1.0, x=x2, y=np.zeros(n_bunches),
        zeta=zeta_bunches.copy(), weight=bunch_intensity))
    bb2.update_from_other_beam(xp.Particles(
        p0c=p0c, mass0=mass0, q0=1.0, x=x1, y=np.zeros(n_bunches),
        zeta=zeta_bunches.copy(), weight=bunch_intensity))

    mbtw1 = line1.twiss_multibunch(zeta_bunches=zeta_bunches)
    mbtw2 = line2.twiss_multibunch(zeta_bunches=zeta_bunches)

    x1 = np.array([tw['x', 'bb'] for tw in mbtw1])
    x2 = np.array([tw['x', 'bb'] for tw in mbtw2])
    history[it] = x1
    print(f'iter {it}: beam1 per-bunch closed orbit x [um] = '
          f'{np.array2string(x1 * 1e6, precision=2)}')

print('\nConverged beam1 per-bunch closed orbit x [um]:', x1 * 1e6)
print('(bare reference orbit was %.1f um)' % (separation / 2 * 1e6))

# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(11, 4))

# Part A: tune footprint vs bunch
axs[0].plot(range(n_bunches), mbtw.qx, 'o-', label='$q_x$')
axs[0].plot(range(n_bunches), mbtw.qy, 's-', label='$q_y$')
axs[0].axhline(qx0, color='C0', ls='--', alpha=0.5)
axs[0].axhline(qy0, color='C1', ls='--', alpha=0.5)
axs[0].set_xlabel('bunch index')
axs[0].set_ylabel('tune')
axs[0].set_title('Part A: per-bunch tunes\n(head-on, per-bunch emittance)')
axs[0].legend()

# Part B: convergence of the per-bunch closed orbit
for ib in range(n_bunches):
    axs[1].plot(range(n_iter), history[:, ib] * 1e6, '.-',
                label=f'bunch {ib}' if ib in (0, n_bunches - 1) else None)
axs[1].axhline(separation / 2 * 1e6, color='k', ls='--',
               label='bare reference')
axs[1].set_xlabel('iteration')
axs[1].set_ylabel('beam1 closed orbit x [$\\mu$m]')
axs[1].set_title('Part B: self-consistent\nper-bunch closed orbit')
axs[1].legend(fontsize=8)

plt.tight_layout()
plt.show()
