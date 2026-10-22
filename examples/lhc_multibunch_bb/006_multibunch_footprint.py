# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
Tune FOOTPRINTS of different bunch families in the LHC collision scenario
(6.8 TeV squeezed flat optics, head-on + BBLR), with the per-bunch
self-consistent multi-bunch beam-beam closed solution.

Workflow (multi-threaded OpenMP kernels by default, ``LHC_OMP=0`` for
serial):

1. Solve the multi-bunch problem on the sector-map machine (as in
   ``002``/``004``): 2 iterations with the orbit-only ``fast_orbit`` twiss,
   then 2 more with ``dynamic_beta=True`` (per-bunch effective sizes
   recomputed from the live betas each iteration; the element state carries
   over between the two calls, so this continues the same iteration).
2. Transfer the converged solution back to the FULL THICK lattice of beam 1:
   the same multi-bunch beam-beam lenses are installed at the (still
   present) encounter markers and loaded with the final per-bunch orbits
   and dynamic sizes of beam 2, plus the (bunch-averaged) dynamic own sizes
   of beam 1.
3. Generate tune footprints on this thick lattice for 12 equidistant
   bunches along the longest train (first and last included), covering the
   PACMAN transition from the train head through the fully-surrounded
   center to the tail. Each footprint tracks particles with ``zeta`` frozen at the
   bunch's slot label, so the multi-bunch lenses apply that bunch's actual
   encounters (head-on + long-range, around its own closed orbit). Before
   tracking, the lenses are switched to ``coherent=False``: the footprint
   particles are INDIVIDUAL protons that see the field of the opposing
   bunches with their own sizes (incoherent, weak-strong), while the closed
   solution was obtained with the coherent rigid-bunch (convolved-size)
   kicks. The footprints use the linear rescale on the ``beambeam_scale``
   knob (bound to the ``scale_strength`` of all lenses at installation, as
   in the xfields beam-beam config tools): computed at weak beam-beam and
   extrapolated linearly to full strength, which avoids resonance-distorted
   footprints. The footprints are computed on THREE variants of the machine
   (same converged lenses) and plotted side by side with the execution
   times compared:

   - the full thick lattice;
   - a second-order-map line that keeps the lattice octupoles (MO) as
     exact thick elements between the maps, retaining most of the lattice
     amplitude detuning (which is 3rd order, hence absent from the maps
     themselves) at a fraction of the thick tracking cost;
   - the plain second-order-map line, whose arcs carry NO amplitude
     detuning -- its footprint isolates the beam-beam contribution and is
     the cheapest to track.
"""

import os
import re
import time
import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt
import lhc_mb_common as mb

# multi-threaded CPU kernels by default in this demo
os.environ.setdefault('LHC_OMP', 'auto')
env, line_b1, line_b2, par = mb.load_lhc('collision')

# ----------------------------------------------------------------------------
# Build the machine (as in 002/004): install + geometry on the full lattice
# ----------------------------------------------------------------------------
scheme_b1, scheme_b2 = mb.load_scheme()
setup = env.xfields.install_multibunch_beambeam(
    clockwise_line='lhcb1', anticlockwise_line='lhcb2', ips=par['ips'],
    num_long_range_encounters_per_side=par['nparasitic'],
    harmonic_number=mb.HARMONIC_NUMBER,
    bunch_spacing_buckets=mb.BUNCH_SPACING_BUCKETS,
    nemitt_x=par['nemitt'], nemitt_y=par['nemitt'],
    filling_clockwise=mb.filling_from_scheme(scheme_b1, par['bunch_intensity']),
    filling_anticlockwise=mb.filling_from_scheme(scheme_b2, par['bunch_intensity']))

# lattice octupoles (MO), kept exact in the third footprint variant to
# retain the lattice amplitude detuning (split elements are excluded from
# the maps, so splitting at them preserves them exactly)
mo_names = [nn for nn in line_b1.element_names if re.match(r'^mo\.', nn)]

print('  building second-order maps between the beam-beam elements...')
setup_red = setup.second_order_maps(context=par['context'])
print(f'  ... and a variant keeping the {len(mo_names)} lattice octupoles '
      'of beam 1 exact...')
setup_mo = setup.second_order_maps(keep_extra_cw=mo_names,
                                   context=par['context'])
# beam-1 lines and lens handles used for the footprints below
red_b1, red_mo_b1 = setup_red.cw_line, setup_mo.cw_line
bb_b1, bb_thick_b1, bb_mo_b1 = setup_red.bb_cw, setup.bb_cw, setup_mo.bb_cw
slots_b1, slots_b2 = setup_red.bunches_cw, setup_red.bunches_acw
print(f'  populated bunches: B1 = {len(slots_b1)}, B2 = {len(slots_b2)}')

# ----------------------------------------------------------------------------
# 1) Self-consistent solve on the fast sector-map model: 4 iterations
#    orbit-only, 4 with dynamic beta
# ----------------------------------------------------------------------------
print('Self-consistent solve (4 iterations fast_orbit):')
t0 = time.time()
setup_red.solve(max_iterations=4)
print('Self-consistent solve (4 more iterations with dynamic beta):')
mbtw_b1, mbtw_b2 = setup_red.solve(
    max_iterations=4, dynamic_beta=True)
print(f'  total solve time: {time.time() - t0:.1f} s')

# ----------------------------------------------------------------------------
# 2) Transfer the converged solution to the full thick lattice of beam 1 and to
#    the maps+MO line (their beam-beam lenses are loaded with the converged
#    per-bunch orbits and dynamic sizes)
# ----------------------------------------------------------------------------
print('Loading the converged solution on the full thick lattice (B1) and '
      'on the maps+MO line...')
setup.load_solution(mbtw_b1, mbtw_b2, dynamic_beta=True)
setup_mo.load_solution(mbtw_b1, mbtw_b2, dynamic_beta=True)

# ----------------------------------------------------------------------------
# 3) Footprints for the bunch families of the longest train
# ----------------------------------------------------------------------------
# longest contiguous filled run of beam 1
filled = np.asarray(scheme_b1) > 0
best_len = best_start = cur_len = cur_start = 0
for s in range(mb.N_SLOTS):
    if filled[s]:
        cur_start = s if cur_len == 0 else cur_start
        cur_len += 1
        if cur_len > best_len:
            best_len, best_start = cur_len, cur_start
    else:
        cur_len = 0
train = list(range(best_start, best_start + best_len))
print(f'  longest train: {best_len} bunches starting at slot {best_start}')

# 12 equidistant bunches along the train (first and last included)
pos_in_train = np.round(np.linspace(0, len(train) - 1, 12)).astype(int)
family = [(train[k], f'train bunch {k + 1}') for k in pos_in_train]

# cross-check the transferred lattices on the family bunches
zeta_fam = np.array([sl for sl, _ in family]) * setup.slot_len
idx_fam = np.searchsorted(slots_b1, [sl for sl, _ in family])
for label, transferred in (('thick', line_b1), ('maps+MO', red_mo_b1)):
    mb_check = transferred.twiss_multibunch(
        zeta_bunches=zeta_fam, mode='fast_orbit', show_progress=False)
    dq_check = mb.wrap_frac_tune(
        mb_check.qx - np.asarray(mbtw_b1.qx)[idx_fam])
    print(f'  transfer check ({label} vs sector maps, family bunches): '
          f'max |dqx| = {np.max(np.abs(dq_check)):.2e}')

# incoherent footprints: individual particles see the field of the opposing
# bunches with their OWN sizes (weak-strong), not the convolved coherent
# kick used for the rigid-bunch closed solution. NOTE: switched only AFTER
# the transfer check above, which must reproduce the (coherent) sector-map
# solution.
for bb in (list(bb_thick_b1.values()) + list(bb_mo_b1.values())
           + list(bb_b1.values())):
    bb.coherent = False

# Footprints on the full thick lattice, on the maps+MO line (exact
# octupoles between the maps -> keeps most of the lattice amplitude
# detuning) and on the plain second-order-map line (2nd-order Taylor arcs
# carry NO amplitude detuning -- its footprint contains only the (exact)
# beam-beam nonlinearity), all with the converged incoherent lenses.
LINES = (('thick', line_b1), ('maps+MO', red_mo_b1), ('maps', red_b1))
SUFFIX = {'thick': '', 'maps+MO': '_mapmo', 'maps': '_map'}
print('Footprints (thick lattice vs maps + exact MO vs plain maps):')
footprints = {}
timing = {tag: 0.0 for tag, _ in LINES}
for sl, label in family:
    fp = {}
    for tag, line in LINES:
        t0 = time.time()
        # linear rescale on the beam-beam strength (as in the xmask
        # footprint example): the footprint is computed at weak beam-beam
        # (beambeam_scale = v0 and v0 + dv) and extrapolated linearly to
        # the actual strength, avoiding resonance-distorted footprints.
        fp[tag] = line.get_footprint(
            nemitt_x=par['nemitt'], nemitt_y=par['nemitt'],
            r_range=(0.3, 6), theta_range=(0.15, np.pi / 2 - 0.15),
            freeze_longitudinal=True, zeta0=sl * setup.slot_len,
            linear_rescale_on_knobs=[xt.LinearRescale(
                knob_name='beambeam_scale', v0=0.0, dv=0.1)])
        fp[tag + '_t'] = time.time() - t0
        timing[tag] += fp[tag + '_t']
    footprints[sl] = dict(label=label)
    for tag, _ in LINES:
        footprints[sl]['qx' + SUFFIX[tag]] = fp[tag].qx
        footprints[sl]['qy' + SUFFIX[tag]] = fp[tag].qy
    print(f'  slot {sl:4d} ({label:14s}): '
          + ' / '.join(f'{tag} {fp[tag + "_t"]:5.1f} s' for tag, _ in LINES)
          + f'   qx [{fp["thick"].qx.min():.4f}, {fp["thick"].qx.max():.4f}]')
print('footprint time: '
      + ', '.join(f'{tag} {timing[tag]:.0f} s' for tag, _ in LINES)
      + '  -> speed-up vs thick: '
      + ', '.join(f'{tag} x{timing["thick"] / timing[tag]:.1f}'
                  for tag, _ in LINES[1:]))

import pandas as pd
pd.to_pickle(footprints, os.path.join(mb.HERE, 'footprints_coll.pkl'))
print('saved footprints_coll.pkl')

# ----------------------------------------------------------------------------
# Plot: footprints colored by the bunch position in the train
# ----------------------------------------------------------------------------
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

fig, axes = plt.subplots(
    1, 3, figsize=(18, 6.5), sharex=True, sharey=True)
cmap = plt.get_cmap('viridis')
norm = Normalize(vmin=1, vmax=len(train))
for ax, suffix, title in ((axes[0], '', 'full thick lattice'),
                          (axes[1], '_mapmo', 'second-order maps\n'
                           '+ exact MO octupoles'),
                          (axes[2], '_map', 'second-order maps\n'
                           '(no lattice/octupole detuning)')):
    for k, (sl, _) in zip(pos_in_train, family):
        fp = footprints[sl]
        color = cmap(norm(k + 1))
        ax.plot(fp['qx' + suffix], fp['qy' + suffix], color=color, lw=1)
        ax.plot(fp['qx' + suffix].T, fp['qy' + suffix].T, color=color, lw=1)
    ax.set_xlabel(r'$q_x$')
    ax.set_title(title)
axes[0].set_ylabel(r'$q_y$')
fig.suptitle('Per-bunch tune footprints, LHC collision (head-on + BBLR)')
fig.tight_layout()
fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=list(axes),
             label='bunch position in train')
plt.show()
