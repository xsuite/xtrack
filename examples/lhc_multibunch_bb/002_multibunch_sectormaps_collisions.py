# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
Multi-bunch beam-beam on the LHC in COLLISION (6.8 TeV, fully squeezed
R2025aRP 15 cm flat optics, levelling-style knobs), with second-order maps.

Scenario following LHC 2025/2026 physics at end of levelling: head-on collisions
at IP1/IP5 (flat optics, H crossing in 1, V crossing in 5, separations off),
levelling offsets at IP2/IP8, spectrometers/solenoids on, octupoles powered,
tunes/chromaticity matched to 62.316/60.322 and Q' = 10. 1.1e11 p/bunch,
2.3 um normalized emittance, full 2460-bunch filling scheme.

``env.xfields.install_multibunch_beambeam`` installs the head-on + long-range
lenses and computes the geometry on the full lattice; ``setup.second_order_maps``
then makes a fast copy where the arcs between the encounters are replaced by
second-order Taylor maps (the lenses stay exact), and ``setup_red.solve`` finds
the per-bunch self-consistent closed solution on it. The flattened collision
optics and the pytrain reference are in ``test_data/lhc_2024`` (regenerate with
``test_data/lhc_2024/pytrain/regenerate_collision.py``).
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import lhc_mb_common as mb

N_ITER = int(os.environ.get('LHC_NITER', '6'))
ALL_BUNCHES = os.environ.get('LHC_ALL', '1') == '1'
WINDOW = int(os.environ.get('LHC_WINDOW', '48'))
COMPUTE_OPTICS_PARAMS = os.environ.get('COMPUTE_OPTICS_PARAMS', '1') == '1'

# ----------------------------------------------------------------------------
env, line_b1, line_b2, par = mb.load_lhc('collision')
scheme_b1, scheme_b2 = mb.load_scheme()

# Install the head-on + long-range lenses and compute the geometry on the full
# thick lattice.
setup = env.xfields.install_multibunch_beambeam(
    clockwise_line='lhcb1', anticlockwise_line='lhcb2', ips=par['ips'],
    num_long_range_encounters_per_side=par['nparasitic'],
    harmonic_number=mb.HARMONIC_NUMBER,
    bunch_spacing_buckets=mb.BUNCH_SPACING_BUCKETS,
    nemitt_x=par['nemitt'], nemitt_y=par['nemitt'],
    filling_clockwise=mb.filling_from_scheme(scheme_b1, par['bunch_intensity']),
    filling_anticlockwise=mb.filling_from_scheme(scheme_b2, par['bunch_intensity']))
print(f'  bare tunes B1 {setup.meta["qx_cw"]:.5f}/{setup.meta["qy_cw"]:.5f}  '
      f'B2 {setup.meta["qx_acw"]:.5f}/{setup.meta["qy_acw"]:.5f}')

if not ALL_BUNCHES:
    # restrict to a bounded window with all-IP pairings (offsets from geometry)
    s1, s2 = mb.windowed_slots(setup.ip_offsets, scheme_b1, scheme_b2, WINDOW)
    setup.set_filling(mb.filling_from_slots(s1, par['bunch_intensity']),
                      mb.filling_from_slots(s2, par['bunch_intensity']))

# Fast sector-map copy: the arcs between the encounters become second-order maps
# (the beam-beam elements stay exact). Solving the reduced setup is much faster.
print('  building second-order maps between the beam-beam elements...')
setup_red = setup.second_order_maps(context=par['context'])
slots_b1, slots_b2 = setup_red.bunches_cw, setup_red.bunches_acw
print(f'  populated bunches: B1 = {len(slots_b1)}, B2 = {len(slots_b2)}')

print('Self-consistent solve (head-on + long-range):')
t0 = time.time()
mbtw_b1, mbtw_b2 = setup_red.solve(max_iterations=N_ITER)
print(f'  solve time ({len(slots_b1)}+{len(slots_b2)} bunches, {N_ITER} iters): '
      f'{time.time() - t0:.1f} s')

if COMPUTE_OPTICS_PARAMS:
    print('Final mode="fast" twiss (per-bunch optics + global quantities):')
    t0 = time.time()
    mbtw_b1 = setup_red.cw_line.twiss_multibunch(
        zeta_bunches=slots_b1 * setup_red.slot_len, mode='fast')
    mbtw_b2 = setup_red.acw_line.twiss_multibunch(
        zeta_bunches=slots_b2 * setup_red.slot_len, mode='fast')
    print(f'  final twiss (both beams): {time.time() - t0:.1f} s')

# bare per-bunch tunes: second-order maps preserve the linear optics, so the
# reduced-line tunes equal the full-lattice ones in setup.meta
dqx_b1 = mb.wrap_frac_tune(mbtw_b1.qx - setup.meta['qx_cw'])
print(f"\nB1 tune shift: dqx in [{dqx_b1.min():.2e}, {dqx_b1.max():.2e}]")

df_b1 = mb.results_dataframe(setup_red, mbtw_b1, slots_b1,
                             setup.meta['qx_cw'], setup.meta['qy_cw'],
                             mirror=False)
df_b2 = mb.results_dataframe(setup_red, mbtw_b2, slots_b2,
                             setup.meta['qx_acw'], setup.meta['qy_acw'],
                             mirror=True)
df_b1.to_pickle(os.path.join(mb.HERE, 'results_b1_coll.pkl'))
df_b2.to_pickle(os.path.join(mb.HERE, 'results_b2_coll.pkl'))
print('saved results_b1_coll.pkl / results_b2_coll.pkl')

mb.plot_results(setup_red, slots_b1, mbtw_b1,
                setup.meta['qx_cw'], setup.meta['qy_cw'],
                title_suffix='  [collision, 6.8 TeV]')
if COMPUTE_OPTICS_PARAMS:
    mb.plot_global_quantities(setup_red, slots_b1, mbtw_b1, slots_b2, mbtw_b2)
plt.show()
