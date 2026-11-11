# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
Multi-bunch beam-beam on the LHC at injection, sped up with second-order maps.

Same physics as ``000_lhc_multibunch_bb.py`` (long-range beam-beam at injection
-- separation bumps kept on, so no head-on collisions -- with per-bunch
self-consistent closed solution), but the machine between the beam-beam
encounters is replaced by second-order Taylor maps:

    reduced line  =  [ map(arc), BB, map(arc), BB, ... ]

The beam-beam elements stay EXACT; only the arcs are approximated (to second
order, as in pytrain's sector maps). The reduced line has ~a few hundred
elements instead of ~13000.
"""

import os
import time
import matplotlib.pyplot as plt

import lhc_mb_common as mb

N_ITER = int(os.environ.get('LHC_NITER', '6'))
ALL_BUNCHES = os.environ.get('LHC_ALL', '1') == '1'  # False -> bounded subset
WINDOW = int(os.environ.get('LHC_WINDOW', '48'))
COMPUTE_OPTICS_PARAMS = os.environ.get('COMPUTE_OPTICS_PARAMS', '1') == '1'

env, line_b1, line_b2, par = mb.load_lhc('injection')
scheme_b1, scheme_b2 = mb.load_scheme()

setup = env.xfields.install_multibunch_beambeam(
    clockwise_line='lhcb1', anticlockwise_line='lhcb2', ips=par['ips'],
    num_long_range_encounters_per_side=par['nparasitic'],
    harmonic_number=mb.HARMONIC_NUMBER,
    bunch_spacing_buckets=mb.BUNCH_SPACING_BUCKETS,
    nemitt_x=par['nemitt'], nemitt_y=par['nemitt'],
    filling_clockwise=mb.filling_from_scheme(scheme_b1, par['bunch_intensity']),
    filling_anticlockwise=mb.filling_from_scheme(scheme_b2, par['bunch_intensity']))
if not ALL_BUNCHES:
    s1, s2 = mb.windowed_slots(setup.ip_offsets, scheme_b1, scheme_b2, WINDOW)
    setup.set_filling(mb.filling_from_slots(s1, par['bunch_intensity']),
                      mb.filling_from_slots(s2, par['bunch_intensity']))

print('  building second-order maps between the beam-beam elements...')
setup_red = setup.second_order_maps(context=par['context'])
red_b1 = setup_red.cw_line

# ----------------------------------------------------------------------------
# Timing comparison
# ----------------------------------------------------------------------------
line_b1.twiss(); red_b1.twiss()
t0 = time.time(); line_b1.twiss(); t_full = time.time() - t0
t0 = time.time(); tw_red = red_b1.twiss(); t_red = time.time() - t0
print(f'  one full-lattice twiss: {t_full:.3f} s ({len(line_b1.element_names)} elements)')
print(f'  one reduced-line twiss: {t_red:.3f} s ({len(red_b1.element_names)} elements)'
      f'  ->  speed-up x{t_full/t_red:.1f}')
# The reduced line reproduces the FRACTIONAL tunes (its integer tune is not
# meaningful -- the maps carry phase advance only modulo 2 pi).
print(f'  fractional tunes reduced/full: '
      f'qx {tw_red.qx % 1:.5f} / {setup.meta["qx_cw"] % 1:.5f}   '
      f'qy {tw_red.qy % 1:.5f} / {setup.meta["qy_cw"] % 1:.5f}')

slots_b1, slots_b2 = setup_red.bunches_cw, setup_red.bunches_acw
print(f'  populated bunches: B1 = {len(slots_b1)}, B2 = {len(slots_b2)}')

print('Self-consistent solve on the reduced (second-order-map) lines:')
t0 = time.time()
mbtw_b1, mbtw_b2 = setup_red.solve(max_iterations=N_ITER)
print(f'  solve time ({len(slots_b1)}+{len(slots_b2)} bunches, {N_ITER} iters): {time.time() - t0:.1f} s')

# Optional: per-bunch optics (dynamic beta*) and global quantities (tunes, chromaticity, coupling)
if COMPUTE_OPTICS_PARAMS:
    print('Final mode="fast" twiss (per-bunch optics + global quantities):')
    t0 = time.time()
    mbtw_b1 = setup_red.cw_line.twiss_multibunch(
        zeta_bunches=slots_b1 * setup_red.slot_len, mode='fast')
    mbtw_b2 = setup_red.acw_line.twiss_multibunch(
        zeta_bunches=slots_b2 * setup_red.slot_len, mode='fast')
    print(f'  final twiss (both beams): {time.time() - t0:.1f} s')

# Reference the tune shift to the bare tune (second-order maps preserve the
# linear optics, so the reduced-line tunes equal the full-lattice setup.meta)
dqx_b1 = mb.wrap_frac_tune(mbtw_b1.qx - setup.meta['qx_cw'])
print(f"\nB1 tune shift: dqx in [{dqx_b1.min():.2e}, {dqx_b1.max():.2e}]")

# Save per-bunch results of both beams as DataFrames
df_b1 = mb.results_dataframe(setup_red, mbtw_b1, slots_b1,
                             setup.meta['qx_cw'], setup.meta['qy_cw'], mirror=False)
df_b2 = mb.results_dataframe(setup_red, mbtw_b2, slots_b2,
                             setup.meta['qx_acw'], setup.meta['qy_acw'], mirror=True)
out_b1 = os.path.join(mb.HERE, 'results_b1.pkl')
out_b2 = os.path.join(mb.HERE, 'results_b2.pkl')
df_b1.to_pickle(out_b1)
df_b2.to_pickle(out_b2)
print(f'saved {out_b1}\nsaved {out_b2}')

mb.plot_results(setup_red, slots_b1, mbtw_b1,
                setup.meta['qx_cw'], setup.meta['qy_cw'],
                title_suffix='  [second-order maps]')
if COMPUTE_OPTICS_PARAMS:
    mb.plot_global_quantities(setup_red, slots_b1, mbtw_b1, slots_b2, mbtw_b2)
plt.show()
