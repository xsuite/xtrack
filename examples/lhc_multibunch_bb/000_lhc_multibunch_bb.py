# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
Multi-bunch beam-beam on the FULL (thick) LHC lattice in collisions (6.8 TeV,
fully squeezed R2025aRP 15 cm flat optics, end-of-levelling knobs).

Head-on and long-range beam-beam elements (BeamBeamBiGaussianMultibunch2D) are
installed at IP1/2/5/8 with ``env.xfields.install_multibunch_beambeam`` (see
``xtrack.multibunch_beambeam``), and the per-bunch closed solution (closed orbit
+ tunes) of the two multi-bunch beams is found self-consistently with
``setup.solve()``. The per-IP head-on bunch-pairing offsets are derived from the
ring geometry (the IPs are passed as a list of names); the beams collide head-on
at IP1/IP5 (levelling offsets at IP2/IP8), so the effect is head-on + BBLR.

This "direct" variant twisses the full thick line (no sector-map reduction).
The companion example ``002_multibunch_sectormaps_collisions.py`` replaces the
arcs by second-order maps (much faster) with ``setup.second_order_maps()``.
"""
import os
import time

import matplotlib.pyplot as plt

import lhc_mb_common as mb

N_ITER = int(os.environ.get('LHC_NITER', '3'))
ALL_BUNCHES = os.environ.get('LHC_ALL', '0') == '1'  # False -> bounded subset
WINDOW = int(os.environ.get('LHC_WINDOW', '48'))

# ----------------------------------------------------------------------------
env, line_b1, line_b2, par = mb.load_lhc('collision')
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
    # restrict to a bounded window (the pairing offsets are now known from the
    # geometry, so we can pick the colliding sub-set) to keep the thick-lattice
    # solve fast
    s1, s2 = mb.windowed_slots(setup.ip_offsets, scheme_b1, scheme_b2, WINDOW)
    setup.set_filling(mb.filling_from_slots(s1, par['bunch_intensity']),
                      mb.filling_from_slots(s2, par['bunch_intensity']))

slots_b1, slots_b2 = setup.bunches_cw, setup.bunches_acw
for ip in par['ips']:
    print(f'  {ip}: head-on offset = {setup.geom[f"bb_{ip}_ho"]["offset"]} slots')
print(f'  populated bunches: B1 = {len(slots_b1)}, B2 = {len(slots_b2)}')

print('Self-consistent solve on the full thick lattice:')
t0 = time.time()
mbtw_b1, mbtw_b2 = setup.solve(max_iterations=N_ITER)
print(f'  solve time ({len(slots_b1)}+{len(slots_b2)} bunches): '
      f'{time.time() - t0:.1f} s')

bare = setup.meta
dqx_b1 = mb.wrap_frac_tune(mbtw_b1.qx - bare['qx_cw'])
print(f"\nB1 tune shift: dqx in [{dqx_b1.min():.2e}, {dqx_b1.max():.2e}]")

df_b1 = mb.results_dataframe(setup, mbtw_b1, slots_b1,
                             bare['qx_cw'], bare['qy_cw'], mirror=False)
df_b2 = mb.results_dataframe(setup, mbtw_b2, slots_b2,
                             bare['qx_acw'], bare['qy_acw'], mirror=True)
out_b1 = os.path.join(mb.HERE, 'results_b1_coll_full.pkl')
out_b2 = os.path.join(mb.HERE, 'results_b2_coll_full.pkl')
df_b1.to_pickle(out_b1)
df_b2.to_pickle(out_b2)
print(f'saved {out_b1}\nsaved {out_b2}')

mb.plot_results(setup, slots_b1, mbtw_b1, bare['qx_cw'], bare['qy_cw'],
                title_suffix='  [full thick lattice, collision]')
plt.show()
