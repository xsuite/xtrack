# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
Impact of DYNAMIC BETA on the multi-bunch beam-beam closed solution in the
LHC collision scenario (6.8 TeV squeezed flat optics, head-on + BBLR).

The beam-beam elements take the effective (convolved) transverse sizes of
the colliding bunch pairs. By default these are STATIC, computed once from
the bare optics: ``sigma^2 = (beta_b1 + beta_b2) * nemitt / gamma0``. But
head-on beam-beam changes the per-bunch beta functions at the encounters
(dynamic beta, ~10% spread of beta* in this scenario), which in turn changes
the sizes, the kicks, and hence the per-bunch closed solution.

This script solves the same machine twice:

1. static sizes (as ``002_multibunch_sectormaps_collisions.py``);
2. ``dynamic_beta=True``: at every iteration the per-bunch effective sizes
   of all encounters are recomputed from the LIVE per-bunch betas of both
   beams (requires the optics-carrying mode='fast' twiss in the loop).

and compares per-bunch tunes, orbits and beta* at IP1.

By default the MULTI-THREADED CPU kernels (OpenMP) are used (set
``LHC_OMP=0`` for serial, ``LHC_OMP=<n>`` for a specific thread count; see
``005_multibunch_openmp.py`` for the speed-up measurement).
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

import lhc_mb_common as mb

N_ITER = int(os.environ.get('LHC_NITER', '6'))

# multi-threaded CPU kernels by default in this demo
os.environ.setdefault('LHC_OMP', 'auto')
env, line_b1, line_b2, par = mb.load_lhc('collision')

# ----------------------------------------------------------------------------
# Build the machine (as in 002): install + geometry on the full lattice, then
# a fast second-order-map copy
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
print('  building second-order maps between the beam-beam elements...')
setup_red = setup.second_order_maps(context=par['context'])
slots_b1, slots_b2 = setup_red.bunches_cw, setup_red.bunches_acw
print(f'  populated bunches: B1 = {len(slots_b1)}, B2 = {len(slots_b2)}')

# ----------------------------------------------------------------------------
# Solve with static sizes, then with dynamic beta
# ----------------------------------------------------------------------------
results = {}
for label, dynamic_beta in (('static', False), ('dynamic beta', True)):
    print(f'Self-consistent solve ({label}):')
    t0 = time.time()
    # 'fast' twiss (per-bunch optics) in both cases so the returned tables carry
    # betx/bety for the static-vs-dynamic beta* comparison below (dynamic_beta
    # forces it anyway; the static solve would otherwise default to fast_orbit).
    mbtw_b1, mbtw_b2 = setup_red.solve(
        max_iterations=N_ITER, tol_sigma=0.0,
        twiss_mode='fast', dynamic_beta=dynamic_beta)
    print(f'  solve time ({N_ITER} iters): {time.time() - t0:.1f} s')
    results[label] = (mbtw_b1, mbtw_b2)

# ----------------------------------------------------------------------------
# Compare per-bunch tunes, orbit and beta* at IP1 (B1)
# ----------------------------------------------------------------------------
mk_b1 = setup_red.bb_name('bb_ip1_ho', False)


def extract(mbtw):
    return dict(
        qx=np.asarray(mbtw.qx_frac), qy=np.asarray(mbtw.qy_frac),
        x=mbtw['x', mk_b1],
        betx=mbtw['betx', mk_b1],
        bety=mbtw['bety', mk_b1],
    )


stat = extract(results['static'][0])
dyn = extract(results['dynamic beta'][0])

df_b1 = mb.results_dataframe(setup_red, results['dynamic beta'][0], slots_b1,
                             setup.meta['qx_cw'], setup.meta['qy_cw'],
                             mirror=False)
df_b2 = mb.results_dataframe(setup_red, results['dynamic beta'][1], slots_b2,
                             setup.meta['qx_acw'], setup.meta['qy_acw'],
                             mirror=True)
df_b1.to_pickle(os.path.join(mb.HERE, 'results_b1_coll_dynbeta.pkl'))
df_b2.to_pickle(os.path.join(mb.HERE, 'results_b2_coll_dynbeta.pkl'))
print('saved results_b1_coll_dynbeta.pkl / results_b2_coll_dynbeta.pkl')

dqx = mb.wrap_frac_tune(dyn['qx'] - stat['qx'])
dqy = mb.wrap_frac_tune(dyn['qy'] - stat['qy'])
print(f"\nDynamic-beta impact (B1):")
print(f"  tune change dqx in [{dqx.min():+.2e}, {dqx.max():+.2e}] "
      f"(rms {dqx.std():.2e})")
print(f"  tune change dqy in [{dqy.min():+.2e}, {dqy.max():+.2e}] "
      f"(rms {dqy.std():.2e})")
print(f"  orbit change at IP1: rms {np.std(dyn['x'] - stat['x'])*1e9:.1f} nm")
print(f"  betx* at IP1: static [{stat['betx'].min():.4f}, "
      f"{stat['betx'].max():.4f}] m, dynamic [{dyn['betx'].min():.4f}, "
      f"{dyn['betx'].max():.4f}] m")

fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

axs[0].plot(slots_b1, dqx * 1e3, '.', ms=3, label=r'$\Delta q_x$')
axs[0].plot(slots_b1, dqy * 1e3, '.', ms=3, label=r'$\Delta q_y$')
axs[0].set_ylabel(r'tune change [$10^{-3}$]')
axs[0].set_title('Per-bunch effect of dynamic beta on the closed solution '
                 '(B1, collision)')
axs[0].legend()

axs[1].plot(slots_b1, (dyn['x'] - stat['x']) * 1e9, '.', ms=3, label='x')
axs[1].set_ylabel('orbit change at IP1 [nm]')
axs[1].legend()

axs[2].plot(slots_b1, stat['betx'], '.', ms=3, label=r'$\beta_x^*$ static')
axs[2].plot(slots_b1, dyn['betx'], '.', ms=3,
            label=r'$\beta_x^*$ dynamic beta')
axs[2].plot(slots_b1, stat['bety'], '.', ms=3, label=r'$\beta_y^*$ static')
axs[2].plot(slots_b1, dyn['bety'], '.', ms=3,
            label=r'$\beta_y^*$ dynamic beta')
axs[2].set_ylabel(r'$\beta^*$ at IP1 [m]')
axs[2].set_xlabel('25 ns slot')
axs[2].legend(ncol=2, fontsize=8)

plt.tight_layout()
plt.show()
