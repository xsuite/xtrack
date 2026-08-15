# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
Multi-threaded (OpenMP) CPU kernels for the multi-bunch beam-beam machinery.

Builds the collision machine on second-order maps (as
``002_multibunch_sectormaps_collisions.py``), populates the beam-beam
elements with one self-consistent iteration, then times the batched
multibunch twiss of beam 1 -- both the orbit-only ``fast_orbit`` mode (used
in the solver loop) and the optics-carrying ``fast`` mode -- with the SERIAL
and the MULTI-THREADED CPU kernels, and reports the speed-up.

All the multibunch examples accept the environment variable ``LHC_OMP``
(unset/``0`` -> serial kernels; ``auto`` or a thread count -> OpenMP
kernels); prebuilt kernels exist for both flavours, so no compilation is
triggered either way. Here ``LHC_OMP`` defaults to ``auto``.
"""

import os
import time
import numpy as np

import xobjects as xo
import lhc_mb_common as mb

os.environ.setdefault('LHC_OMP', 'auto')
env, line_b1, line_b2, par = mb.load_lhc('collision')
ctx_mt = par['context']
n_threads = os.cpu_count() if ctx_mt.omp_num_threads == 'auto' \
    else ctx_mt.omp_num_threads
print(f'multi-threaded context: {ctx_mt.omp_num_threads} '
      f'({n_threads} threads)')

# ----------------------------------------------------------------------------
# Build the machine (as in 002) and populate the beam-beam elements
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
setup_red = setup.second_order_maps(context=ctx_mt)
red_b1 = setup_red.cw_line
slots_b1, slots_b2 = setup_red.bunches_cw, setup_red.bunches_acw
print(f'  populated bunches: B1 = {len(slots_b1)}, B2 = {len(slots_b2)}')

print('Populating the beam-beam elements (one solve iteration):')
setup_red.solve(max_iterations=1, tol_sigma=0.0)

# ----------------------------------------------------------------------------
# Timing: batched multibunch twiss of B1, serial vs multi-threaded kernels
# ----------------------------------------------------------------------------
zeta_b1 = slots_b1 * setup_red.slot_len
contexts = (('serial', xo.ContextCpu()),
            (f'openmp ({n_threads} threads)', ctx_mt))

print(f'\nBatched multibunch twiss of B1 ({len(slots_b1)} bunches):')
for mode in ('fast_orbit', 'fast'):
    timings = []
    for label, ctx in contexts:
        red_b1.discard_tracker()
        red_b1.build_tracker(_context=ctx)
        red_b1.twiss_multibunch(zeta_bunches=zeta_b1[:16], mode=mode,
                                show_progress=False)   # warm-up
        t0 = time.time()
        red_b1.twiss_multibunch(zeta_bunches=zeta_b1, mode=mode,
                                show_progress=False)
        timings.append(time.time() - t0)
        print(f'  mode={mode!r:13s} {label:22s}: {timings[-1]:7.1f} s')
    print(f'  mode={mode!r:13s} speed-up: x{timings[0] / timings[1]:.2f}')
