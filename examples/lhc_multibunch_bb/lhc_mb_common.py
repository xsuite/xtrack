# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
LHC-specific glue for the multi-bunch beam-beam examples.

The multi-bunch beam-beam machinery itself is machine-independent and lives in
:mod:`xtrack.multibunch_beambeam`. The single entry point
``env.xfields.install_multibunch_beambeam(...)`` returns a ``MultibunchBBSetup``;
all further operations are methods on it (``setup.solve()``,
``setup.second_order_maps()``, ``setup.load_solution(...)``,
``setup.set_filling(...)``). The examples call those directly; this module only
holds the LHC-specific bits the generic tools cannot know about:

* :func:`load_lhc` and the ``SCENARIOS`` presets (sequence, optics files, beam
  parameters for injection / collision);
* the filling-scheme helpers (:func:`load_scheme`, :func:`all_filled_slots`,
  :func:`filling_from_scheme` / :func:`filling_from_slots`, :func:`windowed_slots`
  -- which reuses the ``setup.ip_offsets`` the tools derive from the geometry);
* the DataFrame / plotting utilities (:func:`results_dataframe`,
  :func:`plot_results`, :func:`plot_global_quantities`).

Model (following pytrain / TRAIN): head-on and long-range 2D beam-beam elements
(``xfields.BeamBeamBiGaussianMultibunch2D``) at IP1/2/5/8; the per-IP head-on
bunch-pairing offsets are derived from the ring geometry
(``round(2 * (s_ip - s_ip1) / slot_len)`` -> 0 at IP1/IP5, ~891 at IP2, ~2670
at IP8); coherent (rigid-bunch) convolved-size kicks; beam separation = live
closed-orbit difference + geometric survey separation of the two rings.
"""

import os
import json
import numpy as np

import xobjects as xo
import xtrack as xt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', '..', 'test_data', 'lhc_2024')
SCHEME_FILE = os.path.join(DATA, '25ns_2460b_2448_2092_2239_144bpi_20inj.json')

# LHC RF / slot layout: 3564 25-ns slots (h = 35640, 10 buckets per slot)
N_SLOTS = 3564
HARMONIC_NUMBER = 35640
BUNCH_SPACING_BUCKETS = 10

SCENARIOS = {
    # 450 GeV injection: nominal injection optics, separation bumps ON (so the
    # beams do NOT collide head-on -- long-range only), 1.8e11 p/b, 1.5 um.
    'injection': dict(p0c=450e9, bunch_intensity=1.8e11, nemitt=1.5e-6,
                      optics='injection_optics.madx'),
    # 6.8 TeV collision: fully squeezed R2025aRP 15 cm flat optics with
    # end-of-levelling knobs (flattened MAD-X state in
    # collision_optics_15cm_flat_2026.madx), 1.1e11 p/b, 2.3 um, head-on at
    # IP1/5 + BBLR.
    'collision': dict(p0c=6800e9, bunch_intensity=1.1e11, nemitt=2.3e-6,
                      optics='collision_optics_15cm_flat_2026.madx'),
}


def wrap_frac_tune(v):
    """Tune difference on the fractional-tune circle, wrapped to (-0.5, 0.5]
    (fast-mode twiss returns fractional tunes while the bare reference may
    carry an integer part)."""
    return (np.asarray(v) + 0.5) % 1.0 - 0.5


# ----------------------------------------------------------------------------
# Environment-variable defaults shared by the examples
# ----------------------------------------------------------------------------
def default_context():
    """CPU context from ``LHC_OMP`` (unset/``0``/``serial`` -> serial; a thread
    count or ``auto`` -> OpenMP kernels). Prebuilt kernels exist for both."""
    omp = os.environ.get('LHC_OMP', '0')
    if omp in ('0', '', 'serial'):
        return xo.ContextCpu()
    return xo.ContextCpu(omp_num_threads=('auto' if omp == 'auto' else int(omp)))


def default_ips():
    """IP element names from ``LHC_IPS`` (default ``1,2,5,8`` -> ip1/2/5/8)."""
    return [f'ip{v.strip()}' for v in
            os.environ.get('LHC_IPS', '1,2,5,8').split(',')]


def default_nparasitic():
    """Long-range encounters per IP side from ``LHC_NPAR`` (default 45)."""
    return int(os.environ.get('LHC_NPAR', '45'))


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_lhc(scenario=None, *, p0c=None, bunch_intensity=None, nemitt=None,
             optics_file=None, ips=None, nparasitic=None, context=None):
    """Load both LHC beams (sequence + optics) for a scenario.

    Pass a preset name (``'injection'`` / ``'collision'``) or the explicit beam
    parameters (``p0c``, ``bunch_intensity``, ``nemitt``, ``optics_file``) for a
    custom configuration. ``ips`` / ``nparasitic`` / ``context`` default from
    the ``LHC_IPS`` / ``LHC_NPAR`` / ``LHC_OMP`` environment variables.

    Returns ``(env, line_b1, line_b2, par)``; ``line_b2`` is the reversed line
    and ``par`` collects the beam parameters (``p0c``, ``bunch_intensity``,
    ``nemitt``, ``gamma0``, ``ips``, ``nparasitic``, ``context``) for the install
    / solve calls.
    """
    if scenario is not None:
        sc = SCENARIOS[scenario]
        p0c, bunch_intensity, nemitt = (sc['p0c'], sc['bunch_intensity'],
                                        sc['nemitt'])
        optics_file = os.path.join(DATA, sc['optics'])
    if context is None:
        context = default_context()
    if ips is None:
        ips = default_ips()
    if nparasitic is None:
        nparasitic = default_nparasitic()

    env = xt.load(os.path.join(DATA, 'lhc.seq'), format='madx',
                  reverse_lines=['lhcb2'])
    for ln in (env.lhcb1, env.lhcb2):
        ln.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=p0c)
    env.vars.load(optics_file)
    for ln in (env.lhcb1, env.lhcb2):
        ln.twiss_default['method'] = '4d'
        ln.cycle(name_first_element='ip3', inplace=True)  # no IP at s=0
        ln.build_tracker(_context=context)
    par = dict(p0c=p0c, bunch_intensity=bunch_intensity, nemitt=nemitt,
               gamma0=float(env.lhcb1.particle_ref.gamma0[0]),
               ips=list(ips), nparasitic=nparasitic, context=context)
    return env, env.lhcb1, env.lhcb2, par


# ----------------------------------------------------------------------------
# Filling scheme
# ----------------------------------------------------------------------------
def load_scheme():
    with open(SCHEME_FILE) as fid:
        scheme = json.load(fid)
    return np.array(scheme['schemebeam1']), np.array(scheme['schemebeam2'])


def all_filled_slots(scheme_b1, scheme_b2):
    return (sorted(np.where(scheme_b1 > 0)[0].tolist()),
            sorted(np.where(scheme_b2 > 0)[0].tolist()))


def filling_from_scheme(scheme, intensity):
    """Per-slot population array from a 0/1 filling scheme (uniform
    ``intensity`` at every filled slot)."""
    return (np.asarray(scheme) > 0) * float(intensity)


def filling_from_slots(slots, intensity, n_slots=N_SLOTS):
    """Per-slot population array (length ``n_slots``) populated at ``slots``.
    ``intensity`` is a scalar or a per-slot array aligned with ``slots``."""
    filling = np.zeros(n_slots)
    filling[np.asarray(slots, dtype=int)] = intensity
    return filling


def windowed_slots(ho_offsets, scheme_b1, scheme_b2, window, n_slots=N_SLOTS):
    """A bounded subset of the filling: a reference window (the longest
    contiguous filled run of beam 1) plus the windows it collides with at every
    distinct head-on offset (so all IPs get realistic PACMAN pairings).
    ``ho_offsets`` is the ``{ip: offset}`` mapping the tools derive from the
    ring geometry, i.e. ``setup.ip_offsets`` after
    ``env.xfields.install_multibunch_beambeam``.
    """
    offsets = sorted(set(ho_offsets.values()))
    filled = scheme_b1 > 0
    best_len = best_start = cur_len = cur_start = 0
    for s in range(n_slots):
        if filled[s]:
            cur_start = s if cur_len == 0 else cur_start
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    window = min(window, best_len)
    ref_start = best_start
    cand = set()
    for o in offsets:
        for shift in (o, -o):
            for k in range(window):
                cand.add((ref_start + shift + k) % n_slots)
    return (sorted(s for s in cand if scheme_b1[s]),
            sorted(s for s in cand if scheme_b2[s]))


# ----------------------------------------------------------------------------
# Results as a DataFrame
# ----------------------------------------------------------------------------
def results_dataframe(setup, mbtw, slots, bare_qx, bare_qy, mirror=False,
                      ip='ip1'):
    """Per-bunch results as a pandas DataFrame, indexed by 25 ns slot.

    Columns: qx, qy (per-bunch tunes), dqx, dqy (beam-beam tune shift vs the
    bare tune), x, y (closed orbit at the head-on marker of ``ip``, in the
    physical frame -- for the reversed beam-2 line pass ``mirror=True`` to flip
    x). ``dx``/``dy`` are the per-bunch orbit deviations from the beam average.
    """
    import pandas as pd
    marker = setup.bb_name(f'bb_{ip}_ho', mirror)
    x = mbtw['x', marker] * (-1.0 if mirror else 1.0)
    y = mbtw['y', marker]

    df = pd.DataFrame({
        'slot': np.asarray(slots),
        'qx': mbtw.qx, 'qy': mbtw.qy,
        'dqx': wrap_frac_tune(mbtw.qx - bare_qx),
        'dqy': wrap_frac_tune(mbtw.qy - bare_qy),
        'x': x, 'y': y,
        'dx': x - x.mean(), 'dy': y - y.mean(),
    }).set_index('slot')
    return df


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def plot_results(setup, slots_b1, mbtw_b1, bare_qx, bare_qy, title_suffix=''):
    import matplotlib.pyplot as plt
    mk = setup.bb_name('bb_ip1_ho', False)
    co_x = mbtw_b1['x', mk]
    co_y = mbtw_b1['y', mk]
    # per-bunch orbit deviation from the bunch-averaged orbit (removes the common
    # crossing/separation-bump orbit, leaving the bunch-by-bunch beam-beam part)
    dco_x = (co_x - co_x.mean()) * 1e6
    dco_y = (co_y - co_y.mean()) * 1e6
    fig, axs = plt.subplots(2, 1, figsize=(9, 7))
    axs[0].plot(slots_b1, wrap_frac_tune(mbtw_b1.qx - bare_qx) * 1e3, '.',
                label=r'$\Delta q_x$')
    axs[0].plot(slots_b1, wrap_frac_tune(mbtw_b1.qy - bare_qy) * 1e3, '.',
                label=r'$\Delta q_y$')
    axs[0].set_xlabel('25 ns slot')
    axs[0].set_ylabel(r'beam-beam tune shift [$10^{-3}$]')
    axs[0].set_title('LHC: per-bunch beam-beam tune shift (B1)'
                     + title_suffix)
    axs[0].legend()
    axs[1].plot(slots_b1, dco_x, '.', label='x')
    axs[1].plot(slots_b1, dco_y, '.', label='y')
    axs[1].set_xlabel('25 ns slot')
    axs[1].set_ylabel('orbit dev. from mean at IP1 [$\\mu$m]')
    axs[1].set_title('Per-bunch beam-beam closed-orbit deviation at IP1 (B1)')
    axs[1].legend()
    plt.tight_layout()
    return fig


def plot_global_quantities(setup, slots_b1, mbtw_b1, slots_b2, mbtw_b2):
    """Bunch-by-bunch orbit at IP1, beta* at IP1, tunes, chromaticity and
    coupling |C-| of both beams, from mode='fast' MultiBunchTwiss results
    (which carry per-bunch optics and global quantities)."""
    import matplotlib.pyplot as plt
    mk = {False: setup.bb_name('bb_ip1_ho', False),
          True: setup.bb_name('bb_ip1_ho', True)}

    def at_ip1(mbtw, col, mirror):
        return mbtw[col, mk[mirror]]

    fig, axs = plt.subplots(3, 2, figsize=(13, 10), sharex=True)

    ax = axs[0, 0]   # orbit deviation at IP1 (physical frame for both beams)
    for slots, mbtw, mirror, lab in [(slots_b1, mbtw_b1, False, 'B1'),
                                     (slots_b2, mbtw_b2, True, 'B2')]:
        sgn = -1.0 if mirror else 1.0
        x = sgn * at_ip1(mbtw, 'x', mirror)
        y = at_ip1(mbtw, 'y', mirror)
        ax.plot(slots, (x - x.mean()) * 1e6, '.', ms=3, label=f'{lab} x')
        ax.plot(slots, (y - y.mean()) * 1e6, '.', ms=3, label=f'{lab} y')
    ax.set_ylabel(r'orbit dev. at IP1 [$\mu$m]')
    ax.set_title('Per-bunch closed-orbit deviation at IP1')
    ax.legend(ncol=2, fontsize=8)

    ax = axs[0, 1]   # beta* at IP1
    for slots, mbtw, mirror, lab in [(slots_b1, mbtw_b1, False, 'B1'),
                                     (slots_b2, mbtw_b2, True, 'B2')]:
        ax.plot(slots, at_ip1(mbtw, 'betx', mirror), '.', ms=3,
                label=fr'{lab} $\beta_x^*$')
        ax.plot(slots, at_ip1(mbtw, 'bety', mirror), '.', ms=3,
                label=fr'{lab} $\beta_y^*$')
    ax.set_ylabel(r'$\beta^*$ at IP1 [m]')
    ax.set_title('Per-bunch $\\beta^*$ at IP1 (dynamic beta)')
    ax.legend(ncol=2, fontsize=8)

    ax = axs[1, 0]   # fractional tunes
    for slots, mbtw, lab in [(slots_b1, mbtw_b1, 'B1'),
                             (slots_b2, mbtw_b2, 'B2')]:
        ax.plot(slots, mbtw.qx_frac, '.', ms=3, label=f'{lab} $q_x$')
        ax.plot(slots, mbtw.qy_frac, '.', ms=3, label=f'{lab} $q_y$')
    ax.set_ylabel('fractional tune')
    ax.set_title('Per-bunch tunes')
    ax.legend(ncol=2, fontsize=8)

    ax = axs[1, 1]   # chromaticity
    for slots, mbtw, lab in [(slots_b1, mbtw_b1, 'B1'),
                             (slots_b2, mbtw_b2, 'B2')]:
        ax.plot(slots, mbtw.dqx, '.', ms=3, label=f"{lab} $q'_x$")
        ax.plot(slots, mbtw.dqy, '.', ms=3, label=f"{lab} $q'_y$")
    ax.set_ylabel("chromaticity $q'$")
    ax.set_title('Per-bunch chromaticity')
    ax.legend(ncol=2, fontsize=8)

    ax = axs[2, 0]   # coupling
    for slots, mbtw, lab in [(slots_b1, mbtw_b1, 'B1'),
                             (slots_b2, mbtw_b2, 'B2')]:
        ax.plot(slots, mbtw.c_minus, '.', ms=3, label=lab)
    ax.set_xlabel('25 ns slot')
    ax.set_ylabel('$|C^-|$')
    ax.set_title('Per-bunch coupling (closest tune approach)')
    ax.legend(fontsize=8)

    axs[2, 1].axis('off')
    axs[1, 1].set_xlabel('25 ns slot')
    axs[1, 1].tick_params(labelbottom=True)
    plt.suptitle('Per-bunch optics & global quantities '
                 '(mode="fast" multibunch twiss)')
    plt.tight_layout()
    return fig
