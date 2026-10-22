# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
Plot the per-bunch beam-beam results of both beams for BOTH scenarios:

* injection (BBLR only), saved by ``001_multibunch_sectormaps_injection.py``
  as ``results_b{1,2}.pkl``;
* collision (6.8 TeV squeezed, head-on + BBLR), saved by
  ``002_multibunch_sectormaps_collisions.py`` as ``results_b{1,2}_coll.pkl``.

One figure per scenario. Top row: per-bunch tune offsets (beam-beam tune
shift) dqx, dqy for B1 and B2. Bottom row: per-bunch closed-orbit offsets
(deviation from the beam-averaged orbit, which removes the common
crossing/separation bump) dx, dy at IP1.

The pytrain references from
``test_data/lhc_2024/pytrain/pytrain_{injection,collision}.json`` (the same
data the regression test ``test_lhc_multibunch_train.py`` checks against) are
overlaid. Scenarios whose xsuite result files are missing are skipped.
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PYTRAIN_DIR = os.path.join(HERE, '..', '..', 'test_data', 'lhc_2024',
                           'pytrain')

SCENARIOS = [
    # (title, xsuite suffix, scenario key, tune-panel ylim or None)
    ('LHC injection multi-bunch beam-beam (BBLR)', '', 'injection', (-4, 4)),
    ('LHC collision 6.8 TeV multi-bunch beam-beam (head-on + BBLR)',
     '_coll', 'collision', None),
]


def load(name):
    path = os.path.join(HERE, name)
    return pd.read_pickle(path) if os.path.exists(path) else None


def load_pytrain(scenario):
    """pytrain reference from the test-data JSON, as per-beam DataFrames with
    the same columns/index as the xsuite result pickles."""
    path = os.path.join(PYTRAIN_DIR, f'pytrain_{scenario}.json')
    if not os.path.exists(path):
        return None, None
    with open(path) as fid:
        ref = json.load(fid)

    def beam_df(beam):
        dd = ref[beam]
        return pd.DataFrame(
            {kk: dd[kk] for kk in
             ('qx', 'qy', 'dqx', 'dqy', 'x', 'y', 'dx', 'dy')},
            index=pd.Index(dd['slots'], name='slot'))

    return beam_df('b1'), beam_df('b2')


def plot_scenario(title, df_b1, df_b2, pt_b1, pt_b2, tune_ylim):
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), sharex=True)

    def panel(ax, col, scale, ylabel, panel_title):
        if pt_b1 is not None:
            bb = np.full((3564,), np.nan)
            bb[pt_b1.index] = pt_b1[col] * scale
            ax.plot(bb, '-', ms=3, color='C2', alpha=1, label='B1 pytrain')
            bb = np.full((3564,), np.nan)
            bb[pt_b2.index] = pt_b2[col] * scale
            ax.plot(bb, '-', ms=3, color='C3', alpha=1, label='B2 pytrain')
        ax.plot(df_b1.index, df_b1[col] * scale, '.', ms=3, color='C0',
                label='B1 xsuite')
        ax.plot(df_b2.index, df_b2[col] * scale, '.', ms=3, color='C1',
                label='B2 xsuite')
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        ax.legend(ncol=2, fontsize=8)

    panel(axs[0, 0], 'dqx', 1e3, r'$\Delta q_x$ [$10^{-3}$]',
          r'Per-bunch beam-beam tune offset $\Delta q_x$')
    panel(axs[0, 1], 'dqy', 1e3, r'$\Delta q_y$ [$10^{-3}$]',
          r'Per-bunch beam-beam tune offset $\Delta q_y$')
    panel(axs[1, 0], 'dx', 1e6, r'orbit offset x at IP1 [$\mu$m]',
          'Per-bunch beam-beam closed-orbit offset (x)')
    panel(axs[1, 1], 'dy', 1e6, r'orbit offset y at IP1 [$\mu$m]',
          'Per-bunch beam-beam closed-orbit offset (y)')
    if pt_b1 is not None and tune_ylim is not None:
        # pytrain tune tails can be large (unconverged in slicing)
        axs[0, 0].set_ylim(*tune_ylim)
        axs[0, 1].set_ylim(*tune_ylim)
    axs[1, 0].set_xlabel('25 ns slot')
    axs[1, 1].set_xlabel('25 ns slot')

    plt.suptitle(title + ': xsuite vs pytrain' if pt_b1 is not None else title)
    plt.tight_layout()
    return fig


for title, xs_suffix, scenario, tune_ylim in SCENARIOS:
    df_b1 = load(f'results_b1{xs_suffix}.pkl')
    df_b2 = load(f'results_b2{xs_suffix}.pkl')
    if df_b1 is None or df_b2 is None:
        print(f'skipping "{title}" (results_b*{xs_suffix}.pkl not found)')
        continue
    pt_b1, pt_b2 = load_pytrain(scenario)
    plot_scenario(title, df_b1, df_b2, pt_b1, pt_b2, tune_ylim)

plt.show()
