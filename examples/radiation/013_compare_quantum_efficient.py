# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import matplotlib.pyplot as plt

import xobjects as xo
import xpart as xp
import xtrack as xt


context = xo.ContextCpu()

n_particles = 2_000_000
p0c = 182.5e9
length = 22.653765579198428
angle = 0.0022799344662676477

particles0 = xp.Particles(
    _context=context,
    p0c=p0c,
    mass0=xp.ELECTRON_MASS_EV,
    q0=-1,
    x=np.zeros(n_particles),
    px=1e-4 * np.ones(n_particles),
    y=np.zeros(n_particles),
    py=-1e-4 * np.ones(n_particles),
    delta=np.zeros(n_particles),
)

modes = {
    "quantum": 2,
    "quantum-efficient": 3,
}

results = {}

for mode, radiation_flag in modes.items():
    bend = xt.Bend(
        _context=context,
        length=length,
        angle=angle,
        k0_from_h=True,
        radiation_flag=radiation_flag,
    )

    particles = particles0.copy()
    if radiation_flag in (2, 3):
        particles._init_random_number_generator()

    before_px = particles.px.copy()
    before_py = particles.py.copy()
    before_delta = particles.delta.copy()

    bend.track(particles)

    results[mode] = {
        "dpx": context.nparray_from_context_array(particles.px - before_px),
        "dpy": context.nparray_from_context_array(particles.py - before_py),
        "ddelta": context.nparray_from_context_array(
            particles.delta - before_delta),
    }

    print(
        f"{mode:17s}"
        f" <dpx>={np.mean(results[mode]['dpx']): .6e}"
        f" rms(dpx)={np.std(results[mode]['dpx']): .6e}"
        f" <dpy>={np.mean(results[mode]['dpy']): .6e}"
        f" rms(dpy)={np.std(results[mode]['dpy']): .6e}"
        f" <ddelta>={np.mean(results[mode]['ddelta']): .6e}"
        f" rms(ddelta)={np.std(results[mode]['ddelta']): .6e}"
    )


plt.close("all")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig_cdf, axes_cdf = plt.subplots(1, 3, figsize=(14, 4.5))

plot_specs = [
    ("dpx", r"$\Delta p_x$"),
    ("dpy", r"$\Delta p_y$"),
    ("ddelta", r"$\Delta\delta$"),
]

for ax, ax_cdf, (key, label) in zip(axes, axes_cdf, plot_specs):
    all_values = np.concatenate([results[mode][key] for mode in modes])
    lo, hi = np.quantile(all_values, [1e-5, 1 - 1e-5])
    bins = np.linspace(lo, hi, 300)

    for mode in modes:
        ax.hist(
            results[mode][key],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=mode,
        )
        hist, bin_edges = np.histogram(results[mode][key], bins=bins)
        cdf = np.cumsum(hist) / np.sum(hist)
        ax_cdf.step(
            bin_edges[1:],
            cdf,
            where="post",
            linewidth=1.5,
            label=mode,
        )

    ax.set_xlabel(label)
    ax.set_ylabel("density")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()

    ax_cdf.set_xlabel(label)
    ax_cdf.set_ylabel("CDF")
    ax_cdf.grid(True)
    ax_cdf.legend()

fig.suptitle(
    "Synchrotron radiation kick distributions in one FCC-ee tt-like bend")
fig_cdf.suptitle(
    "Synchrotron radiation kick CDFs in one FCC-ee tt-like bend")
fig.tight_layout()
fig_cdf.tight_layout()
plt.show()
