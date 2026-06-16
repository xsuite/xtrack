# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import os

import numpy as np
import matplotlib.pyplot as plt

import xobjects as xo
import xpart as xp
import xtrack as xt


context = xo.ContextCpu()

n_particles = int(os.environ.get("N_PARTICLES", "256_000_000"))
batch_size = int(os.environ.get("BATCH_SIZE", "500_000"))
tail_probability = 1e-10
p0c = 182.5e9
length = 22.653765579198428
angle = 0.0022799344662676477

modes = {
    "quantum": 2,
    "quantum-efficient": 3,
}

results = {}


def make_particles(n_batch, i_start):
    return xp.Particles(
        _context=context,
        p0c=p0c,
        mass0=xp.ELECTRON_MASS_EV,
        q0=-1,
        particle_id=np.arange(i_start, i_start + n_batch),
        x=np.zeros(n_batch),
        px=1e-4 * np.ones(n_batch),
        y=np.zeros(n_batch),
        py=-1e-4 * np.ones(n_batch),
        delta=np.zeros(n_batch),
    )


for mode, radiation_flag in modes.items():
    bend = xt.Bend(
        _context=context,
        length=length,
        angle=angle,
        k0_from_h=True,
        radiation_flag=radiation_flag,
    )

    chunks = {
        "dpx": [],
        "dpy": [],
        "ddelta": [],
    }
    n_done = 0

    while n_done < n_particles:
        n_batch = min(batch_size, n_particles - n_done)
        particles = make_particles(n_batch, n_done)
        if radiation_flag in (2, 3):
            particles._init_random_number_generator()

        before_px = particles.px.copy()
        before_py = particles.py.copy()
        before_delta = particles.delta.copy()

        bend.track(particles)

        chunks["dpx"].append(
            context.nparray_from_context_array(particles.px - before_px))
        chunks["dpy"].append(
            context.nparray_from_context_array(particles.py - before_py))
        chunks["ddelta"].append(
            context.nparray_from_context_array(particles.delta - before_delta))

        n_done += n_batch
        print(f"{mode:17s} tracked {n_done}/{n_particles}")

    results[mode] = {key: np.concatenate(value) for key, value in chunks.items()}

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
    if key == "ddelta":
        lo, hi = np.quantile(all_values, [tail_probability, 1 - 1e-5])
        bins = np.linspace(lo, hi, 500)
    else:
        lo, hi = np.quantile(all_values, [1e-5, 1 - 1e-5])
        bins = np.linspace(lo, hi, 300)

    for mode in modes:
        values = results[mode][key]
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=mode,
        )
        hist, bin_edges = np.histogram(values, bins=bins)
        n_below = np.sum(values < bin_edges[0])
        cdf = (n_below + np.cumsum(hist)) / values.size
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

fig_tail, ax_tail = plt.subplots(1, 1, figsize=(7, 5))
all_ddelta = np.concatenate([results[mode]["ddelta"] for mode in modes])
tail_lo, tail_hi = np.quantile(all_ddelta, [tail_probability, 5e-3])
tail_bins = np.linspace(tail_lo, tail_hi, 700)

for mode in modes:
    values = results[mode]["ddelta"]
    hist, bin_edges = np.histogram(values, bins=tail_bins)
    n_below = np.sum(values < bin_edges[0])
    cdf = (n_below + np.cumsum(hist)) / values.size
    ax_tail.step(
        bin_edges[1:],
        cdf,
        where="post",
        linewidth=1.5,
        label=mode,
    )

ax_tail.set_xlabel(r"$\Delta\delta$")
ax_tail.set_ylabel(r"$P(\Delta\delta \leq x)$")
ax_tail.set_yscale("log")
ax_tail.grid(True)
ax_tail.legend()
fig_tail.suptitle("Lower-tail CDF of the energy kick")
fig_tail.tight_layout()

plt.show()
