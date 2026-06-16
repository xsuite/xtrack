# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import os
import time

import numpy as np
import matplotlib.pyplot as plt

import xobjects as xo
import xpart as xp
import xtrack as xt


context = xo.ContextCpu()

n_particles = int(os.environ.get("N_PARTICLES", "8_000_000"))
batch_size = int(os.environ.get("BATCH_SIZE", "500_000"))
pilot_particles = int(os.environ.get("PILOT_PARTICLES", "1_000_000"))
tail_probability = 1e-10
ddelta_low_sigmas = float(os.environ.get("DDELTA_LOW_SIGMAS", "8.0"))
p0c = 182.5e9
length = 22.653765579198428
angle = 0.0022799344662676477
initial_px = 1e-4
initial_py = -1e-4
initial_delta = 0.0

modes = {
    "quantum": 2,
    "quantum-efficient": 3,
}

plot_specs = [
    ("dpx", r"$\Delta p_x$"),
    ("dpy", r"$\Delta p_y$"),
    ("ddelta", r"$\Delta\delta$"),
]


def make_particles(n_batch, i_start):
    return xp.Particles(
        _context=context,
        p0c=p0c,
        mass0=xp.ELECTRON_MASS_EV,
        q0=-1,
        particle_id=np.arange(i_start, i_start + n_batch),
        x=np.zeros(n_batch),
        px=initial_px * np.ones(n_batch),
        y=np.zeros(n_batch),
        py=initial_py * np.ones(n_batch),
        delta=initial_delta * np.ones(n_batch),
    )


def make_bend(radiation_flag):
    return xt.Bend(
        _context=context,
        length=length,
        angle=angle,
        k0_from_h=True,
        radiation_flag=radiation_flag,
    )


def track_batch(bend, radiation_flag, n_batch, i_start):
    particles = make_particles(n_batch, i_start)
    if radiation_flag in (2, 3):
        particles._init_random_number_generator()

    t_start = time.perf_counter()
    bend.track(particles)
    t_track = time.perf_counter() - t_start

    return {
        "dpx": context.nparray_from_context_array(particles.px) - initial_px,
        "dpy": context.nparray_from_context_array(particles.py) - initial_py,
        "ddelta": (
            context.nparray_from_context_array(particles.delta)
            - initial_delta
        ),
        "t_track": t_track,
    }


def collect_pilot():
    pilot = {
        mode: {key: [] for key, _ in plot_specs}
        for mode in modes
    }
    n_pilot = min(pilot_particles, n_particles)

    for mode, radiation_flag in modes.items():
        bend = make_bend(radiation_flag)
        n_done = 0

        while n_done < n_pilot:
            n_batch = min(batch_size, n_pilot - n_done)
            values = track_batch(bend, radiation_flag, n_batch, n_done)
            for key in pilot[mode]:
                pilot[mode][key].append(values[key])

            n_done += n_batch
            print(f"{mode:17s} pilot {n_done}/{n_pilot}")

        for key in pilot[mode]:
            pilot[mode][key] = np.concatenate(pilot[mode][key])

    return pilot


def make_bins(pilot):
    bins = {}

    for key, _ in plot_specs:
        all_values = np.concatenate([pilot[mode][key] for mode in modes])
        if key == "ddelta":
            mean = np.mean(all_values)
            sigma = np.std(all_values)
            lo_quantile = np.quantile(all_values, tail_probability)
            lo_sigma = mean - ddelta_low_sigmas * sigma
            lo = min(lo_quantile, lo_sigma)
            hi = np.quantile(all_values, 1 - 1e-5)
            bins[key] = np.linspace(lo, hi, 500)
        else:
            lo, hi = np.quantile(all_values, [1e-5, 1 - 1e-5])
            bins[key] = np.linspace(lo, hi, 300)

    all_ddelta = np.concatenate([pilot[mode]["ddelta"] for mode in modes])
    mean = np.mean(all_ddelta)
    sigma = np.std(all_ddelta)
    tail_lo_quantile = np.quantile(all_ddelta, tail_probability)
    tail_lo_sigma = mean - ddelta_low_sigmas * sigma
    tail_lo = min(tail_lo_quantile, tail_lo_sigma)
    tail_hi = np.quantile(all_ddelta, 5e-3)
    bins["ddelta_tail"] = np.linspace(tail_lo, tail_hi, 700)

    return bins


def make_accumulators(bins):
    acc = {}
    for mode in modes:
        acc[mode] = {}
        for key, _ in plot_specs:
            acc[mode][key] = {
                "hist": np.zeros(bins[key].size - 1, dtype=np.int64),
                "below": 0,
                "above": 0,
                "n": 0,
                "sum": 0.0,
                "sum2": 0.0,
                "min": np.inf,
                "max": -np.inf,
            }
        acc[mode]["ddelta_tail"] = {
            "hist": np.zeros(bins["ddelta_tail"].size - 1, dtype=np.int64),
            "below": 0,
            "above": 0,
            "n": 0,
        }
        acc[mode]["timing"] = {
            "n_batches": 0,
            "n_particles": 0,
            "total": 0.0,
            "min": np.inf,
            "max": 0.0,
        }
    return acc


def update_histogram(acc_item, values, bin_edges):
    hist, _ = np.histogram(values, bins=bin_edges)
    acc_item["hist"] += hist
    acc_item["below"] += np.count_nonzero(values < bin_edges[0])
    acc_item["above"] += np.count_nonzero(values >= bin_edges[-1])
    acc_item["n"] += values.size


def update_moments(acc_item, values):
    acc_item["sum"] += np.sum(values)
    acc_item["sum2"] += np.sum(values * values)
    acc_item["min"] = min(acc_item["min"], np.min(values))
    acc_item["max"] = max(acc_item["max"], np.max(values))


def mean_and_rms(acc_item):
    mean = acc_item["sum"] / acc_item["n"]
    variance = acc_item["sum2"] / acc_item["n"] - mean * mean
    return mean, np.sqrt(max(variance, 0.0))


def update_timing(timing, n_batch, t_track):
    timing["n_batches"] += 1
    timing["n_particles"] += n_batch
    timing["total"] += t_track
    timing["min"] = min(timing["min"], t_track)
    timing["max"] = max(timing["max"], t_track)


def print_summary(mode, mode_acc):
    mean_dpx, rms_dpx = mean_and_rms(mode_acc["dpx"])
    mean_dpy, rms_dpy = mean_and_rms(mode_acc["dpy"])
    mean_ddelta, rms_ddelta = mean_and_rms(mode_acc["ddelta"])

    timing = mode_acc["timing"]
    mean_batch_time = timing["total"] / timing["n_batches"]
    throughput = timing["n_particles"] / timing["total"]

    print(
        f"{mode:17s}"
        f" <dpx>={mean_dpx: .6e}"
        f" rms(dpx)={rms_dpx: .6e}"
        f" <dpy>={mean_dpy: .6e}"
        f" rms(dpy)={rms_dpy: .6e}"
        f" <ddelta>={mean_ddelta: .6e}"
        f" rms(ddelta)={rms_ddelta: .6e}"
        f" min(ddelta)={mode_acc['ddelta']['min']: .6e}"
    )
    print(
        f"{mode:17s}"
        f" track_time_total={timing['total']:.6f} s"
        f" mean_batch={mean_batch_time:.6f} s"
        f" min_batch={timing['min']:.6f} s"
        f" max_batch={timing['max']:.6f} s"
        f" throughput={throughput:.6e} particles/s"
    )


def stream_statistics(bins):
    acc = make_accumulators(bins)

    for mode, radiation_flag in modes.items():
        bend = make_bend(radiation_flag)
        n_done = 0

        while n_done < n_particles:
            n_batch = min(batch_size, n_particles - n_done)
            values = track_batch(bend, radiation_flag, n_batch, n_done)
            update_timing(acc[mode]["timing"], n_batch, values["t_track"])

            for key, _ in plot_specs:
                update_histogram(acc[mode][key], values[key], bins[key])
                update_moments(acc[mode][key], values[key])

            update_histogram(
                acc[mode]["ddelta_tail"],
                values["ddelta"],
                bins["ddelta_tail"],
            )

            n_done += n_batch
            print(
                f"{mode:17s} tracked {n_done}/{n_particles}"
                f" batch_time={values['t_track']:.6f} s"
                f" rate={n_batch / values['t_track']:.6e} particles/s"
            )

        print_summary(mode, acc[mode])

    return acc


def plot_results(acc, bins):
    plt.close("all")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig_cdf, axes_cdf = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, ax_cdf, (key, label) in zip(axes, axes_cdf, plot_specs):
        bin_edges = bins[key]
        widths = np.diff(bin_edges)

        for mode in modes:
            item = acc[mode][key]
            density = item["hist"] / item["n"] / widths
            cdf = (item["below"] + np.cumsum(item["hist"])) / item["n"]

            ax.step(
                bin_edges[1:],
                density,
                where="post",
                linewidth=1.5,
                label=mode,
            )
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
    tail_bins = bins["ddelta_tail"]

    for mode in modes:
        item = acc[mode]["ddelta_tail"]
        cdf = (item["below"] + np.cumsum(item["hist"])) / item["n"]
        ax_tail.step(
            tail_bins[1:],
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


pilot = collect_pilot()
bins = make_bins(pilot)
del pilot
acc = stream_statistics(bins)
plot_results(acc, bins)

plt.show()
