# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import importlib.util
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _load_synrad_helpers():
    here = Path(__file__).parent
    helper_path = here / "011_plot_total_energy_loss.py"
    spec = importlib.util.spec_from_file_location("synrad_loss_helpers", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


synrad_helpers = _load_synrad_helpers()


def make_power2_tables(rng, max_power=128, n_table_samples=100_000,
                       batch_photons=1_000_000):
    powers = []
    power = 1
    while power <= max_power:
        powers.append(power)
        power *= 2

    tables = {}
    for power in powers:
        values = np.empty(n_table_samples)
        n_done = 0

        while n_done < n_table_samples:
            n_batch = min(n_table_samples - n_done,
                          max(1, batch_photons // power))
            photon_energy = synrad_helpers.sample_photon_energy_normalized(
                rng, n_batch * power)
            values[n_done:n_done + n_batch] = photon_energy.reshape(
                n_batch, power).sum(axis=1)
            n_done += n_batch

        values.sort()
        tables[power] = values
        print(f"built table for {power:4d} photons")

    return tables


def sample_from_inverse_cdf(rng, sorted_values):
    u = rng.random()
    q = u * (sorted_values.size - 1)
    i0 = int(q)
    i1 = min(i0 + 1, sorted_values.size - 1)
    w = q - i0
    return (1.0 - w) * sorted_values[i0] + w * sorted_values[i1]


def decompose_power2(n_photons, max_power):
    chunks = []
    n_left = int(n_photons)
    while n_left > 0:
        chunk = 1 << (n_left.bit_length() - 1)
        chunk = min(chunk, max_power)
        chunks.append(chunk)
        n_left -= chunk
    return chunks


def sample_fixed_n_power2(rng, n_photons, tables):
    if n_photons == 0:
        return 0.0

    max_power = max(tables)
    total = 0.0
    for chunk in decompose_power2(n_photons, max_power):
        total += sample_from_inverse_cdf(rng, tables[chunk])
    return total


def sample_total_loss_power2(rng, lambdas, n_particles, tables):
    out = np.empty((len(lambdas), n_particles))
    counts_out = np.empty((len(lambdas), n_particles), dtype=np.int64)
    chunks_out = np.empty((len(lambdas), n_particles), dtype=np.int64)

    for ii, lam in enumerate(lambdas):
        n_photons = rng.poisson(lam, size=n_particles)
        counts_out[ii, :] = n_photons
        values = np.empty(n_particles)
        n_chunks = np.empty(n_particles, dtype=np.int64)

        for ipart, count in enumerate(n_photons):
            chunks = decompose_power2(count, max(tables))
            n_chunks[ipart] = len(chunks)
            values[ipart] = sum(
                sample_from_inverse_cdf(rng, tables[chunk])
                for chunk in chunks)

        out[ii, :] = values
        chunks_out[ii, :] = n_chunks

    return out, counts_out, chunks_out


if __name__ == "__main__":
    rng_tables = np.random.default_rng(12345)
    rng_reference = np.random.default_rng(67890)
    rng_power2 = np.random.default_rng(24680)

    lambdas = np.array([2.86, 8.58, 20.0])
    n_particles = 50_000

    tables = make_power2_tables(
        rng_tables,
        max_power=256,
        n_table_samples=40_000,
    )

    print()
    for count in [8, 19, 87, 312]:
        print(f"{count:3d} -> {decompose_power2(count, max(tables))}")

    reference = synrad_helpers.sample_total_loss_normalized(
        rng_reference, lambdas, n_particles)
    power2, counts, chunks = sample_total_loss_power2(
        rng_power2, lambdas, n_particles, tables)

    print()
    print("lambda      <N>   <chunks>   mean ref  mean p2    rms ref   rms p2")
    for ii, lam in enumerate(lambdas):
        print(
            f"{lam:6.2f}"
            f"  {np.mean(counts[ii]):7.3f}"
            f"  {np.mean(chunks[ii]):8.3f}"
            f"  {np.mean(reference[ii]):8.4f}"
            f"  {np.mean(power2[ii]):8.4f}"
            f"  {np.std(reference[ii]):8.4f}"
            f"  {np.std(power2[ii]):8.4f}"
        )

    plt.close("all")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for lam, ref, p2 in zip(lambdas, reference, power2):
        bins = np.linspace(
            0.0,
            np.quantile(np.concatenate([ref, p2]), 0.995),
            120,
        )
        axes[0].hist(ref, bins=bins, density=True, histtype="step",
                     label=rf"brute force, $\lambda={lam:g}$")
        axes[0].hist(p2, bins=bins, density=True, histtype="step",
                     linestyle="--",
                     label=rf"power-2 tables, $\lambda={lam:g}$")

    axes[0].set_xlabel(r"$S(\lambda)$")
    axes[0].set_ylabel("density")
    axes[0].set_yscale("log")
    axes[0].grid(True)
    axes[0].legend(fontsize=8)

    max_count = int(np.quantile(counts, 0.999))
    count_values = np.arange(max_count + 1)
    chunk_values = [
        len(decompose_power2(count, max(tables)))
        for count in count_values
    ]

    axes[1].plot(count_values, count_values, label="photon loop")
    axes[1].step(count_values, chunk_values, where="post",
                 label="power-2 table loop")
    axes[1].set_xlabel("number of photons N")
    axes[1].set_ylabel("number of sampler calls")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    plt.show()
