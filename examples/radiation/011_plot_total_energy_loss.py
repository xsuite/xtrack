# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import matplotlib.pyplot as plt


def synrad(x):
    # Port of SynRad(x) from xtrack/headers/synrad_spectrum.h.
    x = np.asarray(x)
    out = np.zeros_like(x, dtype=float)

    mask_low = (x > 0.0) & (x < 6.0)
    if np.any(mask_low):
        xx = x[mask_low]
        z = xx * xx / 16.0 - 2.0
        b = np.full_like(xx, 0.00000000000000000012)
        a = z * b + 0.00000000000000000460
        b = z * a - b + 0.00000000000000031738
        a = z * b - a + 0.00000000000002004426
        b = z * a - b + 0.00000000000111455474
        a = z * b - a + 0.00000000005407460944
        b = z * a - b + 0.00000000226722011790
        a = z * b - a + 0.00000008125130371644
        b = z * a - b + 0.00000245751373955212
        a = z * b - a + 0.00006181256113829740
        b = z * a - b + 0.00127066381953661690
        a = z * b - a + 0.02091216799114667278
        b = z * a - b + 0.26880346058164526514
        a = z * b - a + 2.61902183794862213818
        b = z * a - b + 18.65250896865416256398
        a = z * b - a + 92.95232665922707542088
        b = z * a - b + 308.15919413131586030542
        a = z * b - a + 644.86979658236221700714
        p = 0.5 * z * a - b + 414.56543648832546975110

        a = np.full_like(xx, 0.00000000000000000004)
        b = z * a + 0.00000000000000000289
        a = z * b - a + 0.00000000000000019786
        b = z * a - b + 0.00000000000001196168
        a = z * b - a + 0.00000000000063427729
        b = z * a - b + 0.00000000002923635681
        a = z * b - a + 0.00000000115951672806
        b = z * a - b + 0.00000003910314748244
        a = z * b - a + 0.00000110599584794379
        b = z * a - b + 0.00002581451439721298
        a = z * b - a + 0.00048768692916240683
        b = z * a - b + 0.00728456195503504923
        a = z * b - a + 0.08357935463720537773
        b = z * a - b + 0.71031361199218887514
        a = z * b - a + 4.26780261265492264837
        b = z * a - b + 17.05540785795221885751
        a = z * b - a + 41.83903486779678800040
        q = 0.5 * z * a - b + 28.41787374362784178164

        y = np.power(xx, 2.0 / 3.0)
        out[mask_low] = (p / y - q * y - 1.0) * 1.81379936423421784215530788143

    mask_high = (x >= 6.0) & (x < 800.0)
    if np.any(mask_high):
        xx = x[mask_high]
        z = 20.0 / xx - 2.0
        a = np.full_like(xx, 0.00000000000000000001)
        b = z * a - 0.00000000000000000002
        a = z * b - a + 0.00000000000000000006
        b = z * a - b - 0.00000000000000000020
        a = z * b - a + 0.00000000000000000066
        b = z * a - b - 0.00000000000000000216
        a = z * b - a + 0.00000000000000000721
        b = z * a - b - 0.00000000000000002443
        a = z * b - a + 0.00000000000000008441
        b = z * a - b - 0.00000000000000029752
        a = z * b - a + 0.00000000000000107116
        b = z * a - b - 0.00000000000000394564
        a = z * b - a + 0.00000000000001489474
        b = z * a - b - 0.00000000000005773537
        a = z * b - a + 0.00000000000023030657
        b = z * a - b - 0.00000000000094784973
        a = z * b - a + 0.00000000000403683207
        b = z * a - b - 0.00000000001785432348
        a = z * b - a + 0.00000000008235329314
        b = z * a - b - 0.00000000039817923621
        a = z * b - a + 0.00000000203088939238
        b = z * a - b - 0.00000001101482369622
        a = z * b - a + 0.00000006418902302372
        b = z * a - b - 0.00000040756144386809
        a = z * b - a + 0.00000287536465397527
        b = z * a - b - 0.00002321251614543524
        a = z * b - a + 0.00022505317277986004
        b = z * a - b - 0.00287636803664026799
        a = z * b - a + 0.06239591359332750793
        p = 0.5 * z * a - b + 1.06552390798340693166
        out[mask_high] = p * np.sqrt(0.5 * np.pi / xx) / np.exp(xx)

    return out


def sample_photon_energy_normalized(rng, n_samples):
    # Same rejection sampler used by synrad_gen_photon_energy_normalized.
    xlow = 1.0
    a1 = 2.149528241534391
    a2 = 1.770750801624037
    ratio = 0.908250405131381

    samples = np.empty(n_samples)
    n_done = 0
    while n_done < n_samples:
        n_try = max(1024, int(1.3 * (n_samples - n_done)))
        use_low = rng.random(n_try) < ratio
        candidate = np.empty(n_try)
        approx = np.empty(n_try)

        u_low = rng.random(np.count_nonzero(use_low))
        candidate[use_low] = u_low**3
        approx[use_low] = a1 / np.maximum(u_low * u_low, np.finfo(float).tiny)

        u_high = rng.random(np.count_nonzero(~use_low))
        candidate[~use_low] = xlow - np.log(np.maximum(u_high, np.finfo(float).tiny))
        approx[~use_low] = a2 * np.exp(-candidate[~use_low])

        accepted = synrad(candidate) >= approx * rng.random(n_try)
        accepted_values = candidate[accepted]
        n_take = min(accepted_values.size, n_samples - n_done)
        samples[n_done : n_done + n_take] = accepted_values[:n_take]
        n_done += n_take

    return samples


def sample_total_loss_normalized(rng, lambdas, n_particles):
    lambdas = np.asarray(lambdas, dtype=float)
    out = np.empty((lambdas.size, n_particles))

    for ii, lam in enumerate(lambdas):
        n_photons = rng.poisson(lam, size=n_particles)
        total_photons = int(np.sum(n_photons))
        out[ii, :] = 0.0
        if total_photons == 0:
            continue

        photon_energy = sample_photon_energy_normalized(rng, total_photons)
        particle_index = np.repeat(np.arange(n_particles), n_photons)
        np.add.at(out[ii], particle_index, photon_energy)

    return out


if __name__ == "__main__":
    rng = np.random.default_rng(12345)

    lambdas = np.array([0.05, 0.2, 1.0, 5.0, 20.0])
    n_particles = 200_000
    total_loss = sample_total_loss_normalized(rng, lambdas, n_particles)

    mean_x = 8.0 / (15.0 * np.sqrt(3.0))
    second_moment_x = 11.0 / 27.0

    plt.close("all")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for lam, values in zip(lambdas, total_loss):
        nonzero = values[values > 0.0]
        zero_probability = np.mean(values == 0.0)
        axes[0].hist(
            nonzero,
            bins=160,
            density=True,
            histtype="step",
            label=rf"$\lambda={lam:g}$, $P_0={zero_probability:.3g}$",
        )

    axes[0].set_xlabel(r"$S(\lambda) = \sum_{i=1}^{N_\lambda} x_i$")
    axes[0].set_ylabel("conditional density for S > 0")
    axes[0].set_yscale("log")
    axes[0].grid(True)
    axes[0].legend()

    lambda_scan = np.logspace(-2, 2, 100)
    sample_mean = np.mean(total_loss, axis=1)
    sample_rms = np.std(total_loss, axis=1)

    axes[1].loglog(lambda_scan, lambda_scan * mean_x, label="theory mean")
    axes[1].loglog(
        lambda_scan,
        np.sqrt(lambda_scan * second_moment_x),
        label="theory rms",
    )
    axes[1].plot(lambdas, sample_mean, "o", label="sample mean")
    axes[1].plot(lambdas, sample_rms, "s", label="sample rms")
    axes[1].set_xlabel(r"$\lambda = \langle N_\gamma \rangle$")
    axes[1].set_ylabel(r"moments of $S(\lambda)$")
    axes[1].grid(True, which="both")
    axes[1].legend()

    fig.tight_layout()
    plt.show()
