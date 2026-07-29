# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

################################################################################
# Required packages
################################################################################
import numpy as np
import matplotlib.pyplot as plt

################################################################################
# User parameters
################################################################################

########################################
# Monte Carlo settings
########################################
SEED                    = 12345
N_PARTICLES             = int(1E6)

########################################
# Photon-count regimes
########################################
TEST_LAMBDAS            = np.array([0.05, 0.2, 1.0, 5.0, 20.0])

########################################
# Plot settings
########################################
PLOT_RESULTS            = True
TAIL_ENERGY_X_MIN       = 1E-2
TAIL_RELATIVE_X_MIN     = 1E-1

RNG                     = np.random.default_rng(SEED)

################################################################################
# Synchrotron spectrum
################################################################################

########################################
# Evaluate normalized photon spectrum
########################################
def synrad(x):
    """Evaluate the normalized synchrotron-radiation photon spectrum.

    Parameters
    ----------
    x : float or array-like
        Photon energy normalized by the critical energy,
        ``x = E_gamma / E_c``. This quantity is dimensionless.

    Returns
    -------
    out : float or ndarray
        Value of the normalized photon spectrum used by Xtrack to sample
        individual synchrotron-radiation photons. In this example it is
        interpreted as the probability density of the dimensionless random
        variable ``x``.

    Notes
    -----
    This is a Python port of ``SynRad(x)`` from
    ``xtrack/headers/synrad_spectrum.h``. The source comment there identifies
    the approximation as Chebyshev series from H.H. Umstaetter,
    ``CERN/PS/SM/81-13`` (10-3-1981), with a reference to LEP Note 632
    (12/1990), converted to C++ by H. Burkhardt on 21-4-1996.

    The purpose of keeping the function visible in this example is pedagogical:
    figure 1 shows the baseline single-photon distribution from which the
    existing photon-by-photon quantum radiation model samples.
    """
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


################################################################################
# Photon-by-photon sampling
################################################################################

########################################
# Sample photon energies
########################################
def sample_photon_energy_normalized(rng, n_samples):
    """Sample individual photon energies from the Xtrack SynRad spectrum.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator used for the Monte Carlo sampling.
    n_samples : int
        Number of individual photon energies to generate.

    Returns
    -------
    samples : ndarray
        Samples of ``x = E_gamma / E_c``.

    Notes
    -----
    This is the same rejection-sampling strategy used by
    ``synrad_gen_photon_energy_normalized`` in
    ``xtrack/headers/synrad_spectrum.h``. The proposal distribution is split
    into a low-energy approximation and a high-energy exponential tail. The
    rejection step then corrects the proposal back to the exact ``SynRad(x)``
    spectrum used by Xtrack.
    """
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

########################################
# Sample total energies
########################################
def sample_total_loss_normalized(rng, lambdas, n_particles,
                                 return_photon_counts=False):
    """Sample total normalized radiation loss for a compound-Poisson process.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator used for photon counts and photon energies.
    lambdas : array-like
        Mean photon counts, ``lambda = <N_gamma>``. One distribution is sampled
        for each value.
    n_particles : int
        Number of independent particles/events sampled per ``lambda``.
    return_photon_counts : bool
        If true, also return the sampled Poisson photon counts.

    Returns
    -------
    out : ndarray
        Array with shape ``(len(lambdas), n_particles)`` containing
        ``S(lambda) = sum_i E_i/E_c``.
    counts : ndarray, optional
        Sampled photon counts with the same leading dimensions. Returned only
        when ``return_photon_counts=True``.

    Notes
    -----
    For each particle/event the process is:

    1. draw ``N_gamma ~ Poisson(lambda)``;
    2. draw ``N_gamma`` independent photon energies from ``SynRad(x)``;
    3. sum them to obtain the total normalized emitted energy.

    This is the baseline process sampled by photon-by-photon quantum radiation.
    """
    lambdas = np.asarray(lambdas, dtype=float)
    out = np.empty((lambdas.size, n_particles))
    counts = np.empty((lambdas.size, n_particles), dtype=np.int64)

    for ii, lam in enumerate(lambdas):
        n_photons = rng.poisson(lam, size=n_particles)
        counts[ii, :] = n_photons
        total_photons = int(np.sum(n_photons))
        out[ii, :] = 0.0
        if total_photons == 0:
            continue

        photon_energy = sample_photon_energy_normalized(rng, total_photons)
        particle_index = np.repeat(np.arange(n_particles), n_photons)
        np.add.at(out[ii], particle_index, photon_energy)

    if return_photon_counts:
        return out, counts

    return out


################################################################################
# Diagnostics
################################################################################

########################################
# Compound-Poisson theory
########################################
def compound_poisson_theory(lambdas):
    """Return theoretical mean and RMS of the normalized total loss.

    The total loss is a compound-Poisson random variable,
    ``S = x_1 + ... + x_N``, where ``N ~ Poisson(lambda)``. For a compound
    Poisson process:

    ``E[S] = lambda E[x]``

    and

    ``Var(S) = lambda E[x^2]``.

    For the normalized synchrotron-radiation photon spectrum used here,
    ``E[x] = 8/(15 sqrt(3))`` and ``E[x^2] = 11/27``.
    """
    lambdas = np.asarray(lambdas, dtype=float)
    mean_single_photon_x = 8.0 / (15.0 * np.sqrt(3.0))
    second_moment_single_photon_x = 11.0 / 27.0

    mean = lambdas * mean_single_photon_x
    rms = np.sqrt(lambdas * second_moment_single_photon_x)
    return mean, rms

########################################
# Print summary
########################################
def relative_error_string(observed, expected):
    """Format a relative error, using ``n/a`` when the reference is zero."""
    if expected == 0:
        if observed == 0:
            return "       n/a"
        return "       inf"
    return f"{(observed / expected - 1):+10.3e}"


def print_summary(lambdas, total_loss, photon_counts):
    """Print a human-readable statistical summary of the sampled process.

    This table is meant to be useful even without looking at the plots. It
    checks that the sampled photon count behaves like a Poisson process and
    that the sampled total energy loss has the expected compound-Poisson
    moments.
    """
    theory_mean, theory_rms = compound_poisson_theory(lambdas)

    print("Compound-Poisson total normalized radiation loss")
    print(f"n_particles = {total_loss.shape[1]}")
    print()
    print(
        " lambda"
        "   <N> sample"
        "   <N> theory"
        "      P(N=0)"
        "    exp(-lambda)"
        "    mean sample"
        "    mean theory"
        "    rel.err"
        "     rms sample"
        "     rms theory"
        "    rel.err"
        "      q99 [1]"
        "    q99.9 [1]"
    )

    for ii, lam in enumerate(lambdas):
        sample_mean = np.mean(total_loss[ii])
        sample_rms = np.std(total_loss[ii])
        print(
            f"{lam:7.3g}"
            f" {np.mean(photon_counts[ii]):12.6f}"
            f" {lam:12.6f}"
            f" {np.mean(photon_counts[ii] == 0):11.6f}"
            f" {np.exp(-lam):14.6f}"
            f" {sample_mean:14.7e}"
            f" {theory_mean[ii]:14.7e}"
            f" {relative_error_string(sample_mean, theory_mean[ii])}"
            f" {sample_rms:14.7e}"
            f" {theory_rms[ii]:14.7e}"
            f" {relative_error_string(sample_rms, theory_rms[ii])}"
            f" {np.quantile(total_loss[ii], 0.99):10.4e}"
            f" {np.quantile(total_loss[ii], 0.999):10.4e}"
        )


########################################
# Baseline checks
########################################
def assert_baseline_checks(lambdas, total_loss, photon_counts):
    """Print and assert the baseline statistical checks.

    These are not precision tests of the physics model. They are sanity checks
    for this example: if any of them fails, either the sampler is broken or the
    Monte Carlo sample is too small for the configured tolerances.
    """
    theory_mean, theory_rms = compound_poisson_theory(lambdas)

    sample_count_mean = np.mean(photon_counts, axis=1)
    sample_zero_probability = np.mean(photon_counts == 0, axis=1)
    sample_loss_mean = np.mean(total_loss, axis=1)
    sample_loss_rms = np.std(total_loss, axis=1)

    checks = [
        (
            "Poisson mean <N_gamma>",
            sample_count_mean,
            lambdas,
            0.03,
            0.01,
        ),
        (
            "Zero-photon probability P(N_gamma=0)",
            sample_zero_probability,
            np.exp(-lambdas),
            0.05,
            5e-3,
        ),
        (
            "Mean total loss <S>",
            sample_loss_mean,
            theory_mean,
            0.06,
            3e-3,
        ),
        (
            "RMS total loss sigma_S",
            sample_loss_rms,
            theory_rms,
            0.06,
            3e-3,
        ),
    ]

    print()
    print("Baseline checks")
    print(
        "check"
        "                                max abs.err"
        "     max rel.err"
        "       rtol"
        "       atol"
        "    status"
    )

    for label, observed, expected, rtol, atol in checks:
        abs_error = np.abs(observed - expected)
        rel_error = abs_error / np.maximum(np.abs(expected), atol)
        passed = np.allclose(observed, expected, rtol=rtol, atol=atol)

        print(
            f"{label:34s}"
            f" {np.max(abs_error):13.4e}"
            f" {np.max(rel_error):13.4e}"
            f" {rtol:10.3e}"
            f" {atol:10.3e}"
            f"    {'pass' if passed else 'FAIL'}"
        )

        assert passed, label


################################################################################
# Plotting
################################################################################

########################################
# Single photon spectrum
########################################
def plot_single_photon_spectrum():
    """Plot the baseline single-photon probability density.

    The plot establishes the distribution sampled by the photon-by-photon
    quantum radiation model. The horizontal axis is dimensionless photon energy
    in units of the critical energy. The vertical axis is a probability density
    with respect to that dimensionless variable.
    """
    x_grid = np.logspace(-6, 2, 1200)
    y_grid = synrad(x_grid)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.loglog(x_grid, y_grid)
    ax.set_xlabel(
        r"Normalized photon energy, $x = E_\gamma / E_c$ [1]")
    ax.set_ylabel(r"Photon probability density, $p(x)$ [1]")
    ax.set_title("Baseline single-photon synchrotron radiation spectrum")
    ax.grid(True, which="both")
    fig.tight_layout()

########################################
# Sample total energies
########################################
def plot_total_loss_distributions(lambdas, total_loss, photon_counts):
    """Plot the total emitted energy for the selected photon-count means.

    This figure shows the nonzero part of the compound-Poisson distribution.
    The zero-loss probability is not visible on a density plot, so it is given
    explicitly in the legend as ``P(N_gamma=0)``.

    Two views are shown because they emphasize different physics:

    - the linear horizontal axis makes the bulk of the total-loss distribution
      easier to compare with the original exploratory plot;
    - the logarithmic horizontal axis makes the rare high-loss tail easier to
      inspect, which is important for lifetime studies.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_linear, ax_log = axes

    for ii, (lam, values, counts) in enumerate(
            zip(lambdas, total_loss, photon_counts)):
        if lam == 0:
            continue

        nonzero = values[values > 0.0]
        zero_probability = np.mean(counts == 0)

        linear_bins = np.linspace(0.0, np.quantile(nonzero, 0.999), 180)
        log_bins = np.logspace(
            np.log10(max(np.quantile(nonzero, 1e-4), 1e-10)),
            np.log10(np.quantile(nonzero, 0.9995)),
            160,
        )
        label = (
            rf"$\lambda=\langle N_\gamma\rangle={lam:g}$, "
            rf"$P(N_\gamma=0)={zero_probability:.3g}$"
        )

        ax_linear.hist(
            nonzero,
            bins=linear_bins,
            density=True,
            histtype="step",
            color=f"C{ii}",
            label=label,
        )
        ax_log.hist(
            nonzero,
            bins=log_bins,
            density=True,
            histtype="step",
            color=f"C{ii}",
        )

    for ax in axes:
        ax.set_xlabel(
            r"Total normalized emitted energy, "
            r"$S=\sum_i E_{\gamma,i}/E_c$ [1]")
        ax.set_ylabel(r"Probability density conditioned on $S>0$ [1]")
        ax.set_yscale("log")
        ax.grid(True, which="both")

    ax_linear.set_title("Bulk view")
    ax_log.set_title("Tail view")
    ax_log.set_xscale("log")
    ax_log.set_xlim(left=TAIL_ENERGY_X_MIN)
    ax_linear.legend(fontsize=8)

    fig.suptitle(
        "Compound-Poisson total energy-loss distribution\n"
        r"Total normalized emitted energy, "
        r"$S=\sum_i E_{\gamma,i}/E_c$ is dimensionless"
    )
    fig.tight_layout()

########################################
# Moment checks
########################################
def plot_moment_checks(lambdas, total_loss):
    """Plot sampled moments and residuals against analytic expectations."""
    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    lambda_scan = np.logspace(-2, 2, 200)
    theory_mean_scan, theory_rms_scan = compound_poisson_theory(lambda_scan)
    theory_mean, theory_rms = compound_poisson_theory(lambdas)

    sample_mean = np.mean(total_loss, axis=1)
    sample_rms = np.std(total_loss, axis=1)
    positive = lambdas > 0

    ax.loglog(lambda_scan, theory_mean_scan, label="theory mean")
    ax.loglog(lambda_scan, theory_rms_scan, label="theory RMS")
    ax.plot(lambdas[positive], sample_mean[positive], "o", label="sample mean")
    ax.plot(lambdas[positive], sample_rms[positive], "s", label="sample RMS")

    mean_residual = sample_mean[positive] / theory_mean[positive] - 1.0
    rms_residual = sample_rms[positive] / theory_rms[positive] - 1.0
    ax_res.axhline(0.0, color="black", linewidth=0.8)
    ax_res.plot(lambdas[positive], 100 * mean_residual, "o-", label="mean")
    ax_res.plot(lambdas[positive], 100 * rms_residual, "s-", label="RMS")

    ax.set_ylabel(r"Moment of $S$ [1]")
    ax.set_title("Compound-Poisson moment check")
    ax.grid(True, which="both")
    ax.legend()

    ax_res.set_xscale("log")
    ax_res.set_xlabel(r"Mean photon count, $\lambda=\langle N_\gamma\rangle$ [1]")
    ax_res.set_ylabel("Residual [%]")
    ax_res.grid(True, which="both")
    ax_res.legend(fontsize=8)
    fig.tight_layout()

########################################
# Tail survival probability
########################################
def plot_tail_survival(lambdas, total_loss):
    """Plot the high-loss tail in units of the mean loss.

    This figure is intended to make the lifetime-relevant part of the
    distribution visible. The vertical axis is the probability that one local
    radiation event loses more than a given multiple of the mean total loss.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    theory_mean, _ = compound_poisson_theory(lambdas)

    for lam, values, mean_value in zip(lambdas, total_loss, theory_mean):
        if lam == 0:
            continue

        scaled_values = values / mean_value
        sorted_values = np.sort(scaled_values)
        survival = 1.0 - (np.arange(values.size) + 1) / values.size
        positive = sorted_values > 0.0
        ax.step(
            sorted_values[positive],
            np.maximum(survival[positive], 0.5 / values.size),
            where="post",
            label=rf"$\lambda=\langle N_\gamma\rangle={lam:g}$",
        )

    ax.set_xlabel(r"Total loss relative to mean, $S/\langle S\rangle$ [1]")
    ax.set_ylabel(r"Survival probability, $P(S/\langle S\rangle > r)$ [1]")
    ax.set_title("High-loss tail of the compound-Poisson process")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=TAIL_RELATIVE_X_MIN)
    ax.grid(True, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()

################################################################################
# Run
################################################################################

total_loss, photon_counts = sample_total_loss_normalized(
    RNG, TEST_LAMBDAS, N_PARTICLES, return_photon_counts=True)

print_summary(TEST_LAMBDAS, total_loss, photon_counts)
assert_baseline_checks(TEST_LAMBDAS, total_loss, photon_counts)

print()
print("OVERALL STATUS: PASS")

if PLOT_RESULTS:
    plt.close("all")
    plot_single_photon_spectrum()
    plot_total_loss_distributions(TEST_LAMBDAS, total_loss, photon_counts)
    plot_moment_checks(TEST_LAMBDAS, total_loss)
    plot_tail_survival(TEST_LAMBDAS, total_loss)
    plt.show()
