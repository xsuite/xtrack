# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from pathlib import Path

import numpy as np


SEED = 20260616
POWERS = [1 << ii for ii in range(9)]
SAMPLE_COUNTS = {
    1: 2_000_000,
    2: 2_000_000,
    4: 1_500_000,
    8: 1_000_000,
    16: 700_000,
    32: 500_000,
    64: 300_000,
    128: 200_000,
    256: 150_000,
}
BATCH_PHOTONS = 1_500_000


def synrad(x):
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
        out[mask_low] = (
            p / y - q * y - 1.0) * 1.81379936423421784215530788143

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
        approx[use_low] = a1 / np.maximum(
            u_low * u_low, np.finfo(float).tiny)

        u_high = rng.random(np.count_nonzero(~use_low))
        candidate[~use_low] = xlow - np.log(
            np.maximum(u_high, np.finfo(float).tiny))
        approx[~use_low] = a2 * np.exp(-candidate[~use_low])

        accepted = synrad(candidate) >= approx * rng.random(n_try)
        accepted_values = candidate[accepted]
        n_take = min(accepted_values.size, n_samples - n_done)
        samples[n_done:n_done + n_take] = accepted_values[:n_take]
        n_done += n_take

    return samples


def make_probability_grid():
    u_tail = np.logspace(-7, -2, 900)
    u_center = np.linspace(1e-2, 1 - 1e-2, 1601)
    return np.unique(np.concatenate((
        [0.0], u_tail, u_center, 1.0 - u_tail[::-1], [1.0])))


def sample_sum_for_power(rng, power, n_samples):
    values = np.empty(n_samples)
    n_done = 0
    while n_done < n_samples:
        n_batch = min(n_samples - n_done, max(1, BATCH_PHOTONS // power))
        photons = sample_photon_energy_normalized(rng, n_batch * power)
        values[n_done:n_done + n_batch] = photons.reshape(
            n_batch, power).sum(axis=1)
        n_done += n_batch
    values.sort()
    return values


def write_c_array(fid, name, values):
    fid.write(f"static const double {name}[XTRACK_SYNRAD_TOTAL_ENERGY_TABLE_SIZE] = {{\n")
    for ii in range(0, values.size, 4):
        chunk = values[ii:ii + 4]
        fid.write("    " + ", ".join(f"{vv:.17e}" for vv in chunk))
        if ii + 4 < values.size:
            fid.write(",")
        fid.write("\n")
    fid.write("};\n\n")


def main():
    rng = np.random.default_rng(SEED)
    u_grid = make_probability_grid()
    out_path = Path(__file__).with_name("synrad_total_energy_tables.h")

    with out_path.open("w") as fid:
        fid.write("// copyright ############################### //\n")
        fid.write("// This file is part of the Xtrack Package.  //\n")
        fid.write("// Copyright (c) CERN, 2021.                 //\n")
        fid.write("// ######################################### //\n\n")
        fid.write("// Generated by xtrack/headers/_generate_synrad_total_energy_tables.py\n")
        fid.write(f"// RNG seed: {SEED}\n")
        fid.write("// Sample counts per photon block:\n")
        for power in POWERS:
            fid.write(f"//   {power}: {SAMPLE_COUNTS[power]}\n")
        fid.write("// Probability grid is concentrated in both tails; table values store\n")
        fid.write("// log(total normalized energy) for interpolation.\n\n")
        fid.write("#ifndef XTRACK_SYNRAD_TOTAL_ENERGY_TABLES_H\n")
        fid.write("#define XTRACK_SYNRAD_TOTAL_ENERGY_TABLES_H\n\n")
        fid.write(
            f"#define XTRACK_SYNRAD_TOTAL_ENERGY_TABLE_SIZE {u_grid.size}\n\n")

        write_c_array(fid, "synrad_total_energy_u_grid", u_grid)

        for power in POWERS:
            samples = sample_sum_for_power(rng, power, SAMPLE_COUNTS[power])
            quantiles = np.quantile(samples, u_grid)
            quantiles[0] = max(quantiles[1] * 0.1, np.finfo(float).tiny)
            write_c_array(
                fid,
                f"synrad_total_energy_log_table_{power}",
                np.log(quantiles),
            )
            print(f"generated {power} with {SAMPLE_COUNTS[power]} samples")

        fid.write("GPUFUN\n")
        fid.write("const double* synrad_get_total_energy_log_table_power2(int64_t nphot){\n")
        fid.write("    switch(nphot){\n")
        for power in POWERS:
            fid.write(
                f"        case {power}: return synrad_total_energy_log_table_{power};\n")
        fid.write("        default: return 0;\n")
        fid.write("    }\n")
        fid.write("}\n\n")
        fid.write("#endif /* XTRACK_SYNRAD_TOTAL_ENERGY_TABLES_H */\n")


if __name__ == "__main__":
    main()
