"""CPU-vs-GPU *timing* benchmark for the RandomRutherford sampler.

Companion to 002_rutherford_cpu_vs_gpu.py (which checks distributional
agreement). This measures the speed-up from enabling RandomRutherford on the
CUDA context: it times the sampling kernel on ContextCpu (single core) and
ContextCupy across a range of parallel-stream counts and reports throughput
(Rutherford draws per second) and the GPU/CPU speed-up.

Methodology (fair-timing):
  * the parallel dimension is the number of RNG streams `n_seeds` (one thread
    per seed); each stream draws SPP samples;
  * particles + RNG state are built once per size and EXCLUDED from the timer,
    so only the Rutherford sampling kernel is measured (not host->device setup);
  * kernels are warmed up once (the one-off NVRTC/cffi compile is excluded);
  * GPU timing brackets cp.cuda.runtime.deviceSynchronize(); the median of
    several repeats is reported.

Caveats: the CPU baseline is a single core (ContextCpu is serial); the absolute
numbers are hardware-specific (here: one consumer GPU).

Run with cupy's CUDA libs on the loader path, e.g.:
    LDP=$(ls -d $(python -c 'import sys;print(sys.prefix)')/lib/python*/site-packages/nvidia/*/lib | tr '\n' ':')
    LD_LIBRARY_PATH="$LDP" python 003_rutherford_cpu_vs_gpu_benchmark.py
"""
import json
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xobjects as xo
import xtrack as xt

# Rutherford parameters (same as tests/test_random_gen_ruth.py + example 002).
rA, rB = 0.0012306225579197868, 53.50625
t0, t1, iterations = 0.001, 0.02, 7

# Scan the number of parallel RNG streams (= one thread/particle per seed).
SIZES = [1_000, 10_000, 100_000, 1_000_000, 3_000_000, 6_000_000]
REPEATS = {1_000: 9, 10_000: 9, 100_000: 7, 1_000_000: 5,
           3_000_000: 5, 6_000_000: 5}
SPP = 1  # Rutherford draws per stream (the per-particle per-interaction cost)


def is_gpu(context):
    return isinstance(context, getattr(xo, "ContextCupy", ()))


def sync(context):
    if is_gpu(context):
        import cupy as cp
        cp.cuda.runtime.deviceSynchronize()


def make_rng(context):
    ran = xt.RandomRutherford(_context=context)
    ran.A, ran.B = rA, rB
    ran.lower_val, ran.upper_val = t0, t1
    ran.Newton_iterations = iterations
    return ran


def time_sample(ran, context, n_seeds, spp, repeats):
    """Median wall time (s) of one Rutherford sampling kernel launch.

    Particles + RNG state are built once and excluded from the timer."""
    particles = xt.Particles(state=np.ones(n_seeds), x=np.ones(n_seeds),
                             _context=context)
    particles._init_random_number_generator()
    samples = context.zeros(shape=(n_seeds * spp,), dtype=np.float64)
    # warm up: triggers kernel build + first-launch costs (excluded from timing)
    ran._sample(particles=particles, samples=samples, n_samples_per_seed=spp)
    sync(context)
    times = []
    for _ in range(repeats):
        sync(context)
        t_start = time.perf_counter()
        ran._sample(particles=particles, samples=samples,
                    n_samples_per_seed=spp)
        sync(context)
        times.append(time.perf_counter() - t_start)
    return float(np.median(times))


def main():
    cpu_ctx = xo.ContextCpu()
    gpu_ctx = xo.ContextCupy()
    cpu_rng = make_rng(cpu_ctx)
    gpu_rng = make_rng(gpu_ctx)

    rows = []
    print(f"{'streams':>9} {'draws':>10} {'CPU [ms]':>10} {'GPU [ms]':>10} "
          f"{'speedup':>9} {'CPU Md/s':>9} {'GPU Md/s':>9}")
    for n_seeds in SIZES:
        reps = REPEATS[n_seeds]
        n_draws = n_seeds * SPP
        t_cpu = time_sample(cpu_rng, cpu_ctx, n_seeds, SPP, reps)
        t_gpu = time_sample(gpu_rng, gpu_ctx, n_seeds, SPP, reps)
        speedup = t_cpu / t_gpu
        cpu_mds = n_draws / t_cpu / 1e6
        gpu_mds = n_draws / t_gpu / 1e6
        rows.append({"n_seeds": n_seeds, "n_draws": n_draws, "cpu_s": t_cpu,
                     "gpu_s": t_gpu, "speedup": speedup,
                     "cpu_mdraws_s": cpu_mds, "gpu_mdraws_s": gpu_mds})
        print(f"{n_seeds:>9d} {n_draws:>10d} {t_cpu*1e3:>10.3f} "
              f"{t_gpu*1e3:>10.3f} {speedup:>8.2f}x {cpu_mds:>9.2f} "
              f"{gpu_mds:>9.2f}")

    ns = np.array([r["n_draws"] for r in rows])
    sp = np.array([r["speedup"] for r in rows])
    cpu_mds = np.array([r["cpu_mdraws_s"] for r in rows])
    gpu_mds = np.array([r["gpu_mdraws_s"] for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.suptitle("RandomRutherford sampler — CPU (1 core) vs GPU (Cupy)",
                 fontsize=13)
    ax1.plot(ns, sp, "o-", color="C2")
    ax1.axhline(1.0, color="0.5", ls="--", lw=1, label="break-even (1x)")
    ax1.set_xscale("log")
    ax1.set_xlabel("number of Rutherford draws")
    ax1.set_ylabel("GPU speed-up  (CPU time / GPU time)")
    ax1.set_title("speed-up vs problem size")
    for x, y in zip(ns, sp):
        ax1.annotate(f"{y:.0f}x", (x, y), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=8)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(ns, cpu_mds, "o-", color="C0", label="CPU (1 core)")
    ax2.plot(ns, gpu_mds, "s-", color="C3", label="GPU (Cupy)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("number of Rutherford draws")
    ax2.set_ylabel("throughput [million draws / s]")
    ax2.set_title("sampling throughput")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png = "rutherford_cpu_vs_gpu_benchmark.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    with open("rutherford_cpu_vs_gpu_benchmark.json", "w") as handle:
        json.dump({"spp": SPP, "rows": rows}, handle, indent=2)
    best = max(rows, key=lambda r: r["speedup"])
    print(f"  wrote {out_png}")
    print(f"peak speed-up {best['speedup']:.0f}x at {best['n_draws']:,} draws")


if __name__ == "__main__":
    main()
