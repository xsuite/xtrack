"""Visual CPU-vs-GPU comparison of the RandomRutherford sampler.

Samples the Rutherford distribution on ContextCpu and ContextCupy with identical
parameters and overlays the two distributions against the analytic PDF. This is
the documentation figure for enabling RandomRutherford on the CUDA context:
agreement is distributional (parallel RNG streams + FP ordering differ across
contexts, so a per-particle bitwise match is neither expected nor required).

Run with cupy's CUDA libs on the loader path, e.g.:
    LDP=$(ls -d $(python -c 'import sys;print(sys.prefix)')/lib/python*/site-packages/nvidia/*/lib | tr '\n' ':')
    LD_LIBRARY_PATH="$LDP" python 002_rutherford_cpu_vs_gpu.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

import xobjects as xo
import xtrack as xt

# Rutherford parameters (same as tests/test_random_gen_ruth.py)
rA, rB, t0, t1, iterations = 0.0012306225579197868, 53.50625, 0.001, 0.02, 7
N_SAMPLES, N_SEEDS = int(3e6), 3000


def sample(context):
    ran = xt.RandomRutherford(_context=context)
    ran.A, ran.B = rA, rB
    ran.lower_val, ran.upper_val = t0, t1
    ran.Newton_iterations = iterations
    s = ran.generate(n_samples=N_SAMPLES, n_seeds=N_SEEDS)
    return np.asarray(context.nparray_from_context_array(s)).ravel()


def analytic_pdf(t):
    # (A/t^2) exp(-B t), normalised to unit area on [t0, t1]
    raw = (rA / t**2) * np.exp(-rB * t)
    grid = np.linspace(t0, t1, 20000)
    y = (rA / grid**2) * np.exp(-rB * grid)
    norm = np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(grid))  # trapezoid
    return raw / norm


cpu = sample(xo.ContextCpu())
gpu = sample(xo.ContextCupy())

ks = ks_2samp(cpu, gpu)

# binned KL divergence (CPU || GPU)
edges = np.linspace(t0, t1, 201)
centers = 0.5 * (edges[:-1] + edges[1:])
h_cpu, _ = np.histogram(cpu, bins=edges, density=True)
h_gpu, _ = np.histogram(gpu, bins=edges, density=True)
p = h_cpu * np.diff(edges) + 1e-12
q = h_gpu * np.diff(edges) + 1e-12
p, q = p / p.sum(), q / q.sum()
kl = float(np.sum(p * np.log(p / q)))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
fig.suptitle(
    f"RandomRutherford  —  CPU vs GPU (Cupy)   |   {N_SAMPLES:,} samples"
    f"   |   KS = {ks.statistic:.4f} (p = {ks.pvalue:.2f})   |   KL = {kl:.2e}",
    fontsize=12,
)

# (1) PDF overlay, log-y
ax = axes[0]
ax.hist(cpu, bins=edges, density=True, histtype="stepfilled", alpha=0.35,
        color="C0", label="CPU")
ax.hist(gpu, bins=edges, density=True, histtype="step", lw=1.8,
        color="C3", label="GPU (Cupy)")
ax.plot(centers, analytic_pdf(centers), "k--", lw=1.2, label="analytic Rutherford")
ax.set_yscale("log")
ax.set_xlabel("t  (momentum-transfer variable)")
ax.set_ylabel("probability density")
ax.set_title("Sampled distribution (log scale)")
ax.legend()

# (2) empirical CDF overlay with the KS gap marked
ax = axes[1]
xs = np.linspace(t0, t1, 2000)
cdf_cpu = np.searchsorted(np.sort(cpu), xs) / cpu.size
cdf_gpu = np.searchsorted(np.sort(gpu), xs) / gpu.size
ax.plot(xs, cdf_cpu, color="C0", lw=1.8, label="CPU")
ax.plot(xs, cdf_gpu, color="C3", lw=1.2, ls="--", label="GPU (Cupy)")
imax = int(np.argmax(np.abs(cdf_cpu - cdf_gpu)))
ax.annotate(f"max |ΔCDF| = {ks.statistic:.4f}",
            xy=(xs[imax], cdf_cpu[imax]), xytext=(0.45, 0.35),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="0.3"))
ax.set_xlabel("t")
ax.set_ylabel("cumulative probability")
ax.set_title("Empirical CDF (KS statistic)")
ax.legend()

# (3) per-bin ratio GPU/CPU with Poisson noise band
ax = axes[2]
counts_cpu, _ = np.histogram(cpu, bins=edges)
counts_gpu, _ = np.histogram(gpu, bins=edges)
mask = counts_cpu > 0
ratio = np.full_like(centers, np.nan)
ratio[mask] = counts_gpu[mask] / counts_cpu[mask]
noise = np.full_like(centers, np.nan)
noise[mask] = np.sqrt(1.0 / counts_cpu[mask] + 1.0 / counts_gpu[mask])
ax.axhline(1.0, color="k", lw=1.0)
ax.fill_between(centers, 1 - noise, 1 + noise, color="0.8",
                label="±1σ counting noise")
ax.plot(centers, ratio, ".", ms=3, color="C2", label="GPU / CPU")
ax.set_ylim(0.9, 1.1)
ax.set_xlabel("t")
ax.set_ylabel("GPU / CPU  bin-count ratio")
ax.set_title("Bin-by-bin ratio")
ax.legend()

fig.tight_layout(rect=(0, 0, 1, 0.95))
out = "rutherford_cpu_vs_gpu.png"
fig.savefig(out, dpi=130)
print(f"KS={ks.statistic:.5f} (p={ks.pvalue:.3f})  KL={kl:.3e}")
print(f"wrote {out}")
