# Tracking Time Examples

This folder contains small timing examples for comparing Xtrack tracking
performance on the same FCC-ee example line. The scripts are intended as
interactive benchmarks: edit the switches at the top of a file, run it, and
compare the generated particle-turn timing curves.

These examples are useful when checking whether a change affects:

- the cost of a radiation model for one execution context;
- the relative performance of CPU serial, CPU OpenMP, and GPU CuPy tracking;
- the crossover with particle count, where fixed overheads become less
  important than the cost per particle-turn.

The timings are hardware- and configuration-dependent. They should be read as
local benchmarking tools, not as fixed reference numbers for Xtrack.

## Common Setup

All scripts use:

```text
examples/fcc_ee_solenoid/fccee_z_lcc.json
line: fccee_p_ring
```

Each scan doubles the number of particles until the last tracked batch exceeds
`TIME_LIMIT`. The plotted quantity is:

```text
track time / number of turns / number of particles
```

in microseconds per particle-turn.

Radiation-enabled cases first compensate mean radiation energy loss, then time
the requested radiation model. Stochastic radiation modes initialize the random
number generator before tracking.

Most scripts expose:

- `TIME_LIMIT`: stop condition for the particle-count scan;
- `N_TURNS`: number of turns per timed track call;
- `N_PARTICLES_INIT`: first particle count;
- `OPTIMIZE_FOR_TRACKING`: whether to call `line.optimize_for_tracking()`
  before building timed trackers.

`OPTIMIZE_FOR_TRACKING` changes the interpretation of the result. Leaving it
off measures the imported example line more directly. Turning it on measures a
line representation closer to production tracking use.

## Comparing Radiation Modes

The `compare_radiations/` scripts fix one execution context and compare:

```text
None
mean
quantum
quantum-kick
```

Files:

- `compare_radiations/001_cpu_single.py`: radiation-mode comparison on the
  default single-thread CPU context.
- `compare_radiations/002_cpu_openmp.py`: radiation-mode comparison on CPU
  OpenMP. Set `OPEN_MP_THREADS` at the top of the file.
- `compare_radiations/003_gpu_cupy.py`: radiation-mode comparison on GPU CuPy.

These scripts answer: for a fixed tracker context, how much does each radiation
model cost?

The `quantum` mode samples individual photons. The `quantum-kick` mode samples
the stochastic total radiation kick without producing individual photon records.
This distinction is central when evaluating GPU performance.

## Comparing Tracker Contexts

The `compare_trackers/` scripts fix one radiation mode and compare selected
tracker contexts:

```text
CPU Single Thread
CPU OpenMP
GPU CuPy
```

Files:

- `compare_trackers/001_none.py`: tracker-context comparison with radiation
  disabled.
- `compare_trackers/002_mean.py`: tracker-context comparison with mean
  radiation.
- `compare_trackers/003_quantum.py`: tracker-context comparison with
  photon-by-photon quantum radiation.
- `compare_trackers/004_quantum_kick.py`: tracker-context comparison with
  quantum-kick radiation.

These scripts answer: for a fixed physics mode, which execution context is
fastest over the scanned particle-count range?

## CPU Versus GPU Across Radiation Modes

`compare_default_cupy.py` compares the default CPU context and GPU CuPy for all
radiation modes in one plot. This is the broad end-to-end view: it shows both
the context difference and how that difference changes when radiation is
enabled.

Use this script when the question is practical turnaround time between local
CPU tracking and GPU tracking for the same line and radiation mode.

## Interpreting Results

The low-particle-count region is often dominated by fixed costs such as tracker
setup, kernel launch, and synchronization. The high-particle-count region is
more representative of per-particle tracking throughput.

For GPU radiation studies, compare at least:

- no radiation versus mean radiation, to estimate the baseline radiation code
  path cost;
- `quantum` versus `quantum-kick`, to isolate the cost of photon-by-photon
  sampling;
- CPU OpenMP versus GPU CuPy, to check whether the GPU speedup is meaningful
  against a realistic multi-core CPU baseline.
