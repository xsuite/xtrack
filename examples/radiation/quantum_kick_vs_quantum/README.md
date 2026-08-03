# Quantum-kick versus quantum

This folder is for physics validation of `quantum-kick` against the
photon-by-photon `quantum` radiation model.

The scope is deliberately narrow:

- compare only the public models `quantum` and `quantum-kick`;
- validate total emitted-energy and kick distributions, including tails;
- check stored total-energy-loss tables without regenerating the full
  production table set;
- keep timing benchmarks out of this folder, except for algorithmic call-count
  diagnostics.

`quantum` generates individual photons. `quantum-kick` does not. It samples an
equivalent stochastic total-energy kick, so equivalence is expected only for
total radiation kicks and downstream tracking observables, not for photon
records.

## Validation sequence

The implemented active sequence currently contains six examples:

```text
001_review_total_energy_tables.py
002_validate_table_sampling.py
003_single_bend_kick_distribution.py
004_regime_scan_kick_distribution.py
005_sampler_call_counts.py
006_fcc_emittance_cpu.py
```

`007_lifetime_tail_proxy.py` is retained as a documented, non-executable
placeholder. It should become active only when a machine acceptance and loss
model are available for a defensible lattice-level lifetime comparison.

These are interactive, deliberately high-statistics validation examples. They
are intended to be run manually when reviewing radiation-physics changes, not
as routine automated tests. The automated test suite checks model
configuration, while this folder provides the more expensive distribution,
tail and lattice-level comparisons.

### 001 Review total-energy tables

Inspect the stored total-energy-loss tables before sampling or tracking:

- table metadata, grid size, direct-table range, and tail coverage;
- monotonicity and finite values of the stored inverse CDFs;
- selected CDF/inverse-CDF plots;
- regenerate a small direct subset, expected to be around `N=1..4`, and compare
  it against the stored defaults.

This gives a quick check that the table data and generator have not drifted in
the low-count region where mistakes would be most visible.

### 002 Validate table sampling

Compare table-sampled fixed-photon-count losses against brute-force
photon-by-photon Monte Carlo:

- fixed counts `1, 2, 4, 8, 16, 32, 64, 128`;
- mean, rms, quantiles, and tail probabilities;
- explicit high-quantile checks such as `q99`, `q999`, and `q9999`.

This validates the random variables sampled by the `quantum-kick` machinery
before any tracking element is involved.

### 003 Single-bend kick distribution

Track many particles through one bend with `quantum` and `quantum-kick`:

- compare `dpx`, `dpy`, `ddelta`, and emitted-energy proxies;
- include both bulk and tail metrics;
- make the single default case easy to inspect with plots.

This checks the full element kick path in a controlled setting.

### 004 Regime scan of kick distributions

Repeat the single-bend kick comparison across emitted-photon regimes:

- very low lambda, mostly zero photons;
- one-photon dominated;
- medium lambda;
- high lambda;
- counts above the direct-table range.

Each regime should report zero-photon fraction, one-photon fraction,
direct-table usage, decomposed-table usage, and tail metrics.

This checks that the physics agreement holds where the algorithm changes
behavior.

### 005 Sampler call counts

Compare algorithmic work rather than wall-clock timing:

- expected photon samples for `quantum`;
- expected table lookups for `quantum-kick`;
- direct-table versus power-of-two decomposition use.

This explains why `quantum-kick` is faster while leaving real timing
comparisons to `examples/tracking_time`.

### 006 FCC emittance CPU check

Run a CPU-only FCC-ee tt tracking-observable check:

- `quantum` versus `quantum-kick`;
- `1024` particles and `150` turns by default;
- all particles initially at zero;
- compare normal modes I and III within explicit tolerances;
- use an `xcoll.EmittanceMonitor` and configurable turn-window smoothing.

This is a first lattice-level physics sanity check, not a high-statistics
equilibrium-emittance certification. The current MAD-X tt sequence is an ideal
planar lattice, so its equilibrium mode-II emittance is zero and only
modes I and III are meaningful here. This should be upgraded to a committed,
native-Xsuite, coupled FCC-ee tt LCC lattice when one becomes available, at
which point the full I/II/III comparison should be restored.

### 007 Lifetime tail proxy

This file is intentionally a future-work specification. Examples 002-004
already cover distribution and kick tails, so another standalone tail sampler
would be redundant. A useful 007 must instead provide a lattice-level loss
observable with:

- a documented RF bucket, momentum acceptance and physical aperture;
- an explicit loss definition;
- survival and loss-location comparisons for `quantum` and `quantum-kick`;
- confidence intervals and PASS, FAIL or LOWSTAT reporting.

It must not present an arbitrary kick threshold as a machine lifetime.
