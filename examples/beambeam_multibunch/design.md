# Multibunch beam-beam design rationalization

This note records a proposed cleanup of the multibunch beam-beam work before
the API becomes established. The goal is to retain the new physics and solver
capabilities while integrating them into the existing Xfields and Xtrack
abstractions.

## Motivation

The current implementation introduces two concepts parallel to existing ones:

1. `BeamBeamBiGaussianMultibunch2D` duplicates most of
   `BeamBeamBiGaussian2D`.
2. `env.xfields.install_multibunch_beambeam(...)` duplicates much of the
   existing `install_beambeam_interactions(...)` and
   `configure_beambeam_interactions(...)` workflow.

Keeping the parallel implementations would create two places to maintain the
same bi-Gaussian kick, encounter placement, survey geometry, optics, coordinate
conventions, scale knobs, serialization and prebuilt-kernel registration. It
would also make it unclear which API should receive future beam-beam features
and fixes.

The multibunch-specific behavior is narrower than these duplicated surfaces:

- Match a tracked bunch to an opposing bunch using `zeta`, with periodic ring
  wrapping.
- Store and update per-bunch centroids, populations and covariances.
- In coherent mode, use the convolution of the two matched bunch covariances.
- Solve the two beams self-consistently and optionally update dynamic beta.

These capabilities can be added behind the existing element and installation
interfaces without changing the established single-bunch behavior.

## Rationalization 1: extend `BeamBeamBiGaussian2D`

`BeamBeamBiGaussian2D` should remain the public 2D bi-Gaussian beam-beam
element. It should gain an optional multibunch source rather than having a
second public element class.

The legacy scalar fields and constructor arguments must remain unchanged. This
is important for existing line files, xdeps references, collective updates and
user code. Optional multibunch arrays should be added under distinct internal
names, together with:

- the active number and allocated capacity of own and opposing bunches;
- the sorted own- and opposing-beam `zeta` grids;
- `zeta_offset`, `zeta_match_tol` and `zeta_period`;
- per-bunch centroids, populations and transverse covariance components;
- a flag selecting incoherent weak-strong or coherent rigid-bunch behavior.

The tracking kernel should have two stages:

1. Select scalar kick parameters. In legacy mode these come directly from the
   existing scalar fields. In multibunch mode they come from the bunch matched
   on `zeta`; if no opposing bunch is matched, no kick is applied.
2. Call one common BB2D kick implementation with the selected centroid,
   population and covariance.

For coherent tracking, the effective covariance is the sum of the matched own
and opposing bunch covariances. Using covariances instead of separate
`sigma_x`/`sigma_y` logic preserves the existing BB2D representation and leaves
room for transverse coupling.

The existing `ref_shift_*`, `post_subtract_*`, covariance rotation,
`scale_strength` and relativistic scaling must be shared by both modes.
Initially, pipeline-based `config_for_update` and the multibunch source should
be mutually exclusive: the current pipeline updater exchanges the aggregate
moments of one bunch and has different semantics from the rigid multibunch
model.

Once migrated, remove:

- `BeamBeamBiGaussianMultibunch2D`;
- its separate C tracking header;
- its top-level export and prebuilt-kernel registration.

Its tests should instead exercise the multibunch mode of
`BeamBeamBiGaussian2D`, including an exact comparison with the scalar mode.

## Rationalization 2: extend the existing installation workflow

The public entry point should remain
`env.xfields.install_beambeam_interactions(...)`, followed by
`env.xfields.configure_beambeam_interactions(...)`. A parallel
`install_multibunch_beambeam(...)` entry point should not be needed.

The existing configuration machinery already owns:

- encounter generation and placement;
- clockwise/anticlockwise conventions;
- bunch-slot delays and encounter pairing;
- Twiss and survey geometry;
- beam covariance computation;
- opposing-beam coordinate transformations and separation;
- beam-beam strength knobs;
- storage of beam-beam configuration on the environment.

These operations should be factored into element-independent helpers that
produce an encounter description and its geometry. The conventional and
multibunch workflows should consume the same description.

The installation API can select a mode, for example:

```python
env.xfields.install_beambeam_interactions(
    clockwise_line='lhcb1',
    anticlockwise_line='lhcb2',
    ip_names=['ip1', 'ip2', 'ip5', 'ip8'],
    num_long_range_encounters_per_side=45,
    harmonic_number=35640,
    bunch_spacing_buckets=10,
    mode='multibunch',
)

setup = env.xfields.configure_beambeam_interactions(
    filling_pattern_cw=filling_cw,
    filling_pattern_acw=filling_acw,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
)
```

The default mode must preserve the existing sliced head-on and long-range
workflow. Multibunch mode installs the extended BB2D element with one 2D lens
per head-on or long-range encounter and allocates its bunch arrays.
Configuration loads the filling, populations, geometry and design
covariances, then returns a `MultibunchBBSetup`.

`MultibunchBBSetup` remains useful, but should contain only the genuinely
stateful multibunch operations:

- `set_filling(...)`;
- `solve(...)`;
- `second_order_maps(...)`;
- `load_solution(...)`;
- convergence and optional dynamic-beta updates.

Element placement, survey geometry, design covariance calculation, reversed
beam transformations and beam-beam knob creation should move to the shared
configuration layer. In particular, beam sizes should come from the standard
Twiss beam-covariance API rather than a separate `sqrt(beta * nemitt / gamma)`
calculation.

`configure_orbit_dependent_parameters_for_bb(...)` cannot be applied unchanged
to the coherent multibunch solve: it subtracts the reference dipole kick,
whereas the solver intentionally retains that kick to find the distorted
closed orbit. Its lower-level coordinate conventions may still be shared.

## Implementation plan

1. **Protect current behavior.** Add focused tests for scalar BB2D,
   multibunch matching, coherent covariance convolution, missing encounters,
   periodic wrapping and per-bunch updates.
2. **Share the C kick.** Extract the existing scalar BB2D field and kick
   calculation into a helper without changing results.
3. **Add optional multibunch storage to BB2D.** Preserve all existing scalar
   fields and defaults; validate that multibunch and pipeline-update modes are
   not combined.
4. **Migrate the multibunch tests and examples.** Instantiate
   `BeamBeamBiGaussian2D` in multibunch mode and verify equivalence with the
   current branch.
5. **Remove the duplicate element.** Delete its Python class, header, export
   and prebuilt-kernel entry.
6. **Extract shared encounter and geometry helpers.** Reuse the existing
   beam-beam configuration tables and the standard Twiss covariance results.
7. **Add multibunch mode to install/configure.** Preserve the existing default
   API and behavior. Make the returned setup the owner of only the stateful
   solver operations.
8. **Remove the parallel installer.** Delete
   `install_multibunch_beambeam(...)` and its environment façade after all
   examples use the shared workflow.
9. **Run compatibility checks.** Cover line serialization, xdeps strength
   knobs, prebuilt kernels, CPU/OpenMP contexts, existing beam-beam tests and
   the pytrain regression.

## Non-goals

- This rationalization does not change the intended coherent multibunch
  physics.
- It does not make the existing pipeline strong-strong updater multibunch
  aware.
- It does not decide whether the fast multibunch Twiss implementation should
  later be refactored to reuse more of standard Twiss; that is a separate
  cleanup topic.
