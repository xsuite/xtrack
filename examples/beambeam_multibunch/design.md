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
    filling_scheme_cw=filling_scheme_cw,
    filling_scheme_acw=filling_scheme_acw,
    bunch_intensity_particles_cw=bunch_intensity_cw,
    bunch_intensity_particles_acw=bunch_intensity_acw,
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

## Bunch-pattern API consistency

The multibunch beam-beam API should use the same bunch-pattern concepts as
Xpart, Xwakes and `BeamStatsMonitor`. Their common public model is:

- `filling_scheme` is a slot-indexed boolean/integer occupancy pattern;
- `filled_slots` contains the corresponding physical slot numbers;
- `bunch_spacing_zeta` is the positive physical distance between adjacent
  slots of that pattern; and
- physical slot `i` is centred at `zeta = -i * bunch_spacing_zeta`.

Occupancy and intensity must remain separate. In particular, a floating-point
array containing the population of every slot should not be called a filling
scheme. The multibunch beam-beam configuration should therefore accept
`filling_scheme_cw` / `filling_scheme_acw` separately from
`bunch_intensity_particles_cw` / `bunch_intensity_particles_acw`. An intensity
can be uniform for all filled slots or slot-indexed when bunch populations are
not uniform. The setup should expose the derived physical slot identifiers as
`filled_slots_cw` and `filled_slots_acw`, rather than the ambiguous
`bunches_cw` and `bunches_acw`.

The normalized filling schemes also determine the exact sizes of the own- and
opposing-bunch arrays allocated in each beam-beam element. The public API
should not expose a separate `num_bunches` argument as reserve capacity: for a
clockwise element, for example, the own arrays are sized from
`len(filled_slots_cw)` and the opposing arrays from
`len(filled_slots_acw)`, with the converse used for the anticlockwise element.
This matches `BeamStatsMonitor` and the Xwakes slicer, where storage follows the
configured pattern rather than a user-visible maximum capacity.

If `set_filling(...)` changes the number of filled slots, it should explicitly
reconfigure or reallocate the affected element arrays. An internal Xobject
capacity may still exist as an implementation detail, but shrinking an active
prefix within a larger reserved allocation must not be part of the public
behavioral contract.

Keeping `harmonic_number` and `bunch_spacing_buckets` in the high-level
beam-beam installation API is useful because encounter pairing needs the
integer ring topology. The normalized setup should additionally expose
`bunch_spacing_zeta`. Any different phase or sign convention needed inside a
beam-beam kernel should be converted at the implementation boundary and should
not change the public bunch-position convention.

There is an existing selection-semantic difference which this refactoring must
not spread further:

- `BeamStatsMonitor.selected_slots` contains physical slot numbers;
- Xpart and Xwakes `bunch_selection` contains ordinal indices into the compact
  list of filled bunches.

For example, with `filling_scheme=[1, 0, 1, 1]`,
`BeamStatsMonitor(selected_slots=[0, 2])` selects physical slots 0 and 2,
whereas the Xpart/Xwakes `bunch_selection=[0, 2]` selects physical slots 0 and
3. The new beam-beam API should use physical slot numbers whenever it exposes
a selection and should call that argument `selected_slots`.

### Scope decision

This PR will not change Xpart, Xwakes or `BeamStatsMonitor`. The latter already
provides the desired physical-slot terminology. Changing the meaning of the
established Xpart/Xwakes `bunch_selection` argument would require a separate
backward-compatible API migration. A possible follow-up is to add
`selected_slots` to those packages, make it mutually exclusive with the legacy
`bunch_selection`, and perform the physical-slot to compact-index conversion
internally.

Add a small cross-package contract test, without Xmask, using a sparse pattern
such as:

```python
filling_scheme = [1, 0, 1, 1]
bunch_spacing_zeta = 5
selected_slots = [0, 3]
```

It should establish that `filled_slots == [0, 2, 3]` and that the selected
physical bunch centres are `[0, -15]`. This catches confusion between physical
slots and compact bunch ordinals, as well as spacing and `zeta`-sign mistakes.

## Implementation plan

The work should be staged so that each intermediate commit remains usable and
the two repositories can be migrated without requiring an atomic Xfields and
Xtrack update.

### Phase 1: freeze current behavior

Add the fast characterization tests before changing either implementation:

- scalar BB2D behavior and serialization;
- current multibunch matching and coherent convolution;
- one-bunch scalar/multibunch equivalence;
- sparse fillings, unequal intensities and periodic matching;
- a small deterministic Xtrack installer/setup test;
- multibunch Twiss mode comparisons; and
- the cross-package bunch-pattern contract described above.

These tests are the normal development loop. Pytrain and Xmask remain the
realistic final acceptance tests.

### Phase 2: extend BB2D without removing the old element

In Xfields, first extract the scalar BB2D field and kick calculation into a
shared C helper. Then add the optional multibunch storage and mode selection to
`BeamBeamBiGaussian2D`, preserving all existing scalar fields, constructor
defaults and pipeline behavior.

Keep `BeamBeamBiGaussianMultibunch2D` temporarily and compare the two elements
directly from identical input data. This Xfields stage is backward compatible,
so the current Xtrack train implementation continues to work until it is
explicitly migrated.

### Phase 3: migrate the Xtrack train to the extended BB2D

Change the current train installer to construct `BeamBeamBiGaussian2D` in
multibunch mode. Apply the bunch-pattern API decisions in the same stage:

- separate `filling_scheme_cw` / `filling_scheme_acw` from the corresponding
  `bunch_intensity_particles_*` inputs;
- expose `filled_slots_cw`, `filled_slots_acw` and `bunch_spacing_zeta`; and
- translate the common public negative-`zeta` slot convention at the kernel
  boundary if the internal matching implementation needs another convention.

Do not otherwise change the solver physics or iteration algorithm during this
migration.

### Phase 4: consolidate installation and configuration

Refactor the existing workflow incrementally:

1. Extract encounter generation from the current installer.
2. Extract Twiss/survey geometry and coordinate transformations.
3. Make the conventional and multibunch paths consume those helpers.
4. Add `mode='multibunch'` to `install_beambeam_interactions(...)`.
5. Add the multibunch filling, intensity and emittance inputs to
   `configure_beambeam_interactions(...)`.
6. Return `MultibunchBBSetup` for the genuinely stateful operations.

Keep `install_multibunch_beambeam(...)` as a temporary bridge and compare its
normalized output against the consolidated path after each step.

### Phase 5: migrate examples and remove the duplicate paths

Once the old/new equivalence tests pass:

- migrate all examples to the consolidated install/configure workflow;
- remove `install_multibunch_beambeam(...)` and its environment façade;
- remove `BeamBeamBiGaussianMultibunch2D`, its C header, export and prebuilt
  kernel entry; and
- replace temporary bridge comparisons with permanent behavioral tests.

The duplicate Xfields element is removed only after Xtrack no longer depends
on it, keeping the cross-repository history bisectable.

### Phase 6: full validation

Run checks in increasing order of cost:

1. Focused Xfields element tests.
2. Focused Xtrack installer and multibunch-Twiss tests.
3. Existing Xfields and Xtrack beam-beam suites.
4. Serialization, xdeps knobs, prebuilt kernels and CPU/OpenMP contexts.
5. The pytrain regression.
6. The slow Xmask tests.

## Test plan before refactoring

The xmask tests provide important end-to-end protection for the established
LHC beam-beam workflow, but they are too slow to be the main development loop.
Before changing the element or installer, add a compact test layer that runs in
seconds and isolates failures. Keep the xmask and pytrain tests as final
realistic acceptance tests.

### 1. Xfields element tests

Extend `xfields/tests/test_beambeam_multibunch2d.py` using construction helpers
such as `_make_scalar_bb(...)` and `_make_multibunch_bb(...)`. During the
migration only the multibunch factory should need to change from
`BeamBeamBiGaussianMultibunch2D` to the multibunch mode of
`BeamBeamBiGaussian2D`.

Protect the current multibunch behavior with focused tests for:

- one-bunch scalar/multibunch equivalence for round and elliptical beams;
- several bunches with distinct centroids, intensities and sizes;
- coherent and incoherent covariance handling;
- an unmatched bunch receiving exactly zero kick;
- positive and negative offsets with periodic wrapping;
- unsorted updates preserving the association between `zeta`, centroids,
  populations and covariances;
- scalar size broadcasting and per-bunch sizes;
- capacity errors and changes in the active bunch count;
- `scale_strength` at `0`, an intermediate value and `1`.

Strengthen the scalar BB2D characterization tests to cover:

- modern and legacy constructors producing identical fields and kicks;
- `ref_shift_x/y` and `post_subtract_px/py`;
- nontrivial transverse covariance fields;
- copy and dictionary/JSON round trips;
- construction inside a line and xdeps control of `scale_strength`;
- the existing pipeline updater remaining functional.

Where the two paths execute the same kick formula, require agreement close to
machine precision. Use broader tolerances only when comparing genuinely
different numerical algorithms.

### 2. Xtrack installation and configuration tests

Add `xtrack/tests/test_multibunch_beambeam.py` based on a small deterministic
two-beam environment: four or eight RF slots, two IP markers, at most one
long-range encounter per side and three populated bunches with different
intensities. Most tests should set `survey_separation=False`; use a separate
small curved lattice for the survey-sign checks.

Test the installation layout exactly:

- element names, order and number of head-on/LR encounters;
- longitudinal positions and CW/ACW suffixes;
- pairing offsets and their signs;
- `zeta_period` and matching tolerance;
- allocated and active bunch counts;
- `beambeam_scale` references.

Test configuration and geometry:

- explicit IP offsets supplied as a mapping;
- inferred offsets supplied through an IP list;
- scalar and per-IP long-range encounter counts;
- assigned populations and covariances;
- CW/ACW transverse coordinate transformations;
- nonzero survey separation on the curved toy lattice.

Test the stateful setup independently:

- `set_filling(...)` with a smaller filling;
- rejection of a filling that exceeds the allocated capacity;
- a short symmetric two-beam solve;
- `load_solution(...)`;
- static and dynamic-beta updates;
- `second_order_maps(...)` preserving the exact beam-beam elements.

### 3. Multibunch Twiss tests

Add `xtrack/tests/test_twiss_multibunch.py` using a `LineSegmentMap` and one 2D
beam-beam lens, following the small generic example without plotting or
external model data.

Cover:

- `full`, `fast` and `fast_orbit` on the same bunches;
- closed-orbit and fractional-tune agreement against `full`;
- beta, alpha, dispersion and phase from `fast` against `full`;
- `MultiBunchTwiss` integer, named-row and attribute access;
- explicit `zeta` input and bunch positions read from `Particles`;
- invalid argument combinations, unsupported kwargs and empty inputs;
- lost particles and closed-orbit failure handling.

The tests should explicitly document the current difference in `qx/qy`
semantics: `fast_orbit` exposes fractional tunes while `fast` exposes
accumulated tunes.

### 4. Temporary old/new equivalence tests

Keep both implementations available briefly during each migration.

For the element, construct the old multibunch element and the new multibunch
mode of BB2D from identical data, track identical particles and compare the
kicks directly.

For installation, build two copies of the toy environment. Configure one with
`install_multibunch_beambeam(...)` and the other with the multibunch mode of
the existing install/configure workflow. Compare normalized snapshots
containing:

- encounter names and positions;
- IP offsets and geometry;
- all active per-bunch element arrays;
- scale-knob expressions;
- per-bunch Twiss results after a small number of solver iterations.

After equivalence is established and the duplicate implementation is removed,
replace these bridge comparisons with permanent behavioral assertions against
the normalized expected result.

### 5. Test and refactoring sequence

Keep the preparatory tests and implementation changes in separate commits:

1. `Add BB2D and multibunch characterization tests`.
2. `Add toy multibunch installer and Twiss tests`.
3. `Share the BB2D kick implementation`.
4. `Add multibunch mode to BeamBeamBiGaussian2D`.
5. `Migrate multibunch train to BB2D`.
6. `Align multibunch bunch-pattern API`.
7. `Share beam-beam encounter and geometry configuration`.
8. `Add multibunch mode to install/configure workflow`.
9. `Migrate examples and remove duplicate APIs`.
10. `Add final regression coverage`.

The first two commits are the immediate next work. Old/new equivalence tests
remain active through commits 4--8, and the pytrain and Xmask suites are run as
final acceptance after the duplicate paths have been removed.

## Non-goals

- This rationalization does not change the intended coherent multibunch
  physics.
- It does not make the existing pipeline strong-strong updater multibunch
  aware.
- It does not decide whether the fast multibunch Twiss implementation should
  later be refactored to reuse more of standard Twiss; that is a separate
  cleanup topic.
