# Multibunch beam-beam design rationalization

This note records a proposed cleanup of the multibunch beam-beam work before
the API becomes established. The goal is to retain the new physics and solver
capabilities while integrating them into the existing Xfields and Xtrack
abstractions.

## Motivation

The current implementation introduces two concepts parallel to existing ones:

1. `BeamBeamBiGaussianRigidBunch2D` originally duplicated the physical kick
   calculation in `BeamBeamBiGaussian2D`.
2. `env.xfields.install_multibunch_beambeam(...)` duplicates much of the
   existing `install_beambeam_interactions(...)` and
   `configure_beambeam_interactions(...)` workflow.

The physical kick and the installation/configuration workflow should not have
parallel implementations. The element storage, however, has a real structural
difference: scalar BB2D has fixed-size data, while the rigid-bunch element owns
filling-dependent arrays and must be reallocated when the number of bunches
changes. Combining those layouts would enlarge every scalar BB2D element,
introduce a per-particle mode branch and complicate the API without avoiding
rigid-bunch reallocation.

The multibunch-specific behavior is narrower than these duplicated surfaces:

- Match a tracked bunch to an opposing bunch using `zeta`, with periodic ring
  wrapping.
- Store and update per-bunch centroids, populations and covariances.
- In coherent mode, use the convolution of the two matched bunch covariances.
- Solve the two beams self-consistently and optionally update dynamic beta.

These capabilities should therefore keep a dedicated element class, while the
physical kick and installation machinery are shared wherever their behavior is
genuinely common.

## Rationalization 1: keep two element classes and share the kick

Keep both public element classes because they represent different storage and
tracking contracts:

- `BeamBeamBiGaussian2D` keeps its compact scalar layout, established
  constructor, weak-strong behavior and pipeline strong-strong updater.
- `BeamBeamBiGaussianRigidBunch2D` owns the filling-dependent bunch arrays and
  the rigid-bunch train behavior.

The classes must call one common BB2D kick helper that owns:

- covariance rotation;
- the bi-Gaussian field calculation;
- relativistic and charge scaling;
- strength scaling; and
- the final transverse momentum kick and optional post-subtraction.

Each wrapper remains responsible for selecting those inputs. Scalar BB2D reads
its scalar fields. The rigid-bunch wrapper matches the opposing bunch on the
periodic `zeta` grid, selects its centroid, population and covariance, and, in
coherent mode, adds the matched own-beam covariance before calling the helper.
The bunch-matching code is multibunch-specific and does not need to be added to
the scalar element.

For coherent tracking, the effective covariance is the sum of the matched own
and opposing bunch covariances. Using covariances instead of separate
`sigma_x`/`sigma_y` logic remains consistent with scalar BB2D and leaves room
for transverse coupling.

Rigid-bunch array lengths should be inferred exactly from the normalized own and
opposing filling data. There is no public reserve-capacity contract. Changing
the number of bunches explicitly reconfigures/reallocates the rigid-bunch
elements; updates that preserve the filling can modify their data in place.

Keeping the classes separate also makes the semantics visible in the API:
pipeline strong-strong remains a configuration of scalar BB2D, whereas
`mode='rigid_bunch'` constructs `BeamBeamBiGaussianRigidBunch2D`.
“Multibunch” describes the capability shared by both approaches; “rigid bunch”
identifies the physical approximation used by this workflow. “Fixed point” is
reserved for the solver algorithm, and “train” for the collection of bunches.
Permanent tests should
include one-bunch and per-selected-bunch comparisons against scalar BB2D so the
shared physical model remains locked down.

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
rigid-bunch workflows should consume the same description.

The installation API can select a mode, for example:

```python
env.xfields.install_beambeam_interactions(
    clockwise_line='lhcb1',
    anticlockwise_line='lhcb2',
    ip_names=['ip1', 'ip2', 'ip5', 'ip8'],
    num_long_range_encounters_per_side=45,
    harmonic_number=35640,
    bunch_spacing_buckets=10,
    mode='rigid_bunch',
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
workflow. Rigid-bunch mode installs the extended BB2D element with one 2D lens
per head-on or long-range encounter and allocates its bunch arrays.
Configuration loads the filling, populations, geometry and design
covariances, then returns a `RigidBunchBBSetup`.

`RigidBunchBBSetup` remains useful, but should contain only the genuinely
stateful rigid-bunch operations:

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
to the coherent rigid-bunch solve: it subtracts the reference dipole kick,
whereas the solver intentionally retains that kick to find the distorted
closed orbit. Its lower-level coordinate conventions may still be shared.

## Bunch-pattern API consistency

The rigid-bunch beam-beam API should use the same bunch-pattern concepts as
Xpart, Xwakes and `BeamStatsMonitor`. Their common public model is:

- `filling_scheme` is a slot-indexed boolean/integer occupancy pattern;
- `filled_slots` contains the corresponding physical slot numbers;
- `bunch_spacing_zeta` is the positive physical distance between adjacent
  slots of that pattern; and
- physical slot `i` is centred at `zeta = -i * bunch_spacing_zeta`.

Occupancy and intensity must remain separate. In particular, a floating-point
array containing the population of every slot should not be called a filling
scheme. The rigid-bunch beam-beam configuration should therefore accept
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

### Phase 2: share Xfields physics without merging element layouts

In Xfields, extract the scalar BB2D field and kick calculation into a shared C
helper and call it from both element kernels. Preserve the scalar BB2D Xobject
layout, constructor defaults and pipeline behavior. Keep rigid-bunch matching,
filling-dependent arrays and update methods on
`BeamBeamBiGaussianRigidBunch2D`.

Then change the rigid-bunch element to allocate exactly from the supplied own and
opposing bunch data. Remove the public reserve-capacity arguments. A filling
change should explicitly reconstruct/reconfigure the affected elements.

### Phase 3: align the Xtrack train API

Keep the train on `BeamBeamBiGaussianRigidBunch2D` and apply the bunch-pattern
API decisions:

- separate `filling_scheme_cw` / `filling_scheme_acw` from the corresponding
  `bunch_intensity_particles_*` inputs;
- expose `filled_slots_cw`, `filled_slots_acw` and `bunch_spacing_zeta`; and
- translate the common public negative-`zeta` slot convention at the kernel
  boundary if the internal matching implementation needs another convention;
- reconfigure the filling-dependent elements when the number of bunches
  changes.

Do not otherwise change the solver physics or iteration algorithm during this
migration.

### Phase 4: consolidate installation and configuration

Refactor the existing workflow incrementally:

1. Extract encounter generation from the current installer.
2. Extract Twiss/survey geometry and coordinate transformations.
3. Make the conventional and rigid-bunch paths consume those helpers.
4. Add `mode='rigid_bunch'` to `install_beambeam_interactions(...)`.
5. Add the rigid-bunch filling, intensity and emittance inputs to
   `configure_beambeam_interactions(...)`.
6. Return `RigidBunchBBSetup` for the genuinely stateful operations.

Keep `install_multibunch_beambeam(...)` as a temporary bridge and compare its
normalized output against the consolidated path after each step.

Step 1 is complete: Xfields now provides one logical encounter table containing
the IP, encounter type, signed long-range index, orientation-specific
displacement from the IP and CW/ACW bunch-pairing offsets. The conventional
installer expands head-on slices from this table, while the temporary
rigid-bunch installer renders its own element names and consumes the same
placement and pairing data.

Step 2 is complete for the rigid-bunch path: Xfields now provides an
element-independent geometry helper that evaluates both Twiss tables, obtains
the standard transverse covariance with `twiss.get_beam_covariance()` and
computes local-survey separation without crossing the line seam. The temporary
rigid-bunch installer consumes this result.

The standard two-beam conventional path now also consumes the shared result for
its explicitly oriented CW/ACW Twiss tables, closed-orbit coordinates,
transverse covariance components, local-survey separation and crossing slopes.
A curved toy-ring test compares every paired encounter against the former
MadPoint/survey calculation before checking the configured CW and
counter-rotating ACW elements. Crabbing, the final counter-rotating element
conversion, orbit-dependent kick subtraction and one-beam antisymmetry remain
mode-specific and unchanged.

### Phase 5: migrate examples and remove the duplicate installer path

Once the installer equivalence tests pass:

- migrate all examples to the consolidated install/configure workflow;
- remove `install_multibunch_beambeam(...)` and its environment façade;
- replace temporary bridge comparisons with permanent behavioral tests.

`BeamBeamBiGaussianRigidBunch2D` remains public, with its own serialized layout
and prebuilt-kernel entry, but its physical kick continues to use the common
BB2D helper.

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

Extend `xfields/tests/test_beambeam_rigid_bunch2d.py` with focused scalar and
rigid-bunch construction helpers.

Protect the current rigid-bunch behavior with focused tests for:

- one-bunch scalar/rigid-bunch equivalence for round and elliptical beams;
- several bunches with distinct centroids, intensities and sizes;
- coherent and incoherent covariance handling;
- an unmatched bunch receiving exactly zero kick;
- positive and negative offsets with periodic wrapping;
- unsorted updates preserving the association between `zeta`, centroids,
  populations and covariances;
- scalar size broadcasting and per-bunch sizes;
- exact allocation from the bunch data and explicit reconfiguration when the
  bunch count changes;
- `scale_strength` at `0`, an intermediate value and `1`.

Strengthen the scalar BB2D characterization tests to cover:

- modern and legacy constructors producing identical fields and kicks;
- `ref_shift_x/y` and `post_subtract_px/py`;
- nontrivial transverse covariance fields;
- copy and dictionary/JSON round trips;
- construction inside a line and xdeps control of `scale_strength`;
- the existing pipeline updater remaining functional.

Before migrating the conventional installer/configurer, add a fast
characterization in `xfields/tests/test_beambeam_config_tools.py`. It must run
the public two-line install/configure workflow with sliced head-on and
long-range encounters and protect names, partner mapping, positions, delays,
Twiss covariances, nonzero orbit and separation geometry, configured 2D/3D
fields, orbit-dependent subtraction and the global strength knob. This test is
now in place; the slower Xmask LHC tests remain the acceptance layer.

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
- own and opposing array lengths inferred from the fillings;
- `beambeam_scale` references.

Test configuration and geometry:

- explicit IP offsets supplied as a mapping;
- inferred offsets supplied through an IP list;
- scalar and per-IP long-range encounter counts;
- assigned populations and covariances;
- CW/ACW transverse coordinate transformations;
- nonzero survey separation on the curved toy lattice.

Test the stateful setup independently:

- `set_filling(...)` rebuilding the filling-dependent elements when the bunch
  count changes;
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

### 4. Temporary installer equivalence tests

The scalar/rigid-bunch element comparisons are permanent physics tests, not a
temporary migration bridge. They compare each selected rigid-bunch kick against
an independently configured scalar BB2D.

For installation, build two copies of the toy environment. Configure one with
`install_multibunch_beambeam(...)` and the other with rigid-bunch mode in
the existing install/configure workflow. Compare normalized snapshots
containing:

- encounter names and positions;
- IP offsets and geometry;
- all active per-bunch element arrays;
- scale-knob expressions;
- per-bunch Twiss results after a small number of solver iterations.

After equivalence is established and the duplicate installer is removed,
replace these bridge comparisons with permanent behavioral assertions against
the normalized expected result.

### 5. Test and refactoring sequence

Keep the preparatory tests and implementation changes in separate commits:

1. `Add BB2D and multibunch characterization tests`.
2. `Add toy multibunch installer and Twiss tests`.
3. `Share the BB2D kick implementation`.
4. `Infer multibunch element storage from bunch data`.
5. `Align rigid-bunch train bunch-pattern API`.
6. `Share beam-beam encounter and geometry configuration`.
7. `Add rigid-bunch mode to install/configure workflow`.
8. `Migrate examples and remove duplicate installer API`.
9. `Add final regression coverage`.

The first six work packages are complete. Fast characterization protects the
shared encounter and geometry output, including exact comparison with the
former conventional survey calculation. Installer equivalence tests remain
active through commits 7--8; Xmask is used at this completed-geometry
checkpoint and again for final acceptance after the duplicate installer path
has been removed.

## Non-goals

- This rationalization does not change the intended coherent rigid-bunch
  physics.
- It does not make the existing pipeline strong-strong updater multibunch
  aware.
- It does not decide whether the fast multibunch Twiss implementation should
  later be refactored to reuse more of standard Twiss; that is a separate
  cleanup topic.
