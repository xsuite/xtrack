# BeamStatsMonitor Design Notes

This note summarizes the proposed design for a new `xtrack.BeamStatsMonitor`.
The goal is to provide one efficient monitor for beam distribution statistics,
covering the main use cases currently spread across:

- `xfields.CollectiveMonitor`
- `xtrack.BeamPositionMonitor`
- `xtrack.BeamSizeMonitor`
- `xcoll.EmittanceMonitor`

The first implementation should focus on the new monitor itself. Wrappers and
deprecation of existing monitors can be considered later.

## Main Concept

`BeamStatsMonitor` records weighted beam statistics at one or more aggregation
levels:

```text
beam  -> one statistic per logged turn
bunch -> one statistic per logged turn and selected bunch
slice -> one statistic per logged turn, selected bunch, and slice
```

The most detailed available level is determined by the constructor inputs. If
no bunch or slice inputs are provided, only whole-beam statistics are recorded.
If bunch inputs are provided without slice inputs, per-bunch statistics are
recorded and whole-beam statistics are available as a reduction. If slice
inputs are provided, per-slice statistics are recorded and per-bunch and
whole-beam statistics are available as reductions.

For bunched beams, the slice grid is defined per selected bunch.

Coasting-beam support is intentionally deferred. The first kernel-first
implementation should focus on whole-beam, bunch-by-bunch, and slice-by-slice
statistics for bunched beams. Coasting behavior, wrapping, and public shape
conventions will be designed after the bunched implementation is stable.

The efficient bunched-beam scaling should be:

```text
n_logged_turns * n_selected_bunches * n_slices
```

The monitor should be a self-contained logging element with its own tracking
kernel. It should not inherit from `ElementWithSlicer` and should not store data
through `CompressedProfile`.

The kernel should directly accumulate weighted primitive moments into the
monitor storage:

```text
particles -> active-particle filter -> turn/slot/slice bin -> atomicAdd
```

`UniformBinSlicer` remains a current-pass slicing primitive for other
collective elements. `BeamStatsMonitor` can share low-level C helper functions
with the slicer where this stays clean, but monitor-specific concepts such as
turn logging, storage shape, reductions, and HDF5 output should remain in
`BeamStatsMonitor`.

## Proposed User API

Example for the whole beam:

```python
mon = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=1000,
    stats=["num_particles", "mean_x", "mean_y", "sigma_x", "sigma_y"],
)

mon.mean_x.shape
# (n_logged_turns,)

mon.get("mean_x", turn=100)
# scalar
```

Example for bunch-by-bunch statistics:

```python
mon = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=1000,
    filled_slots=[0, 1, 2, 3, 4, 5],
    selected_slots=[2, 3],
    bunch_spacing_zeta=25.0,
    stats=["num_particles", "mean_x", "mean_y"],
)

mon.mean_x.shape
# (n_logged_turns, n_selected_bunches)

mon.get("mean_x", slot=2)
# shape (n_logged_turns,)

mon.get("mean_x", level="beam")
# shape (n_logged_turns,)
```

Example for slice-by-slice statistics in one bunch:

```python
mon = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=1000,
    zeta_range=(-0.2, 0.2),
    num_slices=64,
    stats=["num_particles", "mean_x", "mean_y", "sigma_x", "sigma_y"],
)

mon.mean_x.shape
# (n_logged_turns, 1, n_slices)

mon.get("mean_x", level="bunch")
# shape (n_logged_turns, 1)
```

Example for a train of consecutive bunches:

```python
mon = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=1000,
    zeta_range=(-0.2, 0.2),
    num_slices=64,
    num_bunches=72,
    bunch_spacing_zeta=25.0,
    stats=["num_particles", "mean_x", "mean_y", "sigma_x", "sigma_y"],
)
```

Example with explicit filled slots and a monitored subset:

```python
mon = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=1000,
    zeta_range=(-0.2, 0.2),
    num_slices=64,
    filled_slots=[0, 1, 2, 3, 4, 5],
    selected_slots=[2, 3],
    bunch_spacing_zeta=25.0,
    stats=["num_particles", "mean_x", "mean_y"],
)

mon.get("mean_x", level="slice", slot=2)
# shape (n_logged_turns, n_slices)

mon.get("mean_x", level="bunch", slot=[2, 3])
# shape (n_logged_turns, 2)
```

## Filling Scheme and Bunch Selection

The first version should keep the filling API simple.

Accepted inputs:

- `num_bunches`: number of consecutive filled slots, starting at slot 0.
- `filled_slots`: explicit physical filled slot numbers.
- `filling_scheme`: low-level explicit boolean/integer filling scheme.
- `selected_slots`: physical slot numbers to monitor.

Rules:

```text
if filling_scheme is provided:
    use it as authoritative
elif filled_slots is provided:
    use it
else:
    filled_slots = np.arange(num_bunches)
```

Defaults:

```text
without bunch or slice inputs:
    use beam mode with no bunch axis

in bunch or slice mode:
    num_bunches = 1
    selected_slots = all filled slots
```

`selected_slots` should be the public API for monitoring only a subset of
bunches. Internally it can be converted to the existing `bunch_selection`
semantics, namely indices into `filled_slots`.

This preserves the important MPI use case:

```python
filled_slots = np.arange(72)
selected_slots = split_slots_for_rank(filled_slots, rank, size)
```

The output bunch axis should follow `selected_slots` order.

## Turn Selection

The monitor should support logging only a subset of turns.

MVP arguments:

```python
start_at_turn=100
stop_at_turn=200
every_n_turns=10
```

This logs:

```text
100, 110, 120, ..., 190
```

Public arrays should be indexed by logged records, not by all tracked turns:

```python
mon.turns
# array([100, 110, ..., 190])

mon.mean_x.shape
# (n_logged_turns, ...)
```

A future extension can allow:

```python
turns=[100, 101, 105, 200]
```

with `turns` mutually exclusive with `start_at_turn`, `stop_at_turn`,
and `every_n_turns`.

The tracking kernel should not infer the current turn from particle 0. Mixed
`at_turn` values in one particles object are acceptable. For each active
particle, the kernel should use that particle's own `at_turn` to decide whether
the particle contributes and to compute the logged-record index.

This implies:

```text
if particle.state <= 0:
    skip
elif particle.at_turn is not logged:
    skip
else:
    i_record = record index corresponding to particle.at_turn
    accumulate into that record
```

For the MVP regular-grid turn selection, the record index is:

```text
i_record = (particle.at_turn - start_at_turn) // every_n_turns
```

after checking that:

```text
start_at_turn <= particle.at_turn < stop_at_turn
(particle.at_turn - start_at_turn) % every_n_turns == 0
```

This policy avoids the ambiguity of using `particles.at_turn[0]` when particle
0 is lost or stale, and it naturally supports particles from multiple turns in
the same array.

## Weighted Statistics

The public quantity should be called:

```python
num_particles
```

not `count`.

`num_particles` is the sum of particle weights in the bin:

```text
num_particles = sum(particles.weight)
```

All statistics should be weighted by `particles.weight`.

Primitive accumulated data should be weighted sums:

```text
sum_weight
sum_weight_x
sum_weight_px
sum_weight_y
sum_weight_py
sum_weight_zeta
sum_weight_delta
sum_weight_x_x
sum_weight_x_px
sum_weight_px_px
...
```

Statistics:

```python
mon.num_particles
mon.mean_x
mon.sigma_x
mon.cov_x_px
mon.gemitt_x
mon.nemitt_x
mon.gemitt_x_projected
mon.nemitt_x_projected
```

Definitions:

```text
mean_x = sum(w * x) / sum(w)
cov_x_px = sum(w * x * px) / sum(w) - mean_x * mean_px
sigma_x = sqrt(sum(w * x * x) / sum(w) - mean_x**2)
```

For beam moments, use normalization by total weight. Do not apply an unbiased
statistical estimator correction by default.

## Supported Statistics in the MVP

The public API should accept high-level `stats` and internally expand them to
the required weighted moments.

MVP stats:

```text
num_particles
mean_x, mean_px, mean_y, mean_py, mean_zeta, mean_delta
sigma_x, sigma_px, sigma_y, sigma_py, sigma_zeta, sigma_delta
cov_x_px, cov_y_py, cov_zeta_delta
gemitt_x, gemitt_y, gemitt_zeta
nemitt_x, nemitt_y, nemitt_zeta
gemitt_x_projected, gemitt_y_projected, gemitt_zeta_projected
nemitt_x_projected, nemitt_y_projected, nemitt_zeta_projected
```

The default emittances should be the coupled normal-mode emittances. The labels
`x`, `y`, and `zeta` identify the modes sorted to be closest to the horizontal,
vertical, and longitudinal coordinates.

Projected emittances should be explicitly named with the `_projected` suffix.
For example:

```text
gemitt_x_projected = sqrt(var_x * var_px - cov_x_px**2)
nemitt_x_projected = gemitt_x_projected * beta0 * gamma0
```

and similarly for `y`.

For longitudinal projected emittance, the coordinate convention needs to be
explicit. The preferred full coupled covariance should use:

```text
(x, px, y, py, zeta, pzeta)
```

with:

```text
pzeta = ptau / beta0
```

The existing `xfields.CollectiveMonitor` uses `delta` for longitudinal
projected emittance, while `xcoll.EmittanceMonitor` uses `pzeta`. The new
monitor should make the convention explicit and avoid silently mixing them.

## Coupled Emittances and Optics From Sigma

The monitor should support coupled emittances as done by
`xcoll.EmittanceMonitor`, and should also allow optics estimation from the
measured covariance matrix using the method shown in:

```text
xtrack/examples/twiss/027_optics_from_sigma_mat.py
```

These calculations should be performed as Python-side postprocessing from the
stored weighted moments, not in the tracking kernel.

For each logged record and bin, reconstruct the weighted covariance matrix:

```text
Sigma = cov(x, px, y, py, zeta, pzeta)
```

To support coupled emittances and optics, the `stats` expansion must request
all first moments and all pairwise second moments for:

```python
coords = ["x", "px", "y", "py", "zeta", "pzeta"]
```

Then compute the normal-mode emittances from:

```python
from xtrack.linear_normal_form import S, sort_modes

eival, eivec = np.linalg.eig(Sigma @ S)
modes = sort_modes(eivec, eival)

gemitt_x = eival[modes[0]].imag
gemitt_y = eival[modes[1]].imag
gemitt_zeta = eival[modes[2]].imag

nemitt_x = gemitt_x * beta0 * gamma0
nemitt_y = gemitt_y * beta0 * gamma0
nemitt_zeta = gemitt_zeta * beta0 * gamma0
```

The mode sorting should reuse the `xtrack.linear_normal_form.sort_modes`
convention. This sorts the modes to be closest to:

```text
mode 0 -> x-like
mode 1 -> y-like
mode 2 -> zeta-like
```

based on the eigenvectors. The labels therefore mean "mode closest to x/y/zeta"
according to this sorting, not an invariant identity in pathological cases.

It would be useful to factor the shared logic into a helper in `xtrack`, for
example:

```python
xt.get_modes_from_sigma(Sigma)
```

or initially a private helper used by `BeamStatsMonitor`. The helper should
return the sorted eigenvalues/eigenvectors, the coupled emittances, and enough
pairing information to reconstruct a dummy map for optics estimation.

For optics estimation, follow the example in `027_optics_from_sigma_mat.py`:

1. Compute eigenvalues and eigenvectors of `Sigma @ S`.
2. Sort the modes consistently.
3. Construct a dummy stable one-turn matrix with arbitrary phase advances and
   the same eigenvectors.
4. Call `twiss(..., R_matrix=dummy_R, chrom=False)` on a dummy line.

The monitor API should expose this as a method rather than storing optics
arrays by default:

```python
tw_from_sigma = mon.optics_from_sigma(
    turn=100,
    slot=0,
    slice=None,  # None can mean aggregate over slices
)
```

For vectorized use, a later extension can provide:

```python
tw_table = mon.get_optics_from_sigma(aggregate_slices=True)
```

The implementation needs the full ordered conjugate eigenmode pairs to build
the dummy map. `sort_modes` currently returns one positive-orientation member
of each physical pair; for optics reconstruction we should either extend the
helper to return ordered pairs or add a companion function for this purpose.

Safeguards:

- if `num_particles` is below a configurable threshold, return `nan` or warn;
- if `Sigma @ S` is singular, ill-conditioned, or rank deficient, return `nan`
  or warn;
- sorting can be ambiguous for nearly degenerate modes or very strong coupling,
  and this should be documented.

## Data Shape and Access

The default statistic level is the most detailed level available from the
monitor:

```text
beam mode  -> (n_logged_turns,)
bunch mode -> (n_logged_turns, n_selected_bunches)
slice mode -> (n_logged_turns, n_selected_bunches, n_slices)
```

The available reductions are exposed through:

```python
mon.available_levels
# e.g. ("beam", "bunch", "slice")

mon.default_level
# e.g. "slice"
```

Common public properties:

```python
mon.turns
mon.selected_slots
mon.zeta_centers
mon.num_particles
mon.mean_x
mon.sigma_x
mon.nemitt_x
mon.nemitt_x_projected
```

Raw statistic arrays remain NumPy arrays. For direct indexing, helper methods
translate physical coordinates to array indices:

```python
mon.record_index(100)      # machine turn -> logged-record index
mon.slot_index(3)          # physical slot -> selected-slot axis index
mon.slice_index(zeta=0.03) # zeta coordinate -> slice axis index
```

The primary convenience getter accepts a `level` selector and physical
selectors:

```python
mon.get("mean_x", turn=100)
mon.get("mean_x", level="beam", turn=100)
mon.get("mean_x", level="bunch", turn=100, slot=3)
mon.get("mean_x", level="slice", turn=100, slot=3, slice_index=12)
```

Scalar selectors remove the corresponding axis unless `keepdims=True`.

The monitor should avoid exposing internal ring-buffer indices as the primary
API. If the first logged turn is turn 100, then `mon.mean_x[0]` and
`mon.get("mean_x", turn=100)` refer to the same logged turn at the default
level.

Reductions must be computed from weighted primitive sums, not by averaging
statistics. For example, bunch-level `sigma_x` from slice data is
computed by first summing `sum_weight`, `sum_weight_x`, and `sum_weight_x_x`
over slices, then applying the weighted variance formula.

## Tracking Kernel Architecture

`BeamStatsMonitor` should own the arrays that store raw weighted moments over
logged records. The tracking kernel should write directly into these arrays
with atomic additions.

The kernel responsibilities are:

```text
for each active particle:
    read particle.at_turn
    reject particles outside the logged turn selection
    compute logged-record index
    compute selected slot index when needed
    compute slice index when needed
    atomicAdd requested raw weighted moments
```

The kernel should skip lost particles (`state <= 0`). It should not require the
active particles to be compacted and should not assume that particle 0 is active
or representative of the current turn.

The monitor should support three internal binning configurations:

```text
beam mode:
    one bin per logged turn
    no zeta cut
    no slot or slice axis

bunch mode:
    one bin per logged turn and selected slot
    zeta is used only to identify the selected slot
    no public slice axis

slice mode:
    one bin per logged turn, selected slot, and slice
    zeta is used for both slot and slice assignment
```

The monitor should store primitive sums, not statistics. Python-side
accessors compute means, sigmas, covariances, emittances, and reductions from
the stored primitive sums. Coupled emittances and optics-from-sigma remain
Python-side postprocessing.

`UniformBinSlicer` should not be extended to include monitor logging semantics.
If duplication becomes a maintenance issue, factor only the low-level C
building blocks that are genuinely shared:

```text
active-particle handling
selected-slot lookup
zeta -> slot mapping
zeta -> slice mapping
weighted moment atomic accumulation
```

The public `BeamStatsMonitor` behavior should not depend on `ElementWithSlicer`
or `CompressedProfile`.

## HDF5 Output

Use HDF5 only for the first implementation. Do not add Zarr, Parquet, JSON, or
CSV backends in the MVP.

File output should be optional and import `h5py` lazily:

```python
mon = xt.BeamStatsMonitor(
    ...,
    output_file="monitor.h5",
)
```

If `output_file` is not provided, no `h5py` import should be required.
If `output_file` is provided, `h5py` is imported during monitor construction and
the output file is opened in write mode to create a fresh BeamStatsMonitor HDF5
file. Any existing file at that path is replaced at construction time, not at
the first `save_to_file()` call.

The first implementation should keep file output minimal. The monitor always
keeps one configured frame of primitive moment arrays in memory. Providing
`output_file` enables HDF5 export in addition to the in-memory frame; it is not
a true file-backed streaming storage mode.

There is no public `storage` mode and no `buffer_size`/flush-cadence argument in
the first implementation. The presence or absence of `output_file` fully defines
the behavior:

```text
output_file is None      -> in-memory only
output_file is provided  -> in-memory plus HDF5 output
```

For simulations too large to keep in one monitor allocation, the monitor should
facilitate a user-managed frame loop:

```python
mon = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=1000,
    output_file="monitor.h5",
    stats=["num_particles", "mean_x", "sigma_x"],
)

for start in range(0, num_turns, 1000):
    if start != 0:
        mon.start_new_frame(start_at_turn=start)
    line.track(particles, num_turns=1000)
    mon.save_to_file()
```

`start_new_frame(start_at_turn)` clears the primitive moment arrays and retargets
the same-size logged-turn frame to a new start turn. It also clears the internal
per-record touched flags used by `save_to_file()`. It keeps `every_n_turns` and the
number of logged records fixed, and computes:

```text
stop_at_turn = start_at_turn + n_records_per_frame * every_n_turns
```

The first implementation should not support arbitrary frame resizing through
`start_new_frame`; users who need a different frame size should create a new
monitor.

The file should contain the requested statistics by default. This keeps
the HDF5 output immediately usable without requiring a reader to reconstruct
means, sigmas, covariances, or emittances from primitive sums. Reductions should
still be computed from primitive sums before writing; never derive coarser
statistics by averaging finer-level statistics.

Primitive moment output is intentionally not part of the first HDF5
implementation. It can be added later if exact offline reprocessing becomes
important.

Suggested HDF5 layout:

```text
/attrs:
    schema_version
    class = "BeamStatsMonitor"
    stats
    available_levels
    default_level
    every_n_turns
    n_records_per_frame

/turns                         shape (n_written_records,)
/filled_slots                  shape (n_filled_bunches,)
/selected_slots                shape (n_selected_bunches,)
/zeta_centers                  shape (n_selected_bunches, n_slices)

/stats/beam/num_particles      shape (n_written_records,)
/stats/beam/mean_x             shape (n_written_records,)
/stats/beam/sigma_x            shape (n_written_records,)

/stats/bunch/num_particles     shape (n_written_records, n_selected_bunches)
/stats/bunch/mean_x            shape (n_written_records, n_selected_bunches)
/stats/bunch/sigma_x           shape (n_written_records, n_selected_bunches)

/stats/slice/num_particles     shape (n_written_records, n_selected_bunches, n_slices)
/stats/slice/mean_x            shape (n_written_records, n_selected_bunches, n_slices)
/stats/slice/sigma_x           shape (n_written_records, n_selected_bunches, n_slices)
```

Only levels available for the monitor mode should be present. For example, beam
mode writes only `/stats/beam`, bunch mode writes `/stats/beam` and
`/stats/bunch`, and slice mode writes all three levels.

Datasets should be appendable along the first axis:

```python
maxshape=(None, ...)
chunks=(chunk_records, ...)
```

Provide:

```python
mon.save_to_file()
```

for monitors configured with `output_file`. A monitor created without
`output_file` can also be saved later by passing a filename:

```python
mon.save_to_file("monitor.h5")
```

In that case the provided path becomes the monitor output file. If the file does
not exist, `save_to_file()` creates and initializes it. If it already exists,
`save_to_file()` validates the static metadata and appends only newly touched
records from the current in-memory frame. It does not truncate an existing file.

The cleanup semantics are intentionally tied to construction: passing
`output_file` to `BeamStatsMonitor(...)` initializes that path in write mode,
while `save_to_file(path)` is a create-or-append operation.

Internally the monitor keeps one integer flag per record in the current frame.
The tracking kernel sets the flag when any accepted particle contributes to that
record. `save_to_file()` uses the highest touched record to decide how much of the
current frame is ready to write. It uses the last turn already present in
`/turns` to determine which prefix of the current frame was previously flushed.

The touched-record tracking is part of the monitor kernel. After changing this
kernel logic, prebuilt kernels need to be regenerated before the prebuilt-kernel
path can set the flags. Until regenerated kernels are available, development
tests can force JIT compilation, and the Python `save_to_file()` implementation can
fall back to inferring written records from nonzero `num_particles`.

```text
local_start = first current-frame record not already present in /turns
local_stop = highest touched record + 1
append current-frame records [local_start:local_stop)
```

After appending, call the HDF5 flush method and close the file so another script
can inspect the data by reopening it.

Repeated `save_to_file()` calls without newly touched records are no-ops. Later calls
after additional tracking append only the newly touched suffix, which supports
progressive inspection while keeping the full configured frame in memory:

```python
line.track(particles, num_turns=10)
mon.save_to_file()
line.track(particles, num_turns=10)
mon.save_to_file()
```

The first implementation is append-only and assumes flushed frames are not
modified later. This is appropriate for the normal workflow:

```python
line.track(particles, num_turns=...)
mon.save_to_file()
```

If particles are later tracked with `at_turn` values belonging to already flushed
turns, the in-memory monitor and HDF5 file can diverge. The first implementation
does not try to detect or repair this case.

`start_new_frame(start_at_turn)` is a convenience for memory management, not an
automatic tracking controller. The user remains responsible for calling it with a
start turn consistent with the particles' `at_turn` values before tracking the
next frame. When `output_file` is active, users should call `save_to_file()` before
`start_new_frame(...)`; otherwise the current in-memory frame will be discarded.

## Implementation Status

The current implementation is the kernel-first standalone monitor described
above. `xtrack.BeamStatsMonitor` is implemented in
`xtrack/xtrack/monitors/beam_stats_monitor.py`, with its tracking kernel in
`xtrack/xtrack/monitors/beam_stats_monitor.h`.

Implemented:

1. `xt.BeamStatsMonitor` is a standalone `BeamElement`, not a subclass of
   `ElementWithSlicer`.

2. The monitor owns xobjects storage for primitive weighted sums. The record
   object is an `xobjects.HybridClass`, so its fields are exposed as NumPy-like
   arrays on the Python side and can be zeroed in place.

3. The tracking kernel writes directly into the monitor storage with atomic
   additions. It does not use `UniformBinSlicer` or `CompressedProfile`.

4. The kernel filters lost particles, uses each active particle's own
   `at_turn`, and therefore accepts mixed-turn particle arrays.

5. Regular-grid turn selection is implemented through:

   ```text
   start_at_turn
   stop_at_turn
   every_n_turns
   turns property
   ```

6. Filling and selected-bunch inputs are implemented:

   ```text
   filled_slots
   selected_slots
   filling_scheme
   num_bunches
   bunch_spacing_zeta
   ```

7. Beam, bunch, and slice modes are implemented, with reductions accessed by:

   ```text
   level="beam"
   level="bunch"
   level="slice"
   available_levels
   default_level
   ```

8. Statistics are computed from primitive weighted sums. The
   implemented statistics are:

   ```text
   num_particles
   mean_<coord>
   sigma_<coord>
   cov_<coord1>_<coord2>
   gemitt_<plane>_projected
   nemitt_<plane>_projected
   ```

   where coordinates are currently `x`, `px`, `y`, `py`, `zeta`, and `delta`,
   and projected-emittance planes are `x`, `y`, and `zeta`.

9. Optional HDF5 output is implemented with:

   ```text
   output_file
   save_to_file()
   save_to_file(path)
   touched-record tracking
   append-only newly touched suffix writes
   start_new_frame(start_at_turn)
   statistic datasets by default
   appendable datasets
   stable file metadata
   ```

   Passing `output_file` to the constructor initializes that file in write mode
   immediately. Calling `save_to_file(path)` later creates the file if it does
   not exist, or validates and appends to it if it already exists. Files remain
   flat time series; they do not contain frame groups.

10. `start_new_frame(start_at_turn)` clears the in-memory primitive moment
    arrays and touched-record flags in place, keeps the frame size and
    `every_n_turns` fixed, and retargets the monitor to a new turn interval.

11. Serialization through `to_dict()` stores the monitor configuration only,
    not the recorded data arrays.

Covered by focused tests in `xtrack/tests/test_beam_stats_monitor.py`:

- weighted means, sigmas, covariances, and projected transverse emittance
- beam, bunch, and slice aggregation levels
- selected bunch behavior
- `every_n_turns` turn selection
- mixed `at_turn` particles in one tracking call
- lost-particle filtering, including lost particle 0
- in-place reset of `HybridClass` record arrays
- HDF5 schema, statistics layout, and absence of `/frames` and `/moments`
- construction-time file initialization/truncation through `output_file`
- `save_to_file(path)` create-or-append behavior
- rejection of incompatible non-empty existing HDF5 files
- progressive `save_to_file()` calls without rewriting prior records
- `start_new_frame(start_at_turn)` followed by flat append into the same file
- configuration-only `to_dict()` / `Line.to_dict()` behavior

Still deferred:

1. Coupled normal-mode emittances without the `_projected` suffix. Requests for
   these currently raise `NotImplementedError`.

2. Full 6D covariance reconstruction using the preferred
   `(x, px, y, py, zeta, pzeta)` convention.

3. Optics-from-sigma helper methods.

4. Coasting-beam support.

5. Arbitrary turn lists such as `turns=[100, 101, 105, 200]`.

6. Arbitrary frame resizing through `start_new_frame`; users who need a
   different frame size should create a new monitor.

7. Primitive moment output in HDF5. The current HDF5 output writes requested
   statistics only.

8. GPU/OpenMP test coverage where available, and coupled-emittance /
   optics-from-sigma tests once those features exist.

9. Factoring shared C helpers with `UniformBinSlicer`, if this becomes useful
   after the standalone monitor behavior is stable.

10. Wrappers and deprecation strategy for older monitors.

Important convention caveat: longitudinal projected emittance currently uses
`(zeta, delta)` internally. Before implementing full coupled covariance work,
the longitudinal momentum convention needs to be made explicit and aligned with
the preferred `(zeta, pzeta)` convention described above.

Kernel caveat: touched-record tracking is now part of the monitor kernel.
Prebuilt kernels need to be regenerated before the prebuilt-kernel path can set
the touched flags. Development tests can force JIT compilation, and
`save_to_file()` has a Python fallback that infers written records from nonzero
`num_particles` for stale prebuilt kernels.

Examples live in `xtrack/examples/beam_stats_monitor/`:
`000_beam_stats.py` (whole beam), `001_bunch_by_bunch_stats.py` (per bunch),
and `002_slice_by_slice_stats.py` (per slice). The examples are included in
the user guide through
`xsuite/docs/conf.py`, and the guide section is in
`xsuite/docs/particles_monitor.rst`. The API reference entry is in
`xsuite/docs/apireference.rst`.
