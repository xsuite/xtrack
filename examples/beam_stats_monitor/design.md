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

For coasting beams, the monitor should expose one full-turn periodic slice grid
per logged turn. This lets `BeamStatsMonitor` cover the core
functionality of `BeamPositionMonitor` and `BeamSizeMonitor`: sampled
turn-by-turn centroids, intensities, sizes, and any other weighted statistic
supported by the new monitor. Coasting mode uses a different slicing convention
from bunched mode: it does not require a `zeta_range`, and the slice grid spans
one full machine turn.

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

Example for a coasting beam sampled within each turn:

```python
mon = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=1000,
    coasting=True,
    num_slices=256,
    stats=["num_particles", "mean_x", "mean_y", "sigma_x", "sigma_y"],
)

mon.mean_x.shape
# (n_logged_turns, n_slices)

mon.get("mean_x", level="slice", turn=100)
# shape (n_slices,)

tt = mon.time_centers(line_length=line.get_length(), beta0=particles.beta0[0])
plt.plot(tt.ravel(), mon.mean_x.ravel())
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

## Coasting Beams

Coasting mode should be enabled explicitly:

```python
mon = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=1000,
    coasting=True,
    num_slices=256,
    stats=["num_particles", "mean_x", "mean_y", "sigma_x", "sigma_y"],
)
```

In coasting mode, `num_slices` is required and defines the number of periodic
samples per machine turn. Internally the monitor stores one pseudo-bunch per
logged turn to reuse the slice-mode kernel indexing, but this pseudo-bunch axis
is not exposed in the public API:

```text
public shape = (n_logged_turns, n_slices)
internal storage shape = (n_logged_turns, 1, n_slices)
available_levels = ("beam", "slice")
default_level = "slice"
```

Coasting mode should reject bunched-beam inputs whose meaning would be
ambiguous:

```text
zeta_range
filled_slots
filling_scheme
selected_slots
bunch_spacing_zeta
num_bunches != 1
```

The kernel should recover the circumference from the local particle, as done in
`track_rf.h`:

```c
double const line_length = part->line_length;
```

For each active particle, coasting mode computes the continuous arrival-turn
coordinate:

```text
u = particle.at_turn - particle.zeta / line_length
```

This is the same physical convention as the existing sampled monitors, whose
sample index is based on:

```text
(particle.at_turn - start_at_turn) / frev
    - particle.zeta / (particle.beta0 * c)
```

using `line_length = beta0 * c / frev`.

The particle is assigned to the nearest logged reference turn and a periodic
slice within that turn:

```text
effective_turn = floor(u + 0.5)
relative_turn_fraction = u - effective_turn  # approximately [-0.5, 0.5)
i_slice = floor((relative_turn_fraction + 0.5) * num_slices)
```

Boundary handling should be periodic. If roundoff gives `i_slice == num_slices`,
wrap it to `0` and increment the effective turn consistently, or otherwise use
an equivalent robust implementation that keeps the slice index in:

```text
0 <= i_slice < num_slices
```

After `effective_turn` is computed, regular turn selection applies to that
effective turn:

```text
start_at_turn <= effective_turn < stop_at_turn
(effective_turn - start_at_turn) % every_n_turns == 0
```

and:

```text
i_record = (effective_turn - start_at_turn) // every_n_turns
```

Statistics involving `zeta` should use the wrapped within-turn coordinate, not
the unbounded particle coordinate:

```text
zeta_wrapped = -relative_turn_fraction * line_length
```

Therefore `mean_zeta`, `sigma_zeta`, `cov_zeta_delta`, `cov_zeta_pzeta`, and
the full covariance moment set remain meaningful for the sampled coasting beam.
The other particle coordinates (`x`, `px`, `y`, `py`, `delta`, `pzeta`) are
accumulated unchanged.

For coasting mode, meter-valued slice centers require the line length. The
plain `zeta_centers` property should therefore remain reserved for bunched
slice grids, while line-length-aware helpers compute the periodic coasting
centers when needed. The periodic within-turn centers have shape:

```text
(n_slices,)
```

centered on the reference particle:

```text
zeta_centers[i] = -(((i + 0.5) / n_slices) - 0.5) * line_length
```

This is the center of the same periodic binning used by the kernel for
`zeta_wrapped`, and should be computed inside the line-length-aware helpers.

To make multi-turn sampled plots convenient, the monitor should expose a
data-oriented helper:

```python
tt = mon.time_centers(line_length=line.get_length(), beta0=particles.beta0[0])
```

For slice-like modes this returns an array with the same longitudinal-grid
shape as the most detailed statistics:

```text
coasting mode -> (n_logged_turns, n_slices)
bunched slice mode -> (n_logged_turns, n_selected_bunches, n_slices)
```

The helper should use:

```text
time_centers = (turns * line_length - zeta_centers) / (beta0 * c)
```

with broadcasting over turns, selected slots, and slices. This is useful for
coasting beams and also for bunched slice plots over multiple turns:

```python
tt = mon.time_centers(line_length=line.get_length(), beta0=particles.beta0[0])
plt.plot(tt.ravel(), mon.mean_x.ravel())
```

A lower-level helper returning unwrapped longitudinal centers can also be added
if useful:

```python
zz = mon.zeta_centers_unwrapped(line_length=line.get_length())
```

with:

```text
zeta_centers_unwrapped = zeta_centers - turns * line_length
```

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
num_particles
sum_x
sum_px
sum_y
sum_py
sum_zeta
sum_delta
sum_pzeta
sum_x_x
sum_x_px
sum_px_px
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
mean_x, mean_px, mean_y, mean_py, mean_zeta, mean_delta, mean_pzeta
sigma_x, sigma_px, sigma_y, sigma_py, sigma_zeta, sigma_delta, sigma_pzeta
cov_x_px, cov_y_py, cov_zeta_delta, cov_zeta_pzeta
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
gemitt_zeta_projected = sqrt(var_zeta * var_pzeta - cov_zeta_pzeta**2)
nemitt_x_projected = gemitt_x_projected * beta0 * gamma0
```

and similarly for `y` and `zeta`.

For longitudinal projected emittance, the coordinate convention is explicit.
The canonical full coupled covariance uses:

```text
(x, px, y, py, zeta, pzeta)
```

with:

```text
pzeta = ptau / beta0
```

The existing `xfields.CollectiveMonitor` uses `delta` for longitudinal
projected emittance, while `xcoll.EmittanceMonitor` uses `pzeta`. The new
monitor uses `pzeta` for longitudinal projected emittance and coupled
covariance work. `delta` remains available as an ordinary logged coordinate and
can be requested through statistics such as `mean_delta`, `sigma_delta`, and
`cov_zeta_delta`.

The stored pairwise moments intentionally include `zeta_delta` and
`zeta_pzeta`, but not `delta_pzeta`. Therefore `cov_zeta_delta` and
`cov_zeta_pzeta` are supported, while `cov_delta_pzeta` is rejected.

## Coupled Emittances and Optics From Covariance

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

Coupled normal-mode emittances should be ordinary requestable statistics:

```python
mon = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=1000,
    stats=[
        "gemitt_x", "gemitt_y", "gemitt_zeta",
        "nemitt_x", "nemitt_y", "nemitt_zeta",
    ],
)

mon.gemitt_x
mon.get("nemitt_y", level="bunch", slot=3)
```

Internally these statistics are computed from:

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

The same covariance reconstruction should also support optics-like scalar
statistics inferred from the covariance matrix. These should be requestable
through `stats` so that they are available as arrays, through `get(...)`, and in
HDF5 output like other statistics:

```python
mon = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=1000,
    stats=[
        "betx", "alfx",
        "bety", "alfy",
        "betzeta", "alfzeta",
        "dx", "dpx", "dy", "dpy",
    ],
)

mon.betx
mon.get("dx", level="slice", turn=100, slot=3, slice_index=12)
```

The optics-like stats should expand to the same full 6D covariance moment set as
the coupled normal-mode emittances. This keeps the tracking kernel unchanged:
the kernel still only accumulates primitive weighted moments, and all mode and
optics calculations remain Python-side postprocessing.

For convenience, grouped stat aliases can be added:

```python
stats=["normal_mode_emittances"]
stats=["covariance_optics"]
```

with expansions:

```text
normal_mode_emittances ->
    gemitt_x, gemitt_y, gemitt_zeta,
    nemitt_x, nemitt_y, nemitt_zeta

covariance_optics ->
    betx, alfx, bety, alfy, betzeta, alfzeta,
    dx, dpx, dy, dpy
```

The scalar names should remain accepted directly, so users can request only the
small subset they need.

It would be useful to factor the shared covariance-mode logic into a helper in
`xtrack`, for example:

```python
xt.get_modes_from_sigma(Sigma)
```

or initially a private helper used by `BeamStatsMonitor`. The helper should
return the sorted eigenvalues/eigenvectors, the coupled emittances, the
normalizing matrix, and the derived optics-like quantities.

For covariance optics, follow the example in `027_optics_from_sigma_mat.py`:

1. Compute eigenvalues and eigenvectors of `Sigma @ S`.
2. Sort the modes consistently.
3. Build the conjugate eigenvector pairs from the three sorted mode
   representatives and their complex conjugates.
4. Construct a dummy stable one-turn matrix with arbitrary phase advances and
   the same eigenvectors, if this remains the cleanest way to reuse existing
   normal-form/Twiss machinery.
5. Extract the `W_matrix`, Twiss parameters, and dispersion. The dummy map is an
   implementation detail and should not be part of the public result.

The monitor API should also expose a scalar diagnostic method returning a dict
for one selected bin:

```python
out = mon.optics_from_covariance(
    level="beam",
    turn=100,
)

out = mon.optics_from_covariance(
    level="slice",
    turn=100,
    slot=3,
    slice_index=12,
)
```

The dict should omit the internal dummy `R_matrix`, but include the measured
covariance, emittances, optics-like quantities, and enough metadata to diagnose
the result:

```python
{
    "status": "ok",
    "message": "",
    "covariance_matrix": Sigma,
    "covariance_order": ("x", "px", "y", "py", "zeta", "pzeta"),
    "W_matrix": W,
    "gemitt_x": gemitt_x,
    "gemitt_y": gemitt_y,
    "gemitt_zeta": gemitt_zeta,
    "nemitt_x": nemitt_x,
    "nemitt_y": nemitt_y,
    "nemitt_zeta": nemitt_zeta,
    "betx": betx,
    "alfx": alfx,
    "bety": bety,
    "alfy": alfy,
    "betzeta": betzeta,
    "alfzeta": alfzeta,
    "dx": dx,
    "dpx": dpx,
    "dy": dy,
    "dpy": dpy,
    "num_particles": num_particles,
    "beta0_gamma0": beta0_gamma0,
    "condition_number": condition_number,
}
```

The method should be scalar in the first implementation: one selected
turn/slot/slice or aggregation level in, one dict out. Vectorized table-like
output can be added later if needed. If the monitor did not store the full
covariance moment set, this method should raise a clear error explaining that
the user must request one of the coupled-emittance or covariance-optics stats.

Safeguards:

- if `num_particles` is below a configurable threshold, return `nan` values for
  the optics/emittance fields in the diagnostic dict and for the corresponding
  stat arrays;
- if `Sigma @ S` is singular, ill-conditioned, or rank deficient, return `nan`
  values rather than raising during ordinary statistic access;
- sorting can be ambiguous for nearly degenerate modes or very strong coupling,
  and this should be documented.

Warnings should be opt-in or limited to the scalar diagnostic method. Per-slice
workflows can naturally produce many sparse or ill-conditioned bins, and warning
on every failed bin would be too noisy for normal use.

## Data Shape and Access

The default statistic level is the most detailed level available from the
monitor:

```text
beam mode  -> (n_logged_turns,)
bunch mode -> (n_logged_turns, n_selected_bunches)
slice mode -> (n_logged_turns, n_selected_bunches, n_slices)
coasting mode -> (n_logged_turns, n_slices)
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
mon.zeta_centers  # bunched slice grids
mon.time_centers(line_length=line.get_length(), beta0=particles.beta0[0])
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

In coasting mode, no public bunch level or slot selector is exposed.
`slice_index(...)` maps wrapped within-turn `zeta` to the periodic slice grid.
The `time_centers(...)` helper returns an array suitable for plotting sampled
statistics over many turns.

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
computed by first summing `num_particles`, `sum_x`, and `sum_x_x` over slices,
then applying the weighted variance formula.

## Tracking Kernel Architecture

`BeamStatsMonitor` should own the arrays that store raw weighted moments over
logged records. The tracking kernel should write directly into these arrays
with atomic additions.

The kernel responsibilities are:

```text
for each active particle:
    read particle.at_turn
    in coasting mode:
        compute effective_turn and wrapped zeta from zeta / line_length
    otherwise:
        effective_turn = particle.at_turn
        wrapped zeta = particle.zeta
    reject particles outside the logged effective-turn selection
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

coasting mode:
    one periodic full-turn slice grid
    zeta and at_turn jointly define effective turn and slice
    zeta moments use the wrapped within-turn coordinate
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
`/stats/bunch`, bunched slice mode writes all three levels, and coasting mode
writes `/stats/beam` and `/stats/slice` with slice datasets shaped
`(n_written_records, n_slices)`. Coasting mode should not write the internal
pseudo-slot as `/stats/bunch`, `/filled_slots`, or `/selected_slots`.

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
kernel logic or the record layout, prebuilt kernels need to be regenerated before
running the prebuilt-kernel path.

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
   gemitt_<plane>
   nemitt_<plane>
   gemitt_<plane>_projected
   nemitt_<plane>_projected
   betx, alfx, bety, alfy, betzeta, alfzeta
   dx, dpx, dy, dpy
   ```

   where coordinates are currently `x`, `px`, `y`, `py`, `zeta`, `delta`, and
   `pzeta`, and emittance planes are `x`, `y`, and `zeta`. Coupled normal-mode
   emittances and covariance optics use the canonical
   `(x, px, y, py, zeta, pzeta)` covariance matrix. The `zeta`
   projected-emittance plane uses `(zeta, pzeta)`.
   The `delta_pzeta` covariance pair is intentionally not stored or exposed.

   The grouped stat aliases `normal_mode_emittances` and `covariance_optics`
   are expanded at construction time to the corresponding scalar statistic
   names.

9. The scalar `optics_from_covariance(...)` method returns a diagnostic dict for
   one selected bin. It includes the measured covariance matrix, covariance
   coordinate order, `W_matrix`, coupled emittances, covariance-optics scalar
   quantities, `num_particles`, `beta0_gamma0`, condition number, and
   status/message fields. The internal dummy map is not exposed.

10. Optional HDF5 output is implemented with:

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

11. `start_new_frame(start_at_turn)` clears the in-memory primitive moment
    arrays and touched-record flags in place, keeps the frame size and
    `every_n_turns` fixed, and retargets the monitor to a new turn interval.

12. Serialization through `to_dict()` stores the monitor configuration only,
    not the recorded data arrays.

Covered by focused tests in `xtrack/tests/test_beam_stats_monitor.py`:

- weighted means, sigmas, supported covariances, pzeta statistics, and projected
  transverse/longitudinal emittance
- coupled normal-mode emittances, covariance-optics scalar stats, grouped stat
  aliases, and `optics_from_covariance(...)`
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

Next validation and review work:

1. Test the covariance-derived emittances and optics with realistic generated
   distributions, for example particles produced by `generate_gaussian_bunch`
   or another matched/generated distribution workflow. These tests should check
   that the monitor recovers the expected coupled emittances, Twiss-like
   parameters, and dispersion within statistical tolerance.

2. Review the covariance-optics implementation before extending the public
   surface further. In particular, check the numerical failure policy, mode
   sorting conventions, `W_matrix` normalization, sparse-bin behavior, and
   whether any helper logic should be moved out of `BeamStatsMonitor` into a
   shared xtrack utility.

3. Check whether closed-orbit offsets are handled correctly in covariance
   optics. The monitor computes covariance from weighted central moments, but
   the interpretation of the returned `W_matrix`, Twiss-like quantities, and
   dispersion should be validated when the particle distribution is centered on
   a non-zero closed orbit or when the monitor is placed in a line with a
   non-zero local orbit.

Still deferred:

1. Coasting-beam implementation according to the design above.

2. Arbitrary turn lists such as `turns=[100, 101, 105, 200]`.

3. Arbitrary frame resizing through `start_new_frame`; users who need a
   different frame size should create a new monitor.

4. Primitive moment output in HDF5. The current HDF5 output writes requested
   statistics only.

5. GPU/OpenMP test coverage where available.

6. Vectorized table-like output for covariance optics if this becomes useful.

7. Factoring shared C helpers with `UniformBinSlicer`, if this becomes useful
   after the standalone monitor behavior is stable.

8. Wrappers and deprecation strategy for older monitors.

Longitudinal convention: both `delta` and `pzeta` are logged coordinates.
Longitudinal projected emittance and future coupled covariance work use the
canonical `(zeta, pzeta)` pair.
The `delta_pzeta` cross term is not part of the current stored moment set.

Kernel note: touched-record tracking is part of the monitor kernel. When the
record layout or kernel logic changes, prebuilt kernels need to be regenerated
before running the prebuilt-kernel path.

Examples live in `xtrack/examples/beam_stats_monitor/`:
`000_beam_stats.py` (whole beam), `001_bunch_by_bunch_stats.py` (per bunch),
and `002_slice_by_slice_stats.py` (per slice). The examples are included in
the user guide through
`xsuite/docs/conf.py`, and the guide section is in
`xsuite/docs/particles_monitor.rst`. The API reference entry is in
`xsuite/docs/apireference.rst`.
