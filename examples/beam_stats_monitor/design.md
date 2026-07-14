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

`BeamStatsMonitor` records weighted beam statistics on a longitudinal grid.
For bunched beams, the grid is defined per selected bunch. For coasting beams,
the grid covers one longitudinal domain, typically the full circumference.

The efficient bunched-beam scaling should be:

```text
n_logged_turns * n_selected_bunches * n_slices
```

not:

```text
n_turns * sampling_frequency / frev
```

The monitor should build on the capabilities currently provided by
`xfields.ElementWithSlicer` and `xfields.UniformBinSlicer`. These should be
moved to `xtrack`, with `xfields` importing them from there.

## Proposed User API

Example for one bunch:

```python
mon = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=1000,
    zeta_range=(-0.2, 0.2),
    num_slices=64,
    stats=["num_particles", "mean_x", "mean_y", "sigma_x", "sigma_y"],
)
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
```

Example for a coasting beam:

```python
mon = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=1000,
    coasting=True,
    zeta_range=(-circumference / 2, circumference / 2),
    num_slices=1024,
    stats=["num_particles", "mean_x", "sigma_x"],
)
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

## Coasting Beams

A coasting beam can be represented internally as one longitudinal domain with
many slices:

```text
n_domains = 1
zeta_range = full monitored range, often full circumference
num_slices = number of ring bins
```

The public API should not expose a fake bunch axis by default. For coasting
mode:

```python
mon.mean_x.shape
# (n_logged_turns, n_slices)
```

For bunched mode:

```python
mon.mean_x.shape
# (n_logged_turns, n_selected_bunches, n_slices)
```

An advanced getter can preserve a uniform shape:

```python
mon.get("mean_x", keep_bunch_axis=True)
# always (n_logged_turns, n_domains, n_slices)
```

For coasting beams, consider a later option:

```python
wrap_zeta=True
circumference=circumference
```

Without wrapping, particles outside `zeta_range` are not counted.

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

Public derived quantities:

```python
mon.num_particles
mon.mean_x
mon.sigma_x
mon.cov_x_px
mon.gemitt_x
mon.nemitt_x
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
gemitt_x, gemitt_y, gemitt_zeta_delta
nemitt_x, nemitt_y, nemitt_zeta_delta
```

The transverse emittance definitions are:

```text
gemitt_x = sqrt(var_x * var_px - cov_x_px**2)
nemitt_x = gemitt_x * beta0 * gamma0
```

and similarly for `y`.

For longitudinal emittance, use explicit naming in the first version:

```text
gemitt_zeta_delta
nemitt_zeta_delta
```

This avoids hiding the convention difference between existing monitors that use
`delta` and `pzeta = ptau / beta0`.

## Data Shape and Access

Canonical stored shape:

```text
(n_logged_turns, n_selected_bunches, n_slices)
```

For coasting mode, the public accessors may drop the bunch axis:

```text
(n_logged_turns, n_slices)
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
```

The monitor should avoid exposing internal ring-buffer indices as the primary
API. If the first logged turn is turn 100, then:

```python
mon.mean_x[0]
```

should correspond to turn 100.

## HDF5 Output

Use HDF5 only for the first implementation. Do not add Zarr, Parquet, JSON, or
CSV backends in the MVP.

File output should be optional and import `h5py` lazily:

```python
mon = xt.BeamStatsMonitor(
    ...,
    output_file="monitor.h5",
    buffer_size=1000,
)
```

If `output_file` is not provided, no `h5py` import should be required.

`buffer_size` is the number of logged records kept before flushing, not the
number of machine turns. With `every_n_turns=10`, a `buffer_size` of 1000
corresponds to 1000 logged records, i.e. 10000 tracked turns.

Suggested HDF5 layout:

```text
/turns                         shape (n_records_total,)
/selected_slots                shape (n_selected_bunches,)
/zeta_centers                  shape (n_selected_bunches, n_slices)
/stats/num_particles           shape (n_records_total, n_selected_bunches, n_slices)
/stats/mean_x                  shape (n_records_total, n_selected_bunches, n_slices)
/stats/sigma_x                 shape (n_records_total, n_selected_bunches, n_slices)
...
```

Datasets should be appendable along the first axis:

```python
maxshape=(None, n_selected_bunches, n_slices)
chunks=(chunk_size, n_selected_bunches, n_slices)
```

After each flush, call the HDF5 flush method so another script can inspect the
file by reopening it.

Provide:

```python
mon.flush()
```

to manually flush a partially filled buffer.

## Storage Modes

The first implementation can support:

```text
memory
file
memory_and_file
```

Suggested behavior:

- `memory`: keep all logged records in memory.
- `file`: keep only the current buffer in memory and append flushed data to HDF5.
- `memory_and_file`: keep all logged records and also append to HDF5.

If `output_file` is provided and `storage` is not specified, a reasonable
default is:

```text
storage = "file"
```

For short simulations without `output_file`:

```text
storage = "memory"
```

## Implementation Steps

1. Move `UniformBinSlicer` and `ElementWithSlicer` capabilities from `xfields`
   to `xtrack`.

2. Update `xfields` to import the moved slicer infrastructure from `xtrack`.

3. Add `xt.BeamStatsMonitor` using the moved slicer machinery.

4. Implement turn selection:

   ```text
   start_at_turn
   stop_at_turn
   every_n_turns
   turns property
   ```

5. Implement weighted primitive accumulation for the required moments.

6. Implement high-level `stats` expansion to required moments.

7. Implement public derived properties:

   ```text
   num_particles
   means
   sigmas
   covariances
   projected geometric and normalized emittances
   ```

8. Implement selected bunch support:

   ```text
   filled_slots
   selected_slots
   filling_scheme
   num_bunches
   ```

9. Implement coasting mode as a one-domain slicer with a public API that hides
   the fake bunch axis by default.

10. Implement optional HDF5 output:

    ```text
    output_file
    buffer_size
    storage mode
    flush()
    appendable datasets
    stable file metadata
    ```

11. Add tests comparing:

    - weighted means and sigmas against NumPy calculations
    - selected bunch behavior
    - coasting mode shape and values
    - `every_n_turns` turn selection
    - HDF5 append/flush layout
    - transverse emittance against covariance calculations

12. Only after the new monitor is stable, decide on wrappers and deprecation
    strategy for older monitors.
