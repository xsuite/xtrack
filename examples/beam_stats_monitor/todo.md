# BeamStatsMonitor follow-up plan

## Naming and terminology

- Addressed in docstring and user guide.
- Keep the class name `BeamStatsMonitor`.
- Keep the public API centered on "slot" because the implementation is
  slot-based.
- Add a short terminology note to the docstring and user guide:
  - RF bucket: smallest RF time bucket.
  - Bunch slot: allowed bunch position, separated by `bunch_spacing_zeta`.
  - Filled slot: a slot that contains a bunch.
  - Selected slot: a filled slot whose statistics are recorded.
  - Slice: longitudinal subdivision inside a bunch, or full-turn subdivision
    for coasting mode.
- Clarify `filling_scheme` as a slot-indexed filling scheme.

## Whole-beam statistics

- Document that `level="beam"` means all accepted particles for the same
  effective turn.
- In bunched and sliced modes, beam-level statistics are obtained by summing
  the recorded weighted sums over the selected slots and slices.
- Clarify that if `selected_slots` is a subset, beam-level statistics cover
  the selected/recorded slots, not unrecorded filled slots.
- Clarify that coasting mode also provides whole-beam statistics, even though
  coasting mode requires `num_slices`.

## Coasting slice boundary behavior

- Fix the coasting slice assignment in `beam_stats_monitor.h`.
- Current behavior clamps numerically out-of-range slice indices to the first
  or last slice.
- Desired behavior:
  - Keep periodic full-turn folding for coasting beams.
  - Avoid accumulating genuinely out-of-range particles into endpoint slices.
  - Handle the periodic upper boundary consistently, wrapping to slice 0 only
    when appropriate.
- Add regression tests with particles exactly on and close to coasting slice
  boundaries.

## Charge and mass statistics

- First add low-risk statistics computed with the existing
  `particles.weight` weighting:
  - `sum_charge_ratio`
  - `mean_charge_ratio`
  - `sum_mass_ratio`
  - `mean_mass_ratio`
- Treat true alternate weighting modes as a separate design decision because
  they change the meaning of means, covariances, and `num_particles`.
- If alternate weighting is added later, consider a clearly named denominator
  such as `sum_weights` while keeping `num_particles` as
  `sum(particles.weight)`.

## Particle-id selection

- Add `particle_id_range=(start, stop)` to `BeamStatsMonitor`.
- Prefer not to add a `num_particles` constructor argument because
  `num_particles` is already a public statistic.
- Default behavior should remain all particles.
- Implement filtering in the C kernel before slot/slice classification.
- Include `particle_id_range` in `to_dict` and HDF5 metadata.
- Add tests for whole-beam, bunched, sliced, and coasting modes.

## Suggested implementation order

1. Documentation and docstring terminology cleanup.
2. Coasting slice boundary fix and tests.
3. `particle_id_range` support.
4. Charge and mass statistics.
5. Revisit true alternate weighting modes only after the above are stable.
