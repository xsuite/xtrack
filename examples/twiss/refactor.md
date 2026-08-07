# Refactoring plan for `xtrack/xtrack/twiss.py`

## Context

`xtrack/xtrack/twiss.py` is currently a large module, about 6100 lines, that
contains several distinct responsibilities:

- the public `twiss_line(...)` entry point;
- input compatibility, deprecation handling, defaults, and validation;
- open and periodic Twiss orchestration;
- closed-orbit search;
- R/T matrix finite-difference computations;
- chromatic functions;
- radiation analysis and radiation integrals;
- spin and polarization analysis;
- coupling helpers;
- `TwissInit`;
- `TwissTable`;
- strength extraction and reversal helpers.

This makes the file difficult to scan, review, and modify safely. The refactor
should preserve the public API while moving related code into smaller modules.

## Main recommendation

Convert `xtrack/xtrack/twiss.py` into a package:

```text
xtrack/xtrack/twiss/
    __init__.py
    constants.py
    input_normalization.py
    core.py
    open_twiss.py
    lattice_functions_from_W.py
    ring_quantities.py
    periodic_solution.py
    twiss_init.py
    twiss_table.py
    beam_covariance.py
    closed_orbit.py
    transfer_matrices.py
    trajectory_curvatures.py
    non_linear_chromaticity.py
    radiation.py
    spin.py
    coupling_edw_teng.py
    strengths.py
```

The existing public import surface should remain available from
`xtrack.twiss`. In practice, `twiss/__init__.py` should re-export the public
names expected by `xtrack/xtrack/line.py` and `xtrack/xtrack/__init__.py`.

Example public facade:

```python
from .core import twiss_line
from .twiss_init import TwissInit
from .twiss_table import TwissTable
from .closed_orbit import ClosedOrbitSearchError, find_closed_orbit_line
from .transfer_matrices import (
    compute_R_matrix,
    compute_T_matrix_line,
    get_R_matrix,
    get_T_matrix_line,
)
from .non_linear_chromaticity import get_non_linear_chromaticity
```

## Important import consideration

Do not leave both of these in place:

```text
xtrack/xtrack/twiss.py
xtrack/xtrack/twiss/
```

Python import resolution would become brittle and confusing. The cleaner
migration is:

1. Create `xtrack/xtrack/twiss/`.
2. Move `xtrack/xtrack/twiss.py` to `xtrack/xtrack/twiss/__init__.py`.
3. Update relative imports inside the moved file.
4. Run tests to verify that `from .twiss import ...` still resolves.
5. Extract internal modules one by one.

## Proposed module responsibilities

### `constants.py`

Move shared constants and field lists:

- `DEFAULT_STEPS_R_MATRIX`
- `DEFAULT_CO_SEARCH_TOL`
- `DEFAULT_MATRIX_RESPONSIVENESS_TOL`
- `DEFAULT_MATRIX_STABILITY_TOL`
- `DEFAULT_NUM_TURNS_SEARCH_T_REV`
- `AT_TURN_FOR_TWISS`
- `VARS_FOR_TWISS_INIT_GENERATION`
- `CYCLICAL_QUANTITIES`
- strength and element-field lists
- `DEFAULT_COL_ORDER`

There is likely a small typo in `DEFAULT_COL_ORDER`: `'dy'` and `'dpx'` appear
to be missing a comma, producing `'dydpx'`. Fix this with a focused test.

### `input_normalization.py`

Move small, non-numerical helpers that prepare `twiss_line(...)` inputs before
the main computation:

- `_handle_deprecated_twiss_kwargs`
- `_apply_twiss_defaults`

This keeps deprecation warning text and default-value compatibility out of the
main orchestration path. It also makes it clearer which recursive calls still
depend on the original `input_kwargs` payload, which is useful before replacing
those recursive paths with explicit helpers.

### `twiss_init.py`

Move:

- `TwissInit`
- `_complete_twiss_init`
- `_2d_w_matrix`
- `_6d_w_matrix`
- `_W_phys2norm` if both `TwissInit` and `TwissTable` continue to use it

This isolates initial-condition construction, reference-frame handling, and
normal-coordinate conversion.

### `twiss_table.py`

Move:

- `TwissTable`
- `_build_sigma_table`

Consider whether radiation-integral table methods should stay here initially
or move later to `radiation.py`. A conservative first step is to move the class
as-is and only split radiation methods after behavior is covered by tests.

### `closed_orbit.py`

Move:

- `ClosedOrbitSearchError`
- `find_closed_orbit_line`
- `_one_turn_map`
- `_error_for_co_search_6d`
- `_error_for_co_search_4d_delta0`
- `_error_for_co_search_4d_zeta0`
- `_error_for_co_search_4d_delta0_zeta0`
- `_merit_function_co_t_rev`
- `_find_closed_orbit_search_t_rev`

This separates closed-orbit search from Twiss table construction and avoids a
generic `orbit.py` module name.

### `transfer_matrices.py`

Move:

- `get_R_matrix`
- `compute_R_matrix`
- `get_T_matrix_line`
- `compute_T_matrix_line`
- `_complete_steps_r_matrix_with_default`

These functions are public or directly tied to public methods on `Line`.
Keep compatibility imports stable.

### `core.py`

Move the main computation orchestration:

- `twiss_line`
- `_handle_loop_around`
- `_handle_init_inside_range`
- `_multiturn_twiss`
- `_updated_kwargs_from_locals`
- `_str_to_index`

After the package split works, refactor `twiss_line` internally to remove
recursive re-entry. The signature should remain unchanged for API compatibility
and documentation propagation. The `zero_at` post-processing branch has already
been converted from recursive re-entry into final result handling. The
deprecated `at_s` path now switches to a temporary marker line and falls through
to the normal computation path instead of recursively calling `twiss_line`.

### `finalize.py`

Move:

- `_finalize_twiss_result`

This helper centralizes final result post-processing such as `zero_at` and
attaches the `ActionTwiss` object used by matching workflows, while leaving
`TwissInit` returns unchanged.

### `extra_markers.py`

Move:

- `_build_auxiliary_tracker_with_extra_markers`

This helper belongs to the deprecated `at_s` path. It constructs a temporary
tracker with inserted markers so `twiss_line` can reuse the normal
`at_elements` selection path.

### `open_twiss.py`

Move:

- `_twiss_open`

This helper propagates a completed `TwissInit` through a selected range and
builds the element-by-element `TwissTable`. It is the open/range Twiss engine,
distinct from periodic-solution preparation and optional result enrichment.

### `periodic_solution.py`

Move:

- `_find_periodic_solution`

This helper finds or accepts the closed orbit, builds or validates the one-turn
matrix, computes the periodic normal form, and returns the completed periodic
`TwissInit`. Keeping it separate from `core.py` makes the main `twiss_line`
path smaller while avoiding a callback into `twiss_line`.

### `lattice_functions_from_W.py`

Move:

- `_get_lattice_functions`
- `_renormalize_eigenvectors`
- `_extract_twiss_parameters_with_inverse`

These helpers transform the propagated normal-form matrices `W` into Twiss
lattice functions and phase advances. Keeping them out of `core.py` makes the
main orchestration easier to scan without introducing a vague helper module.

### `ring_quantities.py`

Move:

- `_add_ring_quantities`

This helper enriches periodic Twiss results with ring-level quantities:
revolution time, line length, slip factor, momentum compaction, tunes, and
global coupling columns. The name is more specific than global quantities and
matches the fact that these values are only added on periodic/ring results.

### `non_linear_chromaticity.py`

Move:

- `get_non_linear_chromaticity`

This is a coherent physics feature and can be extracted after `core.py` is
stable.

### `chromatic_functions.py`

Move:

- `_get_chromatic_functions`

This helper builds first-order chromatic Twiss columns from off-momentum Twiss
results. It is distinct from `non_linear_chromaticity.py`, which exposes the
public tune-scan helper for higher-order chromaticity.

### `trajectory_curvatures.py`

Move:

- `_get_trajectory_curvatures`

This helper is currently used by radiation-integral calculations and by
`TwissTable` methods that expose trajectory-curvature columns. Keeping it in a
specific module avoids a vague `trajectory.py` bucket.

### `beam_covariance.py`

Move:

- `_build_sigma_table`

This keeps beam covariance table construction out of both `core.py` and
`twiss_table.py`, while preserving the existing `TwissTable.get_beam_covariance`
API.

### `radiation.py`

Move:

- `_get_eneloss_and_damping_rates`
- `_extract_sr_distribution_properties`
- `_get_equilibrium_emittance_kick_as_co`
- `_get_equilibrium_emittance_full`
- `_compute_radiation_integrals`

Radiation is relatively specialized and has many optional output fields, so it
should be isolated from core Twiss flow.

Keep `TwissTable._get_radiation_integrals` as the table-facing API in
`twiss_table.py`, but delegate the actual radiation-integral computation to
`radiation.py`. This keeps the public table behavior, including `add_to_tw`,
while moving the formula-heavy implementation into the radiation module.

### `spin.py`

Move:

- `_find_spin_fixed_point`
- `_errfun_spin`
- `_get_spin_polarization`

Spin depends on radiation-integral quantities for polarization analysis, so
expect imports from `radiation.py` or clear call ordering from `core.py`.

### `coupling_edw_teng.py`

Move:

- `_get_coupling_elements_edwards_teng`
- `_get_coupling_rdts`
- `_get_edwards_teng_initial`
- `_conj_mat`
- `_edwards_teng_from_one_turn_at_all_locations`
- `_propagate_edwards_teng`

This keeps Edwards-Teng coupling logic together without implying it covers all
coupling formalisms.

### `strengths.py`

Move:

- `_add_strengths_to_twiss_res`
- `_reverse_strengths`

This code is table-enrichment logic and is separate from optics computation.

## Refactor `twiss_line` after the package split

Once imports are stable, the next readability win is to reduce the size of
`twiss_line`. Suggested internal helpers:

- `_handle_deprecated_twiss_kwargs(...)`
- `_apply_twiss_defaults(...)`
- `_resolve_start_end_locations(...)`
- `_resolve_periodic_mode(...)`
- `_prepare_line_state_for_twiss(...)`
- `_compute_base_twiss(...)`
- `_add_ring_quantities(...)`
- `_add_optional_twiss_outputs(...)`
- `_finalize_twiss_result(...)`

An internal config object can also help. For example:

```python
@dataclass
class TwissConfig:
    method: str
    start: str | None
    end: str | None
    periodic: bool
    reverse: bool
    chrom: bool | None
    radiation_analysis: bool
    radiation_integrals: bool
    spin: bool
```

This should be internal only. The public `twiss_line(...)` signature should not
change.

## Plan to remove recursive `twiss_line` re-entry

Several branches in `twiss_line` still call `twiss_line(...)` again after
mutating a kwargs dictionary. That pattern makes control flow hard to follow and
keeps `_updated_kwargs_from_locals` alive as scaffolding. The goal is not to
remove every multi-segment Twiss computation in one step, but to replace hidden
top-level re-entry with explicit phases and named helpers.

Already converted:

- `zero_at`: now handled during finalization instead of by recomputing Twiss.
- deprecated `at_s`: now switches to a temporary marker line and falls through
  to the normal path instead of recursively calling `twiss_line`.

### Phase 1: configuration preflight

Convert the branches that only set line/tracker state before Twiss computation:

- `disable_apertures`
- `freeze_longitudinal`
- `freeze_energy`
- `method == "4d"` cavity-kill flag setup
- radiation flag setup for `kick_as_co` and `scale_as_co`

Target shape:

- normalize input arguments once;
- determine all required context managers and flag changes before the base
  computation;
- enter those contexts once;
- run the main computation path without retrying through `twiss_line`.

This phase should remove several uses of `_updated_kwargs_from_locals` while
preserving the existing context-manager boundaries. It should be tested with:

- a normal 4d Twiss;
- `test_twiss_disable_apertures`;
- a focused `freeze_longitudinal` or `freeze_energy` case if the local kernel
  configuration supports it;
- a radiation-method smoke test where practical.

### Phase 2: input rewriting

Convert branches that rewrite the requested range or initialization:

- `start is not None and end is None`;
- `init == "full_periodic"` with a range.

These still need auxiliary Twiss computations, but those computations should be
named explicitly instead of expressed as top-level recursion. Candidate helpers:

- `_compute_one_turn_twiss_from_start(...)`
- `_compute_range_from_full_periodic_init(...)`

These helpers can initially call a lower-level internal Twiss routine or, as an
intermediate step, call `twiss_line` in one isolated place. The important
improvement is to remove kwargs mutation from the main body and make the
composition behavior obvious. Test with:

- start-only periodic Twiss;
- start-only open Twiss with explicit init;
- `test_part_from_full_periodic`;
- a `full_periodic + zero_at` smoke comparison.

### Phase 3: range composition

The remaining recursive helpers intentionally compute and concatenate multiple
Twiss segments:

- `_handle_loop_around`
- `_handle_init_inside_range`
- `_multiturn_twiss`

Do these last. They probably need a lower-level private engine that assumes
inputs are already normalized and can compute one segment without finalization
or input compatibility handling. Candidate shape:

- `twiss_line(...)`: public compatibility wrapper;
- `_twiss_line_normalized(...)`: validates/prepares normalized state and
  orchestrates optional outputs;
- `_compute_twiss_segment(...)`: computes one already-normalized segment;
- range-composition helpers call `_compute_twiss_segment(...)` rather than the
  public wrapper.

Test with:

- loop-around open range;
- init inside range at a marker;
- `tests/test_ps_multiturn_twiss.py`;
- reverse Twiss if touched by the helper split.

### Exit criteria

The recursion cleanup is complete when:

- `twiss_line` no longer calls itself directly;
- `_updated_kwargs_from_locals` is removed;
- finalization happens once for public `TwissTable` results;
- `TwissInit` early returns remain unchanged;
- the public `twiss_line` signature and `ActionTwiss.kwargs` behavior are
  preserved.

## Suggested migration order

1. Create `xtrack/xtrack/twiss/` and move `twiss.py` to
   `twiss/__init__.py`.
2. Update imports in the moved file from single-dot to double-dot where needed,
   for example `from .table import Table` becomes `from ..table import Table`.
3. Run a minimal import check and a focused Twiss test.
4. Extract `constants.py`.
5. Extract `twiss_init.py`.
6. Extract `twiss_table.py`.
7. Extract `transfer_matrices.py`.
8. Extract `closed_orbit.py`.
9. Extract `spin.py`.
10. Extract `non_linear_chromaticity.py`, `radiation.py`, `coupling_edw_teng.py`, and
   `strengths.py`.
11. Refactor `twiss_line` into smaller orchestration helpers, following the
    recursion-removal phases above.

## Characterization tests to run during the refactor

Before moving numerical internals, keep a behavior snapshot for representative
cases:

- default periodic 6d Twiss;
- periodic 4d Twiss;
- open Twiss with `start`, `end`, and `init`;
- `init="full_periodic"` with a range;
- reverse Twiss;
- `chrom=True`;
- `strengths=True`;
- `radiation_integrals=True`;
- `only_twiss_init=True`;
- `only_orbit=True`;
- loop-around range;
- `TwissTable.get_twiss_init(...)`;
- `TwissTable.reverse()`;
- `Line.get_R_matrix(...)`;
- `Line.find_closed_orbit(...)`.

The safest process is to run the focused tests after each extraction, not only
after the full package split.

## Risk notes

- Recursive calls inside `twiss_line` rely on mutating a kwargs dictionary.
  Extracting too aggressively before tests could change behavior.
- `TwissTable` methods reach into many columns conditionally; extraction should
  preserve deprecated fields and warning behavior.
- Radiation and spin paths change line configuration and tracking flags inside
  context managers; keep those transitions close to their current behavior.
- Public imports from `xtrack/xtrack/line.py` and `xtrack/xtrack/__init__.py`
  should not change from the user perspective.

## Near-term low-risk improvements

- Fix the `DEFAULT_COL_ORDER` comma issue with a focused test.
- Move constants first.
- Move `TwissInit` and `TwissTable` without changing their internals.
- Add small helpers around deprecation handling and default assignment in
  `twiss_line`.
- Replace the current `twiss_line` recursion following the explicit plan above:
  configuration preflight, input rewriting, then range composition.
- Avoid changing numerical formulas during structural refactoring.
