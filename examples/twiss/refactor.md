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
    computation_plan.py
    open_propagation.py
    open_table_composition.py
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
- `_normalize_twiss_inputs`

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
- `_kwargs_for_multiturn_continuation`
- `_str_to_index`

After the package split works, refactor `twiss_line` internally so recursive
re-entry is removed from non-multiturn composition paths. The multi-turn path is
the exception: recursive continuation is a readable expression of "propagate one
turn, derive the next turn's init, then continue". The signature should remain
unchanged for API compatibility and documentation propagation. The `zero_at`
post-processing branch has already been converted from recursive re-entry into
final result handling. The deprecated `at_s` path now switches to a temporary
marker line and falls through to the normal computation path instead of
recursively calling `twiss_line`.

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

### `open_propagation.py`

Move pure planning code for composed open Twiss propagation:

- `_TwissPropagationRequest`
- `_OpenTwissPiecePlan`
- `_OpenTwissPropagationPlan`
- `_LoopAroundTwissPlan`
- `_OpenOneTurnTwissPlan`
- `_plan_open_twiss_propagation`
- route-specific pure planners for loop-around, init-inside-range, and open
  one-turn-from-start

Keep segment execution in `core.py` while it still calls back into
`_compute_twiss_segment`.

### `open_table_composition.py`

Move table-composition code for composed open Twiss propagation:

- `_combine_loop_around_twiss_tables`
- `_combine_init_inside_range_twiss_tables`

This keeps table merging, phase re-alignment, unsupported-column cleanup, and
metadata reconciliation separate from segment execution.

### `computation_plan.py`

Move pure planning code for the high-level Twiss flow:

- `_TwissInitAcquisitionPlan`
- `_TwissComputationPlan`
- `_plan_twiss_computation`
- `_plan_twiss_init_acquisition`

This module records the target orchestration shape: acquire or compute a
`TwissInit` first, then pass it to open propagation.

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
- `_normalize_twiss_inputs(...)`
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

Historically, several branches in `twiss_line` called `twiss_line(...)` again
after mutating a kwargs dictionary. The refactor below replaces that hidden
top-level re-entry with explicit phases and named helpers. Multi-turn
continuation remains recursive by design.

### Target orchestration shape

The non-multiturn Twiss flow should make the conceptual phases visible:

1. Normalize inputs and enter temporary line/tracker contexts.
2. If the request is periodic, find the periodic solution for the requested
   scope, either the full line or the selected range. This produces a
   `TwissInit`.
3. From that point on, treat periodic and open Twiss the same way: propagate
   from an available `TwissInit` through an open segment plan.
4. If the init is at a requested boundary and the range does not cross the line
   start, the open propagation is a single piece.
5. If the init is inside the requested range, split at the init location.
6. If the requested range crosses the line start, split at the line boundary as
   well. In the most general case this gives three table pieces to compose.
7. Keep reverse handling explicit in the plan instead of hiding it in endpoint
   rewrites.

Two reverse-mode constraints are part of the target design:

- With `reverse=True`, `start` and `end` are already interpreted in reversed
  traversal order. Planning code should preserve that public meaning instead of
  converting the endpoints back and forth between forward and reverse views.
- For periodic Twiss with `reverse=True`, prefer computing the periodic solution
  in the forward direction and then producing the requested reverse-ordered
  output. Backtracking should only be used for cases where the forward periodic
  solution cannot represent the requested result.

This means the code should distinguish:

- requested traversal order;
- periodic solution computation order;
- final table output order.

Init computation and Twiss propagation should remain separate in code:

- init acquisition computes or extracts a `TwissInit`;
- propagation consumes a completed `TwissInit` and produces table pieces;
- the orchestration layer may connect the two phases, but should avoid hiding
  init computation inside segment propagation helpers.

The planner structures in `computation_plan.py` and `open_propagation.py`
document this target structure:

- `_TwissInitAcquisitionPlan`
- `_PeriodicTwissInitData`
- `_OpenTwissPropagationPlan`
- `_OpenTwissPiecePlan`
- `_plan_twiss_computation`

The high-level computation planner supplies open propagation plans for
post-init ranges, full-periodic ranges, and one-turn-from-start routes. Base
periodic init acquisition is also selected by the computation plan.

The computation plan is built before explicit init completion, while
`init_at` is still available. The base computation now dispatches supplied-init
and periodic-solution acquisition from the init-acquisition plan, and uses its
scope to select a full-line or requested-range periodic solution. Propagation
remains a separate phase after acquisition.

The plan also selects the pre-init orchestration route (`base`,
`periodic_one_turn_from_start`, `open_one_turn_from_start`, or
`full_periodic_range`). The composed-route dispatcher therefore consumes an
explicit route instead of rediscovering it from the normalized inputs.

For `full_periodic_range`, the plan now describes full-line periodic init
acquisition separately from open propagation over the requested range. The
acquired init is propagated through the planned boundary/init pieces instead
of hiding range routing inside a generic segment call.

One-turn-from-start planning is also explicit. Periodic requests record the
requested output direction while retaining forward periodic-solution
acquisition; open requests receive their two boundary pieces directly from the
high-level computation plan.

The base Twiss path now also uses `_compute_periodic_twiss_init_and_data` and
`_propagate_twiss_from_init` names at the call site. This is a small step toward
keeping periodic init computation separate from propagation through the line.
Init completion now receives the normalized computation dictionary as a single
phase input, avoiding a large positional-style handoff and preparing the same
boundary for private segment execution.

Non-multiturn composed pieces now call `_compute_base_twiss` directly. This
private engine accepts already-normalized state, completes/acquires the init,
propagates one non-composed segment, and finishes its table without re-entering
the public `twiss_line` compatibility wrapper. Multi-turn continuation remains
the intentional recursive exception.
Three-piece loop-around requests are executed as the three pieces expressed by
the open propagation plan, with the line-boundary init transfer made explicit;
they no longer depend on nested range routing to split a joined piece.
Composed non-multiturn routes pass normalized state dictionaries directly to
their private helpers. Broad public-kwargs refresh is retained only as the
narrowly named `_kwargs_for_multiturn_continuation`, supporting the intentional
recursive multi-turn path.
The public base fall-through and private segment engine now share
`_compute_base_twiss_after_explicit_init_completion`. This keeps explicit init
completion outside the shared lifecycle while centralizing periodic acquisition,
propagation, and result finishing.
The public entry point now uses `_normalize_twiss_inputs` as the single
deprecated-alias/default/derived-flag normalization phase. The returned
dictionary supplies normalized computation kwargs while preserving the
pre-default public kwargs used by `ActionTwiss`.
Line/range normalization and temporary tracker/config state now live in
`line_context.py`. The public `twiss_line` entry point enters one context stack,
prepares a normalized state dictionary, and delegates orchestration to
`_compute_twiss_with_prepared_line_context`.

The first internal class boundary is `_TwissBaseComputation`, which owns the
base-path preparation, periodic init acquisition, propagation from init, and
result enrichment. Normalization and composed range routing remain outside the
class for now to avoid turning the first class step into a broad behavioral
rewrite. The computation object stores its working data as attributes rather
than as a nested state dictionary; this keeps the phase methods readable while
the public `twiss_line(...)` signature remains unchanged.

Already converted:

- `zero_at`: now handled during finalization instead of by recomputing Twiss.
- deprecated `at_s`: now switches to a temporary marker line and falls through
  to the normal path instead of recursively calling `twiss_line`.
- `disable_apertures`: now enters the aperture-flag preservation context before
  the main computation path instead of recursively calling `twiss_line`.
- `method == "4d"` cavity-kill setup: now enters the track-flag preservation
  context before the main computation path instead of recursively calling
  `twiss_line`.
- `freeze_longitudinal` and `freeze_energy`: now enter their line-state
  contexts before the main computation path instead of recursively calling
  `twiss_line`.
- radiation flag setup for `kick_as_co` and `scale_as_co`: now enters the
  required track/config preservation contexts before the main computation path
  instead of recursively calling `twiss_line`.
- `start is not None and end is None`: now delegates the one-turn table
  composition to `_compute_one_turn_twiss_from_plan`, which dispatches to
  explicit periodic and open one-turn helpers.
- open one-turn-from-start now builds an explicit `_OpenOneTurnTwissPlan` with
  separate line-boundary pieces and transfer-init metadata. Periodic one-turn
  remains on the existing table-rotation path.
- `init == "full_periodic"` with a range: now separates full-periodic init
  acquisition (`_acquire_full_periodic_twiss_init`) from planned open
  propagation over the requested range.
- loop-around open ranges: `_handle_loop_around` now builds an explicit
  `_LoopAroundTwissPlan`, keeping final table order separate from execution
  order. The table combination remains unchanged.
- init-inside-range open ranges: `_handle_init_inside_range` now separates marker
  support validation, segment construction, and table combination into named
  helpers.
- init-inside-range open ranges now route segment construction through the
  open-propagation planner, while keeping the existing validation and table
  combination behavior.
- multi-turn Twiss: `_multiturn_twiss` now separates turn-table construction,
  continuation to the next turn, and final table concatenation into named
  helpers.
- non-multiturn segment composition: helper code now calls
  `_compute_twiss_segment`, which is the single boundary to replace when the
  normalized segment engine is extracted.
- open one-turn composition no longer runs a dummy one-row Twiss just to recover
  the first element name; it now reads the normalized start element name through
  `_line_start_element_name`.
- the early input-rewrite branches for start-only and `init="full_periodic"`
  now share a single finalization return instead of finalizing separately.
- the early input-rewrite branches now run after the flag/config setup, so
  freeze, cavity, radiation, aperture, and temporary-marker setup happen before
  composed segment computation.
- `_compute_twiss_segment` now builds prepared segment kwargs, avoiding
  repeated setup-only requests such as aperture disabling, freeze flags, and
  `at_s` marker insertion in each composed segment.
- freeze setup is now isolated in `_enter_twiss_freeze_context`, making the
  context state transition explicit before composed segment computation.
- the later range-composition branches for loop-around and init-inside-range now
  share a single finalization return instead of finalizing separately.
- periodic solution preparation for the normal one-pass Twiss path is now
  isolated in `_prepare_periodic_solution_for_base_twiss`, which owns the default
  `steps_R_matrix` completion and delegates to `_find_periodic_solution`.
- element-by-element propagation for the normal one-pass Twiss path is now
  isolated in `_propagate_base_twiss_element_by_element`, which delegates to
  `_twiss_open` with explicit arguments.
- periodic solution metadata attachment is now isolated in
  `_add_periodic_solution_data_to_base_twiss`, including R matrices, ring
  quantities, eigenvalues, and rotation matrix.
- base Twiss initialization completion is now isolated in
  `_complete_init_for_base_twiss`; the caller still clears the init-generation
  kwargs afterward because that state is reused by the remaining multiturn
  continuation path.
- open Twiss phase alignment with the provided init is now isolated in
  `_align_open_twiss_phases_with_init`.
- first-order chromatic Twiss result enrichment is now isolated in
  `_add_chromatic_functions_to_twiss_result`.
- radiation-analysis Twiss result enrichment is now isolated in
  `_add_radiation_analysis_to_twiss_result`.
- small result-convention and table-enrichment steps are now isolated in
  `_apply_4d_longitudinal_result_convention`,
  `_set_twiss_result_values_at`, and
  `_add_strengths_and_radiation_integrals_to_twiss_result`.
- spin polarization and Edwards-Teng coupling result enrichment are now isolated
  in `_add_spin_polarization_to_twiss_result` and
  `_add_edwards_teng_coupling_to_twiss_result`.
- final base-result shaping is now isolated in helpers for metadata,
  optional reverse, measured revolution period, multiturn extension,
  `at_elements` selection, and periodic/completed-init tagging.
- base Twiss preconditions and setup are now isolated in helpers for reverse
  range handling, boundary init validation, matrix settings, particle-reference
  preparation, method validation, init-mode validation, and open-Twiss momentum
  offset validation.
- non-multiturn segment composition no longer re-enters the public
  `twiss_line` wrapper. `_compute_twiss_segment` now prepares reusable
  segment kwargs and calls `_compute_base_twiss` directly. Multi-turn
  continuation remains the only intentional recursive `twiss_line` call.
- public Twiss input normalization is now consolidated in
  `_normalize_twiss_inputs`, which handles deprecated aliases, default values,
  derived polarization/spin flags, and defensive `TwissInit` copying.
- Twiss line/context setup is now consolidated in `_prepare_twiss_line_context`,
  which resolves open/range defaults, periodic mode, temporary tracker flags,
  freeze/radiation contexts, `at_s` marker insertion, and `TwissTable` init
  extraction.

### Phase 1: line/context setup

Converted the branches that only set line/tracker state before Twiss
computation:

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

This phase should remove several uses of `_kwargs_for_composed_twiss_call` while
preserving the existing context-manager boundaries. It should be tested with:

- a normal 4d Twiss;
- `test_twiss_disable_apertures`;
- a focused `freeze_longitudinal` or `freeze_energy` case if the local kernel
  configuration supports it;
- a radiation-method smoke test where practical.

### Phase 2: input rewriting

Convert branches that rewrite the requested range or initialization:

- `start is not None and end is None`. The branch is isolated in
  `_compute_one_turn_twiss_from_start`, with separate
  `_compute_periodic_one_turn_twiss_from_start` and
  `_compute_open_one_turn_twiss_from_start` helpers; the helper paths now call
  `_compute_twiss_segment`, which calls `_compute_base_twiss` directly instead
  of re-entering the public `twiss_line` wrapper.
- `init == "full_periodic"` with a range. The branch now prepares full-periodic
  kwargs with `_prepare_kwargs_for_full_periodic_twiss`, extracts the init with
  `_compute_full_periodic_twiss_init`, then calls `_compute_twiss_segment` for
  the requested range.

These still need auxiliary Twiss computations, but those computations should be
named explicitly instead of expressed as top-level recursion. Candidate helpers:

- `_compute_one_turn_twiss_from_start(...)`
- `_compute_periodic_one_turn_twiss_from_start(...)`
- `_compute_open_one_turn_twiss_from_start(...)`
- `_prepare_kwargs_for_full_periodic_twiss(...)`
- `_compute_full_periodic_twiss_init(...)`

These helpers now call `_compute_twiss_segment`, which prepares reusable
segment kwargs and runs `_compute_base_twiss`. The important improvement is to
remove public-wrapper recursion and make the composition behavior obvious. Test
with:

- start-only periodic Twiss;
- start-only open Twiss with explicit init;
- `test_part_from_full_periodic`;
- a `full_periodic + zero_at` smoke comparison.

### Phase 3: range composition

The remaining recursive helpers intentionally compute and concatenate multiple
Twiss segments:

- `_handle_loop_around`. The branch is split into direction-specific segment
  construction helpers and a table-combination helper; its segment calls now go
  through `_compute_twiss_segment`.
- `_handle_init_inside_range`. The branch is split into support validation,
  `_compute_init_inside_range_twiss_parts`, and
  `_combine_init_inside_range_twiss_tables`; its segment calls now go through
  `_compute_twiss_segment`.
- `_multiturn_twiss`. The branch is split into
  `_compute_multiturn_twiss_parts`, `_continue_multiturn_twiss`, and
  `_combine_multiturn_twiss_tables`; this is the one place where recursive
  `twiss_line` continuation can remain.

Do these last. They probably need a lower-level private engine that assumes
inputs are already normalized and can compute one segment without finalization
or input compatibility handling. Candidate shape:

- `twiss_line(...)`: public compatibility wrapper;
- `_twiss_line_normalized(...)`: validates/prepares normalized state and
  orchestrates optional outputs;
- `_compute_twiss_segment(...)`: prepares already-normalized segment kwargs and
  calls `_compute_base_twiss(...)` directly;
- `_compute_base_twiss(...)`: normalized one-pass Twiss engine used by the public
  wrapper and by composed non-multiturn segments;
- `_complete_init_for_base_twiss(...)`: extracted slice of the normal one-pass
  Twiss path, covering `TwissInit` completion and `completed_init` preservation;
- `_prepare_periodic_solution_for_base_twiss(...)`: extracted slice of the
  normal one-pass Twiss path, covering periodic init and one-turn matrix
  preparation;
- `_propagate_base_twiss_element_by_element(...)`: extracted slice of the normal
  one-pass Twiss path, covering propagation of a completed init into the
  element-by-element Twiss table;
- `_add_periodic_solution_data_to_base_twiss(...)`: extracted slice of the
  normal one-pass Twiss path, covering periodic matrix and ring-level result
  data attachment;
- `_align_open_twiss_phases_with_init(...)`: extracted slice of the normal
  one-pass Twiss path, covering open-range phase and `dzeta` alignment after
  reverse handling;
- `_add_chromatic_functions_to_twiss_result(...)`: extracted optional
  enrichment slice, covering first-order chromatic columns and scalars;
- `_add_radiation_analysis_to_twiss_result(...)`: extracted optional enrichment
  slice, covering damping, energy loss, radiation columns, and equilibrium
  emittance updates;
- `_apply_4d_longitudinal_result_convention(...)`,
  `_set_twiss_result_values_at(...)`, and
  `_add_strengths_and_radiation_integrals_to_twiss_result(...)`: extracted
  result-convention and table-enrichment slices;
- `_add_spin_polarization_to_twiss_result(...)` and
  `_add_edwards_teng_coupling_to_twiss_result(...)`: extracted optional physics
  enrichment slices;
- `_add_base_twiss_metadata(...)`, `_reverse_twiss_result_if_needed(...)`,
  `_add_measured_revolution_period_if_requested(...)`,
  `_extend_twiss_result_to_multiple_turns(...)`,
  `_select_twiss_result_at_elements(...)`, and
  `_add_periodicity_and_completed_init_to_twiss_result(...)`: extracted final
  result-shaping slices;
- `_apply_base_twiss_reverse_range(...)`,
  `_validate_base_twiss_boundary_init(...)`,
  `_prepare_base_twiss_matrix_settings(...)`,
  `_prepare_base_twiss_line_and_particle_ref(...)`,
  `_validate_base_twiss_method(...)`,
  `_validate_base_twiss_init_mode(...)`, and
  `_validate_base_twiss_open_momentum_offsets(...)`: extracted precondition and
  setup slices that should move into the eventual base Twiss engine;
- range-composition helpers call `_compute_twiss_segment(...)` rather than the
  public wrapper.

Test with:

- loop-around open range;
- init inside range at a marker;
- `tests/test_ps_multiturn_twiss.py`;
- reverse Twiss if touched by the helper split.

### Exit criteria

The readability cleanup is complete when:

- recursive calls are removed outside the multi-turn continuation path;
- `_kwargs_for_composed_twiss_call` is removed, or no longer used to hide broad
  state rewriting in the main body;
- finalization happens once for public `TwissTable` results, with composed
  branches feeding that same finalization path instead of returning early;
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
  line/context setup, input rewriting, then range composition.
- Avoid changing numerical formulas during structural refactoring.
