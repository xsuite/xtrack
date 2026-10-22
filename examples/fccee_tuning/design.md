# Externally evaluated FCC-ee correction

## Agreed boundary

`Optimize` owns targets, tolerances, weights, varies, response matrices, the
least-squares solve, and application of a requested change to its knobs.
`FCCeeCorrection` owns Twiss computation and caching, the sequence of corrections,
acceptance decisions, retries, and whole-machine snapshots. The user script owns
the error/nonlinear ramp and the separate final refinement at the requested error
fraction (including fractions below one).

Do not change `003b_correct_with_optimizers.py` or its shared numerical helpers.
It remains the reference. Do not reintroduce resume/checkpoints. Keep logging
minimal and settings, optimizers, accepted state, and history inspectable.

## Supplying model results

Add optional `action_results={action: result}` to target-evaluation entry points.
Results are the ordinary outputs of actions, e.g. a Twiss table, not externally
assembled residual vectors. Targets still extract observables and apply their
transformations, weights, tolerances, and activation masks.

`opt.run_actions()` runs each configured action once and returns the complete
`{action: result}` mapping. It does not evaluate targets, apply optimizer knob
changes, or add log entries. Action exceptions propagate.

- Without results, run actions as before.
- With results, do not run actions for the supplied point. Require all configured
  actions; missing data is an error, never an implicit model evaluation.
- Results belong to the call, not a persistent implicit cache inside Optimize.
- Supplying results for inspection does not change knobs.
- `target_status(ret=True, action_results=...)` returns the existing status table.
  Use it for acceptance checks without logging or running Twiss.
- `target_mismatch`, `tag`, and `add_point_to_log` use the same mechanism.
  A tag records a candidate; it does not accept the machine. Logged knob values
  and supplied results must describe the same machine state.
- Constructor evaluation and diagnostic restoration must support supplied results
  too. Multi-evaluation algorithms (`step`/`solve`) still need new evaluations for
  their candidates; supplied results apply only to their starting point (also
  through `run_jacobian`). A single supplied table cannot replace later
  evaluations or numerical derivative probes.

## One explicit least-squares step

`run_lstsq(action_results=None, jacobian=None, *, rcond=None,
sing_val_cutoff=None, step_scale=1.0, tag_outcome=True)` evaluates the starting
targets, computes one direction, and applies it. By default, it then runs the
actions at the changed knobs and records the evaluated outcome with `tag('lstsq')`.
Supplied `action_results` describe only the starting point and are never reused
for the outcome. It does not backtrack or decide whether to accept the change.
It returns the optimizer, as `step()` does.

The FCC-ee workflow passes `tag_outcome=False` to skip the outcome evaluation and
log entry. The caller evaluates the candidate and can record it with `tag()`.

Jacobian precedence:

1. An explicit Jacobian applies only to this call.
2. Otherwise use the installed fixed response and its cached SVD.
3. With neither, compute the usual Jacobian only when action results were omitted.
4. Supplied results without a supplied/stored Jacobian are an error: no hidden
   derivative probes are allowed.

Explicit Jacobians follow `set_response`: full target/vary order, transformed but
unweighted residuals differentiated with respect to physical knobs. Optimize
applies target weights, knob scales, active masks, and SVD cutoffs. Preserve the
existing `Vary.max_step` convention of the Jacobian solver. Limit the direction
by max_step before multiplying by step_scale; shorten it further if necessary to
stay within physical knob limits. Inactive varies do not change. An unsuccessful
calculation must not leave finite-difference probes installed.

The existing `get_correction` API remains available for the unchanged reference.
The new workflow uses `run_lstsq`, not manual direction application.

## External retries and acceptance

Capture the baseline, supply its Twiss, and call `run_lstsq(tag_outcome=False)`.
Compute the new Twiss and run auxiliary corrections, then inspect target data. On rejection,
restore the entire baseline and call `run_lstsq` again with the original Twiss
and a smaller step_scale. Recompute the inexpensive linear solve while reusing
the SVD; no separate public proposal/reapplication object is needed.

Preserve the current numerical acceptance decisions, iteration budgets, and
ordering. In particular:

- Orbit directions use laboratory residuals; acceptance uses the BPM-frame orbit.
- Optics directions include tunes, but the acceptance cost excludes tune rows
  after orbit/tune cleanup. The initial direction is limited to 0.5 in the
  normalized quadrupole coordinates, followed by ten halving trials.
- Coupling settles the baseline and each candidate before comparing cost.
- Orbit/optics/coupling use the existing relative improvement threshold; tune
  cleanup uses strict norm improvement.
- RMS/max global goals are not replaced by per-response-row tolerances. Express
  aggregate goals as targets where useful, while preserving their strict versus
  inclusive comparisons and ramp/final tolerances.
- Rollback includes all knobs changed by auxiliary corrections and orbit guesses.
  Optimizer diagnostics describe evaluations, whereas correction history contains
  accepted ramp/refinement points only.

## Twiss cache

`correction.twiss()` returns data for the current machine, not necessarily the
last accepted ramp point. Cache a successful result together with the tracked
machine-variable values and Twiss options. Direct optimizer knob writes must be
detected, including writes followed by rollback. Never reuse an orbit-only table
where full optics is required. Seeds are search hints; they must not make a stale
machine table valid. Failed evaluations must not be cached.

Track all live variable values (including dependent variables) and Twiss settings.
Read numeric values directly rather than formatting the whole expression table.
Explicit invalidation is required
after direct element/lattice edits that do not pass through variables. Provide a
forced fresh evaluation for final verification and numerical comparisons.

## Validation

Use `/Users/giadarol/miniforge3/envs/py313`. Test supplied-results calls with actions
that fail if invoked; compare weighted/scaled/masked solves to an independent
linear solution. Cover temporary Jacobians, limits, retries, missing results,
target transformations, and unchanged default evaluation. Check real Twiss
targets, state rollback, cache invalidation, and new trial decisions against the
unchanged reference helpers. Report separately whether a full FCC-ee lattice run
was possible; small state tests are not a full-machine physics validation.

### Implementation checks (2026-10-22)

- xdeps suite: 107 passed, 2 expected failures.
- Focused Xtrack response/action/state tests: 12 passed, including with
  FutureWarning promoted to an error.
- A real FCC-ee lattice comparison against the pre-refactor workflow (which
  delegates numerical trials to the unchanged 003b helpers) rejected the same
  0.0125 trial and accepted 0.00625 in one iteration. Knob values agree within
  atol=1e-10/rtol=1e-5 and diagnostics within atol=1e-9/rtol=1e-4.
- After preparing the SVDs before timing, a representative accepted 0.00625 trial
  took approximately 7.2 seconds with this workflow and 6.0 seconds with the
  reference. This is a single-point comparison, not a full-ramp benchmark.
  All variable values are checked for cache validity, and inner trials retain
  whole-machine snapshots; this trades some overhead for inspectable state.
- The full ramp to one and its final refinement have not been validated by an
  end-to-end run in this change. The original script and numerical helper files
  are unchanged.

Local comparison output: `/tmp/check_fccee_lstsq.log`; the earlier comparison
including the rejected 0.0125 trial is `/tmp/check_fccee_lstsq_initial.log`.
