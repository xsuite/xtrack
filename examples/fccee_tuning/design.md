# Externally evaluated FCC-ee correction

## Agreed boundary

`Optimize` owns targets, tolerances, weights, varies, response matrices, the
least-squares solve, and application of a requested change to its knobs.
`FCCeeCorrection` owns Twiss computation, the sequence of corrections,
acceptance decisions, retries, and whole-machine snapshots. The user script owns
the error/nonlinear ramp and the separate final refinement at the requested error
fraction (including fractions below one).

Do not change `003b_correct_with_optimizers.py` or its shared numerical helpers.
It remains the reference. Do not reintroduce resume/checkpoints. Keep logging
minimal and settings, optimizers, accepted state, and history inspectable.

## Optimizer setup modules

- `optics_corr.py`: normal-optics and horizontal-dispersion targets, knob scales,
  weights, and response construction.
- `orbit_corr.py`: horizontal and vertical orbit targets and response matrices.
- `coupling_corr.py`: coupling and vertical-dispersion targets, scales, weights,
  and response construction.
- `tune_corr.py`: nominal tune targets and the two-knob finite-difference response.

Each builder takes the line, nominal reference Twiss, and shared action and
returns a configured optimizer (two for orbit) with its response installed.
Builders reuse the unchanged analytical response helpers. The workflow owns
the reference and observation locations, precomputes response SVDs, and retains
the correction sequence, candidate evaluations, acceptance, rollback, and history.
The existing `correction.optimizers` and inspection attributes remain available.

## Relative beta objective

The beta objective is relative beating, `beta / beta_ref - 1`, rather than
`ln(beta / beta_ref)`. Ordinary `xt.Target` definitions expose physical beta
values with their reference values. Multiply nominal relative-beta response rows
by beta_ref and divide their target weights by beta_ref; scale their tolerances
by beta_ref as well. This preserves the weighted linear system and acceptance
cost of the relative-beta objective.

Alpha, dispersion, and tunes also use ordinary parameter/at targets. Alpha now
targets `alpha_ref` directly, without the `beta_ref / beta` factor. Convert the
shared helper's nominal normalized-alpha response using
`d(alpha) = d(alpha * beta_ref / beta) + alpha_ref * d(log(beta))` at the reference.
The response builder includes alpha locations in its internal beta observations
to supply these derivatives. Alpha weights are unchanged.
No callable targets remain in optics optimizer setup. The workflow's
`optics_vector` returns the optimizer's unweighted physical target residuals;
`optics_weights` supplies the matching weights. Other settings are unchanged;
the shared 003b helpers retain their logarithmic convention.

## Optics response from the optimizer

`build_normal_quad_optics_response_matrix(optimizer, reference)` lives at the
bottom of `optics_corr.py`. It returns an array in full target/vary order,
including inactive entries, without applying weights or knob scales. The caller
supplies the periodic, uncoupled reference Twiss; the function runs no actions
or Twiss calculations.

Ordinary betx/bety, alfx/alfy, dx and qx/qy targets determine the observation
locations and row order. The varies determine the column order and probe steps.
For each independent line variable, xdeps `controlled_targets` identifies the
affected quadrupoles, including dependencies through intermediate variables.
Central knob probes read only those elements' `get_total_knl_ksl()[0][1]`, and
restore the knob in `finally`. This includes k1, knl[1], and relative multipole
contributions without inspecting which field holds the expression.

The unchanged analytical helper supplies the four-point thick-quadrupole
response, converted to physical beta/alpha and per integrated normal strength.
Each knob column sums its affected quadrupole responses times the measured
strength derivatives. No one-to-one knob/magnet pairing is assumed. This model
assumes fixed geometry and normal-strength changes in unsliced, positive-length
Quadrupoles, distributed over their lengths; it does not resolve a different
longitudinal placement of individual multipole kicks.

The FCC-ee setup installs this matrix and keeps its existing target weighting.
For its individual trims, the tune rows give the same integrated-beta knob
scales as before. The directional tracking check now evaluates the optimizer's
physical targets. `003b` and its shared helpers remain unchanged.

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

## Explicit Twiss lifetime

`correction.twiss()` always computes the current machine. The orchestration
retains the returned table and supplies it to target checks and least-squares
steps. After changing knobs, compute a new table before checking the candidate.
For retries, restore the baseline machine and reuse its original table. Final
refinement starts from `accepted_twiss`; final saving independently recomputes
Twiss. There is no variable tracking, cache, invalidation API, or force flag.
Orbit-only tables are promoted to full optics when required.

Construction explicitly sets `method='4d', delta0=0.0`, matching the intended
configuration in `000_install_errors.py`. Do not rely on these defaults being
present in the serialized input: without them Twiss defaults to 6D, changing
both the correction problem and its convergence.

## Validation

Use `/Users/giadarol/miniforge3/envs/py313`. Test supplied-results calls with actions
that fail if invoked; compare weighted/scaled/masked solves to an independent
linear solution. Cover temporary Jacobians, limits, retries, missing results,
target transformations, and unchanged default evaluation. Check real Twiss
targets, state rollback, explicit evaluations, and new trial decisions against the
unchanged reference helpers. Report separately whether a full FCC-ee lattice run
was possible; small state tests are not a full-machine physics validation.

### Implementation checks (2026-10-22)

- Optimizer-inspected optics response: 14 focused response/state tests pass in
  py313 with FutureWarning treated as an error. Coverage includes family knobs,
  intermediate expressions, combined k1/knl[1]/relative-knl contributions,
  reordered and duplicate targets, tune-only targets, inactive entries, and
  restoration after a failed probe. Tracking comparisons use resolved multipole
  kicks consistent with the distributed-strength approximation.
- The real FCC-ee 22173-by-2679 matrix agrees with the previous builder to a
  relative norm difference of 3.5e-11. Weights, scales, tolerances and tags agree;
  both builders' directional tracking checks pass. Complete optics setup took
  5.8 s versus 3.7 s previously in a single comparison. This validation did not
  rerun the full correction ramp.
- xdeps suite: 107 passed, 2 expected failures.
- Focused Xtrack response/action/state tests after cache removal: 11 passed, with
  FutureWarning promoted to an error.
- A real FCC-ee lattice comparison against the pre-refactor workflow (which
  delegates numerical trials to the unchanged 003b helpers) rejected the same
  0.0125 trial and accepted 0.00625 in one iteration. Knob values agree within
  atol=1e-10/rtol=1e-5 and diagnostics within atol=1e-9/rtol=1e-4.
- After preparing the SVDs before timing, a representative accepted 0.00625 trial
  took approximately 7.2 seconds with this workflow and 6.0 seconds with the
  reference. This is a single-point comparison, not a full-ramp benchmark.
  This measurement predates removal of the Twiss cache and its variable checks.
- The first full run hit its 250-second cap near error fraction 0.021. Its
  regenerated input had empty Twiss defaults, unlike the earlier successful
  input with `method='4d', delta0=0.0`. The lattice and variable data matched;
  the missing defaults selected 6D optics. The earlier single-point comparison
  above also used that input and did not establish parity with the intended 4D run.
- With 4D configured explicitly and the cache removed, the full ramp to one,
  final refinement, fresh final Twiss goal checks, and output saving completed
  successfully in 217.9 seconds wall time (211.5 seconds inside the script).
  There were 19 accepted history entries and 254 Twiss calls, including 91
  orbit-only calls. Twiss accounted for 75.8 seconds. This was below the
  250-second cap. The original script and numerical helper files are unchanged.
- The 18 saved ramp fractions exactly match the earlier verified reference run;
  all history diagnostics agree with atol=1e-8 and rtol=1e-3.

Full-run log: `/tmp/fccee_explicit_4d_zb5z1o5t/run.log`; timing and verification
are in `timing.json` and `verification.json` in the same directory. The saved
machine is `response_api_class/machine.json` there.

After extracting the four optimizer setup modules, the full run completed in
214.6 seconds wall time with all final goals met. Its saved history matches the
pre-extraction run exactly, with the same 19 accepted entries and 254 Twiss calls.
The 11 focused response/action/state tests pass. Full-run log and verification:
`/tmp/fccee_corr_modules__i4fb09r/run.log` and `verification.json` in that directory.

Changing only the beta objective to relative beating completed the full ramp,
refinement, fresh final goal checks, and saving in 222.2 seconds wall time,
within the 250-second cap. There were 261 Twiss calls and four final-refinement
iterations (previously 254 and three). All final goals passed; beta RMS was
0.680%/0.764%. The 11 focused tests also pass. Log and verification:
`/tmp/fccee_relative_beta_fjleyqm4/run.log` and `verification.json` there.

Expressing beta, dispersion, and tunes as ordinary targets completed the full
run in 223.6 seconds, with all final goals met and the same 261 Twiss calls and
accepted fractions as the relative-beta run. History agrees within atol=1e-8,
rtol=1e-3. The 12 focused tests include an independent check of the weighted
linear system, correction, and acceptance cost. Log and verification:
`/tmp/fccee_physical_targets_7bvco2p8/run.log` and `verification.json` there.

Dropping alpha normalization, with the corresponding response conversion, passed
all final goals in a 220.0-second full run. A central-difference tracking probe
checked the physical-alpha response to 9.1e-6 relative error. The 260 Twiss calls
include those two additional validation probes; refinement took three iterations.
The 12 focused tests pass, including finite-difference validation of the alpha
row conversion. Log and verification:
`/tmp/fccee_plain_alpha_v19jmngm/run.log` and `verification.json` there.

Local comparison output: `/tmp/check_fccee_lstsq.log`; the earlier comparison
including the rejected 0.0125 trial is `/tmp/check_fccee_lstsq_initial.log`.
