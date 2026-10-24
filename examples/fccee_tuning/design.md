# Externally evaluated FCC-ee correction

## Agreed boundary

`Optimize` owns targets, tolerances, weights, varies, response matrices, the
least-squares solve, and application of a requested change to its knobs.
`FCCeeCorrection` owns Twiss computation, the sequence of corrections,
acceptance decisions, retries, whole-machine snapshots, increasing errors and
the separate final refinement at the requested error fraction (including
fractions below one). The user script configures, loads, runs and saves.

Run the example scripts in order:

1. `000_install_errors.py` installs static lattice errors.
2. `001_install_correction_knobs.py` installs the correction knobs.
3. `002_correct_with_state.py` increases errors, corrects, refines and saves.
4. `003_momentum_scan.py` analyzes the saved machine versus momentum offset.

The older correction and analysis scripts have been removed, together with
unused helpers. Their reference implementations remain in Git history (commit
`11b5f7f58`). Historical validation notes below retain the old script names.
Do not reintroduce resume/checkpoints. Keep logging minimal and settings,
optimizers and accepted state inspectable.

## Momentum-offset analysis

`003_momentum_scan.py` loads `response_api_class/machine.json` independently of
the correction object. All settings are near the top of the script: `INPUT`
and `OUTPUT_DIR` select the saved standard machine and artifact directory.
The default scan is 21 points over delta = -0.01 to +0.01, configured through
`DELTA_MAX` and `NUM_POINTS`. There is no argument parser. Delta means
`(p-p0)/p0`. Radiation is disabled, nonzero magnet tapering is rejected, and no
radiation compensation is performed. All optics calculations use 4D Twiss.

An independent zero-guess search at `ipd` recovers the corrected orbit. Searching
at the line start (`ipa`) fails for the validated error realization; the script
does not reuse the old correction's accepted orbit. Positive and negative delta
branches start independently from the recovered delta=0 solution. Subsequent
Twiss calls use the last successful `particle_on_co` as `co_guess`, keeping the
orbit search active. Steps are limited to 5e-4 and failed steps are halved down
to 1e-6. A tune jump larger than 0.25 per continuation step is also retried;
finite but discontinuous Twiss tunes must not silently enter the plots.
Failed requested points are logged and saved as NaN with their errors and the
last delta reached. Once continuation fails, subsequent points on that branch
are explicitly marked as not reached, also NaN; the opposite branch still runs.
This probes the optics branch connected to delta=0. Twiss failure or a tune
continuity rejection is not a measurement of momentum acceptance.

The ideal reference is reconstructed on the identical element layout by turning
off errors and correction switches, resetting the independent tune offsets and
original tune families to their nominal values, and retaining nominal nonlinear
strengths (`on_nonlinear=1`). All changed controls are restored afterward.
Both the clean and corrected machines are scanned on the same delta grid, using
independent closed-orbit continuation and failure reporting. Only tunes are
plotted versus delta, with clean and corrected curves overlaid in each plane.
The reference momentum is always delta=0: beta beating is
`beta_machine(delta)/beta_ideal(0)-1` for each machine, including chromatic changes.
Thus the saved clean beta beating is zero at delta=0, but generally not elsewhere.
This common denominator allows direct comparison of the saved results. It does not
subtract the ideal machine's own off-momentum beta variation.
The correction's optics target used nonlinear strengths off, so its reported
beta-beating RMS need not exactly equal this analysis with nominal sextupoles on.

Observations are the union of horizontal-correction BPM entrances and the four
IP entrances `ipa/ipd/ipg/ipj`, matching the correction's optics observations.
Orbit is expressed in lattice coordinates, without subtracting BPM alignment
offsets. Orbit and off-momentum betas remain available in the saved data.
The second figure overlays nominal and corrected beta along the ring at delta=0,
with signed beating below. Ring plots use those same observations, sorted by s;
they are not maxima over every lattice element.

Artifacts in `analysis/momentum_scan` are `tunes_vs_delta.png`,
`beta_along_ring.png`,
`momentum_scan.npz` (coordinates, names, s, betas, reference betas, beating,
tunes, and failures), and `summary.json` (definitions, settings and failures).
NPZ fields without a prefix describe the corrected machine; `clean_*` fields
store the clean scan, including its own tunes, errors and last reached delta.
The JSON records these separately in `failed_points` and `clean_failed_points`.
Blue/orange vertical lines mark failed or unreached corrected/clean points.
For reproducible runs set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, and `VECLIB_MAXIMUM_THREADS=1` before starting Python.

For interactive figures in IPython, enable a GUI backend with `%matplotlib`,
then run `%run examples/fccee_tuning/003_momentum_scan.py` from the repository
root. Set `INPUT` to the desired machine path before running. `SHOW_PLOTS = True`
by default keeps both figures open while still saving the artifacts.
Set it to `False` for batch runs to close figures after saving.

DA/MA implementation and tracking scans are deferred at the user's request.
The chosen MA definition for that future work is off-momentum closed-orbit
launches at `ipa` with a small transverse amplitude. Turns, amplitudes and
scan ranges still need to be settled and timed before a substantial scan.

## Executing the correction

After a successful trial, each execution method directly updates
`last_accepted_twiss` and prints `Accepted`. Subsequent Twiss calls use
`last_accepted_twiss.particle_on_co` as `co_guess`, keeping the closed-orbit search
active after knob changes. There is no separate default-guess copy to update
or restore; snapshots retain the accepted table. This seed is not stored in
`line.twiss_default`, so saving the line does not serialize a particle there.
The acceptance update does not evaluate metrics or retain history. Snapshots contain only the state
needed for rollback. Saving belongs to the caller through `line.to_json(path)`;
there is no correction history list or `history.csv` output.

`FCCeeCorrection.run(target_error_fraction=1., initial_increment=0.0125,
min_increment=1e-5, max_increment=0.10)` calls two separate public methods:

- `iterative_error_increase_and_correction(...)` increases errors and nonlinear strengths
  together, extrapolates existing corrections, evaluates Twiss, and runs the
  custom fixed-strength sequence. Rejected trials are restored and the increment
  is halved. Successful trials update the accepted state. An accepted trial needing at most
  three cycles grows the next increment by 1.25, capped at `max_increment`.
- `refine_correction()` switches to final tolerances and corrects the current accepted state,
  including at fractions below one. It starts from `last_accepted_twiss` and does no
  extra work when the final goals are already met. A failed refinement restores
  its starting machine while retaining the final tolerances.

Both stages and `run()` return a success boolean. `run()` skips refinement when
increasing errors fails. The object exposes `correction_failed`, `failure_reason`,
`target_error_fraction`. The failure flag is derived from `failure_reason`. Invalid targets or increment
bounds raise before changing execution or machine state. Numerical failures
leave the accepted machine available for inspection and saving; they do not
raise out of these stage methods.

`run()` returns success at the requested error fraction, including final refinement.
A successful run to a smaller fraction remains a partial-strength output. `002`
uses the returned success flag together with the achieved error fraction to choose
`machine.json` or `partial.json`, and saves with `line.to_json(...)` before raising
on a failed run. The workflow has
no saving method or correction-specific metadata writer. Saving does not run an
extra Twiss calculation; callers can explicitly validate or add metadata if needed.
There is no checkpoint or disk-resume mechanism. The explicit
`correct_at_fixed_strengths` sequence remains the place to add correction stages.

Twiss evaluation is centralized in `FCCeeCorrection.twiss(seed=None, *,
only_orbit=False, chrom=False)`. A supplied table provides `co_guess=seed.particle_on_co`;
without a seed, the line defaults apply. Error-increase trials explicitly pass
`last_accepted_twiss`, while nominal-reference calculations are unseeded.

## Correction optimizer modules

`correction_optimizer.py` defines the example-local `CorrectionOptimizer`, a
subclass of `xtrack.match.OptimizeLine`. Each correction module builds an instance
of this shared class using ordinary functions; the modules contain no class
definitions:

- `corr_optics.build_opt_corr_optimizer`: normal optics and horizontal dispersion.
- `corr_orbit.build_orbit_corr_optimizer`: one instance for each plane, with independent
  response solves and BPM-frame orbit goals.
- `corr_coupling.build_coupling_corr_optimizer`: RDTs, vertical dispersion and c_minus.
- `corr_tune.build_tune_corr_optimizer`: nominal tune targets and response.
- `corr_chromaticity.build_chromaticity_corr_optimizer`: ideal chromaticities using
  four arc sextupole family offsets.

Tune correction varies independent offsets `dkqf2_tune` and `dkqd1_tune`.
The original `kqf2` and `kqd1` family variables remain unchanged. The installer
follows their quadrupole connections and adds equivalent offsets to `k1`,
recording the connections so reloading does not attach them twice.

Chromaticity correction varies `dksf1_chromaticity`, `dksf2_chromaticity`,
`dksd1_chromaticity` and `dksd2_chromaticity`. These independent offsets act on
arc sextupoles only; the original family values and straight-section sextupoles
remain unchanged. Installation is idempotent, including after saving/reloading.
The module evaluates the aligned machine with nominal nonlinear strengths to
obtain constant `dqx`/`dqy` targets. Its 2-by-4 response uses central differences
of the family offsets and restores each probe. The default tolerance is 0.1
in each plane; both targets and tolerance are inspectable on the optimizer.

`correct_at_fixed_strengths(..., chrom=False)` retains ordinary Twiss during the
error increase and excludes the chromaticity goal. `refine_correction()` calls
it with `chrom=True`: a fresh chromatic Twiss is obtained when the input lacks
chromaticities, and subsequent evaluations also include them. A conditional
chromaticity correction follows the tune step. Every enabled goal is checked
afterward, so sextupole changes that disturb other quantities require another
cycle. No persistent phase flag is stored. Chromatic Twiss uses `delta_chrom=1e-6`.
Diagnostics omit unavailable chromaticities; missing data never satisfies the
chromaticity goal. Response preparation adds nine chromatic Twiss evaluations
at nominal full nonlinear strength before the error increase begins.

Each module owns its settings constant, targets, varies, response construction,
solve cutoff, reference/observation data, nominal knob values, `compute_metrics(tw)`,
`goal(tw)` and `print_metrics(tw, log=...)`. These diagnostic methods consume the
supplied table without tracking or implicitly retaining it. The normal optimizer
API remains available for target inspection and least-squares steps.

The builder passes `compute_metrics`, `goal` and `print_metrics`
functions to `CorrectionOptimizer`. Each function takes `optimizer` as its first
argument, followed by Twiss and any reporting options. The shared class binds that first
argument, so the workflow still calls `optimizer.goal(tw)`, for example.
All stages apply corrections through `run_lstsq`; there is no specialized tune
correction method. Helpers used only within a module remain ordinary function calls.

For example:

```python
from copy import copy

settings = copy(corr_optics.DEFAULT_OPTICS_CORR_SETTINGS)
settings.beta_err_rms_tol = 0.02
knobs = corr_optics.install_optics_knobs(line)
opt_corr = corr_optics.build_opt_corr_optimizer(
    line, reference, knobs=knobs, observation_names=bpm_and_ip_names,
    interaction_points=interaction_points, ip_names=ip_names,
    settings=settings,
)
metrics = opt_corr.compute_metrics(tw)
goals_met = opt_corr.goal(tw)
```

`DEFAULT_OPTICS_CORR_SETTINGS`, `DEFAULT_ORBIT_CORR_SETTINGS`, `DEFAULT_COUPLING_CORR_SETTINGS` and `DEFAULT_TUNE_CORR_SETTINGS` are
module-level `SimpleNamespace` objects. Builders copy these defaults when no
settings are supplied, so each optimizer's settings can be edited independently.
Explicitly supplied settings are used as provided. Derived response weights are
calculated in the builders. No subclass
or method override is needed to define another correction.

`FCCeeCorrection` accepts `intermediate_max_iterations` and `final_max_iterations`
directly and stores them as editable attributes. The example
passes module settings through an `optimizer_settings` dictionary at construction;
inspect or change them afterwards through each `optimizer.settings`. Response
weights and cutoffs are configured during construction; `optimizer.rcond` remains
directly editable. There is no central schema of correction-specific settings.

Builders initialize constant tolerances and the active intermediate limits.
`iterative_error_increase_and_correction()` explicitly sets the intermediate optics Dx and
coupling RDT/c_minus/Dy limits; `refine_correction()` explicitly sets the final limits.
Each passes its iteration budget to `correct_at_fixed_strengths(tw, *, max_iterations)`.
There is no stored refinement phase. Snapshots contain only machine controls and
`last_accepted_twiss`; restoring does not change optimizer tolerances.
Changing the error fraction does not change any tolerances. Goal calls need only
`tw`, without mode/context arguments. Active tolerances, such as
`optics.dx_err_ip_tol` or `orbit_x.orbit_tol`, remain inspectable and directly editable.
There is no goal-configuration callback. When adding a correction with different
error-increase/refinement limits, include its assignments in the two execution methods.

Orbit goals use inclusive RMS comparisons and a fixed tolerance (10 micrometres
by default) throughout error increases and refinement.
Optics/coupling goals retain strict comparisons and their error-increase/refinement limits.
Optics still checks tunes as well as beta and Dx. Tune targets and global
acceptance both use the configured tune tolerance, without a separate cleanup limit.

The workflow collects metrics and prints one short summary from each optimizer.
Metric names must be unique. Orbit metrics are `orbit_x_rms` and `orbit_y_rms`.
Optics metrics are `beta_err_rms`/`beta_err_max` for relative beta errors and
`dx_err_rms`/`dx_err_ip` for absolute dispersion errors against the reference.
Additional named metrics remain available through `compute_metrics(tw)`. Detailed targets, weights,
scales and singular values are inspected on the optimizer itself, without mirrored
attributes on `FCCeeCorrection`.

After constructing the optimizers, the workflow collects their varies and nominal
knob values for scaling as errors increase and snapshots. Shared knobs appear once;
conflicting nominal values raise an error. Each module declares nonzero nominal
values where needed (currently tune knobs); ordinary correction trims default to
zero. A new correction therefore needs its own module, an instance added in
`FCCeeCorrection.__init__`, and its desired placement in `correct_at_fixed_strengths`.
It automatically participates in overall goals, metrics, printing and rollback.
The correction sequence remains custom Python; no generic pipeline is introduced.

Each optimizer owns a local action and uses supplied results at construction.
The workflow still owns the common reference and observation policy, precomputes
response SVDs, and controls candidate evaluations, acceptance and rollback.

Each optimizer sets `rcond` from its module settings (zero for tunes).
`run_lstsq`, `get_correction`, `step`, `solve`, and `run_jacobian` use this default
when no per-call value is supplied. Explicit per-call values, including zero,
override it without changing the default or rebuilding the response SVD.

## Relative beta objective

Beta, alpha and horizontal dispersion use one observation list: BPMs plus the
four main IPs. The optics builder takes this list as `observation_names` and
constructs all five local targets in one loop, assigning each target's value,
weight, tolerance and tag immediately. Tune targets follow the local targets.
The optics optimizer evaluates global beta RMS/max goals across the full Twiss
table, including locations outside the fit observations.

The beta objective is relative beating, `beta / beta_ref - 1`, rather than
`ln(beta / beta_ref)`. Ordinary `xt.Target` definitions expose physical beta
values with their reference values. Multiply nominal relative-beta response rows
by beta_ref and divide their target weights by beta_ref; scale their tolerances
by beta_ref as well. This preserves the weighted linear system and acceptance
cost of the relative-beta objective.

Beta targets within 500 m of an interaction point receive the original extra
weight of 5, accounting for ring wraparound. Horizontal dispersion retains its
original extra IP weight (50 with the default tolerances). Both weightings were
restored after testing uniform weights. IP stopping tolerances are unchanged.

Alpha, dispersion, and tunes also use ordinary parameter/at targets. Alpha now
targets `alpha_ref` directly, without the `beta_ref / beta` factor. Convert the
shared helper's nominal normalized-alpha response using
`d(alpha) = d(alpha * beta_ref / beta) + alpha_ref * d(log(beta))` at the reference.
The response builder includes alpha locations in its internal beta observations
to supply these derivatives. Alpha weights are unchanged.
No callable targets remain in optics optimizer setup. Use the optimizer's
`target_status(ret=True, action_results=[tw]).residue` for physical residuals
and its targets' weights for the matching objective weights. The analytical response helpers retain their logarithmic convention.

## Optics response from the optimizer

`build_normal_quad_optics_response_matrix(optimizer, reference)` lives at the
bottom of `corr_optics.py`. It returns an array in full target/vary order,
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

`corr_optics.install_optics_knobs(line)` installs one independent
`dk1_optics_<name>` trim per quadrupole, adds it to `k1` through `on_optics_corr`,
records the optics circuit in metadata, and returns the knob list. Repeated
occurrences of the same element share one trim. Both 001 and the correction
workflow call this function. Existing variables are checked against the actual
`k1` expression dependencies, so a loaded lattice is reused without adding a
second trim term, resetting its values or enabling a disabled switch. A conflicting
variable name raises an error before any edits; the metadata alone is not used
as evidence that the connection exists.

The optics builder takes the returned `knobs` list explicitly, without reading
line metadata. Custom pre-existing knob lists can still be passed directly to
the builder. It installs this matrix with the original relative-beta,
alpha, horizontal-dispersion and tune weights, including the IP emphasis.
All optics knobs use the constant `vary.weight = 1.`, so their columns are no
longer normalized using the tune derivatives. The SVD uses the physical knob
coordinates and the existing target weights. Optics setup does not run a
directional tracking check or compute Twiss; response validation belongs in tests.

## Supplying model results

Accept `action_results={action: result}` or an ordered list/tuple of results at
all target-evaluation entry points. Sequence entries follow `optimizer.actions`
(first occurrence across targets, with duplicates removed) and must have exactly
one result per action. Dictionary input remains supported; `run_actions()` still
returns a dictionary.

FCC-ee uses `action_results=[tw]` and `[reference]` at construction. Each optimizer
owns a local action, so builders and tune correction no longer accept a shared
action and the workflow has no `twiss_action` attribute. Optics/orbit actions
wrap `line.twiss`; coupling/tune actions wrap the existing full-Twiss callback.
Supplied results bypass action preparation and execution, including construction.
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

The example's `CorrectionOptimizer` defaults to `tag_outcome=False`, skipping the
outcome evaluation and log entry. The caller evaluates the candidate and can
record it with `tag()`, or explicitly pass `tag_outcome=True`. Ordinary optimizers
retain the default of `True`.

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

## Goal checks and iteration limits

The fixed-strength loop runs **orbit, optics, coupling, tune**, followed by
**chromaticity** when `chrom=True`.
Each stage is skipped when its goal is already met. At the end of a cycle,
if `tunes.goal(tw)` is false, one `tunes.run_lstsq(action_results=[tw])` is followed
by a fresh Twiss evaluation. An already satisfied overall goal returns before
starting a cycle.
Candidate application and Twiss evaluation are explicit in
`correct_at_fixed_strengths()`. There are no penalty comparisons, penalty helper
methods or inner step-halving retries. Intermediate errors may increase.

`intermediate_max_iterations=5` limits complete cycles at each error fraction;
`final_max_iterations=40` applies to final refinement. The loop checks all goals
before any work and after every completed cycle, including the last allowed
cycle. If goals remain unmet at the limit, it raises. The error-increase/refinement caller
owns rollback; a rejected error-increase trial also halves the error-fraction increment.
Twiss failures propagate immediately to the outer rollback.

The coupling stage applies one ordinary correction in both orbit planes before
checking the coupling goal again. If a skew correction is needed, it evaluates
Twiss, applies another orbit correction in both planes, and evaluates Twiss again.
These orbit steps are unconditional within the coupling stage and use full Twiss;
there is no separate settling loop, tolerance, projection, or local rollback.
Tune correction remains explicit after coupling. Optics does not nest orbit correction.

There is no nested tune loop: further tune corrections happen in subsequent
main cycles, within the same iteration budget as the other corrections. The
optics optimizer also retains its qx/qy targets. Orbit, optics, tune and
chromaticity use step_scale=1; coupling
retains its configured scale (0.5 by default). Optimizer knob limits and
maximum-step handling are unchanged.

The RMS/max goals remain in the correction modules, with their strict versus
inclusive comparisons and intermediate/final tolerances unchanged. The optimizer's scalar
`target_status().penalty` remains available for inspection, but the workflow
no longer uses it to decide whether to retain a candidate.

## Explicit Twiss lifetime

`correction.twiss()` always computes the current machine. The orchestration
retains the returned table and supplies it to target checks and least-squares
steps. After changing knobs, compute a new table before checking the candidate.
For retries, restore the baseline machine and reuse its original table. Final
refinement starts from `last_accepted_twiss`; final saving independently recomputes
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

- Using BPMs plus the four main IPs for all optics quantities passed the full
  ramp, final refinement and fresh final Twiss goal check in 222.2 s wall time
  (250 s limit). The common list has 2643 locations and produces 13217 targets.
  All IP weights are retained. Final whole-ring beta RMS x/y was 0.72%/0.91%,
  with maxima 5.32%/25.97%; IP Dx/Dy errors were 88.9/10.3 um, below their
  100/20 um limits. All other final goals passed. The 15 focused tests pass,
  including per-location target order, weights and tolerances. Artifacts:
  `/tmp/fccee_bpm_ip_zvymp8lm/{run.log,verification.json,timing.json}`.
- Removing the horizontal-dispersion IP weight as well passed the 14 focused
  tests, but the full run hit its 250 s wall-time limit during final refinement.
  The ramp reached full error strength at 193.7 s after two rejected large steps.
  At refinement iteration 4 (243.0 s), horizontal dispersion RMS was 0.273 mm
  and beta RMS x/y was 0.44%/0.96%; completion and all final goals were not
  verified. The log does not include individual IP residuals, so it does not
  identify the remaining failing goal. Both beta and horizontal-dispersion IP
  weights were subsequently restored. Artifacts:
  `/tmp/fccee_uniform_dx_bz_rg2xd/{run.log,timing.json}`.
- With uniform relative-beta weights (no extra IP beta weight), the full ramp,
  final refinement and fresh final Twiss goal check passed in 196.2 s wall time,
  below the 250 s limit. There were 17 accepted points, 233 full Twiss calls and
  87 orbit calls. Final beta RMS x/y was 1.80%/0.99% (previous weighted run:
  0.49%/0.70%); global maxima were 10.33%/37.50%, within the 50% goal. Maximum
  absolute beta beating across the four IPs was 2.44%/3.24%. All dispersion,
  orbit, coupling and tune goals passed. The run also includes the earlier
  removal of the two setup validation Twiss calls and the generic response
  builder, so its timing is not an isolated weighting benchmark. Artifacts:
  `/tmp/fccee_uniform_beta_93kgi3aw/{run.log,verification.json,timing.json}`.
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

Moving metrics and goal checks into their correction modules passed all 17
focused tests. The full ramp, refinement, fresh final goal checks and saving
completed in 211.5 seconds wall time, below the 250-second cap. All final goals
passed, with 19 accepted entries and 256 Twiss calls (93 orbit-only). Saved
history agrees with the preceding shared-BPM/IP run within atol=1e-8, rtol=1e-3.
Log and verification: `/tmp/fccee_module_metrics_t28ffb3m/run.log` and
`verification.json` in the same directory. The 003b reference is unchanged.

Using `target_status().penalty` for inner-trial acceptance passed 32 xdeps
optimizer/action/response tests and 18 xtrack correction/response tests. These
include weighted, transformed, logarithmic and inactive targets, no extra model
evaluation or log entry, and an external retry that restores the baseline and
halves an overshooting step twice before accepting it.
The full run passed all final goals in 233.7 seconds wall time, within the
250-second cap (previous run: 211.5 seconds). History agrees with the preceding
run within atol=1e-8, rtol=1e-3, with the same 19 accepted entries and 256 Twiss
calls (93 orbit-only). Log and verification:
`/tmp/fccee_status_penalty_29fjkqk3/run.log` and `verification.json` there.

The fully separate orbit/tune/optics/tune/coupling/tune trial hit the 250-second
cap at an accepted error fraction of 0.243853, with 11 rejected ramp trials.
Repeated coupling-step failures motivated restoring orbit settling inside the
coupling stage. Log: `/tmp/fccee_explicit_stages_zmp60_s7/run.log`.

With combined orbit/coupling correction restored and tune correction kept
explicit, the full ramp, refinement, final goal checks and saving passed in
215.5 seconds wall time (previous penalty-based workflow: 233.7 seconds).
There were 20 accepted entries and 240 Twiss calls, including 83 orbit-only
calls, versus 19 entries and 256 calls previously. The numerical history changes
with the new ordering; all final goals pass. The 22 focused correction/response
tests pass, including stage ordering, tune rollback and orbit settling without
nested tune correction. Log and verification:
`/tmp/fccee_explicit_tunes_coupled_orbit_fmhwnxdm/run.log` and `verification.json`
in the same directory. The 003b reference and its shared helpers are unchanged.

Removing all inner step-halving retries passed the full ramp, refinement, fresh
final goal checks and saving in 210.5 seconds wall time, below the 250-second
cap (previous run: 215.5 seconds). The accepted history matches the preceding
combined orbit/coupling run within atol=1e-8, rtol=1e-3, with the same 20 entries
and 240 Twiss calls (83 orbit-only). The 23 focused tests pass, including a
single improving candidate and rollback of a rejected candidate without retry.
Only the outer ramp still halves its increment. This validates the tested
machine/error realization; it does not establish equivalence for other errors.
Log and verification: `/tmp/fccee_no_backtracking_s7fp5vnb/run.log` and
`verification.json` in the same directory. The 003b reference is unchanged.

Moving candidate application and acceptance into the fixed-strength loop,
and leaving rollback to the ramp/refinement caller except during orbit settling,
passed all 26 focused tests. The full correction passed all final goals in
212.5 seconds wall time, below the 250-second cap. History matches the preceding
run within atol=1e-8, rtol=1e-3, with the same 20 accepted entries and 240 Twiss
calls (83 orbit-only). Penalty-improvement criteria and the 40-iteration budget
are unchanged. Log and verification: `/tmp/fccee_inline_acceptance_7f98intj/run.log`
and `verification.json` in the same directory.

Removing penalty-based acceptance and using five ramp cycles / forty final
refinement cycles passed the full ramp, refinement, fresh final goal checks
and saving in 197.4 seconds wall time (previous run: 212.5 seconds). Accepted
history agrees within atol=1e-8, rtol=1e-3, with the same 20 entries and 240 Twiss
calls (83 orbit-only). All 30 focused tests pass, including convergence on the
last allowed cycle, exhaustion of each budget, retention of worsening valid
steps, tune iteration exhaustion, and local restoration after a settling
evaluation fails. Log and verification: `/tmp/fccee_goals_only__91z7mem/run.log`
and `verification.json` in the same directory. The 003b reference is unchanged.

Local comparison output: `/tmp/check_fccee_lstsq.log`; the earlier comparison
including the rejected 0.0125 trial is `/tmp/check_fccee_lstsq_initial.log`.

Ordered action results, per-optimizer actions, and optimizer-owned `rcond`
passed 84 focused tests across xdeps and xtrack. Tests cover action ordering and
length validation, supplied-result evaluation without action calls, default
cutoffs and explicit overrides for fixed/numerical responses, SVD reuse, and
clone preservation. The full ramp/refinement passed all final goals in 191.4 s
wall time under the 250 s cap, with the same 20 accepted entries and 240 Twiss
calls (83 orbit-only). History agrees with the preceding goal-only run within
atol=1e-8, rtol=1e-3. Log and verification:
`/tmp/fccee_optimizer_defaults_7jgk0xt0/run.log` and `verification.json` in that
directory. The 003b reference and shared numerical helpers remain unchanged.

Moving setup, settings, metrics, active goals and printing into
`CorrectionOptimizer` subclasses passed 36 focused xtrack tests. Checks include
mode/fraction changes, restoring active tolerances, strict/inclusive goal limits,
unchanged metrics against the shared reference helpers, and an additional
correction participating in goals, printing, metrics, knob scaling and rollback.
The full ramp, final refinement, fresh final goal check and saving completed in
190.4 seconds wall time under the 250-second cap. All final goals pass; there are
20 accepted entries and 240 Twiss calls (83 orbit-only), as before. The saved CSV
history is identical to the preceding optimizer-defaults run. Artifacts:
`/tmp/fccee_correction_classes_gl4bmgt2/run.log`, `verification.json` and
`timing.json`. The 003b reference and all its shared helpers remain unchanged.

Replacing the correction subclasses with module builder functions passed the same
36 focused tests, including independent callback binding and settings on separate
instances. All five correction optimizers are now direct `CorrectionOptimizer`
instances. The full ramp/refinement completed in 198.2 seconds wall time under
the 250-second cap, with all final goals met, 20 accepted entries and 240 Twiss
calls (83 orbit-only). The saved CSV history is exactly equal to the preceding
baseline. Log and verification are in
`/tmp/fccee_correction_factories_jiyby7h_/run.log`, `verification.json` and
`timing.json`. No correction module contains a class definition; the analytical
optics response builder and the shared 003b helpers remain unchanged.

Removing goal-configuration callbacks and using a constant 10-micrometre orbit
tolerance passed all 36 focused tests. Mode-switch tests cover the tighter
optics/coupling limits, rollback across modes, and preservation of active limits
within a mode. The full ramp/refinement completed in 204.4 seconds wall time
under the 250-second cap. All 12 independently checked final criteria passed.
There were 20 accepted entries and 245 Twiss calls (88 orbit-only), compared with
240 calls (83 orbit-only) before this change. The looser early orbit criterion
changes the ramp history; exact history equality is no longer expected. Final
orbit RMS values were 0.0112 and 7.1434 micrometres. Artifacts:
`/tmp/fccee_constant_orbit_tol_oresz_3p/run.log`, `verification.json`,
`final_checks.json` and `timing.json`. Coupling's 50-nanometre orbit settling
tolerance and all shared 003b helpers remain unchanged.

Using the constant `weight=1.` for every optics knob, together with the module
settings constants, passed all 36 focused tests. The full ramp/refinement passed
all 12 independently checked final criteria in 174.9 seconds wall time, versus
204.4 seconds for the preceding run. It accepted 17 history entries without any
rejected ramp steps and used 221 Twiss calls (79 orbit-only), versus 245 (88
orbit-only) previously. Final beta RMS was 1.30%/1.33%, beta maximum was
5.04%/7.42%, and Dx RMS was 0.974 mm against the 1 mm limit. Artifacts:
`/tmp/fccee_constant_optics_weight_6nbea93h/run.log`, `verification.json`,
`final_checks.json` and `timing.json`. The analytical response, target weights,
acceptance tolerances and shared 003b helpers remain unchanged.

Moving error-increase execution and refinement into `FCCeeCorrection` passed
56 focused tests, including rejected-trial rollback, minimum-increment failure,
partial-result saving, refinement failure, and refinement below full strength.
The full run passed all 12 independent final checks in 182.3 seconds wall time,
within the 250-second cap. Its accepted history is exactly equal to the previous
workflow: 17 entries and 221 Twiss calls (79 orbit-only). Artifacts:
`/tmp/fccee_class_execution_91lm1e1u/run.log`, `verification.json`,
`final_checks.json` and `timing.json`. `004` now contains configuration,
construction, one `run(...)` call and saving; the reference and shared 003b
helpers remain unchanged.

The folder cleanup retains only the 000/001/002 entry scripts and their
required modules. Four obsolete entry scripts and 21 unused helper functions
were removed; retained numerical helper implementations are unchanged. All
56 focused tests pass. Running the renamed `002_correct_with_state.py` completed
in 175.2 seconds wall time, with all 12 final checks passing and exactly the
same accepted history (17 entries, 221 Twiss calls, including 79 orbit-only).
Artifacts: `/tmp/fccee_folder_cleanup_umez7zpj/run.log`, `verification.json`,
`final_checks.json` and `timing.json`.

### Ordinary orbit correction around coupling

Replacing the dedicated settling loop with one ordinary two-plane orbit correction
before and after coupling passed a sequential baseline/candidate comparison on the
same prepared input lattice, using one BLAS thread and a 250-second cap per run.
Overall wall time fell from 178.7 s to 150.4 s (15.8%). Twiss calls fell from 219
(140 full, 79 orbit-only) to 170 (all full). Both runs reached full error strength,
passed all final optimizer goals, and had no rejected increments. Final refinement
needed three cycles in the baseline and four in the candidate. All 57 focused
tests pass, including unconditional orbit correction around coupling and propagation
of failed evaluations to the outer rollback. The settling helper, correctable-mode
projection, coupling-specific orbit tolerance, and full-Twiss promotion helper
were removed. These timings are a single paired comparison.
Artifacts: `/tmp/fccee_plain_orbit_8446z1w4/run.log`, `comparison.json`, and each
run's `verification.json` and `timing.json`. Saved machines are in each run's
`response_api_class/machine.json`.

### One conditional tune step per cycle

The tune stage now uses one ordinary `run_lstsq` at the end of each main cycle,
only when `tunes.goal(tw)` is false. The specialized `correct` method, its nested
iteration budget, and half-tolerance cleanup setting were removed. The ordinary
tune tolerance is used by targets and goals. `CorrectionOptimizer.run_lstsq`
defaults to `tag_outcome=False`, while explicit outcome tagging remains available.

Using the same prepared lattice and one BLAS thread, the conditional version
completed in 139.3 s with 132 full Twiss calls, compared with the preceding
three-position tune cleanup's 150.4 s and 170 calls. The intermediate experiment
with an unconditional end-of-cycle tune step took 143.0 s and 140 calls. All
three runs reached full error strength and passed every final optimizer goal.
The conditional run had no rejected increments and needed four refinement cycles.
These are individual run timings, each below the 250-second cap. All 58 focused
tests pass, including default/explicit tagging, tune-step ordering, skipped
satisfied goals, iteration exhaustion and outer rollback on failed evaluations.
Artifacts: `/tmp/fccee_single_tune_kc05pjsp/run.log`, `comparison.json`, and the
`candidate` (unconditional) and `conditional` subdirectories, each containing
`verification.json`, `timing.json` and `response_api_class/machine.json`.


### Chromaticity correction during refinement

The full standard-lattice run passed every optimizer goal at full error strength
and again after saving/reloading. The original tune and sextupole family values
were unchanged. There were no rejected error increments, and refinement required
four cycles. Final chromaticities were [12.08386247, 5.23995844], against ideal
values [12.11101576, 5.24079795]; both errors were below the 0.1 tolerance.

With one BLAS thread, the example took 152.7 s, versus 140.5 s in the preceding
tune-offset run. Twiss calls increased from 132 to 144. The complete validation,
including loading the baseline and reloading the saved result, took 187.9 s,
within the 250-second cap. These are individual run timings, not repeated
benchmarks. All 66 focused tests pass, covering family offsets and preservation,
reload/idempotency, response restoration, refinement-only activation and
rechecking other goals after a chromaticity change.
Artifacts: `/tmp/fccee_chrom_refinement_yw7balkq/run.log`, `verification.json`,
`timing.json`, and `response_api_class/machine.json`.

### Independent momentum scan

The new script was validated on the saved standard machine above, without the
correction object or its accepted Twiss particle. Independent search at `ipa`
failed; search at `ipd` succeeded. Recovered tunes agree with the preceding
validation within 4e-9. Ideal reference betas match a separate load of the
original standard lattice exactly at all 2643 BPM/IP observations. Beta-beating
RMS relative to this nominal-nonlinearity reference is 1.176% / 1.260%.

The final 21-point scan over delta = +/-0.01 took 38.9 s with one BLAS/OMP
thread. Sixteen requested points succeeded, from -0.008 through +0.007.
Continuation could not resolve tune discontinuities on the way to -0.009
and +0.008 (last reached -0.00809375 and +0.007083984375). The remaining
three points were marked not reached. These are optics/continuation limits,
not physical aperture or momentum-acceptance measurements.

Output checks verified the requested grid, finite successful rows, NaN failed
rows, matching failure metadata, beta-beating arithmetic and four plot artifacts.
Plot axes were subsequently fixed to the requested +/-0.01 range and regenerated
from the saved scan. Python compilation and `git diff --check` passed. No DA/MA
tracking scan was run. Artifacts: `/tmp/fccee_analysis_KNaJM2/final_scan.log`,
`final_scan/` (plots, NPZ, summary and verification), and
`/tmp/fccee_analysis_KNaJM2/reference_verification.json`.

Adding the clean-machine scan and overlays completed in 48.2 s with one BLAS/OMP
thread. All 21 clean points succeeded; the corrected scan retained the same 16
successful points and five failed/unreached points. Corrected coordinates,
betas and tunes agree with the preceding output within rtol=1e-8, atol=1e-10.
Clean results were independently compared with the original standard lattice
at delta=0 and +/-0.003: beta/orbit arrays and tunes passed, with maximum
horizontal-beta relative differences below 1.2e-8. Checks also covered NaN
failure rows and both beta-beating definitions. All four plots were saved and
the comparison plots were visually inspected. Artifacts:
`/tmp/fccee_analysis_KNaJM2/clean_comparison.log` and `clean_comparison/`
(plots, NPZ, summary and `verification.json`).
