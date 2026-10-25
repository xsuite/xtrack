# Externally evaluated FCC-ee correction

## Integrated second-order chromaticity correction

`corr_second_order_chrom.py` replaces the dedicated `007_arc_octupoles.py` study.
`001` installs two independent arc quadrupole knl[3] offsets; `002` includes
`second_order_chrom` among its configured optimizers and saves the final result
in the usual `response_api_class/machine.json`. No new elements are added and
arc dipole b4 stays active. Settings are exposed as
`DEFAULT_SECOND_ORDER_CHROM_SETTINGS` and `optimizer_settings['second_order_chrom']`.

The targets are the clean machine's Q''/2 with nominal nonlinear strengths.
The default tolerance is 1 in each plane, while the underlying optimizer targets
Twiss's Q'' (`ddqx`, `ddqy`) with tolerance 2. Central k3 probes of +/-0.01 m^-4
build the fixed 2-by-2 response at the clean reference. Each controlled
quadrupole receives an additive `knl[3] = switch * trim * length` contribution.
The two offsets are
included in the workflow's knob discovery, scaling and snapshot/rollback.
Unlike the earlier standalone study, these correction trims are outside
`on_nonlinear` so the workflow scales their physical strength exactly once.
Their own master switch is registered for clean-machine reconstruction.

Both chromatic corrections are skipped during the error ramp (`chrom=False`).
During refinement, first-order chromaticity correction is followed by
second-order correction, then every goal is rechecked. Orbit, optics, coupling,
tunes or first-order chromaticity disturbed by an octupole step must converge
again before the machine is accepted. Tracking is not part of this loop.

Both orders reuse the same chromatic Twiss calculation. The shared
`delta_chrom=2e-4` is configured through the second-order settings. At the old
1e-6 step, numerical cancellation produced unreliable Q''/2 (errors of thousands
on this lattice). At 2e-4, clean, corrected and quad-octupole cases agree with
the independently validated five-point quartic fit to about 0.25 in Q''/2;
first-order chromaticity changes by much less than its 0.1 tolerance. The
numerical comparison is in `/tmp/fccee_second_order_Tc9y0Y/chromatic_check.json`.

`003` remains the tune-versus-delta analysis. `004` defaults to b4 on and accepts
the ordinary corrected machine. Its optional `CHECK_AMPLITUDE_DETUNING` diagnostic
runs the former 007's 0.5–3 sigma action scans before radiation/tapering is enabled,
with cavity kicks inhibited and delta/zeta unfrozen. A timing pilot precedes
512-turn tracking, and 256/512-turn estimates and separate-action slopes are saved.
The 6D aperture study then proceeds with the selected radiation model, active RF and tapering.
The diagnostic defaults off; it does not add plots or tracking to ordinary correction.

The following integration validation used the former per-magnet K3L circuit;
it does not validate the k3 circuit introduced on 2026-10-25.
Integration validation (2026-10-24) regenerated `001` from the standard error
machine and ran `002` with one BLAS/OMP thread. It reached full errors and passed
all seven goals after three refinement cycles, including an independent reload
using `co_search_at='ipd'`. Correction and saving took 160.4 s including input
loading; the complete supervised validation took 204.9 s, below the 250 s cap.
Final Q''/2 was [18.57463124, -73.96432232], against ideal targets
[18.55382726, -74.35966084]. An independent five-point quartic fit at
delta = [-0.002, -0.001, 0, 0.001, 0.002] gave corrected
[18.70125993, -74.17406517] versus clean [18.44353563, -74.33321068],
within tolerance in both planes. The resulting arc trims were
[-0.81376613, -1.01478430] m^-3. Original tune/sextupole family variables and
element names were preserved; b4 remained on, radiation and tapering off.
The validation report and log are in `/tmp/fccee_second_order_Tc9y0Y/`.
All 80 focused regression tests passed; 58 relevant tests were rerun after
the final installer validation and settings changes and also passed.

The optional `004` diagnostic was exercised on both regenerated machines with
`RUN_SCANS=False`, eight tracking threads and single-threaded BLAS. This exposed
and fixed a context transition: radiation Twiss requires a serial tracker after
the parallel detuning scan; parallel tracking is restored after 6D optics.
The successful smoke test took 114.7 s, including ten launches per machine for
512 turns, independent tapering/6D closure and four 64-turn mean/quantum pilots.
All detuning launches survived, delta stayed fixed and zeta evolved freely.
JSON, NPZ and PNG/PDF outputs are in
`/tmp/fccee_second_order_Tc9y0Y/analysis_006/`; the successful log is
`/tmp/fccee_second_order_Tc9y0Y/analysis_006_retry.log`.
At 0.5–1 sigma the clean [axx, axy, ayx, ayy] coefficients were approximately
[-0.092, 3.212, 3.206, 4.394] x 10^5 m^-1, versus corrected
[-0.498, 6.021, 6.000, -6.206] x 10^5 m^-1. Matching Q'' therefore does not
restore ideal amplitude detuning. No full DA/MA scan or acceptance claim is
part of this integration validation.

The validated prepared input and `response_api_class/machine.json` replace the
standard local files. Their predecessors are preserved as
`previous_correction_knobs.json` and `previous_machine.json` in the validation
directory above.

## Earlier arc octupole response study

Before integration, `007_arc_octupoles.py` used additive `knl[3]` trims in the existing arc
`qf2a` and `qd1a` quadrupoles, following the user's simplification. No new
elements are inserted. Each knob denotes total integrated K3L [m^-3] per
physical quadrupole; where a magnet has multiple pieces, the trim is shared
in proportion to their lengths. Existing `knl[3]` expressions and relative
errors are preserved. The added trim follows `on_arc_octupole_corr` and
`on_nonlinear`; clean-machine reset disables the former.

The component uses the quadrupole's existing body integration (five kicks in
the current selected families), alignment and roll. It is therefore a different
model from an independent thin entrance kick, requiring a new measured response.
The analytic detuning response remains explicitly an entrance-optics estimate;
tracking checks its predictions. New outputs use `analysis/arc_octupoles_quad_knl`,
preserving the earlier thin-element study under `analysis/arc_octupoles`.

The quad-body remeasurement gives K3L = -0.814773 / -1.017821 m^-3 per
physical QF/QD, yielding Q''/2 = [18.721, -74.443] against clean
[18.433, -74.341]. All 25 momentum points over +/-1.2% succeed. The full
candidate's ten 0.5–3 sigma launches survive 512 conservative turns, with a
similar detuning tradeoff to the earlier entrance-kick model. The focused
validation took 101.5 s and verified unchanged element names, zero-trim
preservation, total integrated trim per magnet, unchanged input SHA256, and
independent Twiss after saving/reloading. Switching the trims off after reload
recovers every original quadrupole knl[3]. Two focused tests pass, including
an unequal-length split magnet and preservation of a live existing expression.

The new directory contains `quad_knl_validation.json`, tracking/scan NPZs and
`trial_full_machine.json` for a later 6D comparison. The original machine is
unchanged. The log is `/tmp/fccee_quad_octupoles_jBQqA1/validation.log`.
The guarded and fractional cases have not been rerun for this representation;
the earlier results below remain explicitly those of the entrance-kick model.

### Earlier entrance-kick study

The first implementation explored new normal thin octupoles at the entrances of
regular arc `qf2a` and `qd1a` quadrupoles: 848 and 864 sites respectively, with
one shared integrated-strength knob per family. Physical-magnet metadata
selects one entrance per magnet, avoiding duplicate correctors on split halves.
These are hypothetical additional correctors; no IR octupole or original
sextupole/tune/correction control is changed, and no hardware strength limit
is assumed. The saved corrected input is never overwritten.

The starting point restores arc dipole b4. A central finite-difference response
fits Q and Q' as monitors and Q''/2 as the two objectives. Five delta points
between -0.002 and +0.002 use a quartic fit and independent closed-orbit
continuation from `ipd`; halving the fit interval checks sensitivity. The
two-family linear solution is evaluated at 25%, 50%, and 100%, alongside clean,
corrected and b4-off references. A second, guarded solution minimizes the two
relative chromatic errors while requiring predicted errors in axx, the mean of
axy/ayx, and ayy to be at most 90% of their initial distances from clean. It
uses the measured chromatic response and analytic detuning response, then
checks the result with tracking. This is a small-amplitude coefficient guard,
not a bound on the whole tune footprint or an acceptance guarantee.
Full tune scans cover +/-1.2%, retaining NaNs
and failure messages when continuation fails. The analytic uncoupled response
is recorded only as a comparison with the measured response.
Continuation rejects tune changes larger than 0.02 per accepted substep and
halves the delta step down to 1e-6. This is deliberately tighter than 003's
default: the loose limit could bridge an ambiguous region of the guarded trial.

The conservative diagnostic uses radiation/tapering off and inhibits cavity
kicks with `XS_FLAG_KILL_CAVITY_KICK`; neither energy nor zeta is frozen.
Tracking starts at `septum`, with zero normalized momenta, delta=0, and the
approved geometric emittances 0.71 nm and 1.4 pm. A 64-turn pilot precedes
512-turn tracking, using eight OpenMP threads and one BLAS thread. Separate
action scans keep the other mode at 0.5 sigma; slopes over 0.5–1, 0.5–2 and
0.5–3 sigma retain both cross derivatives. Comparing 256/512-turn estimates
checks tune-extraction convergence. A synthetic test verifies that horizontal
curvature cannot alias into the much smaller vertical action in this fit.
Plots, monitored normal coordinates, tunes, strengths, response and settings
are saved under `analysis/arc_octupoles`. This does not establish DA/MA:
any candidate still needs the separate 6D radiation/tapering comparison.

The full chromatic solution requires K3L = -0.837731 / -0.996423 m^-3 per
QF/QD corrector. Measured Q''/2 changes from [1086.49, -1058.89] to
[18.675, -74.436], against clean [18.433, -74.341]. The analytic and measured
chromatic responses agree within 0.03%. At 0.5–1 sigma, ayy changes from
-3.28e5 to -6.51e5 m^-1 (clean +4.39e5); the mean cross coefficient changes
from 2.23e5 to 6.10e5 m^-1 (clean 3.21e5). These coefficients are farther
from clean even though chromaticity improves. The b4-off reference also has
worse amplitude-detuning coefficients despite its improved earlier DA/MA;
therefore coefficient distances alone cannot select the final candidate.

The guarded solution uses K3L = -1.412819 / +0.544977 m^-3. Tracking confirms
that its three small-amplitude detuning errors are below the starting errors,
but its Q''/2 is [-253.0, -623.8] and its negative-delta tune curve develops a
large excursion. Tighter continuation cannot reach -1.2%, stopping at
-0.01116406. This is a failed optics continuation, not a measured MA boundary.
The full chromatic solution is the better candidate for the next 300-turn
6D comparison; neither candidate has yet been validated with radiation/tapering.
The nominal IR octupoles and original corrected machine remain unchanged.

Validation: the six initial cases took 297.6 s, the additional guarded case
66.9 s, and the final seven-case rescan with the tighter continuation limit
192.4 s (all with one BLAS thread and eight tracking threads). Only the
guarded -1.2% point fails in the final comparison. All tracked 0.5–3 sigma
particles survive 512 conservative turns. The largest 256/512-turn difference
in a small-amplitude detuning coefficient is 250 m^-1; halving the chromatic
fit interval changes Q''/2 by at most 0.18. Artifact verification checks the
1,712 unique arc sites, initial normal coordinates, fixed delta, failed-point
NaNs, unchanged input hash, and all four plot files. The action-fit test passes.
Results and `verification.json` are in `analysis/arc_octupoles`; the live log is
`/tmp/fccee_octupole_survey_1geiXT/survey.log`. No DA/MA scan or saved-machine
replacement was performed in this study.

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
off errors and correction switches, resetting the independent tune offsets to
zero while preserving the family expressions, and retaining nominal nonlinear
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

## 6D DA/MA comparison at the septum

`004_da_ma_6d.py` compares the standard clean and corrected machines. This is
a separate 6D study: the user now requested active RF, synchrotron radiation
and tapering, replacing the earlier proposal of fixed-delta launches at `ipa`.
Both lines are cycled to the exact element `septum`. Clean controls are restored
using the same recipe as `003`; neither the input machine nor correction knobs
are rewritten. Each machine is independently tapered with mean radiation and
zero mean energy deviation. An independent closed-orbit search at `ipd` recovers
the 6D orbit, and an additional physical one-turn track checks closure.

The user selected a preliminary **300-turn** comparison, geometric emittances
0.71 nm / 1.4 pm, and a coarse grid. Both machines use identical settings:

- DA: 16 polar angles covering all quadrants and 11 radii from 0 to 20 sigma;
  zero normalized momenta and zero initial momentum offset from the closed orbit.
- MA: 13 initial momentum offsets from -1.2% to +1.2% and 11 signed equal transverse
  amplitudes from -20 to +20 sigma. Four phases (0, pi/4, pi/2, 3pi/4) are applied
  at each signed amplitude, with the same phase in both transverse modes.
  Positive and negative amplitudes together cover eight directions. The plotted
  color averages completed turns over four phases per sign; a white contour
  separates cells with all four phases surviving from cells with at least one loss.
- One radiation model per run: `RADIATION_MODEL='mean'` by default, or `'quantum'`.
  The same model is used for clean and corrected machines. Tapering always uses
  mean radiation. Quantum tracking uses one reproducible random seed per particle,
  with the same seed assignment in clean/corrected cases. This is one lattice
  error realization and one quantum realization per launch, not an ensemble study.

Transverse coordinates use each machine's 6D normal-form matrix at the septum
and the same geometric emittances. Initial physical zeta equals the synchronous
closed-orbit value; initial delta equals its local closed-orbit value plus the
scan offset. The six launch constraints include the linear dispersive offsets.
They are not nonlinear off-momentum closed orbits. Neither zeta nor delta is
frozen during tracking. No beam-beam or beamstrahlung model is added. Input
physical apertures and the global tracking aperture are retained and recorded;
no chamber aperture is invented. Survival means the tracking state remains
positive through the requested turns. MA denotes this initial-delta versus
transverse-amplitude survival map, not local Touschek acceptance or lifetime.

The recipe is inspired by the [July ESRF presentation, slide 5](https://indico.cern.ch/event/1704042/contributions/7166876/attachments/3307462/5918024/LCC_tuning02072026.pdf)
(eight phases and equal normalized amplitudes) and the
[February GHC study, slides 9–14](https://indico.cern.ch/event/1648395/contributions/6928073/attachments/3219867/5737223/Optics_tuning_of_the_Fccee_xsuite_incl_RDT_correction_Satya_13_FEB_2026.pdf)
(2500 turns, radiation with/without quantum excitation). The septum, 300 turns,
coarse grid, and explicit equal-phase pairing are settings of this comparison;
this is not an exact reproduction of either team's complete simulation.

Settings are at the top, with no argument parser and `SHOW_PLOTS=True`.
`TRACKING_THREADS=8` enables OpenMP for tracking; BLAS stays single-threaded.
Each case runs a small timing pilot before the scan and logs its projection.
`RUN_SCANS=False` performs preparation and pilots only. There are two figures,
DA and MA, each with clean/corrected panels for the selected radiation model. PNG/PDF files,
per-case NPZ launch and loss data (in particle-ID order), optics, and a JSON
summary are saved under `analysis/da_ma_6d`. The summary records input SHA256,
closure residuals, aperture policy, settings, timings and completion status.
Initial transverse coordinates are normal coordinates, not laboratory x/y
divided by a common beam size; the NPZ also retains physical launch coordinates.
The stashed phase/chromatic studies remain separate from this script.

Validation on 2026-10-24 used input SHA256
`2268eecc2db8d2d38dd8383e5bb24f4c3fd85d0bc0dd6dc758d79435d27eddc7`.
The full eight scans finished in 464.9 s on eight tracking threads, with
single-threaded BLAS. The independent 6D closure residuals were below 1.5e-12
in x/y/zeta and 8.6e-14 in delta. All 176 DA launches survived for the clean
machine in both radiation models. The corrected machine retained 163 with mean
radiation and 156 with quantum excitation; the first sampled losses occurred at
radius 16 sigma. The clean DA boundary was not reached within the 20-sigma grid.
The corrected MA map was smaller at finite transverse amplitude. At zero initial
transverse amplitude, all four cases survived the sampled delta points through
both -1% and +1%; the next points, +/-1.5%, did not survive. These coarse,
300-turn results do not establish converged acceptance limits.

Three launch/action and loss-ordering tests passed. Artifact validation checked
all eight cases, matching launch grids, physical zeta/delta constraints, loss
turn bookkeeping, and the unchanged input SHA256. Both optics NPZ files use
string arrays for element names, so the artifacts load without pickle.
Run log: `/tmp/fccee_6d_aperture_OMxkfW/comparison.log`.
Results and validation: `analysis/da_ma_6d/summary.json` and `verification.json`.

### Arc-dipole b4 suppression comparison

The subsequent comparison sets `SUPPRESS_ARC_DIPOLE_B4=True` in `004` and
changes the MA scan to 13 points over +/-1.2% (0.2% spacing). The plot limits
are explicitly set to that same range. The switch
`on_error_arc_dipole_b4=0` suppresses both systematic and random normal octupole
field errors in arc dipoles. It leaves nominal octupoles, other error sources,
and the saved correction settings intact; no re-correction is performed.
This is the same component isolated by the earlier chromaticity ablation.
Both machines are independently tapered again. The clean machine already has
all installed errors disabled, so this additional switch does not change it.

The turns, DA grid, signed MA amplitudes, phase directions, emittances and
random seeds are otherwise unchanged. Results go to `analysis/da_ma_6d_b4_off`,
preserving the previous `analysis/da_ma_6d` baseline. Because the MA delta grid
is finer and narrower, total surviving-particle counts should not be compared
directly between the two runs. Clean and corrected machines within each run
still use identical grids. The script records the b4 switch values and labels
the corrected-machine plot panels explicitly. The source machine remains intact.

The b4-off repeat completed in 698.0 s on eight tracking threads. Corrected DA
survival improved from 163 to 176 of 176 launches with mean radiation and from
156 to 170 with quantum excitation; the clean reference retained all 176 in both
cases. The six remaining quantum DA losses included a radius-16-sigma launch,
so the improved count does not establish a larger minimum loss radius.
On the new MA grid, corrected/clean survival counts were 279/330 (mean) and
251/288 (quantum), each out of 572 launches. These are sampled-particle counts,
not acceptance areas. A finite-amplitude MA deficit remains.

Validation checked 9,935 correction knobs and other error controls against the
saved values, with only `on_error_arc_dipole_b4` excluded. The independent 6D
orbit closure, identical clean/corrected launch grids, physical longitudinal
launch constraints, all eight artifacts, and unchanged input SHA256 passed.
The repeated clean DA artifacts matched the baseline exactly, including the
quantum cases. Three analysis tests passed. The log is
`/tmp/fccee_b4_off_JhowYH/comparison.log`; the new output directory contains
`summary.json` and `verification.json` alongside PNG/PDF plots and NPZ results.

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

Twiss evaluation is centralized in `FCCeeCorrection.twiss(co_guess=None, *,
only_orbit=False, chrom=False)`. Callers pass `co_guess=tw.particle_on_co`
directly; when omitted, a zero closed-orbit guess is passed explicitly. The workflow
and the direct optics/orbit actions pass `method='4d'` and `delta0=0.0`
explicitly, leaving `line.twiss_default` unchanged. `000` no longer writes
correction-specific Twiss defaults into the saved input. Error-increase trials
explicitly pass `last_accepted_twiss.particle_on_co`, while nominal-reference
calculations use the zero guess.

Validation (2026-10-25): after replacing the Twiss-table seed argument with
`co_guess`, 79 focused correction/error-recipe tests passed in 6.07 s
with one BLAS/OMP thread, including Twiss evaluations with empty and conflicting
line defaults and checks that those defaults remain unchanged. Log:
`/tmp/fccee_co_guess_tests.log`. The full-machine correction was not rerun.

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
  four sextupole family offsets.
- `corr_second_order_chrom.build_second_order_chrom_optimizer`: ideal second-order
  chromaticities using two arc quadrupole knl[3] offsets.

Orbit, optics, coupling, tune, chromaticity, and second-order chromaticity installation
asserts that correction knob names are absent before creating zero-valued trims
and enabling their switches. `001` installs the knobs once. Optimizers consume
the recorded knob lists, while reference preparation zeros their values.
Existing correction expressions are not inspected.
Optics and coupling use `table.rows.match(element_type=...)` for element
selection; tune and chromaticity work directly on family variables.
`corr_orbit.install_orbit_knobs(line)` completes the horizontal/vertical
corrector layout at quadrupole entrances, attaches independent zero-valued
trims, and records the per-plane correctors, BPMs, and switch. `001` calls it
before installing the nonlinear ramp. Existing corrector expressions and the
horizontal/vertical kick signs are preserved.

Second-order chromaticity uses xdeps to find quadrupoles controlled by `kqf2`
and `kqd1`, including through derived variables. In the standard lattice this
includes `qf3a` and `qd1am` as well as `qf2a` and `qd1a`. Each of the two
correction knobs specifies k3 [m^-4]; every controlled quadrupole receives
`switch * trim * length` in `knl[3]`. No physical-magnet metadata is needed.
Existing `knl[3]` expressions are preserved, and the new contribution is
independent of k1 and its correction trims. Reinstallation asserts, including
after JSON reload; the workflow resets installed knob values instead.

Validation of second-order k3 installation (2026-10-25): 87 focused tests
passed in 14.32 s with one BLAS/OMP thread. The tests cover direct and indirect
family control, exclusion of unrelated quadrupoles and controlled bends,
unequal lengths, independence from k1 and optics trims, preserved expressions,
switch initialization, rejection of existing knobs, and JSON reload. Log:
`/tmp/fccee_second_order_k3_tests.log`. No full-machine correction was
rerun and no saved machine files were regenerated.

`corr_coupling.install_coupling_knobs(line)` mirrors the optics install-once
pattern: select sextupoles with `rows.match`, assert all correction knob names
are absent, set `on_coupling_corr` to one, initialize zero-valued trims, append
their contributions, record metadata, and return the knob list. `001` calls it
after copying the error metadata onto the line. The physical-magnet groups come
from that metadata: one `dk1sl_coupling_<first_piece>` controls each sextupole,
with integrated `ksl[1]` shared among its pieces in proportion to their lengths.
Existing field expressions remain intact, and the skew trims remain outside
`on_nonlinear`. The optimizer consumes the recorded coupling circuit.

Validation of the extracted coupling installer (2026-10-25): 83 focused tests
passed in 5.86 s with one BLAS/OMP thread. The new checks cover unequal piece
lengths, preservation of existing expressions, switch behavior, independence
from nonlinear scaling, and assertion on existing knobs before any changes,
including after JSON reload. Log: `/tmp/fccee_coupling_installer_tests.log`.
The full-machine correction was not rerun.

Validation (2026-10-25): 79 focused tests passed in 5.72 s with one BLAS/OMP
thread. Tests cover zeroing existing trim expressions, preserved family controls
and switches, no duplicate trim contributions after JSON reload, and unchanged
physical-piece weighting. Log: `/tmp/fccee_installers_cleanup_tests.log`.
The full-machine correction was not rerun.

Tune correction varies independent offsets `dkqf2_tune` and `dkqd1_tune`.
The installer appends each offset, multiplied by `on_tune_corr`, to the existing
`kqf2` or `kqd1` family expression. All dependents follow those families; no
element lookup, dependency traversal, or coefficient measurement is needed.
`001` passes its configured `TUNE_KNOBS` list and registers the tune switch.
The optimizer reads the installed knob list. Reference preparation and clean
reconstruction in `003`/`004` zero only the offsets, preserving family expressions.
Metadata retains zero nominal trim values and the family names; the former
`family_nominal_values` and `k1_coefficients` fields are no longer needed.

Validation of the family-level tune installer (2026-10-25): 91 focused correction
and aperture-analysis tests passed in 16.57 s with one BLAS/OMP thread. Checks
cover constant and expression-defined families, dependent variables, switch
behavior, rejection of existing trims, JSON reload, zeroing offsets without
erasing family expressions, and response construction from installed knobs.
Log: `/tmp/fccee_tune_family_tests.log`. Saved machines were not regenerated and
the full-machine correction was not rerun.

Chromaticity correction varies `dksf1_chromaticity`, `dksf2_chromaticity`,
`dksd1_chromaticity` and `dksd2_chromaticity`. The installer appends each offset,
multiplied by `on_chromaticity_corr`, directly to the existing expression of
`ksf1`, `ksf2`, `ksd1`, or `ksd2`. It does not inspect elements or alter their
expressions. All dependents follow the corrected family, including the derived
straight-section family knobs in the standard machine. Installation asserts
that all correction knob names are absent before making changes. `001` installs
these knobs once; the optimizer reads the recorded knob list, and reference
preparation explicitly zeros the installed trims. Reinstallation also asserts
after JSON reload.
Validation of this install-once behavior (2026-10-25): all 56 chromaticity and
workflow tests passed in 4.02 s. Log:
`/tmp/fccee_chromaticity_install_once_tests.log`.

The offsets inherit the sextupoles' `on_nonlinear` factor, so `scale_corrections`
excludes them from its additional scaling; snapshots still include them.
Prepared inputs with the old element-based chromaticity circuit must be
regenerated with `001_install_correction_knobs.py`. The installer does not check
for old element-based circuit metadata. Existing saved machine files have not
been regenerated.

Validation of the family-level attachment (2026-10-25): 81 focused tests passed
in 5.85 s with one BLAS/OMP thread, including derived-family propagation,
unchanged element expressions, JSON reload, and single nonlinear scaling with
snapshot rollback. Log: `/tmp/fccee_chromaticity_family_tests.log`.
The full-machine correction was not rerun for this changed circuit.

The module evaluates the aligned machine with nominal nonlinear strengths to
obtain constant `dqx`/`dqy` targets. Its 2-by-4 response uses central differences
of the family offsets and restores each probe. The default tolerance is 0.1
in each plane; both targets and tolerance are inspectable on the optimizer.

`correct_at_fixed_strengths(..., chrom=False)` retains ordinary Twiss during the
error increase and excludes both chromaticity goals. `refine_correction()` calls
it with `chrom=True`: a fresh chromatic Twiss is obtained when the input lacks
the required derivatives, and subsequent evaluations also include them.
First-order and then second-order correction follow the tune step. Every enabled
goal is checked afterward, so changes that disturb other quantities require
another cycle. No persistent phase flag is stored. Chromatic Twiss uses the
shared `delta_chrom=2e-4`. Missing derivatives never satisfy their goals.
Response preparation adds nine first-order and five second-order chromatic
Twiss evaluations at nominal full nonlinear strength before the error ramp.

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
`dk1_corr_optimics_<name>` trim per quadrupole, adds it to `k1` through `on_corr_optics`,
records the optics circuit in metadata, and returns the knob list. Repeated
occurrences of the same element share one trim. Like the chromaticity installer,
it asserts that all knob names are absent, sets the switch to one, and creates
zero-valued trims while preserving the original `k1` family expressions. `001`
installs them once. The workflow reads the recorded knob list and zeros the
installed trims before computing the nominal reference. Reinstallation asserts,
including after JSON reload. No expression dependencies are inspected.
Quadrupoles are selected with `table.rows.match(element_type='Quadrupole')`.

Validation of the install-once behavior (2026-10-25): 64 optics and workflow
tests passed in 5.73 s with one BLAS/OMP thread, covering preserved family
expressions, repeated elements, and rejection of existing knobs before changes,
including after JSON reload. Log: `/tmp/fccee_optics_install_once_tests.log`.
Existing machine JSON files were not regenerated for the renamed knobs/switch.

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
**first-order and second-order chromaticity** when `chrom=True`.
Each stage is skipped when its goal is already met. At the end of a cycle,
all goals are checked again; a disturbed tune is corrected on the next cycle.
An already satisfied overall goal returns before starting a cycle.
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
both chromaticity corrections use step_scale=1; coupling
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
