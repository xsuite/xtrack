The dominant discrepancy is the sign of the report's radial bump relative to
the R axis used by `survey_utils`. The supplied GEODE results also reveal a
different missing-exit convention and smaller differences in bend rotations.
The intentional 90-degree corrector cases are excluded from the statistics below.

Following confirmation from section 5.4.2, the reader now negates radial
deviations when returning geometric RST displacements. The analysis below
describes the original, pre-fix snapshots, which are retained. The diagnostic
has been adapted to the corrected reader; its historical comparison is unchanged.

This investigation uses the supplied snapshots, with the py313 environment.
It does not change `001a`, `001b`, the bump report, the JSON or the imported TFS.

1. **The two beam-point files describe different things.**

   The legacy beam-point positions remain nominal despite the bumps. Their
   distance from a freshly computed nominal survey is at most 0.048862 mm;
   most of this is the starting-angle difference described below. The new
   beam points contain the misaligned element endpoints. All 384 matched new
   beam points agree with the supplied TFS within 0.000820 mm (0.820 micrometres),
   consistent with rounding each CSV coordinate to a micrometre. The saved
   misaligned JSON gives the same agreement. The two additional legacy beam
   points are `H4TEST.-.E` and `.S`.

   Consequently, a 100 mm beam-point difference can simply be the intended
   displacement. The socket comparison is the meaningful comparison of the
   two alignment workflows.

2. **The report's radial sign is reversed at the Xsuite conversion boundary.**

   `survey_utils._rst_transform_frames` defines R = -x, S = s, T = y in
   the tilted chord frame. `001a` passes the report's radial totals straight
   into that R coordinate. The legacy GEODE socket displacements correspond
   instead to the negative of that R displacement; S and T have matching signs.
   For a pure translation, the measured difference is therefore

   `socket_new - socket_legacy = 2 * radial_bump * e_R`

   apart from the small nominal-survey difference. The data establish this
   relative sign without requiring an assumption about which convention should
   be named positive R globally.

   | Element | Report radial bump | Socket separation |
   | --- | ---: | ---: |
   | XCSV.X0220179 | -100 mm | 200.011 mm |
   | XFFV.X0220257 | +100 mm | 199.982 mm |
   | MBNV.X0220077 | +100 mm | 200.001–200.003 mm |
   | LSX.X0220195 | -50 mm | 100.013 mm |

   This also reverses radial crab rotations. MBNV.X0220321 requests +20 mm
   at entrance and -20 mm at exit over 5 m. Reversing that crab changes the
   longitudinal coordinate of sockets 0.612 m off the axis by approximately
   `2 * (40 mm / 5 m) * 0.612 m = 9.792 mm`.
   This matches the observed 9.793 mm longitudinal discrepancy. A rigid
   transformation using the opposite radial sign removes it; merely
   subtracting the radial translation does not.

   The appropriate adaptation is to negate both report radial endpoints before
   calling the geometry helper. This is an input-convention adaptation, not
   evidence that the generic helper's internally consistent R basis is wrong.

3. **Missing exit transverse bumps behave as zero in the legacy result.**

   `bumps_report.read_bumps_report` currently copies entrance displacements to
   an unreported exit, creating a pure translation. The legacy sockets instead
   agree with zero transverse bump at that exit and the resulting crab.

   For XSCI.X0220130, the sole socket has `Origine=Centre`, S=0. Its entrance
   vertical bump is +2 mm and its exit is absent. Legacy GEODE moves the centre
   approximately +1 mm; Xsuite moves it +2 mm. Treating the missing exit T as
   zero reduces the discrepancy from 1.0005 mm to 1.27 micrometres after the
   starting-angle correction. XDWC.X0220500 similarly drops from 3.1313 mm to
   1.78 micrometres when both the radial sign and missing-exit treatment change.

   The `.E` suffix alone is insufficient to identify a socket's physical
   location: its `Origine` and S columns matter. An entrance-origin socket can
   receive the full entrance displacement while a centre-origin socket gets
   approximately half. The missing-exit result is inferred from these data;
   GEODE implementation code was not available.

4. **The initial bearing differs slightly.**

   The legacy CSV starts at 7.4198240 gon, whereas `001b` specifies 7.4198200 gon.
   The difference is 6.283e-8 rad. Applying that rigid global rotation reduces
   the largest nominal beam residual from 48.862 to 3.360 micrometres. The
   remaining few micrometres are not resolved by the supplied precision and
   this simple baseline model.

5. **Roll on a bend is applied in a different order.**

   `misalignment_from_rst_offsets` includes the requested roll when solving
   dtheta and dphi, so the specified transverse endpoints stay fixed even when
   a bend is rolled. An alternative model that first infers the crab with
   zero additional roll, then adds the roll about the entrance tangent, closely
   reproduces the legacy sockets. These are distinct operations on a bend:
   the entrance tangent is not parallel to its chord.

   After correcting radial sign, missing exits and initial bearing,
   MBHHE.X0220692.S and MBHHE.X0220699.S retain 125.217 and 124.386 micrometre
   discrepancies. The alternative roll-order model reduces them to 2.701 and
   3.013 micrometres. It also reduces MBNH.X0220399.S from 16.166 to 2.134
   micrometres. This is strong numerical evidence for the operation-order
   explanation, rather than a direct inspection of GEODE's implementation.

The diagnostic maps each new GEODE socket into its Xsuite element's local
frame, then transports that same socket with the alternative element pose.
This keeps the supplied socket geometry and tests the alignment transformations;
it does not independently reimplement GEODE's geodetic or socket calculations.
The report's saved misalignment table supplies design tilts, including those
of drift-modelled instruments.

Excluding the six MCXCA/MDSV elements with intentional corrector roll changes
leaves 205 socket points. After the sign, missing-exit, bearing and roll-order
adaptations, **201/205 agree within 5 micrometres**, with median residual
1.618 micrometres. The four remaining points are the two sockets each of
MBNH.X0220377 (up to 53.836 micrometres) and MBNV.X0220339 (up to 17.990
micrometres). These involve transverse crabs of bends. For MBNH.X0220377,
the reported roll difference is about 144 microradians; multiplied by its
0.36 m socket offset this accounts for about 52 micrometres. A complete common
crab/roll convention still needs to be established before claiming exact
equivalence at the micrometre level.

The large longitudinal exit adjustments reported by `001a` are not the main
problem: for example, QNL.X0220177 discards about 6.696 mm of requested exit S
to preserve rigidity, yet its two sockets agree with legacy GEODE within
2.03 micrometres after the bearing correction.

Reproduce the analysis from this directory with:

```sh
XSUITE_ALLOW_KERNEL_COMPILATION=1 /Users/giadarol/miniforge3/envs/py313/bin/python 002_diagnose_geode.py
```

The script writes `geode_diagnostic_beam.csv` and `geode_diagnostic_socket.csv`.
All residual columns ending in `_mm` are millimetres. The original input
files are read only. The compilation flag permits the small attribute-reading
kernel to compile because this environment's cached kernels have older package
versions; particle tracking kernels are not needed for the analysis.
