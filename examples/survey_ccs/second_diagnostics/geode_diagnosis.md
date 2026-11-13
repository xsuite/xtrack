The new H4TEST3 GEODE reports dated 17-SEP-2026 confirm that the radial-sign
fix worked. The large radial translation and crab discrepancies have disappeared.
The remaining differences follow the same missing-exit and bend-rotation
patterns identified in the first exercise.

Subsequent implementation: `bumps_report.py` now defaults missing exit R/T
bumps to zero, and the working-directory misalignment CSV, JSON and TFS have
been regenerated. The three inputs used for the H4TEST3 import are preserved
in this directory; `002_diagnose_geode.py` uses those copies to keep the
comparison below tied to the actual imported survey. The current GEODE reports
have not yet been refreshed with the missing-exit fix.

The regenerated TFS changes only the 12 single-point instruments. Both
generation scripts pass their checks, and the surveyed exits have zero
transverse displacement while their centres have half the entrance R/T
displacement. `single_point_fix_prediction.csv` records the expected socket
positions relative to legacy GEODE, inferred by transporting the measured
H4TEST3 sockets with the corrected element poses. The maximum predicted
residual is 45.04 micrometres at the current bearing, or 3.08 micrometres when
also matching the legacy starting bearing. These are predictions pending
another GEODE import; only the missing-exit convention has been changed.

Inputs are `geode_beam_points_xsuite_no_bumps.csv` and
`geode_socket_points_xsuite_no_bumps.csv` from the working directory, and the
preserved `bump_misalignments.csv`, `h4_misaligned.json`, and
`h4_survey_comp_vbend_output.tfs` in this directory. The `_v2.tfs` file
has an identical data table. The legacy reference remains the two
`*_mad_and_bumps.csv` files (including `geode_beam_points_from_mad_and_bumps.csv`).
The preceding results in `first_diagnostics/` are preserved.

**Import and input consistency checks**

- All 97 rows of the regenerated misalignment table agree with the corrected
  bump reader and the geometry helper, with maximum numerical difference
  2.22e-16 across the six misalignment parameters.
- All 384 new beam points match the current TFS and the survey of the current
  JSON within 0.820 micrometres in 3D, consistent with the CSV coordinate
  rounding to 1 micrometre.
- All 217 sockets match by point name, with no missing or additional points.
  The beam comparison has the same two legacy-only line boundary points,
  `H4TEST.-.E` and `.S`, as before.
- The legacy beam points remain nominal. Their differences from the new beam
  points still include the intended element displacements and should not be
  treated as socket alignment errors.

**Actual socket differences before further modelling**

As agreed, statistics exclude the 12 sockets of the six MCXCA/MDSV elements
with intentional roll-convention changes. The remaining sample has 205 sockets.
Its largest discrepancy has fallen from 200.011 mm to 1.439 mm; its median
is 0.024372 mm.

| Element | Previous socket difference (mm) | Current socket difference (mm) |
| --- | ---: | ---: |
| XCSV.X0220179 | 200.011 | 0.01118–0.01122 |
| XFFV.X0220257 | 199.982 | 0.01706–0.01803 |
| MBNV.X0220077 | 200.001–200.003 | 0.00510–0.00592 |
| LSX.X0220195 | 100.013 | 0.01304–0.01319 |

These are directly measured GEODE-to-GEODE differences, without another radial
sign correction. The diagnostic no longer subtracts the old 2R prediction.

The remaining large differences are:

- Ten centre-origin XSCI/XDWC sockets: 1.0005–1.4391 mm. The reader still copies
  an unreported exit's transverse displacement from the entrance. Legacy
  GEODE instead matches zero exit transverse bump, producing approximately
  half the entrance displacement at the centre. For example, XSCI.X0220130
  has a +2 mm entrance T request and a centre socket: the two workflows differ
  by about 1 mm. The two entrance-origin single-point sockets do not have
  this millimetre discrepancy.
- MBHHE.X0220692.S and MBHHE.X0220699.S: 0.162631 and 0.161453 mm, respectively.
  These are the largest differences outside the single-point instruments.
  The bend roll-order model from the first analysis still explains them.
- A smooth position offset from the unchanged initial-bearing mismatch:
  7.419824 gon in the legacy CSV versus 7.419820 gon in `001b`. Rotating the
  nominal survey to the legacy bearing reduces its maximum beam residual
  from 48.862 to 3.360 micrometres.

**Diagnostic models, not additional applied fixes**

The script transports each measured new socket between alternative Xsuite
element poses. It keeps socket local coordinates and tests the same alternatives
as before: zero transverse bumps at missing exits, the legacy initial bearing,
and crab inferred before adding roll about the entrance tangent. It does not
modify the line-generation scripts, input files, or GEODE.

| Comparison stage | Maximum residual (mm) | Median residual (mm) |
| --- | ---: | ---: |
| Actual new versus legacy sockets | 1.439098 | 0.024372 |
| Model zero missing-exit transverse bumps | 0.162631 | 0.024083 |
| Also match initial bearing | 0.125929 | 0.001574 |
| Also model entrance-tangent roll ordering | 0.053836 | 0.001574 |

All 12 single-point instruments reach residuals below 3.08 micrometres after
the missing-exit and bearing adjustments. The roll-order model reduces
MBHHE.X0220692.S and MBHHE.X0220699.S to 3.40 and 3.01 micrometres.

With these models, 201 of 205 comparable sockets agree within 5 micrometres.
The remaining four are unchanged from the first exercise:

| Point | Residual (micrometres) |
| --- | ---: |
| MBNH.X0220377.E | 52.896 |
| MBNH.X0220377.S | 53.836 |
| MBNV.X0220339.E | 17.990 |
| MBNV.X0220339.S | 17.087 |

These are transverse crabs of bends; the exact crab/roll convention is still
not fully reconciled. Thus the new import validates the radial-sign fix and
the earlier diagnosis, but does not establish full micrometre equivalence.
The next largest actionable difference is the reader's missing-exit treatment.

Reproduce from the `survey_ccs` directory:

```sh
XSUITE_ALLOW_KERNEL_COMPILATION=1 /Users/giadarol/miniforge3/envs/py313/bin/python 002_diagnose_geode.py
```

Results are written to this directory:

- `geode_diagnostic_beam.csv`: per-beam-point import and nominal-survey checks.
- `geode_diagnostic_socket.csv`: measured residuals and the alternative models.
- `geode_import_comparison.csv`: previous versus current measured socket errors.

All CSV residual columns ending in `_mm` use millimetres.
