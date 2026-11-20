<!--
This file is part of the Xtrack Package.
Copyright (c) CERN, 2026.
-->

# Demonstration ring

One geometry with separately loaded operating strengths:

| File | Purpose |
| --- | --- |
| [geometry.py](geometry.py) | Define magnetic families, live circuits, named devices and reusable sections |
| [strengths.py](strengths.py) | Select the nominal quadrupole strengths, reference momentum and RF setting |
| [lattice.py](lattice.py) | Load both files and assemble the complete ring |

Load an independent nominal machine with `env = xt.load(path_to_lattice)` and
select `env['demo']`. Each load owns independent elements and variables, with
the stored strengths applied directly. Loading does not run a matching solve.

The model has six 160 m arcs and six 90 m straights. One straight contains the
low-beta IP; the others contain symmetric utility optics. Reference particles
are electrons with `p0c=1e9 eV`. The 1.5 km ring has tunes approximately
`8.75483249 / 10.01745242`, IP beta functions `1 m / 0.01 m`, and approximately
zero IP alpha and dispersion. Arc beta functions span about 12–65 m; insertion
peaks reach about 535 m. The six-sector teaching idea draws on
[Szymon Łopaciuk's aperture tutorial](https://github.com/xsuite/aperture-model-tutorial)
and the [Xsuite lattice-design tutorial](https://github.com/xsuite/tutorial_lattice_design).
The dimensions and optics here were constructed independently.

BPMs are named passive `Device` planes. Combined horizontal/vertical correctors
have independent positive-kick controls, with `knl[0]=-h` and `ksl[0]=v`.
Normal trims and gradient errors are separate additive variables. Skew, sextupole,
octupole, error and trim settings start at zero. Quadrupole and skew gradients
are in m^-2, integrated sextupole strengths in m^-2, and integrated octupole
strengths in m^-3. Dipoles carry their design field through their curvature.

Sextupole partners two cells apart have transverse betatron `-I` transport at
the nominal momentum. Their dispersion differs. No complete local chromatic
cancellation or crab-waist transformation is claimed. Bare chromaticities are
large and the vertical tune is near an integer; momentum acceptance and new
nonlinear launch ranges require qualification.

RF voltage starts at zero; radiation is off. Setting `voltage_rf=1e5` powers
the thin cavity at harmonic 200, about 39.9723 MHz, with phase pi in the public
`abs(q0)` kick convention. This gives a stable small-amplitude synchronous mode
with `qs` about 0.007925. Bucket acceptance, acceleration, radiation compensation
and mixed-species synchronous configurations are separate studies.

The ring provides a common input for general ring workflows. It does not replace
controlled one-element or analytic-map demonstrations, open transport lines,
accelerating linacs, or machine-specific physics and regression fixtures.
