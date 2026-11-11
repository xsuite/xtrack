# Particle masses and magnetic moments

This directory contains the saved source data and generation scripts for
`xtrack/particles/masses.py` and `xtrack/particles/magnetic_moments.py`.
For ions, both use fully stripped nuclei in their **nuclear ground state**.
The generated modules contain Python constants. Loading Xtrack does not read
these source files or access the network.

## Reproduce the tables

From the repository root, using only Python's standard library:

```sh
python test_data/particle_data/generate_masses.py
python test_data/particle_data/generate_magnetic_moments.py
python test_data/particle_data/generate_masses.py --check
python test_data/particle_data/generate_magnetic_moments.py --check
```

Regenerate masses first: the moment generator uses these masses to convert
nuclear moments to BMT anomalies. Each script writes only its corresponding
Python module. Both read `sources.json` and check the SHA-256 hashes of the
saved files before calculating anything. Missing source records, checksum
mismatches, duplicate entries and invalid physical assumptions raise errors,
including with optimized Python.

The shared CODATA excerpt contains nine mass/conversion constants and seven
anomaly/moment constants. Both sets came from the same CODATA 2022 table.
Original values and standard uncertainties are retained; trailing padding has
been removed. The mass generator converts MeV to eV. Moment ratios and
anomalies are dimensionless. Nuclear spin data cover every ion in the mass
table, so adding a species requires checking both generated files.

## Sources

`sources.json` records download URLs, retrieval dates, full-response hashes
and retained-excerpt hashes. The full downloads are not needed to
regenerate the tables. AME and NIST binding sources were retrieved on 2026-09-28;
nuclear spin/moment sources on 2026-11-10. CODATA was retrieved on 2026-09-28
for masses and again on 2026-11-11 for moments; `sources.json` records the latter
date for the combined excerpt.

* `ame2020_selected.txt`: original header and 147 selected, unmodified records
  from the AMDC/IAEA distribution of
  [AME2020 mass_1.mas20.txt](https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt).
  M. Wang et al., *The AME 2020 atomic mass evaluation (II). Tables, graphs and
  references*, Chinese Physics C **45**, 030003 (2021),
  [doi:10.1088/1674-1137/abddaf](https://doi.org/10.1088/1674-1137/abddaf).
  These are **neutral-atom** masses. The ASCII precision is used as distributed;
  values are not copied from the old Python table's rounded comments.
* `codata2022_selected.txt`: original header and sixteen selected rows from
  [NIST's 2022 CODATA constants](https://physics.nist.gov/cuu/Constants/Table/allascii.txt).
  Mass/conversion constants, electron/muon anomalies and the proton, neutron,
  deuteron, triton and bare helion moments all use this edition. The helion
  moment is unshielded, not the shielded atomic value.
  See P. J. Mohr et al., Rev. Mod. Phys. **97**, 025002 (2025),
  [doi:10.1103/RevModPhys.97.025002](https://doi.org/10.1103/RevModPhys.97.025002).
* `nist_binding_energies.txt`: the ASCII table and notes extracted from the
  HTML response of **NIST ASD 5.12 (2024)**, with HTML tags removed and entities
  decoded. Kramida, Ralchenko, Reader and NIST ASD Team,
  [doi:10.18434/T4W30F](https://doi.org/10.18434/T4W30F).
  The query requests neutral C, O, Ne, Ar, Fe, Xe, Au and Pb, **total binding
  energy**, eV, uncertainties and references. The full query is in
  `sources.json`. All returned values carry NIST's square-bracket designation
  for a sum involving experimental or semiempirical ionization energies.

* `nubase2020_selected.txt` retains the format header and the 151
  ground-state records matching the ion masses. F. G. Kondev et al.,
  *The NUBASE2020 evaluation of nuclear physics properties*, Chinese Physics C
  **45**, 030001 (2021),
  [doi:10.1088/1674-1137/abddae](https://doi.org/10.1088/1674-1137/abddae).
  [AMDC/IAEA ASCII distribution](https://www-nds.iaea.org/amdc/ame2020/nubase_4.mas20.txt).
* `iaea_recommended_selected.txt` retains the physical values, qualifiers,
  original references, publication citations and evaluation identifiers of
  the 66 matching ground-state records in the IAEA Nuclear Electromagnetic Moments
  [recommended magnetic-moment CSV](https://www-nds.iaea.org/nuclearmoments/magn_mom_recomm.csv).
  They are formatted as a fixed-column text excerpt, matching the mass-data
  layout. Numerical fields, uncertainties and moment annotations are preserved.
  The distributed entries cite N. J. Stone, INDC(NDS)-0915. The report and its
  sign policy are available via
  [doi:10.61092/iaea.qs1q-27sa](https://doi.org/10.61092/iaea.qs1q-27sa).
  Dataset compilation/interface: T. J. Mertzimekis, IAEA Nuclear Data Section.

The four meson masses and their uncertainties are retained from the previous
table, attributed there to [PDG 2024](https://pdg.lbl.gov/2024/listings/particle_properties.html).
They are embedded explicitly in the generator and are not re-evaluated by this
ion-mass correction. Photon mass remains zero.

## Mass calculation

For a neutral atom of atomic number `Z` and relative atomic mass `A_r`, AME2020
equation (1) gives, in eV:

```text
E_nucleus = A_r * E_u - Z * E_e + B_e
E_u = 931494103.72 eV       (CODATA 2022)
E_e =    510998.95069 eV    (CODATA 2022)
```

`B_e` is the positive energy required to remove **all** electrons. Its sign is
positive because the neutral atom's mass already includes the negative
electronic binding energy. Nuclear binding is already contained in the AME
atomic mass and must not be applied again. The AME `BINDING ENERGY/A` column is
nuclear binding energy per nucleon; it is **not** the electronic `B_e` used here.

NIST supplies the sum of successive ionization energies and its uncertainty:

| Element | Z | B_e (eV) | NIST uncertainty (eV) |
|---|---:|---:|---:|
| C | 6 | 1030.1085 | 0.0004 |
| O | 8 | 2043.8429 | 0.0022 |
| Ne | 10 | 3511.696 | 0.019 |
| Ar | 18 | 14400.8 | 2.4 |
| Fe | 26 | 34619 | 11 |
| Xe | 54 | 202400 | 140 |
| Au | 79 | 517100 | 1100 |
| Pb | 82 | 568200 | 1500 |

For example:

```text
C12: 12 * 931494103.72 - 6 * 510998.95069 + 1030.1085
     = 11174864281.04436 eV -> C12_MASS_EV = 11174864281.0

Pb208: 207.976652005 * 931494103.72 - 82 * 510998.95069 + 568200
       -> Pb208_MASS_EV = 193687691340.1
```

The deuteron, triton, He3 and He4 constants are taken directly from CODATA as
nuclear masses, so they receive no further electron correction. The proton
also uses a direct CODATA nuclear mass.

### Mass approximations and uncertainties

1. **Electronic isotope shifts are neglected.** The same NIST elemental
   binding energy is used for every isotope of that element. Changes from
   nuclear recoil and finite nuclear size are not separately calculated.
   Consequently these values are not isotope-specific precision determinations
   at the level of the smallest quoted NIST binding-energy uncertainties.
2. **No empirical binding-energy formula is used.** The approximate
   `14.4381*Z**2.39 + 1.55468e-6*Z**5.35` formula quoted in AME2020 equation (2)
   was used only for the preliminary audit. The corrected table uses the NIST
   sums above, including NIST's evaluated/theoretical inputs and uncertainties.
3. **Uncertainty propagation is approximate.** Each isotope comment reports
   the following quadrature, rounded to two significant figures:

   ```text
   sigma_E = sqrt((E_u * sigma_A_r)**2 + (A_r * sigma_E_u)**2
                  + (Z * sigma_E_e)**2 + sigma_B_e**2)
   sigma_E_u = 0.29 eV; sigma_E_e = 0.00016 eV
   ```

   Input uncertainties are retained in the source files. Cross-correlations
   between atomic masses, CODATA constants and binding energies are ignored.
   Neither the unquantified isotope-shift error nor the at-most-0.05 eV output
   rounding error is included. These comments are **not certified total
   uncertainties**, and should not be used to propagate precision mass
   differences without covariance information.
4. **NIST's binding-energy uncertainty has its own prescription.** Uncertainties
   of ionization energies from the same literature source are added linearly;
   contributions from different sources are combined in quadrature. Missing
   ionization-energy uncertainties are estimated as ten units of the last
   digit. We use NIST's result directly. See the
   [ASD help](https://physics.nist.gov/PhysRefData/ASD/Html/iehelp.html).
5. **AME extrapolations remain extrapolations.** A `#` in the AME numerical
   field is parsed as a decimal point and preserved as a flag in `masses.py`.
   Such masses are not wholly based on experimental data.
6. **Scope:** the constants describe fully stripped ground-state nuclei,
   not partially stripped ions or nuclear isomers. The separate PDG lookup's
   fallback approximation `A * U_MASS_EV` for unlisted isotopes is unchanged.

## Magnetic moments and BMT conventions

CODATA takes precedence for light nuclei. Otherwise the recommended IAEA
moment is used with a definite ground-state spin. NUBASE supplies a definite
spin where available, checked against a definite IAEA spin. This resolves the
older tentative spin label for Au-183. Spin-zero mesons follow the PDG
spin assignments. Photon helicity is outside the massive-particle vector-spin
record; its spin, moment and anomaly fields are `None`.

For a nucleus of spin `I`, charge `Z e`, nuclear mass `m`, and signed magnetic
moment `mu` in nuclear magnetons (`mu_N = e hbar / (2 m_p)`):

```text
g_BMT = mu * m / (I * Z * m_p)
a_BMT = g_BMT / 2 - 1
```

Masses come from the maintained Xtrack mass table. A nuclear g factor quoted
in nuclear magnetons cannot be inserted directly as `g_BMT`. Electron/muon
anomalies are used directly; their moments in nuclear magnetons are derived
from those anomalies and the reference mass ratios.

Particle and antiparticle have the same charge-normalized anomaly. The
accessor reverses their signed magnetic moments. No antiparticle isotope
constructor support is added here.

The IAEA report states that signs are given only when established by the
measurement. An unsigned value is **not** interpreted as positive. Tentative,
ambiguous or systematic spin assignments do not provide automatic anomalies.
Original qualifiers and uncertainties remain in the retained excerpts;
generated decimal places are computational precision, not measurement accuracy.

## Reading the spin data

`magnetic_moments.py` stores named, three-field `SpinProperties` records in the
same species order as `masses.py`. The fields are spin quantum number `I` in
units of hbar, signed magnetic moment in nuclear magnetons, and the
charge-normalized BMT anomaly `(g-2)/2`. Each record supports both tuple
unpacking and access by field name.

`xtrack.particles.pdg.get_spin_properties_from_pdg_id(name_or_id)` returns the
record for a species. `None` means unavailable or inapplicable, never a
measured zero. For example, spin-zero nuclei have `(0.0, 0.0, None)`, while
the neutron has its known spin and moment but no charge-normalized anomaly.
An anomaly requires definite nonzero spin, a known signed moment, nonzero
charge and positive rest mass. Comments and source excerpts explain missing
values, including uncertain spin assignments and undetermined moment signs.
Unknown species and isomers return three `None` values; ground-state data are
never substituted for an isomer.

For compatibility, the existing numeric `Particles.anomalous_magnetic_moment`
array stores zero when no physical default is available. Zero means `g = 2`,
not an unknown or zero magnetic moment. The constructor leaves the spin vector
zero, so no vector precession signal is produced until spin is supplied.
Explicit anomalies, including intentional zero, and serialized values are
preserved. A per-particle PDG label alone does not change reference properties.
Automatic defaults apply only to the nominal species mass and charge; custom
reference definitions retain zero unless an anomaly is supplied explicitly.
Partially stripped ions are not inferred from fully stripped nuclear data.

CPU spin calculations warn about zero anomaly, corresponding to `g = 2`,
except for known spin-zero species. GPU diagnostics can check a CPU reference
when available; they do not check particles stored on the device.

These data support vector BMT precession where applicable. They do not
implement higher-spin tensor polarization or extend electron/positron
radiative-polarization formulae to nuclei.

## Updating sources

Replace the relevant excerpt from its authoritative distribution, record the
edition, retrieval date and full/excerpt hashes in `sources.json`, and review
parser changes. Preserve fixed-width numerical fields, input uncertainties,
AME extrapolation flags and spin/moment qualifiers. Regenerate both modules,
inspect the changes, and run the mass, magnetic-moment and PDG tests. Do not
fetch live values during loading or generation, or edit generated values by hand.
