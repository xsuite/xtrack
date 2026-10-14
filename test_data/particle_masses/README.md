# Sources and calculation of particle masses

`xtrack/particles/masses.py` stores rest energies `m*c**2` in eV. For ions,
the mass is that of the **fully stripped nucleus in its nuclear ground state**.
The atomic mass unit is still defined using a **neutral** carbon-12 atom:
`U_MASS_EV` is a conversion constant, not an ion mass.

## Reproduce the table

From the Xtrack repository root, using Python 3 (standard library only):

```sh
python test_data/particle_masses/generate_masses.py
python test_data/particle_masses/generate_masses.py --check
```

Generation is offline and deterministic. It writes only `masses.py`, retaining
the existing public constant names and isotope coverage. The script verifies
the SHA-256 hashes of the saved source excerpts before calculating anything.
It uses decimal arithmetic for the central values and rounds converted nuclear
masses to 0.1 eV. Extra decimal places are guard digits, not stated accuracy.

## Sources

All excerpts were retrieved on **2026-09-28**. `sources.json` records the exact
download URLs, download dates, hashes of the full downloaded responses, and
hashes of the committed excerpts. The full responses are not needed to
regenerate the table. The excerpts are:

* `ame2020_selected.txt`: original header and 147 selected, unmodified records
  from the AMDC/IAEA distribution of
  [AME2020 mass_1.mas20.txt](https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt).
  M. Wang et al., *The AME 2020 atomic mass evaluation (II). Tables, graphs and
  references*, Chinese Physics C **45**, 030003 (2021),
  [doi:10.1088/1674-1137/abddaf](https://doi.org/10.1088/1674-1137/abddaf).
  These are **neutral-atom** masses. The ASCII precision is used as distributed;
  values are not copied from the old Python table's rounded comments.
* `codata2022_selected.txt`: original header and nine unmodified rows from
  [NIST's 2022 CODATA constants](https://physics.nist.gov/cuu/Constants/Table/allascii.txt).
  The conversion constant, electron, muon, proton, neutron, deuteron, triton,
  helion (He3 nucleus), and alpha particle (He4 nucleus) all use this edition.
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

The four meson masses and their uncertainties are retained from the previous
table, attributed there to [PDG 2024](https://pdg.lbl.gov/2024/listings/particle_properties.html).
They are embedded explicitly in the generator and are not re-evaluated by this
ion-mass correction. Photon mass remains zero.

## Calculation

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

## Approximations and uncertainties

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

## Changes from the previous table and future updates

All 147 C/O/Ne/Ar/Fe/Xe/Au/Pb entries now subtract the electron rest energies
and add the total electronic binding energy. Regeneration from AME also fixes
the old Xe113 conversion discrepancy. Electron, proton, neutron and atomic
mass unit constants move from older values to CODATA 2022; the direct muon,
deuteron, triton, He3 and He4 values are numerically unchanged.

For future updates, replace the relevant source excerpt from its authoritative
distribution, record the edition, retrieval date and both hashes in
`sources.json`, and review any format changes in the parser. Keep the original
fixed-width fields and AME extrapolation flags. Regenerate, inspect the diff,
and run the mass and PDG tests. Do not silently fetch live values during import
or regeneration, or edit the generated constants by hand.
