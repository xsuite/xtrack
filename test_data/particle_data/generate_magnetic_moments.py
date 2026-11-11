# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""Generate species spin defaults offline, using only the standard library."""
import argparse
import ast
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import runpy


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TARGET = ROOT / 'xtrack/particles/magnetic_moments.py'


def mass_entries():
    masses = runpy.run_path(ROOT / 'xtrack/particles/masses.py')
    tree = ast.parse((ROOT / 'xtrack/particles/pdg.py').read_text())
    elements = next(ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'elements'
                            for t in node.targets))
    atomic_numbers = {symbol: z for z, symbol in elements.items()}
    special = dict(ELECTRON=11, MUON=13, PROTON=2212, NEUTRON=2112,
                   PHOTON=22, PION=211, PION0=111, KAON=321, KAON0=311,
                   DEUTERON=1000010020, TRITON=1000010030)
    result = {}
    for name, mass in masses.items():
        if not name.endswith('_MASS_EV') or name == 'U_MASS_EV':
            continue
        species = name[:-8]
        if species in special:
            pid = special[species]
        else:
            symbol, a = re.fullmatch(r'([A-Z][a-z]?)(\d+)', species).groups()
            pid = 1000000000 + atomic_numbers[symbol] * 10000 + int(a) * 10
        if pid in result:
            raise ValueError(f'Duplicate mass entry: {name}')
        result[pid] = (species, mass)
    return result


def definite_spin(raw):
    # Parenthesized parity does not make I uncertain; parenthesized I does.
    match = re.fullmatch(r'(\d+(?:/\d+)?)(?:[+-]|\([+-]\))\*?', raw)
    return float(Fraction(match[1])) if match else None


def read_sources():
    metadata = json.loads((HERE / 'sources.json').read_text())
    required = {'ame2020_selected.txt', 'codata2022_selected.txt',
                'nist_binding_energies.txt', 'nubase2020_selected.txt',
                'iaea_recommended_selected.txt'}
    if set(metadata) != required:
        raise ValueError('Source metadata must cover all five input excerpts')
    for name, source in metadata.items():
        digest = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        if digest != source['excerpt_sha256']:
            raise ValueError(f'Source checksum mismatch: {name}')

    constants = {}
    quantities = set()
    for line in (HERE / 'codata2022_selected.txt').read_text().splitlines()[11:]:
        quantity = line[:60].strip()
        if quantity in quantities:
            raise ValueError(f'Duplicate CODATA quantity: {quantity}')
        quantities.add(quantity)
        is_mass = quantity.endswith(' energy equivalent in MeV')
        expected_unit = 'MeV' if is_mass else ''
        if line[110:].strip() != expected_unit:
            raise ValueError(f'Unexpected CODATA unit for {quantity}')
        if not is_mass:
            constants[quantity] = (
                Decimal(line[60:85].replace(' ', '')),
                Decimal(line[85:110].replace(' ', '')))
    spins = {}
    for line in (HERE / 'nubase2020_selected.txt').read_text().splitlines():
        if not line.startswith('#'):
            if line[7] != '0':
                raise ValueError('Only nuclear ground states are supported')
            pid = 1000000000 + int(line[4:7]) * 10000 + int(line[:3]) * 10
            if pid in spins:
                raise ValueError(f'Duplicate NUBASE ground state: {pid}')
            spins[pid] = line[88:102].strip()
    moments = {}
    for line in (HERE / 'iaea_recommended_selected.txt').read_text().splitlines():
        if line.startswith('#'):
            continue
        pid = 1000000000 + int(line[:4]) * 10000 + int(line[4:9]) * 10
        if pid in moments:
            raise ValueError(f'Ambiguous recommended ground state: {pid}')
        moments[pid] = {'spin': line[14:34].strip(),
                        'magnetic dipole [nm]': line[34:64].strip()}
    return constants, spins, moments


def generate():
    constants, spins, moments = read_sources()
    masses = mass_entries()
    if set(spins) != {pid for pid in masses if pid > 1000000000}:
        raise ValueError('NUBASE excerpt must cover every packaged ion mass')
    if not set(moments) <= set(spins):
        raise ValueError('Moment excerpt contains an unknown ground state')
    codata = {
        11: ('electron mag. mom. anomaly', 0.5),
        13: ('muon mag. mom. anomaly', 0.5),
        2212: ('proton mag. mom. to nuclear magneton ratio', 0.5),
        2112: ('neutron mag. mom. to nuclear magneton ratio', 0.5),
        1000010020: ('deuteron mag. mom. to nuclear magneton ratio', 1.),
        1000010030: ('triton mag. mom. to nuclear magneton ratio', 0.5),
        1000020030: ('helion mag. mom. to nuclear magneton ratio', 0.5),
    }
    if set(constants) != {quantity for quantity, spin in codata.values()}:
        raise ValueError('CODATA excerpt must contain all seven quantities')
    records = {}
    notes = {}
    for pid, (name, mass) in masses.items():
        spin, moment, anomaly = None, None, None
        note = 'PDG'
        if pid in codata:
            quantity, spin = codata[pid]
            value, uncertainty = constants[quantity]
            label = 'a' if quantity.endswith('anomaly') else 'mu/mu_N'
            note = f'CODATA 2022; sigma({label})={uncertainty:g}'
            if quantity.endswith('anomaly'):
                anomaly = float(value)
                # Positive PDG IDs for charged leptons have negative charge.
                moment = float(-(1 + value) * Decimal(str(masses[2212][1]))
                               / Decimal(str(mass)))
            else:
                moment = float(value)
        elif pid in [111, 211, 311, 321]:
            spin, moment = 0., 0.
        elif pid == 22:
            note = 'Photon helicity is outside the massive-particle BMT model'
        elif pid in moments:
            row = moments[pid]
            spin = definite_spin(row['spin'])
            nubase_spin = definite_spin(spins[pid])
            if nubase_spin is not None:
                if spin is not None and spin != nubase_spin:
                    raise ValueError(f'Conflicting spin assignments: {pid}')
                spin = nubase_spin
            raw = row['magnetic dipole [nm]'].strip()
            note = f'IAEA; I={row["spin"]}; mu/mu_N={raw}'
            match = re.fullmatch(r'([+-]\d+(?:\.\d+)?)\(\d+\)', raw)
            if match:
                moment = float(match[1])
            elif raw[:1].isdigit():
                note += '; sign undetermined'
            else:
                note += '; no usable signed moment'
            if spin is None:
                note += f'; uncertain I={spins[pid]} (NUBASE2020)'
            elif definite_spin(row['spin']) is None:
                note += f'; I={spins[pid]} (NUBASE2020)'
        elif pid in spins:
            note = f'NUBASE2020; I={spins[pid]}'
            spin = definite_spin(spins[pid])
            if spin == 0:
                moment = 0.
        charge = ((pid - 1000000000) // 10000 if pid > 1000000000
                  else {11: -1, 13: -1, 2112: 0, 22: 0,
                        111: 0, 311: 0}.get(pid, 1))
        if (anomaly is None and spin is not None and spin > 0
                and moment is not None and mass > 0 and charge != 0):
            # mu = g * (Z e / 2m) * I hbar, mu_N = e hbar / 2m_p.
            anomaly = float(Decimal(str(moment)) * Decimal(str(mass))
                / (2 * Decimal(str(spin)) * charge
                   * Decimal(str(masses[2212][1]))) - 1)
        records[pid] = (spin, moment, anomaly)
        notes[pid] = note
    header = '''# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""Ground-state magnetic moments and charge-normalized BMT anomalies.

Generated offline by test_data/particle_data/generate_magnetic_moments.py; source
excerpts, calculations and conventions are in test_data/particle_data/README.md.
Regenerate from the repository root:
    python test_data/particle_data/generate_magnetic_moments.py

Sources:
  CODATA 2022: https://physics.nist.gov/cuu/Constants/Table/allascii.txt
  NUBASE2020: https://doi.org/10.1088/1674-1137/abddae
  IAEA recommended moments: https://doi.org/10.61092/iaea.qs1q-27sa

For fully stripped nuclei, a = mu*m/(2*I*Z*m_p)-1 with mu in nuclear magnetons.
None means unavailable or inapplicable, not zero. Unsigned moments do not
provide anomalies. Input qualifiers and uncertainties are retained in the
source excerpts. Derived decimal places are computational guard digits,
not measurement accuracy; correlated uncertainties are not propagated here.
"""
from typing import NamedTuple


# Ground-state spin data, generated offline. Sources and regeneration:
# test_data/particle_data/README.md. None means unavailable or inapplicable,
# not a zero magnetic moment. Anomalies use charge-normalized g factors.
class SpinProperties(NamedTuple):
    """Spin I [hbar], signed moment [mu_N], and charge-normalized BMT anomaly.

    Known spin zero and the known neutron moment are retained as data, but
    have no charge-normalized BMT anomaly. Photon helicity is not represented
    by this massive-particle spin record.
    """
    spin: float | None
    magnetic_moment_mu_N: float | None
    anomalous_magnetic_moment: float | None


# Same species order and element groups as masses.py.
# Fields: spin I [hbar], magnetic moment [mu_N], BMT anomaly (g-2)/2.
'''
    body = []
    previous_group = None
    for pid, record in records.items():
        name = masses[pid][0]
        if pid in codata or name == 'He4':
            group = 'Direct CODATA values and light nuclei'
        elif pid < 1000000000:
            group = 'Photon and mesons'
        else:
            group = re.match(r'[A-Z][a-z]?', name)[0]
        if group != previous_group:
            body.append(f'\n# {group}\n')
            previous_group = group
        label = name + '_SPIN_PROPERTIES'
        body.append(f'{label:<28} = SpinProperties{record!r}'
                    f'  # {notes[pid]}\n')
    return header + ''.join(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = generate()
    if args.check:
        if not TARGET.exists() or TARGET.read_text() != result:
            raise SystemExit('magnetic_moments.py needs regeneration')
    else:
        TARGET.write_text(result)


if __name__ == '__main__':
    main()
