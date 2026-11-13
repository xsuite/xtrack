"""Magnetic-moment catalogue, conventions and offline regeneration."""
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

import numpy as np
import pytest

import xtrack as xt
from xtrack.particles import masses, magnetic_moments, pdg


SOURCE = Path(__file__).resolve().parents[1] / 'test_data/particle_data'


@pytest.fixture
def generator(tmp_path, monkeypatch):
    for path in SOURCE.glob('*.txt'):
        shutil.copyfile(path, tmp_path / path.name)
    shutil.copyfile(SOURCE / 'sources.json', tmp_path / 'sources.json')
    script = SOURCE / 'generate_magnetic_moments.py'
    functions = runpy.run_path(str(script))
    monkeypatch.setitem(functions['generate'].__globals__, 'HERE', tmp_path)
    monkeypatch.setitem(functions['generate'].__globals__, 'TARGET',
                        tmp_path / 'magnetic_moments.py')
    monkeypatch.setattr(sys, 'argv', [str(script)])
    return functions, tmp_path


def update_checksum(root, name):
    path = root / 'sources.json'
    metadata = json.loads(path.read_text())
    metadata[name]['excerpt_sha256'] = hashlib.sha256(
        (root / name).read_bytes()).hexdigest()
    path.write_text(json.dumps(metadata))


def test_regenerate_missing_and_stale_output(generator, monkeypatch):
    functions, root = generator
    output = root / 'magnetic_moments.py'
    functions['main']()
    assert output.read_text() == functions['generate']()
    monkeypatch.setattr(sys, 'argv', ['generate', '--check'])
    functions['main']()
    output.write_text('stale output')
    with pytest.raises(SystemExit, match='needs regeneration'):
        functions['main']()
    monkeypatch.setattr(sys, 'argv', ['generate'])
    functions['main']()
    assert output.read_text() == functions['generate']()


@pytest.mark.parametrize('name', ['nubase2020_selected.txt',
                                 'iaea_recommended_selected.txt'])
def test_duplicate_source_records_rejected(generator, name):
    functions, root = generator
    path = root / name
    lines = path.read_text().splitlines(keepends=True)
    record = next(line for line in lines if not line.startswith('#'))
    path.write_text(''.join(lines) + record)
    update_checksum(root, name)
    with pytest.raises(ValueError, match='Duplicate|Ambiguous'):
        functions['generate']()


def test_conflicting_spin_assignments_rejected(generator):
    functions, root = generator
    path = root / 'iaea_recommended_selected.txt'
    lines = path.read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.startswith('#') and int(line[:4]) == 6 and int(line[4:9]) == 13:
            lines[index] = line[:14] + f'{"3/2-":<20}' + line[34:]
    path.write_text(''.join(lines))
    update_checksum(root, path.name)
    with pytest.raises(ValueError, match='Conflicting spin assignments'):
        functions['generate']()


@pytest.mark.parametrize('raw, expected', [('5/2(-)', 2.5), ('(5/2)-', None),
                                        ('0+', 0.), ('5/2+#', None)])
def test_spin_qualifier_conventions(generator, raw, expected):
    functions, _ = generator
    assert functions['definite_spin'](raw) == expected


@pytest.mark.parametrize('species, spin, moment', [
    ('C13', 0.5, 0.702369), ('O17', 2.5, -1.893543), ('Fe57', 0.5, 0.09020)])
def test_iaea_nuclear_moment_convention(species, spin, moment):
    # Independent selected IAEA inputs check the parsed value and recover
    # the moment from the anomaly, including the I, Z and nuclear-mass factors.
    particle = xt.Particles(species, p0c=1e9)
    data = pdg.get_spin_properties_from_pdg_id(species)
    assert data.spin == spin
    assert data.magnetic_moment_mu_N == moment
    recovered = (1 + particle.anomalous_magnetic_moment[0]) * (
        2 * spin * particle.q0 * xt.PROTON_MASS_EV / particle.mass0)
    np.testing.assert_allclose(recovered, moment, rtol=1e-14, atol=0)



def test_named_records_follow_mass_order():
    expected = [name.removesuffix('_MASS_EV') for name in vars(masses)
                if name.endswith('_MASS_EV') and name != 'U_MASS_EV']
    actual = [name.removesuffix('_SPIN_PROPERTIES')
              for name in vars(magnetic_moments)
              if name.endswith('_SPIN_PROPERTIES')]
    assert actual == expected
    assert all(isinstance(getattr(magnetic_moments, name + '_SPIN_PROPERTIES'),
                          magnetic_moments.SpinProperties)
               for name in actual)
    assert magnetic_moments.SpinProperties._fields == (
        'spin', 'magnetic_moment_mu_N', 'anomalous_magnetic_moment')


@pytest.mark.parametrize('kind, message', [
    ('checksum', 'Source checksum mismatch'),
    ('metadata', 'Source metadata must cover'),
    ('quantity', 'CODATA excerpt must contain all seven quantities'),
    ('unit', 'Unexpected CODATA unit'),
    ('duplicate', 'Duplicate CODATA quantity'),
])
def test_moment_source_validation(generator, kind, message):
    functions, root = generator
    path = root / 'codata2022_selected.txt'
    lines = path.read_text().splitlines(keepends=True)
    index = next(i for i, line in enumerate(lines)
                 if line[:60].strip() == 'electron mag. mom. anomaly')
    if kind == 'metadata':
        metadata = root / 'sources.json'
        records = json.loads(metadata.read_text())
        records.pop('codata2022_selected.txt')
        metadata.write_text(json.dumps(records))
    else:
        if kind == 'quantity':
            del lines[index]
        elif kind == 'unit':
            lines[index] = lines[index].rstrip('\n')[:110].ljust(110) + 'MeV\n'
        elif kind == 'duplicate':
            lines.append(lines[index])
        else:
            lines.append('corruption\n')
        path.write_text(''.join(lines))
        if kind != 'checksum':
            update_checksum(root, path.name)
    with pytest.raises(ValueError, match=message):
        functions['generate']()


def test_catalogue_covers_mass_table_with_applicable_anomalies():
    assert set(pdg._get_spin_properties_table()) == {abs(pid) for pid in pdg._get_mass_table()}
    for pid, mass in pdg._get_mass_table().items():
        data = pdg.get_spin_properties_from_pdg_id(pid)
        charge = pdg.get_properties_from_pdg_id(pid)[0]
        applicable = (data.spin is not None and data.spin > 0
                      and data.magnetic_moment_mu_N is not None
                      and charge != 0 and mass > 0)
        if applicable:
            assert np.isfinite(data.anomalous_magnetic_moment)
        else:
            assert data.anomalous_magnetic_moment is None


@pytest.mark.parametrize('species, spin, moment', [
    ('O16', 0., 0.), ('Pb208', 0., 0.),
    ('C15', 0.5, None),
    ('Au178', None, None),
    ('Fe55', 1.5, None),
    ('neutron', 0.5, -1.91304276),
    ('photon', None, None),
])
def test_known_values_are_distinct_from_missing_or_inapplicable_anomalies(
        species, spin, moment):
    data = pdg.get_spin_properties_from_pdg_id(species)
    assert data.spin == spin
    assert data.magnetic_moment_mu_N == moment
    assert data.anomalous_magnetic_moment is None


def test_charge_conjugation_and_unlisted_ground_states():
    proton = pdg.get_spin_properties_from_pdg_id('proton')
    antiproton = pdg.get_spin_properties_from_pdg_id('anti-proton')
    assert antiproton.spin == proton.spin
    assert antiproton.anomalous_magnetic_moment == proton.anomalous_magnetic_moment
    assert antiproton.magnetic_moment_mu_N == -proton.magnetic_moment_mu_N
    for species in [1000822081, 'C16']:
        data = pdg.get_spin_properties_from_pdg_id(species)
        assert data.spin is None
        assert data.magnetic_moment_mu_N is None
        assert data.anomalous_magnetic_moment is None


def test_moment_table_matches_saved_sources():
    subprocess.run([sys.executable,
                    str(SOURCE / 'generate_magnetic_moments.py'), '--check'],
                   check=True)
