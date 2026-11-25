from pathlib import Path
import runpy
import hashlib
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

from xtrack.particles import masses, pdg


def test_bare_nucleus_mass_convention():
    # Independent reference inputs: AME2020 neutral atomic masses and NIST
    # ASD 5.12 total electron binding energies (eV). Recovering B_e detects
    # both the old neutral-atom convention and an omitted/wrong-sign B_e.
    for name, atom_u, z, binding_ev in [
        ('C12', 12.0, 6, 1030.1085),
        ('O16', 15.99491461926, 8, 2043.8429),
        ('Xe113', 112.933221663, 54, 202400),
        ('Pb208', 207.976652005, 82, 568200),
    ]:
        mass = getattr(masses, name + '_MASS_EV')
        recovered_binding = mass - atom_u * masses.U_MASS_EV + z * masses.ELECTRON_MASS_EV
        np.testing.assert_allclose(recovered_binding, binding_ev, rtol=0, atol=0.051)
        a = round(atom_u)
        particle_id = pdg.get_pdg_id_ion(a, z)
        assert pdg.get_mass_from_pdg_id(particle_id) == mass
        assert pdg.get_mass_from_pdg_id(-particle_id) == mass


def test_mass_table_matches_saved_sources():
    repo_root = Path(__file__).resolve().parents[1]
    generator = runpy.run_path(str(
        repo_root / 'test_data' / 'particle_data' / 'generate_masses.py'))
    constants, binding, atoms = generator['read_sources']()
    assert len(atoms) == 147
    assert len(binding) == 8
    assert len(constants) == 9
    assert Path(masses.__file__).read_text() == generator['render']()


@pytest.fixture
def mass_generator(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / 'test_data/particle_data'
    for path in source.glob('*.txt'):
        shutil.copyfile(path, tmp_path / path.name)
    shutil.copyfile(source / 'sources.json', tmp_path / 'sources.json')
    functions = runpy.run_path(str(source / 'generate_masses.py'))
    monkeypatch.setitem(functions['render'].__globals__, 'DATA_DIR', tmp_path)
    monkeypatch.setitem(functions['render'].__globals__, 'OUTPUT', tmp_path / 'masses.py')
    monkeypatch.setattr(sys, 'argv', ['generate_masses'])
    return functions, tmp_path, source


def update_mass_checksum(root, name):
    path = root / 'sources.json'
    metadata = json.loads(path.read_text())
    metadata[name]['excerpt_sha256'] = hashlib.sha256(
        (root / name).read_bytes()).hexdigest()
    path.write_text(json.dumps(metadata))


def test_mass_rejects_missing_metadata(mass_generator):
    functions, root, _ = mass_generator
    path = root / 'sources.json'
    metadata = json.loads(path.read_text())
    metadata.pop('ame2020_selected.txt')
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='Source metadata must cover'):
        functions['read_sources']()


def test_mass_rejects_corrupt_source(mass_generator):
    functions, root, _ = mass_generator
    path = root / 'ame2020_selected.txt'
    path.write_text(path.read_text() + 'corruption\n')
    with pytest.raises(ValueError, match='Source checksum mismatch'):
        functions['read_sources']()


@pytest.mark.parametrize('kind, message', [
    ('missing', 'CODATA excerpt must contain all nine mass quantities'),
    ('unit', 'Unexpected CODATA unit'),
])
def test_codata_rejects_missing_quantities_and_wrong_units(
        mass_generator, kind, message):
    functions, root, _ = mass_generator
    path = root / 'codata2022_selected.txt'
    lines = path.read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line[:60].strip() == 'electron mass energy equivalent in MeV':
            if kind == 'missing':
                del lines[index]
            else:
                lines[index] = line.rstrip('\n')[:110].ljust(110) + 'eV\n'
            break
    path.write_text(''.join(lines))
    update_mass_checksum(root, path.name)
    with pytest.raises(ValueError, match=message):
        functions['read_sources']()


@pytest.mark.parametrize('name', ['codata2022_selected.txt',
                                 'ame2020_selected.txt',
                                 'nist_binding_energies.txt'])
def test_mass_rejects_duplicate_source(mass_generator, name):
    functions, root, _ = mass_generator
    path = root / name
    lines = path.read_text().splitlines(keepends=True)
    if name.startswith('nist'):
        record = next(line for line in lines if line.split('|')[0].strip().isdigit())
    elif name.startswith('codata'):
        record = next(line for line in lines
                      if line[:60].strip() == 'electron mass energy equivalent in MeV')
    else:
        record = next(line for line in lines
                      if line[9:14].strip().isdigit()
                      and line[14:19].strip().isdigit())
    path.write_text(''.join(lines) + record)
    update_mass_checksum(root, name)
    with pytest.raises(ValueError, match='Duplicate'):
        functions['render']()


@pytest.mark.parametrize('kind, message', [
    ('charge', 'Binding energies must refer to neutral atoms'),
    ('energy', 'Expected bracketed NIST total binding energy'),
    ('atomic_number', 'Atomic-number mismatch'),
])
def test_mass_rejects_invalid_binding_input(mass_generator, kind, message):
    _, root, source = mass_generator
    name = 'nist_binding_energies.txt'
    path = root / name
    lines = path.read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        fields = line.split('|')
        if fields[0].strip().isdigit():
            if kind == 'charge':
                fields[2] = fields[2].replace('0', '1')
            elif kind == 'energy':
                fields[3] = fields[3].replace('[', '').replace(']', '')
            else:
                fields[0] = str(int(fields[0].strip()) + 1)
            lines[index] = '|'.join(fields)
            break
    path.write_text(''.join(lines))
    update_mass_checksum(root, name)
    code = ('import runpy,sys; from pathlib import Path; '
            'g=runpy.run_path(sys.argv[1]); '
            'g["render"].__globals__["DATA_DIR"]=Path(sys.argv[2]); '
            'g["render"]()')
    # These checks previously used assert, which disappears with python -O.
    result = subprocess.run([sys.executable, '-O', '-c', code,
        str(source / 'generate_masses.py'), str(root)],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert f'ValueError: {message}' in result.stderr


def test_mass_regenerates_missing_output(mass_generator):
    functions, root, _ = mass_generator
    functions['main']()
    assert (root / 'masses.py').read_text() == Path(masses.__file__).read_text()
