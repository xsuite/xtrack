from pathlib import Path
import runpy

import numpy as np

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
        repo_root / 'test_data' / 'particle_masses' / 'generate_masses.py'))
    constants, binding, atoms = generator['read_sources']()
    assert len(atoms) == 147
    assert len(binding) == 8
    assert len(constants) == 9
    assert Path(masses.__file__).read_text() == generator['render']()
