# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

import numpy as np

import xtrack as xt
import xtrack.particles.pdg as pdg
import xtrack.particles.masses as masses

from xobjects.test_helpers import for_all_test_contexts
import xobjects as xo


def test_names():
    # Test that get_name_from_pdg_id and get_pdg_id_from_name are each other's inverse
    for val in pdg.pdg_table.values():
        name = val[1]
        pdg_id = pdg.get_pdg_id_from_name(name)
        assert pdg.get_name_from_pdg_id(pdg_id, long_name=False) == name
        assert pdg.get_pdg_id_from_name(
            pdg.get_name_from_pdg_id(pdg_id, long_name=False)) == pdg_id
        long_name = val[-1]
        pdg_id = pdg.get_pdg_id_from_name(long_name)
        assert pdg.get_name_from_pdg_id(pdg_id, long_name=True) == long_name
        assert pdg.get_pdg_id_from_name(
            pdg.get_name_from_pdg_id(pdg_id, long_name=True)) == pdg_id
    # Test table names
    _names = [
        [['e⁻', 'e', 'electron'], 11],
        [['e⁺', 'positron'], -11],
        [['νₑ', 'electron neutrino'], 12],
        [['μ⁻', 'μ', 'muon-', 'muon'], 13],
        [['μ⁺', 'muon+', 'anti-muon'], -13],
        [['νμ', 'muon neutrino'], 14],
        [['τ⁻', 'τ', 'tau-', 'tau'], 15],
        [['τ⁺', 'tau+', 'anti-tau'], -15],
        [['ντ', 'tau neutrino'], 16],
        [['γ⁰', 'γ', 'photon'], 22],
        [['π⁰', 'π', 'pion', 'pion0', 'pi0'], 111],
        [['π⁺', 'pion+', 'pi+'], 211],
        [['π⁻', 'pion-', 'pi-'], -211],
        [['K⁰', 'kaon', 'kaon0'], 311],
        [['K⁺', 'kaon+'], 321],
        [['K⁻', 'kaon-'], -321],
        [['KL', 'long kaon'], 130],
        [['Kₛ', 'short kaon'], 310],
        [['D⁰', 'D'], 421],
        [['D⁺'], 411],
        [['D⁻'], -411],
        [['Dₛ⁺'], 431],
        [['Dₛ⁻'], -431],
        [['p⁺', 'p', 'proton'], 2212],
        [[ 'p⁻', 'anti-proton'], -2212],
        [['n⁰', 'n', 'neutron'], 2112],
        [['Δ⁺⁺', 'delta++'], 2224],
        [['Δ⁺', 'delta+'], 2214],
        [['Δ⁰', 'delta0'], 2114],
        [['Δ⁻', 'delta-'], 1114],
        [['Λ⁰', 'Λ', 'lambda'], 3122],
        [['Λc⁺', 'lambdac+'], 4122],
        [['Σ⁺', 'sigma+'], 3222],
        [['Σ⁰', 'Σ', 'sigma', 'sigma0'], 3212],
        [['Σ⁻', 'sigma-'], 3112],
        [['Ξ⁰', 'Ξ', 'xi', 'xi0'], 3322],
        [['Ξ⁻', 'xi-'], 3312],
        [['Ξc⁰', 'Ξc', 'xic', 'xic0'], 4132],
        [['Ξc⁺', 'xic+'], 4232],
        [["Ξ'c⁰", "Ξ'c", "xiprimec", "xiprimec0"], 4312],
        [["Ξ'c⁺", "xiprimec+"], 4322],
        [['Ω⁻', 'omega-'], 3334],
        [['Ωc⁰', 'Ωc', 'omegac', 'omegac0'], 4332],
        [['²H', 'H2', 'hydrogen-2', 'deuteron'], 1000010020],
        [['³H', 'H3', 'hydrogen-3', 'triton'], 1000010030]
    ]
    for names, pdg_id in _names:
        for name in names:
            name_variants = [[nn, nn.lower(), nn.upper(), nn.capitalize()]
                             for nn in [name, pdg._to_normal_script(name),
                                        pdg._digits_to_superscript(name)]]
            name_variants = [nnn for nn in name_variants for nnn in nn]
            pdg_ids = np.unique(pdg.get_pdg_id_from_name(name_variants))
            assert len(pdg_ids) == 1
            assert pdg_ids[0] == pdg_id
            antiname_variants = [[f'anti{pdg._flip_end_sign(nn)}',
                                  f'anti-{pdg._flip_end_sign(nn)}',
                                  f'anti {pdg._flip_end_sign(nn)}']
                                 for nn in name_variants]
            antiname_variants = [nnn for nn in antiname_variants for nnn in nn]
            pdg_ids = np.unique(pdg.get_pdg_id_from_name(antiname_variants))
            assert len(pdg_ids) == 1
            assert pdg_ids[0] == -pdg_id
    # Test ion names
    for Z, name in pdg.elements_long.items():
        assert name == pdg.get_element_full_name_from_Z(Z)
        for A in range(Z+1, 4*Z):
            if Z == 1 and A < 4:
                long_name = 'deuteron' if A == 2 else 'triton'
            else:
                long_name = f'{name}-{A}'
            pdg_id = pdg.get_pdg_id_ion(A, Z)
            assert pdg.get_pdg_id_from_name(f'{name}{A}') == pdg_id
            assert pdg.get_pdg_id_from_name(f'{name} {A}') == pdg_id
            assert pdg.get_pdg_id_from_name(f'{name}-{A}') == pdg_id
            assert pdg.get_pdg_id_from_name(f'{name}_{A}') == pdg_id
            assert pdg.get_pdg_id_from_name(f'{name}.{A}') == pdg_id
            assert pdg.get_name_from_pdg_id(pdg_id, long_name=True) == long_name
            assert pdg.get_name_from_pdg_id(
                        pdg.get_pdg_id_from_name(long_name),
                        long_name=True) == long_name
            assert pdg.get_pdg_id_from_name(
                        pdg.get_name_from_pdg_id(pdg_id,
                        long_name=True)) == pdg_id
    for Z, name in pdg.elements.items():
        assert name == pdg.get_element_name_from_Z(Z)
        for A in range(Z+1, 4*Z):
            short_name = f'{pdg._digits_to_superscript(A)}{name}'
            pdg_id = pdg.get_pdg_id_ion(A, Z)
            assert pdg.get_pdg_id_from_name(f'{A}{name}') == pdg_id
            assert pdg.get_pdg_id_from_name(f'{name}{A}') == pdg_id
            assert pdg.get_pdg_id_from_name(f'{name} {A}') == pdg_id
            assert pdg.get_pdg_id_from_name(f'{name}-{A}') == pdg_id
            assert pdg.get_pdg_id_from_name(f'{name}_{A}') == pdg_id
            assert pdg.get_pdg_id_from_name(f'{name}.{A}') == pdg_id
            assert pdg.get_name_from_pdg_id(pdg_id, long_name=False) == short_name
            assert pdg.get_name_from_pdg_id(
                        pdg.get_pdg_id_from_name(short_name),
                        long_name=False) == short_name
            assert pdg.get_pdg_id_from_name(
                        pdg.get_name_from_pdg_id(pdg_id,
                        long_name=False)) == pdg_id


def test_charge_and_properties():
    for pdg_id, vals in pdg.pdg_table.items():
        q, _, _, _ = pdg.get_properties_from_pdg_id(pdg_id)
        assert q == vals[0]
        anti_q, _, _, _ = pdg.get_properties_from_pdg_id(-pdg_id)
        if vals[1].endswith('⁺') or vals[1].endswith('+'):
            assert np.sign(q) == 1
            assert np.sign(anti_q) == -1
        elif vals[1].endswith('⁻') or vals[1].endswith('-'):
            assert np.sign(q) == -1
            assert np.sign(anti_q) == 1
        elif vals[1].endswith('⁰') or vals[1].endswith('0'):
            assert np.sign(q) == 0
            assert np.sign(anti_q) == 0
        else:
            # Particle names without a charge in the first column of the PDG table
            # should be neutral (except for Hydrogen isotopes)
            if 0 < abs(pdg_id) < 1000000000:
                assert np.sign(q) == 0
                assert np.sign(anti_q) == 0
    for pdg_id in range(100000000, 100119000):
        pdg_id *= 10
        if pdg.is_ion(pdg_id):
            q, A, Z, name = pdg.get_properties_from_pdg_id(pdg_id, long_name=False, subscripts=False)
            assert A > Z > 0
            assert q > 0
            assert q == Z
            name = name.replace(f"{A}", '')
            assert pdg.get_Z_from_element_name(name) == Z
            assert name == pdg.get_element_name_from_Z(Z)
            q, A, Z, name = pdg.get_properties_from_pdg_id(pdg_id, long_name=True)
            if name == 'deuteron':
                assert A == 2
                assert Z == 1
                assert q == 1
            elif name == 'triton':
                assert A == 3
                assert Z == 1
                assert q == 1
            else:
                assert A > Z > 0
                assert q > 0
                assert q == Z
                name_parts = name.split('-')
                assert pdg.get_Z_from_element_name(name_parts[0]) == Z
                assert name_parts[1] == f"{A}"
                assert name_parts[0] == pdg.get_element_full_name_from_Z(Z)


def test_masses():
    assert pdg.get_pdg_id_from_mass_charge(0, 0) == 22
    assert pdg.get_pdg_id_from_mass_charge(511e3, -1) == 11
    assert pdg.get_pdg_id_from_mass_charge(511e3, 1) == -11
    assert pdg.get_pdg_id_from_mass_charge(105.7e6, -1) == 13
    assert pdg.get_pdg_id_from_mass_charge(105.7e6, 1) == -13
    assert pdg.get_pdg_id_from_mass_charge(135.0e6, 0) == 111
    assert pdg.get_pdg_id_from_mass_charge(139.6e6, 1) == 211
    assert pdg.get_pdg_id_from_mass_charge(139.6e6, -1) == -211
    assert pdg.get_pdg_id_from_mass_charge(497.6e6, 0) == 311
    assert pdg.get_pdg_id_from_mass_charge(493.7e6, 1) == 321
    assert pdg.get_pdg_id_from_mass_charge(493.7e6, -1) == -321
    assert pdg.get_pdg_id_from_mass_charge(938.3e6, 1) == 2212
    assert pdg.get_pdg_id_from_mass_charge(938.3e6, -1) == -2212
    assert pdg.get_pdg_id_from_mass_charge(939.6e6, 0) == 2112
    assert pdg.get_pdg_id_from_mass_charge(1875.6e6, 1) == 1000010020
    assert pdg.get_pdg_id_from_mass_charge(2808.9e6, 1) == 1000010030
    assert np.isclose(pdg.get_mass_from_pdg_id(22), 0, rtol=1e-4)
    assert np.isclose(pdg.get_mass_from_pdg_id(11), 511e3, rtol=1e-3)
    assert np.isclose(pdg.get_mass_from_pdg_id(-11), 511e3, rtol=1e-3)
    assert np.isclose(pdg.get_mass_from_pdg_id(13), 105.7e6, rtol=1e-3)
    assert np.isclose(pdg.get_mass_from_pdg_id(-13), 105.7e6, rtol=1e-3)
    assert np.isclose(pdg.get_mass_from_pdg_id(111), 135.0e6, rtol=1e-3)
    assert np.isclose(pdg.get_mass_from_pdg_id(211), 139.6e6, rtol=1e-3)
    assert np.isclose(pdg.get_mass_from_pdg_id(-211), 139.6e6, rtol=1e-3)
    assert np.isclose(pdg.get_mass_from_pdg_id(311), 497.6e6, rtol=1e-3)
    assert np.isclose(pdg.get_mass_from_pdg_id(321), 493.7e6, rtol=1e-3)
    assert np.isclose(pdg.get_mass_from_pdg_id(-321), 493.7e6, rtol=1e-3)
    assert np.isclose(pdg.get_mass_from_pdg_id(2212), 938.3e6, rtol=1e-3)
    assert np.isclose(pdg.get_mass_from_pdg_id(-2212), 938.3e6, rtol=1e-3)
    assert np.isclose(pdg.get_mass_from_pdg_id(2112), 939.6e6, rtol=1e-3)
    assert np.isclose(pdg.get_mass_from_pdg_id(1000010020), 1875.6e6, rtol=1e-3)
    assert np.isclose(pdg.get_mass_from_pdg_id(1000010030), 2808.9e6, rtol=1e-3)
    # Test the ion masses defined in the mass table
    expected_ion_masses = {
        1000010020: masses.DEUTERON_MASS_EV,
        1000010030: masses.TRITON_MASS_EV,
    }
    for massdef, mass in masses.__dict__.items():
        if massdef.endswith('_MASS_EV'):
            massdef = massdef[:-8]
            if any([i.isdigit() for i in massdef]):
                A = int(''.join([i for i in massdef if i.isdigit()]))
                el = massdef.replace(f'{A}', '')
                if el in pdg.elements.values():
                    Z = pdg.get_Z_from_element_name(el)
                    pdg_id = pdg.get_pdg_id_ion(A, Z)
                    assert pdg.get_pdg_id_from_mass_charge(mass, Z) == pdg_id
                    assert np.isclose(pdg.get_mass_from_pdg_id(pdg_id), mass, rtol=1e-7)
                    expected_ion_masses[pdg_id] = mass

    for Z in pdg.elements.keys():
        for A in range(Z+1, 4*Z):
            pdg_id = pdg.get_pdg_id_ion(A, Z)
            assert pdg.get_pdg_id_from_mass_charge(A*masses.U_MASS_EV, Z) == pdg_id
            # A*u is the fallback for unlisted isotopes, not an accuracy bound
            # on tabulated nuclear masses (e.g. Ar36 differs by more than 0.1%).
            expected = expected_ion_masses.get(pdg_id, A*masses.U_MASS_EV)
            assert pdg.get_mass_from_pdg_id(pdg_id, verbose=False) == expected


def test_lead_208():
    pdg_id = 1000822080
    assert pdg.get_pdg_id_ion(208, 82) == pdg_id
    assert pdg.get_pdg_id_from_mass_charge(masses.Pb208_MASS_EV, 82) == pdg_id
    assert pdg.get_name_from_pdg_id(pdg_id, long_name=False) == '²⁰⁸Pb'
    assert pdg.get_name_from_pdg_id(pdg_id, long_name=True) == 'Lead-208'
    assert pdg.get_pdg_id_from_name('Pb208')    == pdg_id
    assert pdg.get_pdg_id_from_name('Pb 208')   == pdg_id
    assert pdg.get_pdg_id_from_name('Pb-208')   == pdg_id
    assert pdg.get_pdg_id_from_name('Pb_208')   == pdg_id
    assert pdg.get_pdg_id_from_name('Pb.208')   == pdg_id
    assert pdg.get_pdg_id_from_name('pb208')    == pdg_id
    assert pdg.get_pdg_id_from_name('208pb')    == pdg_id
    assert pdg.get_pdg_id_from_name('208Pb')    == pdg_id
    assert pdg.get_pdg_id_from_name('lead208')  == pdg_id
    assert pdg.get_pdg_id_from_name('Lead 208') == pdg_id
    assert pdg.get_pdg_id_from_name('Lead_208') == pdg_id
    assert pdg._mass_consistent(pdg_id, masses.Pb208_MASS_EV)
    assert pdg.get_element_name_from_Z(82) == 'Pb'
    assert pdg.get_element_full_name_from_Z(82) == 'Lead'
    xo.assert_allclose(pdg.get_mass_from_pdg_id(pdg_id), masses.Pb208_MASS_EV,
                       rtol=1e-10, atol=0)
    assert pdg.get_properties_from_pdg_id(pdg_id) == (82., 208, 82, '²⁰⁸Pb')


@for_all_test_contexts
def test_build_reference_from_pdg_id(test_context):
    particle_ref_proton  = xt.particles.reference_from_pdg_id(pdg_id='proton',
                                                    _context=test_context)
    particle_ref_proton.move(_context=xo.context_default)
    assert particle_ref_proton.pdg_id == 2212
    particle_ref_lead = xt.particles.reference_from_pdg_id(pdg_id='Pb208',
                                                 _context=test_context)
    particle_ref_lead.move(_context=xo.context_default)
    xo.assert_allclose(particle_ref_lead.q0, 82.)
    xo.assert_allclose(particle_ref_lead.mass0, masses.Pb208_MASS_EV)


def test_particle_type_helpers():
    # Individual particles
    assert pdg.is_proton(2212)
    assert not pdg.is_proton(-2212)
    assert not pdg.is_proton(2112)
    assert pdg.is_antiproton(-2212)
    assert not pdg.is_antiproton(2212)
    assert not pdg.is_antiproton(-2112)

    assert pdg.is_electron(11)
    assert not pdg.is_electron(-11)
    assert not pdg.is_electron(12)
    assert pdg.is_positron(-11)
    assert not pdg.is_positron(11)
    assert not pdg.is_positron(-12)

    assert pdg.is_muon(13)
    assert not pdg.is_muon(-13)
    assert not pdg.is_muon(14)
    assert pdg.is_antimuon(-13)
    assert not pdg.is_antimuon(13)
    assert not pdg.is_antimuon(-14)

    assert pdg.is_lepton(11)
    assert pdg.is_lepton(-11)
    assert pdg.is_lepton(12)
    assert not pdg.is_lepton(22)

    assert pdg.is_neutrino(12)
    assert pdg.is_neutrino(-12)
    assert pdg.is_neutrino(14)
    assert pdg.is_neutrino(16)
    assert not pdg.is_neutrino(11)

    assert pdg.is_pion(111)
    assert pdg.is_pion(211)
    assert pdg.is_pion(-211)
    assert not pdg.is_pion(321)

    assert pdg.is_kaon(130)
    assert pdg.is_kaon(310)
    assert pdg.is_kaon(311)
    assert pdg.is_kaon(-311)
    assert pdg.is_kaon(321)
    assert pdg.is_kaon(-321)
    assert not pdg.is_kaon(211)

    assert pdg.is_meson(211)
    assert pdg.is_meson(-321)
    assert pdg.is_meson(421)
    assert not pdg.is_meson(2212)

    assert pdg.is_baryon(2212)
    assert pdg.is_baryon(-2212)
    assert pdg.is_baryon(3122)
    assert not pdg.is_baryon(211)

    assert pdg.is_hadron(211)
    assert pdg.is_hadron(-321)
    assert pdg.is_hadron(2212)
    assert not pdg.is_hadron(11)
    assert not pdg.is_hadron(22)

    assert pdg.is_photon(22)
    assert not pdg.is_photon(-22)
    assert not pdg.is_photon(11)

    assert pdg.is_antiparticle(-11)
    assert pdg.is_antiparticle(-211)
    assert pdg.is_antiparticle(-2212)
    assert not pdg.is_antiparticle(11)
    assert not pdg.is_antiparticle(22)

    assert pdg.is_ion(1000010020)
    assert pdg.is_ion(1000822080)
    assert pdg.is_ion(-1000822080)
    assert not pdg.is_ion(2212)

    assert pdg.is_neutral(22)
    assert pdg.is_neutral(111)
    assert pdg.is_neutral(2112)
    assert pdg.is_neutral(12)
    assert not pdg.is_neutral(11)
    assert not pdg.is_neutral(211)
    assert not pdg.is_neutral(2212)
    assert not pdg.is_neutral(1000822080)

    # Arrays
    pdg_ids = np.array([
        22,            # photon
        11,            # electron
        -11,           # positron
        12,            # neutrino
        13,            # muon
        -13,           # antimuon
        111,           # pi0
        211,           # pi+
        -211,          # pi-
        130,           # KL
        321,           # K+
        -321,          # K-
        2212,          # proton
        -2212,         # antiproton
        2112,          # neutron
        1000822080,    # Pb208
    ])

    np.testing.assert_array_equal(
        pdg.is_proton(pdg_ids),
        [False, False, False, False, False, False, False, False,
         False, False, False, False, True, False, False, False],
    )

    np.testing.assert_array_equal(
        pdg.is_antiproton(pdg_ids),
        [False, False, False, False, False, False, False, False,
         False, False, False, False, False, True, False, False],
    )

    np.testing.assert_array_equal(
        pdg.is_electron(pdg_ids),
        [False, True, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False],
    )

    np.testing.assert_array_equal(
        pdg.is_positron(pdg_ids),
        [False, False, True, False, False, False, False, False,
         False, False, False, False, False, False, False, False],
    )

    np.testing.assert_array_equal(
        pdg.is_muon(pdg_ids),
        [False, False, False, False, True, False, False, False,
         False, False, False, False, False, False, False, False],
    )

    np.testing.assert_array_equal(
        pdg.is_antimuon(pdg_ids),
        [False, False, False, False, False, True, False, False,
         False, False, False, False, False, False, False, False],
    )

    np.testing.assert_array_equal(
        pdg.is_lepton(pdg_ids),
        [False, True, True, True, True, True, False, False,
         False, False, False, False, False, False, False, False],
    )

    np.testing.assert_array_equal(
        pdg.is_neutrino(pdg_ids),
        [False, False, False, True, False, False, False, False,
         False, False, False, False, False, False, False, False],
    )

    np.testing.assert_array_equal(
        pdg.is_pion(pdg_ids),
        [False, False, False, False, False, False, True, True,
         True, False, False, False, False, False, False, False],
    )

    np.testing.assert_array_equal(
        pdg.is_kaon(pdg_ids),
        [False, False, False, False, False, False, False, False,
         False, True, True, True, False, False, False, False],
    )

    np.testing.assert_array_equal(
        pdg.is_meson(pdg_ids),
        [False, False, False, False, False, False, True, True,
         True, True, True, True, False, False, False, False],
    )

    np.testing.assert_array_equal(
        pdg.is_baryon(pdg_ids),
        [False, False, False, False, False, False, False, False,
         False, False, False, False, True, True, True, False],
    )

    np.testing.assert_array_equal(
        pdg.is_hadron(pdg_ids),
        [False, False, False, False, False, False, True, True,
         True, True, True, True, True, True, True, False],
    )

    np.testing.assert_array_equal(
        pdg.is_photon(pdg_ids),
        [True, False, False, False,False, False, False, False,
         False, False, False, False, False, False, False, False],
    )

    np.testing.assert_array_equal(
        pdg.is_antiparticle(pdg_ids),
        [False, False, True, False, False, True, False, False,
         True, False, False, True, False, True, False, False],
    )

    np.testing.assert_array_equal(
        pdg.is_ion(pdg_ids),
        [False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, True],
    )

    np.testing.assert_array_equal(
        pdg.is_neutral(pdg_ids),
        [True, False, False, True, False, False, True, False,
         False, True, False, False, False, False, True, False],
    )
