# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #
from pathlib import Path
import xobjects as xo
import xtrack as xt
import numpy as np
from scipy.constants import c as clight
import pytest
from scipy.interpolate import interp1d


# Run the scripts in the following folder to regenerate the reference files
BMAD_REF_FILES = Path(xt.__file__).parent / '../test_data/spin_refs_bmad'

COMMON_TEST_CASES = [
    {
        'case': {
            'x': 0.001,
            'px': 1e-05,
            'y': 0.002,
            'py': 2e-05,
            'delta': 0.001,
            'spin_x': 0.1,
            'spin_z': 0.2,
        },
        'id': 'base'
    },
    {
        'case': {
            'x': 0.001,
            'px': 1e-05,
            'y': 0.002,
            'py': 2e-05,
            'delta': -0.01,
            'spin_x': 0.1,
            'spin_z': 0.2,
        },
        'id': 'delta=-0.01'
    },
    {
        'case': {
            'x': 0.001,
            'px': 1e-05,
            'y': 0.002,
            'py': 2e-05,
            'delta': -0.005,
            'spin_x': 0.1,
            'spin_z': 0.2,
        },
        'id': 'delta=-0.005'
    },
    {
        'case': {
            'x': 0.001,
            'px': 1e-05,
            'y': 0.002,
            'py': 2e-05,
            'delta': 0,
            'spin_x': 0.1,
            'spin_z': 0.2,
        },
        'id': 'delta=0'
    },
    {
        'case': {
            'x': 0.001,
            'px': 1e-05,
            'y': 0.002,
            'py': 2e-05,
            'delta': 0.005,
            'spin_x': 0.1,
            'spin_z': 0.2,
        },
        'id': 'delta=0.005'
    },
    {
        'case': {
            'x': 0.001,
            'px': 1e-05,
            'y': 0.002,
            'py': 2e-05,
            'delta': 0.01,
            'spin_x': 0.1,
            'spin_z': 0.2,
        },
        'id': 'delta=0.01'
    },
    {
        'case': {
            'x': 0.001,
            'px': -0.03,
            'y': 0.002,
            'py': -0.02,
            'delta': 0.001,
            'spin_x': 0.1,
            'spin_z': 0.2,
        },
        'id': 'px=-0.03, py=-0.02'
    },
    {
        'case': {
            'x': 0.001,
            'px': -0.015,
            'y': 0.002,
            'py': -0.01,
            'delta': 0.001,
            'spin_x': 0.1,
            'spin_z': 0.2,
        },
        'id': 'px=-0.015, py=-0.01'
    },
    {
        'case': {
            'x': 0.001,
            'px': 0,
            'y': 0.002,
            'py': 0,
            'delta': 0.001,
            'spin_x': 0.1,
            'spin_z': 0.2,
        },
        'id': 'px=0, py=0'
    },
    {
        'case': {
            'x': 0.001,
            'px': 0.015,
            'y': 0.002,
            'py': 0.01,
            'delta': 0.001,
            'spin_x': 0.1,
            'spin_z': 0.2,
        },
        'id': 'px=0.015, py=0.01'
    },
    {
        'case': {
            'x': 0.001,
            'px': 0.03,
            'y': 0.002,
            'py': 0.02,
            'delta': 0.001,
            'spin_x': 0.1,
            'spin_z': 0.2,
        },
        'id': 'px=0.03, py=0.02'
    }
]


@pytest.mark.parametrize(
    'case,atol',
    zip(
        [case['case'].copy() for case in COMMON_TEST_CASES],
        [4e-06, 4e-05, 3e-05, 2e-05, 3e-05, 4e-05, 0.04, 0.02, 3e-05, 0.02, 0.04],
    ),
    ids=[case['id'] for case in COMMON_TEST_CASES],
)
def test_kicker(case, atol):
    case['spin_y'] = np.sqrt(1 - case['spin_x']**2 - case['spin_z']**2)

    ref_file = BMAD_REF_FILES / 'kicker_bmad.json'
    refs = xt.json.load(ref_file)

    ref = None
    for ref_case in refs:
        if ref_case['in'] == case:
            ref = ref_case['out']
            break
    if ref is None:
        raise ValueError(f'Case {case} not found in file {ref_file}')


    p = xt.Particles(
        p0c=700e9, mass0=xt.ELECTRON_MASS_EV,
        anomalous_magnetic_moment=0.00115965218128,
        **case,
    )

    env = xt.Environment()
    line = env.new_line(
        components=[
            env.new('mykicker', xt.Magnet, length=2.0, knl=[1e-3], ksl=[2e-3]),
            env.new('mymarker', xt.Marker),
        ]
    )

    line.configure_spin(spin_model='auto')

    line.track(p)

    xo.assert_allclose(p.spin_x[0], ref['spin_x'], atol=atol, rtol=0)
    xo.assert_allclose(p.spin_y[0], ref['spin_y'], atol=atol, rtol=0)
    xo.assert_allclose(p.spin_z[0], ref['spin_z'], atol=atol, rtol=0)


@pytest.mark.parametrize(
    'case,atol',
    zip(
        [case['case'].copy() for case in COMMON_TEST_CASES],
        [3e-8, 3e-8, 3e-8, 3e-8, 3e-8, 3e-8, 2e-5, 1e-5, 2e-8, 1e-5, 2e-5],
    ),
    ids=[case['id'] for case in COMMON_TEST_CASES],
)
def test_uniform_solenoid(case, atol):
    case['spin_y'] = np.sqrt(1 - case['spin_x']**2 - case['spin_z']**2)

    ref_file = BMAD_REF_FILES / 'solenoid_bmad.json'
    refs = xt.json.load(ref_file)

    ref = None
    for ref_case in refs:
        if ref_case['in'] == case:
            ref = ref_case['out']
            break
    if ref is None:
        raise ValueError(f'Case {case} not found in file {ref_file}')


    p = xt.Particles(
        p0c=700e9, mass0=xt.ELECTRON_MASS_EV,
        anomalous_magnetic_moment=0.00115965218128,
        **case,
    )

    Bz_T = 0.05
    ks = Bz_T / (p.p0c[0] / clight / p.q0)
    env = xt.Environment()
    line = env.new_line(
        components=[
            env.new('mysolenoid', xt.UniformSolenoid, length=0.02, ks=ks),
            env.new('mymarker', xt.Marker),
        ]
    )

    line.configure_spin(spin_model='auto')

    line.track(p)

    xo.assert_allclose(p.spin_x[0], ref['spin_x'], atol=atol, rtol=0)
    xo.assert_allclose(p.spin_y[0], ref['spin_y'], atol=atol, rtol=0)
    xo.assert_allclose(p.spin_z[0], ref['spin_z'], atol=atol, rtol=0)


@pytest.mark.parametrize(
    'case,atol',
    zip(
        [case['case'].copy() for case in COMMON_TEST_CASES],
        [3e-8, 3e-8, 3e-8, 3e-8, 3e-8, 3e-8, 2e-5, 1e-5, 2e-8, 1e-5, 2e-5],
    ),
    ids=[case['id'] for case in COMMON_TEST_CASES],
)
def test_legacy_solenoid(case, atol):
    case['spin_y'] = np.sqrt(1 - case['spin_x']**2 - case['spin_z']**2)

    ref_file = BMAD_REF_FILES / 'solenoid_bmad.json'
    refs = xt.json.load(ref_file)

    ref = None
    for ref_case in refs:
        if ref_case['in'] == case:
            ref = ref_case['out']
            break
    if ref is None:
        raise ValueError(f'Case {case} not found in file {ref_file}')


    p = xt.Particles(
        p0c=700e9, mass0=xt.ELECTRON_MASS_EV,
        anomalous_magnetic_moment=0.00115965218128,
        **case,
    )

    Bz_T = 0.05
    ks = Bz_T / (p.p0c[0] / clight / p.q0)
    env = xt.Environment()
    line = env.new_line(
        components=[
            env.new('mysolenoid', xt.Solenoid, length=0.02, ks=ks),
            env.new('mymarker', xt.Marker),
        ]
    )

    line.configure_spin(spin_model='auto')

    line.track(p)

    xo.assert_allclose(p.spin_x[0], ref['spin_x'], atol=atol, rtol=0)
    xo.assert_allclose(p.spin_y[0], ref['spin_y'], atol=atol, rtol=0)
    xo.assert_allclose(p.spin_z[0], ref['spin_z'], atol=atol, rtol=0)


@pytest.mark.parametrize(
    'case,atol',
    zip(
        [case['case'].copy() for case in COMMON_TEST_CASES],
        [7e-6, 7e-6, 7e-6, 7e-6, 7e-6, 7e-6, 7e-3, 4e-3, 8e-6, 4e-3, 7e-3],
    ),
    ids=[case['id'] for case in COMMON_TEST_CASES],
)
def test_bend(case, atol):
    case['spin_y'] = np.sqrt(1 - case['spin_x']**2 - case['spin_z']**2)

    ref_file = BMAD_REF_FILES / 'bend_bmad.json'
    refs = xt.json.load(ref_file)

    ref = None
    for ref_case in refs:
        if ref_case['in'] == case:
            ref = ref_case['out']
            break
    if ref is None:
        raise ValueError(f'Case {case} not found in file {ref_file}')


    p = xt.Particles(
        p0c=700e9, mass0=xt.ELECTRON_MASS_EV,
        anomalous_magnetic_moment=0.00115965218128,
        **case,
    )

    env = xt.Environment()
    line = env.new_line(
        components=[
            env.new('mybend', xt.Bend, k0=0.01, h=0.01, length=0.02),
            env.new('mymarker', xt.Marker),
        ]
    )

    line.configure_spin(spin_model='auto')

    line.track(p)

    xo.assert_allclose(p.spin_x[0], ref['spin_x'], atol=atol, rtol=0)
    xo.assert_allclose(p.spin_y[0], ref['spin_y'], atol=atol, rtol=0)
    xo.assert_allclose(p.spin_z[0], ref['spin_z'], atol=atol, rtol=0)


@pytest.mark.parametrize(
    'case,atol',
    zip(
        [case['case'].copy() for case in COMMON_TEST_CASES],
        [6e-8, 6e-8, 6e-8, 6e-8, 6e-8, 6e-8, 6e-5, 3e-5, 2e-7, 3e-5, 6e-5],
    ),
    ids=[case['id'] for case in COMMON_TEST_CASES],
)
def test_quadrupole(case, atol):
    case['spin_y'] = np.sqrt(1 - case['spin_x']**2 - case['spin_z']**2)

    ref_file = BMAD_REF_FILES / 'quadrupole_bmad.json'
    refs = xt.json.load(ref_file)

    ref = None
    for ref_case in refs:
        if ref_case['in'] == case:
            ref = ref_case['out']
            break
    if ref is None:
        raise ValueError(f'Case {case} not found in file {ref_file}')


    p = xt.Particles(
        p0c=700e9, mass0=xt.ELECTRON_MASS_EV,
        anomalous_magnetic_moment=0.00115965218128,
        **case,
    )

    env = xt.Environment()
    line = env.new_line(
        components=[
            env.new('mybend', xt.Quadrupole, k1=0.01, length=0.02),
            env.new('mymarker', xt.Marker),
        ]
    )

    line.configure_spin(spin_model='auto')

    line.track(p)

    xo.assert_allclose(p.spin_x[0], ref['spin_x'], atol=atol, rtol=0)
    xo.assert_allclose(p.spin_y[0], ref['spin_y'], atol=atol, rtol=0)
    xo.assert_allclose(p.spin_z[0], ref['spin_z'], atol=atol, rtol=0)


def test_polarization_lep_base():
    line = xt.load(BMAD_REF_FILES / 'lep_lattice_to_bmad.json')

    line['on_sol.2'] = 1
    line['on_sol.4'] = 1
    line['on_sol.6'] = 1
    line['on_sol.8'] = 1
    line['on_spin_bump.2'] = 0
    line['on_spin_bump.4'] = 0
    line['on_spin_bump.6'] = 0
    line['on_spin_bump.8'] = 0
    line['on_coupl_sol.2'] = 0
    line['on_coupl_sol.4'] = 0
    line['on_coupl_sol.6'] = 0
    line['on_coupl_sol.8'] = 0
    line['on_coupl_sol_bump.2'] = 0
    line['on_coupl_sol_bump.4'] = 0
    line['on_coupl_sol_bump.6'] = 0
    line['on_coupl_sol_bump.8'] = 0

    def make_table(data):
        return xt.Table(
            data={k: np.array(v) for k, v in data.items()}, index='name'
        )

    bmad_data = xt.json.load(BMAD_REF_FILES / 'lep_bmad_base.json')
    spin_bmad = make_table(data=bmad_data['spin'])
    spin_summary_bmad = bmad_data['spin_summary']

    # Make the tables the same length
    start, end = 'ip1', 'bemi.ql1a.l1'
    spin_bmad = spin_bmad.rows[start.upper():end.upper()]
    tw = line.twiss4d(polarization=True).rows[start:end]

    bmad_polarization_eq = spin_summary_bmad['Polarization Limit DK']
    bmad_pol_time_s = 60 * spin_summary_bmad['Polarization Time BKS (minutes, turns)'][0]
    bmad_depol_time_s = 60 * spin_summary_bmad['Depolarization Time (minutes, turns)'][0]
    xo.assert_allclose(tw.spin_polarization_eq, bmad_polarization_eq, atol=0, rtol=3e-2)
    xo.assert_allclose(tw.spin_t_pol_component_s, bmad_pol_time_s, atol=0, rtol=1e-2)
    xo.assert_allclose(tw.spin_t_depol_component_s, bmad_depol_time_s, atol=0, rtol=3e-2)

    xo.assert_allclose(tw.spin_t_pol_buildup_s,
        (1/tw.spin_t_pol_component_s + 1/tw.spin_t_depol_component_s)**-1,
        atol=0, rtol=1e-5)


    for kk in ['spin_x', 'spin_y', 'spin_z',
        'spin_dn_dpz_x', 'spin_dn_dpz_y', 'spin_dn_dpz_z']:
        spin_bmad[kk] *= -1

    spin_x_interp = interp1d(spin_bmad.s, spin_bmad.spin_x)(tw.s)
    xo.assert_allclose(tw.spin_x, spin_x_interp, atol=6e-9, rtol=0)

    spin_y_interp = interp1d(spin_bmad.s, spin_bmad.spin_y)(tw.s)
    xo.assert_allclose(tw.spin_y, spin_y_interp, atol=5e-8, rtol=0)

    spin_z_interp = interp1d(spin_bmad.s, spin_bmad.spin_z)(tw.s)
    xo.assert_allclose(tw.spin_z, spin_z_interp, atol=6e-9, rtol=0)

    spin_dn_dpz_x_interp = interp1d(spin_bmad.s, spin_bmad.spin_dn_dpz_x)(tw.s)
    xo.assert_allclose(
        tw.spin_dn_ddelta_x, spin_dn_dpz_x_interp, atol=0.15, rtol=0
    )

    spin_dn_dpz_y_interp = interp1d(spin_bmad.s, spin_bmad.spin_dn_dpz_y)(tw.s)
    xo.assert_allclose(
        tw.spin_dn_ddelta_y, spin_dn_dpz_y_interp, atol=0.02, rtol=0
    )

    spin_dn_dpz_z_interp = interp1d(spin_bmad.s, spin_bmad.spin_dn_dpz_z)(tw.s)
    xo.assert_allclose(
        tw.spin_dn_ddelta_z, spin_dn_dpz_z_interp, atol=0.2, rtol=0
    )


def test_polarization_lep_spin_bump():
    line = xt.load(BMAD_REF_FILES / 'lep_lattice_to_bmad.json')

    line['on_sol.2'] = 1
    line['on_sol.4'] = 1
    line['on_sol.6'] = 1
    line['on_sol.8'] = 1
    line['on_spin_bump.2'] = 1
    line['on_spin_bump.4'] = 1
    line['on_spin_bump.6'] = 1
    line['on_spin_bump.8'] = 1
    line['on_coupl_sol.2'] = 0
    line['on_coupl_sol.4'] = 0
    line['on_coupl_sol.6'] = 0
    line['on_coupl_sol.8'] = 0
    line['on_coupl_sol_bump.2'] = 0
    line['on_coupl_sol_bump.4'] = 0
    line['on_coupl_sol_bump.6'] = 0
    line['on_coupl_sol_bump.8'] = 0

    def make_table(data):
        return xt.Table(
            data={k: np.array(v) for k, v in data.items()}, index='name'
        )

    bmad_data = xt.json.load(BMAD_REF_FILES / 'lep_bmad_spin_bump.json')
    spin_bmad = make_table(data=bmad_data['spin'])
    spin_summary_bmad = bmad_data['spin_summary']

    # Make the tables the same length
    start, end = 'ip1', 'bemi.ql1a.l1'
    spin_bmad = spin_bmad.rows[start.upper():end.upper()]
    tw = line.twiss4d(polarization=True).rows[start:end]

    bmad_polarization_eq = spin_summary_bmad['Polarization Limit DK']
    bmad_pol_time_s = 60 * spin_summary_bmad['Polarization Time BKS (minutes, turns)'][0]
    bmad_depol_time_s = 60 * spin_summary_bmad['Depolarization Time (minutes, turns)'][0]
    xo.assert_allclose(tw.spin_polarization_eq, bmad_polarization_eq, atol=0, rtol=3e-2)
    xo.assert_allclose(tw.spin_t_pol_component_s, bmad_pol_time_s, atol=0, rtol=1e-2)
    xo.assert_allclose(tw.spin_t_depol_component_s, bmad_depol_time_s, atol=0, rtol=0.15)

    xo.assert_allclose(tw.spin_t_pol_buildup_s,
        (1/tw.spin_t_pol_component_s + 1/tw.spin_t_depol_component_s)**-1,
        atol=0, rtol=1e-5)


    for kk in ['spin_x', 'spin_y', 'spin_z',
        'spin_dn_dpz_x', 'spin_dn_dpz_y', 'spin_dn_dpz_z']:
        spin_bmad[kk] *= -1

    spin_x_interp = interp1d(spin_bmad.s, spin_bmad.spin_x)(tw.s)
    xo.assert_allclose(tw.spin_x, spin_x_interp, atol=2e-5, rtol=0)

    spin_y_interp = interp1d(spin_bmad.s, spin_bmad.spin_y)(tw.s)
    xo.assert_allclose(tw.spin_y, spin_y_interp, atol=4e-7, rtol=0)

    spin_z_interp = interp1d(spin_bmad.s, spin_bmad.spin_z)(tw.s)
    xo.assert_allclose(tw.spin_z, spin_z_interp, atol=2e-5, rtol=0)

    spin_dn_dpz_x_interp = interp1d(spin_bmad.s, spin_bmad.spin_dn_dpz_x)(tw.s)
    xo.assert_allclose(
        tw.spin_dn_ddelta_x, spin_dn_dpz_x_interp, atol=0.1, rtol=0
    )

    spin_dn_dpz_y_interp = interp1d(spin_bmad.s, spin_bmad.spin_dn_dpz_y)(tw.s)
    xo.assert_allclose(
        tw.spin_dn_ddelta_y, spin_dn_dpz_y_interp, atol=0.003, rtol=0
    )

    spin_dn_dpz_z_interp = interp1d(spin_bmad.s, spin_bmad.spin_dn_dpz_z)(tw.s)
    xo.assert_allclose(
        tw.spin_dn_ddelta_z, spin_dn_dpz_z_interp, atol=0.15, rtol=0
    )


def test_polarization_lep_sext_corr():
    line = xt.load(BMAD_REF_FILES / 'lep_lattice_to_bmad.json')

    line['on_sol.2'] = 1
    line['on_sol.4'] = 1
    line['on_sol.6'] = 1
    line['on_sol.8'] = 1
    line['on_spin_bump.2'] = 1
    line['on_spin_bump.4'] = 1
    line['on_spin_bump.6'] = 1
    line['on_spin_bump.8'] = 1
    line['on_coupl_sol.2'] = 1
    line['on_coupl_sol.4'] = 1
    line['on_coupl_sol.6'] = 1
    line['on_coupl_sol.8'] = 1
    line['on_coupl_sol_bump.2'] = 1
    line['on_coupl_sol_bump.4'] = 1
    line['on_coupl_sol_bump.6'] = 1
    line['on_coupl_sol_bump.8'] = 1

    def make_table(data):
        return xt.Table(
            data={k: np.array(v) for k, v in data.items()}, index='name'
        )

    bmad_data = xt.json.load(BMAD_REF_FILES / 'lep_bmad_sext_corr.json')
    spin_bmad = make_table(data=bmad_data['spin'])
    spin_summary_bmad = bmad_data['spin_summary']

    # Make the tables the same length
    start, end = 'ip1', 'bemi.ql1a.l1'
    spin_bmad = spin_bmad.rows[start.upper():end.upper()]
    tw = line.twiss4d(polarization=True).rows[start:end]

    bmad_polarization_eq = spin_summary_bmad['Polarization Limit DK']
    bmad_pol_time_s = 60 * spin_summary_bmad['Polarization Time BKS (minutes, turns)'][0]
    bmad_depol_time_s = 60 * spin_summary_bmad['Depolarization Time (minutes, turns)'][0]
    xo.assert_allclose(tw.spin_polarization_eq, bmad_polarization_eq, atol=0, rtol=3e-2)
    xo.assert_allclose(tw.spin_t_pol_component_s, bmad_pol_time_s, atol=0, rtol=1e-2)
    xo.assert_allclose(tw.spin_t_depol_component_s, bmad_depol_time_s, atol=0, rtol=0.2)

    xo.assert_allclose(tw.spin_t_pol_buildup_s,
        (1/tw.spin_t_pol_component_s + 1/tw.spin_t_depol_component_s)**-1,
        atol=0, rtol=1e-5)

    for kk in ['spin_x', 'spin_y', 'spin_z',
        'spin_dn_dpz_x', 'spin_dn_dpz_y', 'spin_dn_dpz_z']:
        spin_bmad[kk] *= -1

    spin_x_interp = interp1d(spin_bmad.s, spin_bmad.spin_x)(tw.s)
    xo.assert_allclose(tw.spin_x, spin_x_interp, atol=2e-5, rtol=0)

    spin_y_interp = interp1d(spin_bmad.s, spin_bmad.spin_y)(tw.s)
    xo.assert_allclose(tw.spin_y, spin_y_interp, atol=5e-7, rtol=0)

    spin_z_interp = interp1d(spin_bmad.s, spin_bmad.spin_z)(tw.s)
    xo.assert_allclose(tw.spin_z, spin_z_interp, atol=2e-5, rtol=0)

    spin_dn_dpz_x_interp = interp1d(spin_bmad.s, spin_bmad.spin_dn_dpz_x)(tw.s)
    xo.assert_allclose(
        tw.spin_dn_ddelta_x, spin_dn_dpz_x_interp, atol=0.1, rtol=0
    )

    spin_dn_dpz_y_interp = interp1d(spin_bmad.s, spin_bmad.spin_dn_dpz_y)(tw.s)
    xo.assert_allclose(
        tw.spin_dn_ddelta_y, spin_dn_dpz_y_interp, atol=0.002, rtol=0
    )

    spin_dn_dpz_z_interp = interp1d(spin_bmad.s, spin_bmad.spin_dn_dpz_z)(tw.s)
    xo.assert_allclose(
        tw.spin_dn_ddelta_z, spin_dn_dpz_z_interp, atol=0.1, rtol=0
    )

    line['on_sol.2'] = 0
    line['on_sol.4'] = 0
    line['on_sol.6'] = 0
    line['on_sol.8'] = 0
    line['on_spin_bump.2'] = 0
    line['on_spin_bump.4'] = 0
    line['on_spin_bump.6'] = 0
    line['on_spin_bump.8'] = 0
    line['on_coupl_sol.2'] = 0
    line['on_coupl_sol.4'] = 0
    line['on_coupl_sol.6'] = 0
    line['on_coupl_sol.8'] = 0
    line['on_coupl_sol_bump.2'] = 0
    line['on_coupl_sol_bump.4'] = 0
    line['on_coupl_sol_bump.6'] = 0
    line['on_coupl_sol_bump.8'] = 0

    tw = line.twiss4d(polarization=True)
    xo.assert_allclose(
        line.particle_ref.anomalous_magnetic_moment[0]*line.particle_ref.gamma0[0],
        103.45, rtol=0, atol=1e-9)
    xo.assert_allclose(tw.spin_tune_fractional, 0.45, rtol=0, atol=1e-6)
