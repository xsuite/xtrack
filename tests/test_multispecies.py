import numpy as np
import pytest
from scipy.constants import c as clight
from scipy.constants import e as qe

import xobjects as xo
import xtrack as xt


def test_multispecies_multipole():

    p_ref1 = xt.Particles(
        mass0=xt.PROTON_MASS_EV,
        q0=2,
        gamma0=1.2)

    p_ref2 = xt.Particles(
        mass0=2*xt.PROTON_MASS_EV,
        q0=3,
        gamma0=1.5)

    # Build a particle referred to reference 1
    p1 = p_ref1.copy()
    p1.x = 1e-2
    p1.y = 2e-2
    p1.delta = 0.5
    P_p1 = (1 + p1.delta) * p1.p0c

    # Build the same particle referred to reference 2
    p1_ref2 = p_ref2.copy()
    p1_ref2_mass_ratio = p1.mass0 / p1_ref2.mass0
    p1_ref2_charge_ratio = p1.q0 / p1_ref2.q0

    p1_ref2.x = p1.x
    p1_ref2.y = p1.y
    p1_ref2.zeta = p1.zeta / p1.beta0 * p1_ref2.beta0
    p1_ref2.charge_ratio = p1_ref2_charge_ratio
    p1_ref2.chi = p1_ref2_charge_ratio / p1_ref2_mass_ratio
    p1_ref2.delta = P_p1 / p1_ref2_mass_ratio / p1_ref2.p0c - 1

    xo.assert_allclose(p1_ref2.mass, p1.mass, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.charge, p1.charge, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.energy, p1.energy, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.rvv * p1_ref2.beta0, p1.rvv * p1.beta0, atol=0, rtol=1e-14)

    xo.assert_allclose(p1.rpp, 1 / (1 + p1.delta), atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.rpp, 1 / (1 + p1_ref2.delta), atol=0, rtol=1e-14)

    p1c = p1.p0c / p1.rpp * p1.mass_ratio
    p1c_ref2 = p1_ref2.p0c / p1_ref2.rpp * p1_ref2.mass_ratio
    xo.assert_allclose(p1c, p1c_ref2, atol=0, rtol=1e-14)

    L_bend = 1.

    B_T = 0.5
    BsT = 0.1
    G_Tm = 0.1
    Gs_Tm = -0.05

    P0_J_ref1 = p_ref1.p0c[0] / clight * qe
    h_bend_ref1 = B_T * qe * p_ref1.charge[0] / P0_J_ref1 # This is brho
    theta_bend_ref1 = h_bend_ref1 * L_bend
    theta_skew_ref1 = BsT * qe * p_ref1.charge[0] / P0_J_ref1
    k1l_ref1 = G_Tm * qe * p_ref1.charge[0] / P0_J_ref1
    k1sl_ref1 = Gs_Tm * qe * p_ref1.charge[0] / P0_J_ref1


    P0_J_ref2 = p_ref2.p0c[0] / clight * qe
    h_bend_ref2 = B_T * qe * p_ref2.charge[0] / P0_J_ref2
    theta_bend_ref2 = h_bend_ref2 * L_bend
    theta_skew_ref2 = BsT * qe * p_ref2.charge[0] / P0_J_ref2
    k1l_ref2 = G_Tm * qe * p_ref2.charge[0] / P0_J_ref2
    k1sl_ref2 = Gs_Tm * qe * p_ref2.charge[0] / P0_J_ref2

    n_slices = 100

    dipole_ref1 = xt.Multipole(knl=[theta_bend_ref1/n_slices, k1l_ref1/n_slices],
                            ksl=[theta_skew_ref1/n_slices, k1sl_ref1/n_slices],
                            length=L_bend/n_slices, hxl=0.2/n_slices)
    dipole_ref2 = xt.Multipole(knl=[theta_bend_ref2/n_slices, k1l_ref2/n_slices],
                            ksl=[theta_skew_ref2/n_slices, k1sl_ref2/n_slices],
                            length=L_bend/n_slices, hxl=0.2/n_slices)

    ele_ref1 = []
    for ii in range(n_slices):
        ele_ref1.append(xt.Drift(length=L_bend/n_slices/2))
        ele_ref1.append(dipole_ref1)
        ele_ref1.append(xt.Drift(length=L_bend/n_slices/2))

    ele_ref2 = []
    for ii in range(n_slices):
        ele_ref2.append(xt.Drift(length=L_bend/n_slices/2))
        ele_ref2.append(dipole_ref2)
        ele_ref2.append(xt.Drift(length=L_bend/n_slices/2))

    line_ref1 = xt.Line(elements=ele_ref1)
    line_ref2 = xt.Line(elements=ele_ref2)

    line_ref1.append_element(element=xt.Marker(), name='endmarker')
    line_ref2.append_element(element=xt.Marker(), name='endmarker')

    line_ref1.build_tracker()
    line_ref2.build_tracker()

    line_ref1.track(p1, ele_start=0, ele_stop='endmarker')
    line_ref2.track(p1_ref2, ele_start=0, ele_stop='endmarker')

    xo.assert_allclose(p1.x, p1_ref2.x, atol=0, rtol=1e-14)
    xo.assert_allclose(p1.y, p1_ref2.y, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.mass, p1.mass, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.charge, p1.charge, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.energy, p1.energy, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.rvv * p1_ref2.beta0, p1.rvv * p1.beta0, atol=0, rtol=1e-14)

    # Check absolute time of arrival
    t0_ref1 = p1.s / (p1.beta0 * clight)           # Absolute reference time of arrival
    t0_ref2 = p1_ref2.s / (p1_ref2.beta0 * clight) # Absolute reference time of arrival
    dt_ref1 = -p1.zeta / (p1.beta0 * clight)           # Arrival time relative to reference
    dt_ref2 = -p1_ref2.zeta / (p1_ref2.beta0 * clight) # Arrival time relative to reference

    xo.assert_allclose(t0_ref1 + dt_ref1, t0_ref2 + dt_ref2, atol=1e-11, rtol=0)

@pytest.mark.parametrize('model', ['expanded', 'bend-kick-bend', 'rot-kick-rot'])
@pytest.mark.parametrize('B_T', [0.4, 0])
@pytest.mark.parametrize('hxl', [0.2, 0])
@pytest.mark.parametrize('G_Tm', [0.1, 0])
@pytest.mark.parametrize('S_Tm2', [0.05, 0])
def test_multispecies_bend(model, B_T, hxl, G_Tm, S_Tm2):
    p_ref1 = xt.Particles(
        mass0=xt.PROTON_MASS_EV,
        q0=2,
        gamma0=1.2)

    p_ref2 = xt.Particles(
        mass0=2*xt.PROTON_MASS_EV,
        q0=3,
        gamma0=1.5)

    # Build a particle referred to reference 1
    p1 = p_ref1.copy()
    p1.x = 1e-3
    p1.y = 2e-3
    p1.delta = 0.5
    P_p1 = (1 + p1.delta) * p1.p0c

    # Build the same particle referred to reference 2
    p1_ref2 = p_ref2.copy()
    p1_ref2_mass_ratio = p1.mass0 / p1_ref2.mass0
    p1_ref2_charge_ratio = p1.q0 / p1_ref2.q0

    p1_ref2.x = p1.x
    p1_ref2.y = p1.y
    p1_ref2.zeta = p1.zeta / p1.beta0 * p1_ref2.beta0
    p1_ref2.charge_ratio = p1_ref2_charge_ratio
    p1_ref2.chi = p1_ref2_charge_ratio / p1_ref2_mass_ratio
    p1_ref2.delta = P_p1 / p1_ref2_mass_ratio / p1_ref2.p0c - 1

    xo.assert_allclose(p1_ref2.mass, p1.mass, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.charge, p1.charge, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.energy, p1.energy, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.rvv * p1_ref2.beta0, p1.rvv * p1.beta0, atol=0, rtol=1e-14)

    xo.assert_allclose(p1.rpp, 1 / (1 + p1.delta), atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.rpp, 1 / (1 + p1_ref2.delta), atol=0, rtol=1e-14)

    p1c = p1.p0c / p1.rpp * p1.mass_ratio
    p1c_ref2 = p1_ref2.p0c / p1_ref2.rpp * p1_ref2.mass_ratio
    xo.assert_allclose(p1c, p1c_ref2, atol=0, rtol=1e-14)

    L_bend = 1.

    P0_J_ref1 = p_ref1.p0c[0] / clight * qe
    h_bend_ref1 = B_T * qe * p_ref1.charge[0] / P0_J_ref1 * L_bend # This is brho
    theta_bend_ref1 = h_bend_ref1 * L_bend
    k1l_ref1 = G_Tm * qe * p_ref1.charge[0] / P0_J_ref1 * L_bend
    k2l_ref1 = S_Tm2 * qe * p_ref1.charge[0] / P0_J_ref1 * L_bend


    P0_J_ref2 = p_ref2.p0c[0] / clight * qe
    h_bend_ref2 = B_T * qe * p_ref2.charge[0] / P0_J_ref2 * L_bend
    theta_bend_ref2 = h_bend_ref2 * L_bend
    k1l_ref2 = G_Tm * qe * p_ref2.charge[0] / P0_J_ref2 * L_bend
    k2l_ref2 = S_Tm2 * qe * p_ref2.charge[0] / P0_J_ref2 * L_bend

    n_slices = 10

    dipole_ref1 = xt.Bend(k0=theta_bend_ref1/L_bend, length=L_bend / n_slices,
                        h=hxl/L_bend, k1=k1l_ref1/n_slices/L_bend,
                        knl=[0, 0, k2l_ref1/n_slices])
    dipole_ref2 = xt.Bend(k0=theta_bend_ref2/L_bend, length=L_bend / n_slices,
                        h=hxl/L_bend, k1=k1l_ref2/n_slices/L_bend,
                        knl=[0, 0, k2l_ref2/n_slices])

    if model == 'expanded':
        dipole_ref1.num_multipole_kicks = 5
        dipole_ref2.num_multipole_kicks = 5

    ele_ref1 = []
    for ii in range(n_slices):
        ele_ref1.append(xt.Drift(length=L_bend/n_slices/2))
        ele_ref1.append(dipole_ref1)
        ele_ref1.append(xt.Drift(length=L_bend/n_slices/2))

    ele_ref2 = []
    for ii in range(n_slices):
        ele_ref2.append(xt.Drift(length=L_bend/n_slices/2))
        ele_ref2.append(dipole_ref2)
        ele_ref2.append(xt.Drift(length=L_bend/n_slices/2))

    line_ref1 = xt.Line(elements=ele_ref1)
    line_ref2 = xt.Line(elements=ele_ref2)

    line_ref1.append_element(element=xt.Marker(), name='endmarker')
    line_ref2.append_element(element=xt.Marker(), name='endmarker')

    line_ref1.configure_bend_model(core=model)
    line_ref2.configure_bend_model(core=model)

    line_ref1.build_tracker()
    line_ref2.build_tracker()

    line_ref1.track(p1, ele_start=0, ele_stop='endmarker')
    line_ref2.track(p1_ref2, ele_start=0, ele_stop='endmarker')

    xo.assert_allclose(p1.x, p1_ref2.x, atol=0, rtol=1e-10)
    xo.assert_allclose(p1.y, p1_ref2.y, atol=0, rtol=1e-10)
    xo.assert_allclose(p1_ref2.mass, p1.mass, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.charge, p1.charge, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.energy, p1.energy, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.rvv * p1_ref2.beta0, p1.rvv * p1.beta0, atol=0, rtol=1e-14)

    # Check absolute time of arrival
    t0_ref1 = p1.s / (p1.beta0 * clight)           # Absolute reference time of arrival
    t0_ref2 = p1_ref2.s / (p1_ref2.beta0 * clight) # Absolute reference time of arrival
    dt_ref1 = -p1.zeta / (p1.beta0 * clight)           # Arrival time relative to reference
    dt_ref2 = -p1_ref2.zeta / (p1_ref2.beta0 * clight) # Arrival time relative to reference

    xo.assert_allclose(t0_ref1 + dt_ref1, t0_ref2 + dt_ref2, atol=1e-11, rtol=0)

@pytest.mark.parametrize('model', ['full', 'linear'])
@pytest.mark.parametrize('side', ['entry', 'exit'])
def test_multispecies_dipole_edge(model, side):

    p_ref1 = xt.Particles(
        mass0=xt.PROTON_MASS_EV,
        q0=2,
        gamma0=1.2)

    p_ref2 = xt.Particles(
        mass0=2*xt.PROTON_MASS_EV,
        q0=3,
        gamma0=1.5)

    # Build a particle referred to reference 1
    p1 = p_ref1.copy()
    p1.x = 1e-3
    p1.y = 2e-3
    p1.delta = 0.5
    P_p1 = (1 + p1.delta) * p1.p0c

    # Build the same particle referred to reference 2
    p1_ref2 = p_ref2.copy()
    p1_ref2_mass_ratio = p1.mass0 / p1_ref2.mass0
    p1_ref2_charge_ratio = p1.q0 / p1_ref2.q0

    p1_ref2.x = p1.x
    p1_ref2.y = p1.y
    p1_ref2.zeta = p1.zeta / p1.beta0 * p1_ref2.beta0
    p1_ref2.charge_ratio = p1_ref2_charge_ratio
    p1_ref2.chi = p1_ref2_charge_ratio / p1_ref2_mass_ratio
    p1_ref2.delta = P_p1 / p1_ref2_mass_ratio / p1_ref2.p0c - 1

    xo.assert_allclose(p1_ref2.mass, p1.mass, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.charge, p1.charge, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.energy, p1.energy, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.rvv * p1_ref2.beta0, p1.rvv * p1.beta0, atol=0, rtol=1e-14)

    xo.assert_allclose(p1.rpp, 1 / (1 + p1.delta), atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.rpp, 1 / (1 + p1_ref2.delta), atol=0, rtol=1e-14)

    p1c = p1.p0c / p1.rpp * p1.mass_ratio
    p1c_ref2 = p1_ref2.p0c / p1_ref2.rpp * p1_ref2.mass_ratio
    xo.assert_allclose(p1c, p1c_ref2, atol=0, rtol=1e-14)

    B_T = 0.4
    e1=0.1

    P0_J_ref1 = p_ref1.p0c[0] / clight * qe
    k_bend_ref1 = B_T * qe * p_ref1.charge[0] / P0_J_ref1 # This is brho

    P0_J_ref2 = p_ref2.p0c[0] / clight * qe
    k_bend_ref2 = B_T * qe * p_ref2.charge[0] / P0_J_ref2

    edge_ref1 = xt.DipoleEdge(k=k_bend_ref1, e1=e1, hgap=0.05, fint=0.5, side=side)
    edge_ref2 = xt.DipoleEdge(k=k_bend_ref2, e1=e1, hgap=0.05, fint=0.5, side=side)

    ele_ref1 = [xt.Drift(length=1), edge_ref1, xt.Drift(length=1)]
    ele_ref2 = [xt.Drift(length=1), edge_ref2, xt.Drift(length=1)]

    line_ref1 = xt.Line(elements=ele_ref1)
    line_ref2 = xt.Line(elements=ele_ref2)

    line_ref1.append_element(element=xt.Marker(), name='endmarker')
    line_ref2.append_element(element=xt.Marker(), name='endmarker')

    line_ref1.configure_bend_model(edge=model)
    line_ref2.configure_bend_model(edge=model)

    line_ref1.build_tracker()
    line_ref2.build_tracker()

    line_ref1.track(p1, ele_start=0, ele_stop='endmarker')
    line_ref2.track(p1_ref2, ele_start=0, ele_stop='endmarker')

    xo.assert_allclose(p1.x, p1_ref2.x, atol=0, rtol=1e-5)
    xo.assert_allclose(p1.y, p1_ref2.y, atol=0, rtol=1e-1)          #?????????????
    xo.assert_allclose(p1_ref2.mass, p1.mass, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.charge, p1.charge, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.energy, p1.energy, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.rvv * p1_ref2.beta0, p1.rvv * p1.beta0, atol=0, rtol=1e-14)

    # Check absolute time of arrival
    t0_ref1 = p1.s / (p1.beta0 * clight)           # Absolute reference time of arrival
    t0_ref2 = p1_ref2.s / (p1_ref2.beta0 * clight) # Absolute reference time of arrival
    dt_ref1 = -p1.zeta / (p1.beta0 * clight)           # Arrival time relative to reference
    dt_ref2 = -p1_ref2.zeta / (p1_ref2.beta0 * clight) # Arrival time relative to reference

    xo.assert_allclose(t0_ref1 + dt_ref1, t0_ref2 + dt_ref2, atol=1e-11, rtol=0)


def test_multispecies_quadrupole():

    p_ref1 = xt.Particles(
        mass0=xt.PROTON_MASS_EV,
        q0=2,
        gamma0=1.2)

    p_ref2 = xt.Particles(
        mass0=2*xt.PROTON_MASS_EV,
        q0=3,
        gamma0=1.5)

    # Build a particle referred to reference 1
    p1 = p_ref1.copy()
    p1.x = 1e-3
    p1.y = 2e-3
    p1.delta = 0.5
    P_p1 = (1 + p1.delta) * p1.p0c

    # Build the same particle referred to reference 2
    p1_ref2 = p_ref2.copy()
    p1_ref2_mass_ratio = p1.mass0 / p1_ref2.mass0
    p1_ref2_charge_ratio = p1.q0 / p1_ref2.q0

    p1_ref2.x = p1.x
    p1_ref2.y = p1.y
    p1_ref2.zeta = p1.zeta / p1.beta0 * p1_ref2.beta0
    p1_ref2.charge_ratio = p1_ref2_charge_ratio
    p1_ref2.chi = p1_ref2_charge_ratio / p1_ref2_mass_ratio
    p1_ref2.delta = P_p1 / p1_ref2_mass_ratio / p1_ref2.p0c - 1

    xo.assert_allclose(p1_ref2.mass, p1.mass, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.charge, p1.charge, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.energy, p1.energy, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.rvv * p1_ref2.beta0, p1.rvv * p1.beta0, atol=0, rtol=1e-14)

    xo.assert_allclose(p1.rpp, 1 / (1 + p1.delta), atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.rpp, 1 / (1 + p1_ref2.delta), atol=0, rtol=1e-14)

    p1c = p1.p0c / p1.rpp * p1.mass_ratio
    p1c_ref2 = p1_ref2.p0c / p1_ref2.rpp * p1_ref2.mass_ratio
    xo.assert_allclose(p1c, p1c_ref2, atol=0, rtol=1e-14)

    model = 'bend-kick-bend'
    model = 'rot-kick-rot'
    model = 'expanded'
    L_bend = 1.
    G_Tm = 0.1
    Gs_Tm = -0.05

    P0_J_ref1 = p_ref1.p0c[0] / clight * qe
    k1l_ref1 = G_Tm * qe * p_ref1.charge[0] / P0_J_ref1
    k1sl_ref1 = Gs_Tm * qe * p_ref1.charge[0] / P0_J_ref1


    P0_J_ref2 = p_ref2.p0c[0] / clight * qe
    k1l_ref2 = G_Tm * qe * p_ref2.charge[0] / P0_J_ref2
    k1sl_ref2 = Gs_Tm * qe * p_ref2.charge[0] / P0_J_ref2

    n_slices = 10

    quad_ref1 = xt.Quadrupole(k1=k1l_ref1, k1s=k1sl_ref1, length=L_bend/n_slices)
    quad_ref2 = xt.Quadrupole(k1=k1l_ref2, k1s=k1sl_ref2, length=L_bend/n_slices)

    ele_ref1 = []
    for ii in range(n_slices):
        ele_ref1.append(xt.Drift(length=L_bend/n_slices/2))
        ele_ref1.append(quad_ref1)
        ele_ref1.append(xt.Drift(length=L_bend/n_slices/2))

    ele_ref2 = []
    for ii in range(n_slices):
        ele_ref2.append(xt.Drift(length=L_bend/n_slices/2))
        ele_ref2.append(quad_ref2)
        ele_ref2.append(xt.Drift(length=L_bend/n_slices/2))

    line_ref1 = xt.Line(elements=ele_ref1)
    line_ref2 = xt.Line(elements=ele_ref2)

    line_ref1.append_element(element=xt.Marker(), name='endmarker')
    line_ref2.append_element(element=xt.Marker(), name='endmarker')

    line_ref1.configure_bend_model(core=model)
    line_ref2.configure_bend_model(core=model)

    # line_ref1.config.XTRACK_USE_EXACT_DRIFTS = True
    # line_ref2.config.XTRACK_USE_EXACT_DRIFTS = True

    line_ref1.build_tracker()
    line_ref2.build_tracker()

    line_ref1.track(p1, ele_start=0, ele_stop='endmarker')
    line_ref2.track(p1_ref2, ele_start=0, ele_stop='endmarker')

    xo.assert_allclose(p1.x, p1_ref2.x, atol=0, rtol=1e-10)
    xo.assert_allclose(p1.y, p1_ref2.y, atol=0, rtol=1e-10)
    xo.assert_allclose(p1_ref2.mass, p1.mass, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.charge, p1.charge, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.energy, p1.energy, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.rvv * p1_ref2.beta0, p1.rvv * p1.beta0, atol=0, rtol=1e-14)

    # Check absolute time of arrival
    t0_ref1 = p1.s / (p1.beta0 * clight)           # Absolute reference time of arrival
    t0_ref2 = p1_ref2.s / (p1_ref2.beta0 * clight) # Absolute reference time of arrival
    dt_ref1 = -p1.zeta / (p1.beta0 * clight)           # Arrival time relative to reference
    dt_ref2 = -p1_ref2.zeta / (p1_ref2.beta0 * clight) # Arrival time relative to reference

    xo.assert_allclose(t0_ref1 + dt_ref1, t0_ref2 + dt_ref2, atol=1e-11, rtol=0)

def test_multispecies_sextupole():

    p_ref1 = xt.Particles(
        mass0=xt.PROTON_MASS_EV,
        q0=2,
        gamma0=1.2)

    p_ref2 = xt.Particles(
        mass0=2*xt.PROTON_MASS_EV,
        q0=3,
        gamma0=1.5)

    # Build a particle referred to reference 1
    p1 = p_ref1.copy()
    p1.x = 1e-3
    p1.y = 2e-3
    p1.delta = 0.5
    P_p1 = (1 + p1.delta) * p1.p0c

    # Build the same particle referred to reference 2
    p1_ref2 = p_ref2.copy()
    p1_ref2_mass_ratio = p1.mass0 / p1_ref2.mass0
    p1_ref2_charge_ratio = p1.q0 / p1_ref2.q0

    p1_ref2.x = p1.x
    p1_ref2.y = p1.y
    p1_ref2.zeta = p1.zeta / p1.beta0 * p1_ref2.beta0
    p1_ref2.charge_ratio = p1_ref2_charge_ratio
    p1_ref2.chi = p1_ref2_charge_ratio / p1_ref2_mass_ratio
    p1_ref2.delta = P_p1 / p1_ref2_mass_ratio / p1_ref2.p0c - 1

    xo.assert_allclose(p1_ref2.mass, p1.mass, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.charge, p1.charge, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.energy, p1.energy, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.rvv * p1_ref2.beta0, p1.rvv * p1.beta0, atol=0, rtol=1e-14)

    xo.assert_allclose(p1.rpp, 1 / (1 + p1.delta), atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.rpp, 1 / (1 + p1_ref2.delta), atol=0, rtol=1e-14)

    p1c = p1.p0c / p1.rpp * p1.mass_ratio
    p1c_ref2 = p1_ref2.p0c / p1_ref2.rpp * p1_ref2.mass_ratio
    xo.assert_allclose(p1c, p1c_ref2, atol=0, rtol=1e-14)

    model = 'bend-kick-bend'
    model = 'rot-kick-rot'
    model = 'expanded'
    L_bend = 1.
    S_Tm2 = 0.01
    Ss_Tm2 = -0.05

    P0_J_ref1 = p_ref1.p0c[0] / clight * qe
    k2_ref1 = S_Tm2 * qe * p_ref1.charge[0] / P0_J_ref1
    k2s_ref1 = Ss_Tm2 * qe * p_ref1.charge[0] / P0_J_ref1

    P0_J_ref2 = p_ref2.p0c[0] / clight * qe
    k2_ref2 = S_Tm2 * qe * p_ref2.charge[0] / P0_J_ref2
    k2s_ref2 = Ss_Tm2 * qe * p_ref2.charge[0] / P0_J_ref2

    n_slices = 10

    sext_ref1 = xt.Sextupole(k2=k2_ref1, k2s=k2s_ref1, length=L_bend/n_slices)
    sext_ref2 = xt.Sextupole(k2=k2_ref2, k2s=k2s_ref2, length=L_bend/n_slices)

    ele_ref1 = []
    for ii in range(n_slices):
        ele_ref1.append(xt.Drift(length=L_bend/n_slices/2))
        ele_ref1.append(sext_ref1)
        ele_ref1.append(xt.Drift(length=L_bend/n_slices/2))

    ele_ref2 = []
    for ii in range(n_slices):
        ele_ref2.append(xt.Drift(length=L_bend/n_slices/2))
        ele_ref2.append(sext_ref2)
        ele_ref2.append(xt.Drift(length=L_bend/n_slices/2))

    line_ref1 = xt.Line(elements=ele_ref1)
    line_ref2 = xt.Line(elements=ele_ref2)

    line_ref1.append_element(element=xt.Marker(), name='endmarker')
    line_ref2.append_element(element=xt.Marker(), name='endmarker')

    line_ref1.configure_bend_model(core=model)
    line_ref2.configure_bend_model(core=model)

    # line_ref1.config.XTRACK_USE_EXACT_DRIFTS = True
    # line_ref2.config.XTRACK_USE_EXACT_DRIFTS = True

    line_ref1.build_tracker()
    line_ref2.build_tracker()

    line_ref1.track(p1, ele_start=0, ele_stop='endmarker')
    line_ref2.track(p1_ref2, ele_start=0, ele_stop='endmarker')

    xo.assert_allclose(p1.x, p1_ref2.x, atol=0, rtol=1e-10)
    xo.assert_allclose(p1.y, p1_ref2.y, atol=0, rtol=1e-10)
    xo.assert_allclose(p1_ref2.mass, p1.mass, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.charge, p1.charge, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.energy, p1.energy, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.rvv * p1_ref2.beta0, p1.rvv * p1.beta0, atol=0, rtol=1e-14)

    # Check absolute time of arrival
    t0_ref1 = p1.s / (p1.beta0 * clight)           # Absolute reference time of arrival
    t0_ref2 = p1_ref2.s / (p1_ref2.beta0 * clight) # Absolute reference time of arrival
    dt_ref1 = -p1.zeta / (p1.beta0 * clight)           # Arrival time relative to reference
    dt_ref2 = -p1_ref2.zeta / (p1_ref2.beta0 * clight) # Arrival time relative to reference

    xo.assert_allclose(t0_ref1 + dt_ref1, t0_ref2 + dt_ref2, atol=1e-11, rtol=0)

def test_multispecies_octupole():

    p_ref1 = xt.Particles(
        mass0=xt.PROTON_MASS_EV,
        q0=2,
        gamma0=1.2)

    p_ref2 = xt.Particles(
        mass0=2*xt.PROTON_MASS_EV,
        q0=3,
        gamma0=1.5)

    # Build a particle referred to reference 1
    p1 = p_ref1.copy()
    p1.x = 1e-2
    p1.y = 2e-2
    p1.delta = 0.5
    P_p1 = (1 + p1.delta) * p1.p0c

    # Build the same particle referred to reference 2
    p1_ref2 = p_ref2.copy()
    p1_ref2_mass_ratio = p1.mass0 / p1_ref2.mass0
    p1_ref2_charge_ratio = p1.q0 / p1_ref2.q0

    p1_ref2.x = p1.x
    p1_ref2.y = p1.y
    p1_ref2.zeta = p1.zeta / p1.beta0 * p1_ref2.beta0
    p1_ref2.charge_ratio = p1_ref2_charge_ratio
    p1_ref2.chi = p1_ref2_charge_ratio / p1_ref2_mass_ratio
    p1_ref2.delta = P_p1 / p1_ref2_mass_ratio / p1_ref2.p0c - 1

    xo.assert_allclose(p1_ref2.mass, p1.mass, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.charge, p1.charge, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.energy, p1.energy, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.rvv * p1_ref2.beta0, p1.rvv * p1.beta0, atol=0, rtol=1e-14)

    xo.assert_allclose(p1.rpp, 1 / (1 + p1.delta), atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.rpp, 1 / (1 + p1_ref2.delta), atol=0, rtol=1e-14)

    p1c = p1.p0c / p1.rpp * p1.mass_ratio
    p1c_ref2 = p1_ref2.p0c / p1_ref2.rpp * p1_ref2.mass_ratio
    xo.assert_allclose(p1c, p1c_ref2, atol=0, rtol=1e-14)

    L_bend = 1.
    O_Tm3 = 1
    Os_Tm3 = -2

    P0_J_ref1 = p_ref1.p0c[0] / clight * qe
    k3_ref1 = O_Tm3 * qe * p_ref1.charge[0] / P0_J_ref1
    k3s_ref1 = Os_Tm3 * qe * p_ref1.charge[0] / P0_J_ref1

    P0_J_ref2 = p_ref2.p0c[0] / clight * qe
    k3_ref2 = O_Tm3 * qe * p_ref2.charge[0] / P0_J_ref2
    k3s_ref2 = Os_Tm3 * qe * p_ref2.charge[0] / P0_J_ref2

    n_slices = 10

    oct_ref1 = xt.Octupole(k3=k3_ref1, k3s=k3s_ref1, length=L_bend/n_slices)
    oct_ref2 = xt.Octupole(k3=k3_ref2, k3s=k3s_ref2, length=L_bend/n_slices)

    ele_ref1 = []
    for ii in range(n_slices):
        ele_ref1.append(xt.Drift(length=L_bend/n_slices/2))
        ele_ref1.append(oct_ref1)
        ele_ref1.append(xt.Drift(length=L_bend/n_slices/2))

    ele_ref2 = []
    for ii in range(n_slices):
        ele_ref2.append(xt.Drift(length=L_bend/n_slices/2))
        ele_ref2.append(oct_ref2)
        ele_ref2.append(xt.Drift(length=L_bend/n_slices/2))

    line_ref1 = xt.Line(elements=ele_ref1)
    line_ref2 = xt.Line(elements=ele_ref2)

    line_ref1.append_element(element=xt.Marker(), name='endmarker')
    line_ref2.append_element(element=xt.Marker(), name='endmarker')

    line_ref1.config.XTRACK_USE_EXACT_DRIFTS = True
    line_ref2.config.XTRACK_USE_EXACT_DRIFTS = True

    line_ref1.build_tracker()
    line_ref2.build_tracker()

    line_ref1.track(p1, ele_start=0, ele_stop='endmarker')
    line_ref2.track(p1_ref2, ele_start=0, ele_stop='endmarker')

    xo.assert_allclose(p1.x, p1_ref2.x, atol=0, rtol=1e-12)
    xo.assert_allclose(p1.y, p1_ref2.y, atol=0, rtol=1e-12)
    xo.assert_allclose(p1_ref2.mass, p1.mass, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.charge, p1.charge, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.energy, p1.energy, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.rvv * p1_ref2.beta0, p1.rvv * p1.beta0, atol=0, rtol=1e-14)

    # Check absolute time of arrival
    t0_ref1 = p1.s / (p1.beta0 * clight)           # Absolute reference time of arrival
    t0_ref2 = p1_ref2.s / (p1_ref2.beta0 * clight) # Absolute reference time of arrival
    dt_ref1 = -p1.zeta / (p1.beta0 * clight)           # Arrival time relative to reference
    dt_ref2 = -p1_ref2.zeta / (p1_ref2.beta0 * clight) # Arrival time relative to reference

    xo.assert_allclose(t0_ref1 + dt_ref1, t0_ref2 + dt_ref2, atol=1e-11, rtol=0)

def test_multispecies_cavity():

    p_ref1 = xt.Particles(
        mass0=xt.PROTON_MASS_EV,
        q0=2,
        gamma0=1.2)

    p_ref2 = xt.Particles(
        mass0=2*xt.PROTON_MASS_EV,
        q0=3,
        gamma0=1.5)

    # Build a particle referred to reference 1
    p1 = p_ref1.copy()
    p1.x = 1e-2
    p1.y = 2e-2
    p1.delta = 0.5
    P_p1 = (1 + p1.delta) * p1.p0c

    # Build the same particle referred to reference 2
    p1_ref2 = p_ref2.copy()
    p1_ref2_mass_ratio = p1.mass0 / p1_ref2.mass0
    p1_ref2_charge_ratio = p1.q0 / p1_ref2.q0

    p1_ref2.x = p1.x
    p1_ref2.y = p1.y
    p1_ref2.zeta = p1.zeta / p1.beta0 * p1_ref2.beta0
    p1_ref2.charge_ratio = p1_ref2_charge_ratio
    p1_ref2.chi = p1_ref2_charge_ratio / p1_ref2_mass_ratio
    p1_ref2.delta = P_p1 / p1_ref2_mass_ratio / p1_ref2.p0c - 1

    xo.assert_allclose(p1_ref2.mass, p1.mass, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.charge, p1.charge, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.energy, p1.energy, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.rvv * p1_ref2.beta0, p1.rvv * p1.beta0, atol=0, rtol=1e-14)

    xo.assert_allclose(p1.rpp, 1 / (1 + p1.delta), atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.rpp, 1 / (1 + p1_ref2.delta), atol=0, rtol=1e-14)

    p1c = p1.p0c / p1.rpp * p1.mass_ratio
    p1c_ref2 = p1_ref2.p0c / p1_ref2.rpp * p1_ref2.mass_ratio
    xo.assert_allclose(p1c, p1c_ref2, atol=0, rtol=1e-14)

    s_cav = 100
    frequency = 400e6


    lag2 = 180 / np.pi *  2 * np.pi * frequency * s_cav / clight * (1/p1_ref2.beta0 - 1/p1.beta0)


    cav_ref1 = xt.Cavity(voltage=1e6, frequency=frequency)
    cav_ref2 = xt.Cavity(voltage=1e6, frequency=frequency, lag=lag2)

    ele_ref1 = [xt.Drift(length=s_cav), cav_ref1, xt.Drift(length=0)]
    ele_ref2 = [xt.Drift(length=s_cav), cav_ref2, xt.Drift(length=0)]

    line_ref1 = xt.Line(elements=ele_ref1)
    line_ref2 = xt.Line(elements=ele_ref2)

    line_ref1.append_element(element=xt.Marker(), name='endmarker')
    line_ref2.append_element(element=xt.Marker(), name='endmarker')

    line_ref1.build_tracker()
    line_ref2.build_tracker()

    line_ref1.track(p1, ele_start=0, ele_stop='endmarker')
    line_ref2.track(p1_ref2, ele_start=0, ele_stop='endmarker')



    # Check absolute time of arrival
    t0_ref1 = p1.s / (p1.beta0 * clight)           # Absolute reference time of arrival
    t0_ref2 = p1_ref2.s / (p1_ref2.beta0 * clight) # Absolute reference time of arrival
    dt_ref1 = -p1.zeta / (p1.beta0 * clight)           # Arrival time relative to reference
    dt_ref2 = -p1_ref2.zeta / (p1_ref2.beta0 * clight) # Arrival time relative to reference

    xo.assert_allclose(t0_ref1 + dt_ref1, t0_ref2 + dt_ref2, atol=1e-11, rtol=0)
    xo.assert_allclose(p1_ref2.zeta,
        (1 - p1_ref2.beta0 / p1.beta0) * p1.s + p1_ref2.beta0 / p1.beta0 * p1.zeta,
        atol=1e-11, rtol=0)

    xo.assert_allclose(p1.x, p1_ref2.x, atol=0, rtol=1e-12)
    xo.assert_allclose(p1.y, p1_ref2.y, atol=0, rtol=1e-12)
    xo.assert_allclose(p1_ref2.mass, p1.mass, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.charge, p1.charge, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.energy, p1.energy, atol=0, rtol=1e-14)
    xo.assert_allclose(p1_ref2.rvv * p1_ref2.beta0, p1.rvv * p1.beta0, atol=0, rtol=1e-14)