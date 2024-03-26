import numpy as np
import xtrack as xt
import xobjects as xo
import xpart as xp

import pytest

def test_rf_track_lattice():

    try:
        import RF_Track as RFT
    except ImportError:
        pytest.skip()

    # Bend parameters
    clight = 299792458 # m/s
    p0c = 1.2e9  # eV
    lbend = 3 # m
    angle = np.pi / 2 # rad
    rho = lbend / angle # m
    By = p0c / rho / clight # T

    #############################################
    #######  RF-Track's part starts here  #######
    #############################################

    # Define the RF-Track element
    vol = RFT.Volume()
    vol.dt_mm = 0.1
    vol.odeint_algorithm = 'rk2'
    vol.set_static_Bfield(0.0, By, 0.0)
    vol.set_s0(rho, 0.0, 0.0, 0.0, 0.0, 0.0)
    vol.set_s1(0.0, 0.0, rho, 0.0, 0.0, -angle)
    vol.set_length(lbend)

    #############################################
    #######  RF-Track's part ends here    #######
    #############################################

    # Back to Xsuite
    pi = np.pi
    elements = {
        'd1.1':  xt.Drift(length=1),
        'mb1.1': xt.RFT_Element(element=vol),
        'd2.1':  xt.Drift(length=1),

        'mqd.1': xt.Quadrupole(length=0.3, k1=-0.7),
        'd3.1':  xt.Drift(length=1),
        'mb2.1': xt.RFT_Element(element=vol),
        'd4.1':  xt.Drift(length=1),

        'd1.2':  xt.Drift(length=1),
        'mb1.2': xt.RFT_Element(element=vol),
        'd2.2':  xt.Drift(length=1),

        'mqd.2': xt.Quadrupole(length=0.3, k1=-0.7),
        'd3.2':  xt.Drift(length=1),
        'mb2.2': xt.RFT_Element(element=vol),
        'd4.2':  xt.Drift(length=1),
    }

    # Build the ring
    line = xt.Line(elements=elements, element_names=list(elements.keys()))
    line.particle_ref = xt.Particles(p0c=p0c, mass0=xt.PROTON_MASS_EV)
    line.configure_bend_model(core='full', edge=None)


    ## Transfer lattice on context and compile tracking code=
    line.build_tracker()

    ## Build particle object on context
    n_part = 200

    rng = np.random.default_rng(2021)

    particles0 = xp.Particles(p0c=p0c, #eV
                            q0=1, mass0=xp.PROTON_MASS_EV,
                            x=rng.uniform(-1e-3, 1e-3, n_part),
                            px=rng.uniform(-1e-5, 1e-5, n_part),
                            y=rng.uniform(-2e-3, 2e-3, n_part),
                            py=rng.uniform(-3e-5, 3e-5, n_part),
                            zeta=rng.uniform(-1e-2, 1e-2, n_part),
                            delta=rng.uniform(-1e-2, 1e-2, n_part))

    particles_rft = particles0.copy()

    print('RF-Track tracking starts')
    line.track(particles_rft)   
    print('RF-Track tracking ends')

    ele_xt = elements.copy()
    b_xt = xt.Bend(length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend)
    ele_xt['mb1.1'] = b_xt
    ele_xt['mb1.2'] = b_xt
    ele_xt['mb2.1'] = b_xt
    ele_xt['mb2.2'] = b_xt

    line_ref = xt.Line(elements=ele_xt, element_names=line.element_names)
    line_ref.build_tracker()
    particles_ref = particles0.copy()

    print('Xtrack tracking starts')
    line_ref.track(particles_ref)
    print('Xtrack tracking ends')

    assert_allclose = np.testing.assert_allclose
    assert np.all(particles_rft.particle_id == particles_ref.particle_id)
    assert_allclose(particles_rft.x,  particles_ref.x,  atol=6e-5, rtol=0)
    assert_allclose(particles_rft.px, particles_ref.px, atol=2e-5, rtol=0)
    assert_allclose(particles_rft.y,  particles_ref.y,  atol=6e-7, rtol=0)
    assert_allclose(particles_rft.py, particles_ref.py, atol=2e-6, rtol=0)
    assert_allclose(particles_rft.rpp, particles_ref.rpp, atol=3e-5, rtol=0)
    assert_allclose(particles_rft.rvv, particles_ref.rvv, atol=1e-5, rtol=0)
    assert_allclose(particles_rft.ptau, particles_ref.ptau, atol=2e-5, rtol=0)
    assert_allclose(particles_rft.delta, particles_ref.delta, atol=4e-5, rtol=0)
    assert_allclose(particles_rft.chi, particles_ref.chi, atol=1e-10, rtol=0)
    assert_allclose(particles_rft.p0c, particles_ref.p0c, atol=4e-5, rtol=0)
    assert_allclose(particles_rft.energy0, particles_ref.energy0, atol=4e-5, rtol=0)
    assert_allclose(particles_rft.zeta, particles_ref.zeta, atol=4e-5, rtol=0)
    assert_allclose(particles_rft.beta0, particles_ref.beta0, atol=1e-10, rtol=0)
    assert_allclose(particles_rft.mass0, particles_ref.mass0, atol=1e-10, rtol=0)
    assert_allclose(particles_rft.gamma0, particles_ref.gamma0, atol=1e-10, rtol=0)
