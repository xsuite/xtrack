# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import logging

import numpy as np
import pytest

import xobjects as xo
import xpart as xp
import xtrack as xt


def test_collimation_infrastructure():

    ######################################
    # Create a dummy collimation process #
    ######################################

    class DummyInteractionProcess:

        def __init__(self, fraction_lost, fraction_secondary):

            self.fraction_lost = fraction_lost
            self.fraction_secondary = fraction_secondary

        def interact(self, particles):

            n_part = particles._num_active_particles

            # Kill some particles
            mask_kill = np.random.uniform(size=n_part) < self.fraction_lost
            particles.state[:n_part][mask_kill] = 0

            # Generate some more particles
            mask_secondary = np.random.uniform(size=n_part) < self.fraction_secondary
            n_products = np.sum(mask_secondary)
            if n_products>0:
                products = {
                    's': particles.s[:n_part][mask_secondary],
                    'x': particles.x[:n_part][mask_secondary],
                    'px': particles.px[:n_part][mask_secondary],
                    'y': particles.y[:n_part][mask_secondary],
                    'py': particles.py[:n_part][mask_secondary],
                    'zeta': particles.zeta[:n_part][mask_secondary],
                    'delta': particles.delta[:n_part][mask_secondary],

                    'mass_ratio': particles.x[:n_part][mask_secondary] *0 + .5,
                    'charge_ratio': particles.x[:n_part][mask_secondary] *0 + .5,

                    'parent_particle_id': particles.particle_id[:n_part][
                                                                mask_secondary],
                    'at_element': particles.at_element[:n_part][mask_secondary],
                    'at_turn': particles.at_turn[:n_part][mask_secondary],
                    }
            else:
                products = None

            return products

    #############################################
    # Create the corresponding beam interaction #
    #############################################

    beam_interaction = xt.BeamInteraction(
            interaction_process=DummyInteractionProcess(fraction_lost=0.1,
                                                        fraction_secondary=0.2))

    ############################################
    # Go through the collimator multiple times #
    ############################################

    particles = xp.Particles(_capacity=200,
            p0c=7000, x=np.linspace(-1e-3, 1e-3, 10))

    for _ in range(10):
        beam_interaction.track(particles)

    ###############
    # Some checks #
    ###############

    assert particles._num_lost_particles >= 0
    assert particles._num_active_particles >= 0

    n_all_parts = particles._num_active_particles + particles._num_lost_particles

    assert np.all(np.diff(particles.state) <= 0) # checks that there is no lost after active

    # Check each id is present only once
    ids = particles.particle_id[:n_all_parts]
    assert len(list(set(ids))) == n_all_parts

    # Check parent and secondaries have the same position
    ind_secondaries = np.where(particles.parent_particle_id[:particles._num_active_particles] !=
                        particles.particle_id[:particles._num_active_particles])[0]
    for ii in ind_secondaries:
        parent_id = particles.parent_particle_id[ii]
        parent_x = particles.x[np.where(particles.particle_id == parent_id)[0][0]]

        assert parent_x == particles.x[ii]

@pytest.mark.parametrize(
    'sandwitch_aper',
    [True, False],
    ids=['sadwitch_aper', 'no_sadwitch_aper'])
def test_aperture_refinement(sandwitch_aper):
    n_part=10000
    shift_x = 0.3e-2
    shift_y = 0.5e-2

    ctx = xo.context_default
    buf = ctx.new_buffer()

    logger = logging.getLogger('xtrack')
    logger.setLevel(logging.DEBUG)

    # Define aper_0
    aper_0 = xt.LimitEllipse(_buffer=buf, a=2e-2, b=2e-2)
    shift_aper_0 = (shift_x, shift_y)
    rot_deg_aper_0 = 10.

    # Define aper_1
    aper_1 = xt.LimitEllipse(_buffer=buf, a=1e-2, b=1e-2)
    shift_aper_1 = (shift_x, shift_y)
    rot_deg_aper_1 = 10.

    # aper_0_sandwitch
    if sandwitch_aper:
        line_aper_0 = xt.Line(
            elements=[xt.XYShift(_buffer=buf, dx=shift_aper_0[0], dy=shift_aper_0[1]),
                    xt.SRotation(_buffer=buf, angle=rot_deg_aper_0),
                    aper_0,
                    xt.Multipole(_buffer=buf, knl=[0.00]),
                    xt.SRotation(_buffer=buf, angle=-rot_deg_aper_0),
                    xt.XYShift(_buffer=buf, dx=-shift_aper_0[0], dy=-shift_aper_0[1])])
        line_aper_0.build_tracker(_buffer=buf)
        # aper_1_sandwitch
        line_aper_1 = xt.Line(
            elements=[xt.XYShift(_buffer=buf, dx=shift_aper_1[0], dy=shift_aper_1[1]),
                    xt.SRotation(_buffer=buf, angle=rot_deg_aper_1),
                    aper_1,
                    xt.Multipole(_buffer=buf, knl=[0.00]),
                    xt.SRotation(_buffer=buf, angle=-rot_deg_aper_1),
                    xt.XYShift(_buffer=buf, dx=-shift_aper_1[0], dy=-shift_aper_1[1])])
    else:
        aper_0.shift_x = shift_aper_0[0]
        aper_0.shift_y = shift_aper_0[1]
        aper_0.rot_s_rad = np.deg2rad(rot_deg_aper_0)
        line_aper_0 = xt.Line(
            elements=[aper_0, xt.Multipole(_buffer=buf, knl=[0.0])])
        line_aper_0.build_tracker(_buffer=buf)
        aper_1.shift_x = shift_aper_1[0]
        aper_1.shift_y = shift_aper_1[1]
        aper_1.rot_s_rad = np.deg2rad(rot_deg_aper_1)
        line_aper_1 = xt.Line(
            elements=[aper_1, xt.Multipole(_buffer=buf, knl=[0.00])])

    line_aper_1.build_tracker(_buffer=buf)

    # Build example line
    line=xt.Line(
        elements = ((xt.Drift(_buffer=buf, length=0.5),)
                    + line_aper_0.elements
                    + (xt.Drift(_buffer=buf, length=1),
                       xt.Multipole(_buffer=buf, knl=[0.]),
                       xt.Drift(_buffer=buf, length=1),
                       xt.Cavity(_buffer=buf, voltage=3e6, frequency=400e6),
                       xt.ParticlesMonitor(_buffer=buf,
                            start_at_turn=0, stop_at_turn=10, num_particles=3),
                       xt.Drift(_buffer=buf, length=1.),
                       xt.Marker())
                    + line_aper_1.elements))
    line.build_tracker(_buffer=buf)

    # Test on full line
    r = np.linspace(0, 0.018, n_part)
    theta = np.linspace(0, 8*np.pi, n_part)
    particles = xp.Particles(_context=ctx,
            p0c=6500e9,
            x=r*np.cos(theta)+shift_x,
            y=r*np.sin(theta)+shift_y)

    line.track(particles)


    loss_loc_refinement = xt.LossLocationRefinement(line,
                            n_theta = 360,
                            r_max = 0.5, # m
                            dr = 50e-6,
                            ds = 0.1,
                            save_refine_lines=True,
                            allowed_backtrack_types=[
                                xt.Multipole,
                                xt.Cavity
                                ])

    import time
    t0 = time.time()

    loss_loc_refinement.refine_loss_location(particles)

    t1 = time.time()
    print(f'Took\t{(t1-t0)*1e3:.2f} ms')

    # Automatic checks
    mask_lost = particles.state == 0
    r_calc = np.sqrt((particles.x-shift_x)**2 + (particles.y-shift_y)**2)
    assert np.all(r_calc[~mask_lost]<1e-2)
    assert np.all(r_calc[mask_lost]>1e-2)
    i_aper_1 = line.elements.index(aper_1)
    assert np.all(particles.at_element[mask_lost]==i_aper_1)
    assert np.all(particles.at_element[~mask_lost]==0)
    s0 = line.get_s_elements()[line.elements.index(aper_0)]
    s1 = line.get_s_elements()[line.elements.index(aper_1)]
    r0 = np.sqrt(aper_0.a_squ)
    r1 = np.sqrt(aper_1.a_squ)
    s_expected = s0 + (r_calc-r0)/(r1 - r0)*(s1 - s0)
    # TODO This threshold is a bit large
    xo.assert_allclose(particles.s[mask_lost], s_expected[mask_lost], atol=0.11)


def test_losslocationrefinement_thick_collective_collimator():

    ######################################
    # Create a dummy collimation process #
    ######################################

    class DummyInteractionProcess:
        '''
        I kill some particles and I kick some others by an given angle
        and I generate some secondaries with the opposite angles.
        '''
        def __init__(self, fraction_lost, fraction_secondary, length, kick_x):

            self.fraction_lost = fraction_lost
            self.fraction_secondary = fraction_secondary
            self.kick_x = kick_x
            self.length = length

            self.drift= xt.Drift(length=self.length)


        def interact(self, particles):

            self.drift.track(particles)

            n_part = particles._num_active_particles

            # Kill some particles
            mask_kill = np.random.uniform(size=n_part) < self.fraction_lost
            particles.state[:n_part][mask_kill] = -1 # special flag`


            # Generate some more particles
            mask_secondary = np.random.uniform(size=n_part) < self.fraction_secondary
            n_products = np.sum(mask_secondary)
            if n_products>0:
                products = {
                    's': particles.s[:n_part][mask_secondary],
                    'x': particles.x[:n_part][mask_secondary],
                    'px': particles.px[:n_part][mask_secondary] + self.kick_x,
                    'y': particles.y[:n_part][mask_secondary],
                    'py': particles.py[:n_part][mask_secondary],
                    'zeta': particles.zeta[:n_part][mask_secondary],
                    'delta': particles.delta[:n_part][mask_secondary],

                    'mass_ratio': particles.x[:n_part][mask_secondary] *0 + 1.,
                    'charge_ratio': particles.x[:n_part][mask_secondary] *0 + 1.,

                    'parent_particle_id': particles.particle_id[:n_part][mask_secondary],
                    'at_element': particles.at_element[:n_part][mask_secondary],
                    'at_turn': particles.at_turn[:n_part][mask_secondary],
                    }
            else:
                products = None

            return products

    #############################################
    # Create the corresponding beam interaction #
    #############################################

    interaction_process=DummyInteractionProcess(length=1., kick_x=4e-3,
                                                fraction_lost=0.0,
                                                fraction_secondary=0.2)
    beam_interaction = xt.BeamInteraction(length=interaction_process.length,
                                          interaction_process=interaction_process)

    line = xt.Line(elements=[
        xt.Multipole(knl=[0,0]),
        xt.LimitEllipse(a=2e-2, b=2e-2),
        xt.Drift(length=1.),
        xt.Multipole(knl=[0,0]),
        xt.LimitEllipse(a=2e-2, b=2e-2),
        xt.Drift(length=2.),
        beam_interaction,
        xt.Multipole(knl=[0,0]),
        xt.LimitEllipse(a=2e-2, b=2e-2),
        xt.Drift(length=10.),
        xt.LimitEllipse(a=2e-2, b=2e-2),
        xt.Drift(length=10.),
        ])
    line.build_tracker()

    particles = xp.Particles(
            _capacity=200000,
            x=np.zeros(100000))

    line.track(particles)

    mask_lost = particles.state == 0
    assert np.all(particles.at_element[mask_lost] == 10)


    loss_loc_refinement = xt.LossLocationRefinement(line,
                                                n_theta = 360,
                                                r_max = 0.5, # m
                                                dr = 50e-6,
                                                ds = 0.05,
                                                save_refine_lines=True)

    loss_loc_refinement.refine_loss_location(particles)

    xo.assert_allclose(particles.s[mask_lost], 9.00,
                       rtol=0, atol=loss_loc_refinement.ds*1.0001)

def test_losslocationrefinement_skip_refinement_for_collimators():

    class DummyInteractionProcess:
        '''
        I kill some particles. I kick some others by an given angle
        and I generate some secondaries with the opposite angles.
        '''
        def __init__(self, fraction_lost, fraction_secondary, length, kick_x):

            self.fraction_lost = fraction_lost
            self.fraction_secondary = fraction_secondary
            self.kick_x = kick_x
            self.length = length

            self.drift = xt.Drift(length=self.length)


        def interact(self, particles):

            self.drift.track(particles)

            n_part = particles._num_active_particles

            # Kill some particles
            mask_kill = np.random.uniform(size=n_part) < self.fraction_lost
            particles.state[:n_part][mask_kill] = -1 # special flag`


            # Generate some more particles
            mask_secondary = np.random.uniform(size=n_part) < self.fraction_secondary
            n_products = np.sum(mask_secondary)
            if n_products>0:
                products = {
                    's': particles.s[:n_part][mask_secondary],
                    'x': particles.x[:n_part][mask_secondary] + self.length * (
                        particles.px[:n_part][mask_secondary] + self.kick_x),
                    'px': particles.px[:n_part][mask_secondary] + self.kick_x,
                    'y': particles.y[:n_part][mask_secondary],
                    'py': particles.py[:n_part][mask_secondary],
                    'zeta': particles.zeta[:n_part][mask_secondary],
                    'delta': particles.delta[:n_part][mask_secondary],

                    'mass_ratio': particles.x[:n_part][mask_secondary] *0 + 1.,
                    'charge_ratio': particles.x[:n_part][mask_secondary] *0 + 1.,

                    'parent_particle_id': particles.particle_id[:n_part][mask_secondary],
                    'at_element': particles.at_element[:n_part][mask_secondary],
                    'at_turn': particles.at_turn[:n_part][mask_secondary],
                    }
            else:
                products = None

            return products

    ############################################################
    # Create a beam interaction from the process defined above #
    ############################################################

    # kick_x is large so that the particles are lost on the first aperture element
    interaction_process=DummyInteractionProcess(length=1., kick_x=4e-2,
                                                fraction_lost=0.0,
                                                fraction_secondary=0.2)
    collimator = xt.BeamInteraction(length=interaction_process.length,
                                        interaction_process=interaction_process)

    ########################################################
    # Create a line including the collimator defined above #
    ########################################################

    line = xt.Line(elements=[
        xt.Drift(length=2.),
        xt.Marker(),
        xt.LimitEllipse(a=2e-2, b=2e-2),
        collimator,
        xt.Marker(),
        xt.LimitEllipse(a=2e-2, b=2e-2),
        xt.Drift(length=10.),
        ])

    #################
    # Build line #
    #################

    line.build_tracker()
    line.config.XTRACK_GLOBAL_XY_LIMIT = 1e3

    ##########################
    # Build particles object #
    ##########################

    # We prepare empty slots to store the product particles that will be
    # generated during the tracking.
    particles = xp.Particles(_capacity=200000, x=np.zeros(100000))

    #########
    # Track #
    #########

    line.track(particles)

    ############################
    # Loss location refinement #
    ############################

    part_before = particles.copy()

    loss_loc_refinement = xt.LossLocationRefinement(line,
                                                n_theta = 360,
                                                r_max = 0.5, # m
                                                dr = 50e-6,
                                                ds = 0.05,
                                                save_refine_lines=True)

    loss_loc_refinement.refine_loss_location(particles)

    assert np.all(part_before.s == particles.s)
