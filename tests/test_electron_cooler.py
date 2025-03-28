# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import pytest
import pathlib
from cpymad.madx import Madx
from scipy.stats import linregress
from scipy import constants as cst
import ducktrack as dtk
import xobjects as xo
import xpart as xp
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts, fix_random_seed
from xtrack.beam_elements.elements import _angle_from_trig

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()


@for_all_test_contexts
def test_ecooler_emittance(test_context):
    """Test the electron cooler by comparing the cooling rate with Betacool for LEIR.
    """
    data = np.load(test_data_folder/'electron_cooler/emittance_betacool.npz')
    emittance_betacool = data['emittance']
    time_betacool = data['time']

    gamma0 = 1.004469679
    beta0 = np.sqrt(1 - 1 / gamma0**2)
    mass0 = 193729.0248722061 * 1e6  # eV/c^2
    clight = 299792458.0
    p0c = mass0 * beta0 * gamma0  # eV/c
    q0 = 54
    particle_ref = xp.Particles(p0c=p0c,q0=q0,mass0=mass0,beta0=beta0,gamma0=gamma0)

    circumference = 78.54370266  # m
    T_per_turn = circumference/(clight*beta0)

    qx = 1.82
    qy = 2.72
    beta_x = 5
    beta_y = 5
    qs=0.005247746218929317
    bets0=-2078.673348423543

    arc = xt.LineSegmentMap(
            qx=qx, qy=qy,
            length=circumference,
            betx=beta_x,
            bety=beta_y,
            )

    arc_matching = xt.LineSegmentMap(
            qx=qx, qy=qy,
            length=circumference,
            betx=beta_x,
            bety=beta_y,
            qs=qs,
            bets=bets0)

    line_matching=xt.Line([arc_matching])
    line_matching.build_tracker()

    num_particles=int(1e2)
    sigma_dp = 5e-3    
    gemitt_x = 14e-6
    gemitt_y = 14e-6

    nemitt_x = gemitt_x*beta0*gamma0
    nemitt_y = gemitt_y*beta0*gamma0

    particles = xp.generate_matched_gaussian_bunch(
            num_particles=num_particles,
            nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=4.2,
            particle_ref=particle_ref,
            line=line_matching,        
            )

    particles.delta = np.random.normal(loc=0.0, scale=sigma_dp, size=num_particles)
    particles.zeta = np.random.uniform(-circumference/2, circumference/2, num_particles)

    max_time_s = 1
    int_time_s = 1*1e-3
    num_turns = int((max_time_s / T_per_turn).item())
    save_interval = int((int_time_s / T_per_turn).item())

    monitor = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=1,
                            n_repetitions=int(num_turns/save_interval),
                            repetition_period=save_interval,
                            num_particles=len(particles.x))

    current = 0.6  # amperes
    cooler_length = 2.5  # m cooler length
    radius_e_beam = 25 * 1e-3
    temp_perp = 100e-3 # <E> [eV] = kb*T
    temp_long =  1e-3 # <E> [eV]
    magnetic_field = 0.075  # T for LEIR

    electron_cooler = xt.ElectronCooler(current=current,
                                    length=cooler_length,
                                    radius_e_beam=radius_e_beam,
                                    temp_perp=temp_perp, temp_long=temp_long,
                                    magnetic_field=magnetic_field)

    line = xt.Line(elements=[monitor, electron_cooler, arc],element_names=['monitor','electron_cooler','arc'])                                    
    line.particle_ref = particle_ref
    line.build_tracker()

    line.track(particles, num_turns=num_turns,
            turn_by_turn_monitor=False,with_progress=True)

    x = monitor.x[:,:,0]
    px = monitor.px[:,:,0]
    time = monitor.at_turn[:, 0, 0] * T_per_turn

    action_x = (x**2/beta_x + beta_x*px**2)
    geo_emittance_x=np.mean(action_x,axis=1)/2

    # Match Betacool and Xsuite indices to compare emittance
    valid_indices = ~np.isnan(time_betacool)
    time_betacool = time_betacool[valid_indices]
    matched_indices = [np.abs(time - tb).argmin() for tb in time_betacool]
    emittance_xsuite = geo_emittance_x[matched_indices]
    emittance_betacool = emittance_betacool[:len(emittance_xsuite)]
    
    xo.assert_allclose(emittance_xsuite, emittance_betacool, rtol=0, atol=1e-5)
   
@for_all_test_contexts
def test_ecooler_force(test_context):
    """Test the electron cooler by comparing the cooling force with Betacool for LEIR.
    """
    # Load Betacool force data
    data_betacool = np.load(test_data_folder/'electron_cooler/force_betacool.npz')
    v_diff_betacool = data_betacool['v_diff']
    force_betacool = data_betacool['force']

    beta_rel = 0.09423258405
    gamma = 1.004469679
    current = 0.6  # Amperes
    cooler_length = 2.5  # m
    radius_e_beam = 25 * 1e-3  # m
    temp_perp = 100e-3  # eV
    temp_long = 1e-3  # eV
    magnetic_field = 0.075  # T for LEIR
    mass0 = 193729.0248722061 * 1e6  # eV/c^2
    clight = 299792458.0  # m/s
    p0c = mass0 * beta_rel * gamma  # eV/c
    q0 = 54  # Charge
    beta_x = 5  # m
    emittance = 14 * 1e-6  # Initial geometric emittance (m·rad)

    # Reference particle
    particle_ref = xp.Particles(p0c=p0c, mass0=mass0, q0=q0)

    # Electron cooler
    cooler = xt.ElectronCooler(
        current=current,
        length=cooler_length,
        radius_e_beam=radius_e_beam,
        temp_perp=temp_perp,
        temp_long=temp_long,
        magnetic_field=magnetic_field,
        record_flag=1
    )  
    
    num_particles = int(1e4)
    particles = xp.Particles(
        mass0=mass0,
        p0c=p0c,
        q0=q0,
        x=np.random.normal(0, 1e-20, num_particles),
        px=np.random.normal(0, 4 * np.sqrt(emittance / beta_x), num_particles),
        y=np.random.normal(0, 1e-20, num_particles),
        py=np.random.normal(0, 1e-20, num_particles),
        delta=np.zeros(num_particles),
        zeta=np.zeros(num_particles)
    )
   
    line = xt.Line(elements=[cooler])
    line.particle_ref = particle_ref
    line.build_tracker()

    # Start internal logging for the electron cooler
    record = line.start_internal_logging_for_elements_of_type(
        xt.ElectronCooler, capacity=10000)

    line.track(particles)
    force = record.Fx
    particle_id=record.particle_id[:num_particles]
    particle_id_sort=np.argsort(particle_id)

    force=force[particle_id_sort]

    px_tot = p0c * particles.px
    beta_diff = px_tot / (mass0 * gamma)
    v_diff = beta_diff * clight
    
    sorted_indices = np.argsort(v_diff)
    v_diff = v_diff[sorted_indices]
    force = force[sorted_indices]

    # Match Betacool and Xsuite indices to compare forces
    v_diff_betacool = v_diff_betacool[~np.isnan(v_diff_betacool)]
    matching_indices = [np.abs(v_diff - vb).argmin() for vb in v_diff_betacool]

    force_xsuite = np.array([force[i] for i in matching_indices])
    force_betacool = force_betacool[:len(v_diff_betacool)]

    xo.assert_allclose(force_xsuite, force_betacool, rtol=0, atol=10)