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
from scipy.optimize import curve_fit

# test_data_folder = pathlib.Path(
#     __file__).parent.joinpath('../test_data').absolute()


@for_all_test_contexts
def test_pulsed_laser(test_context):
    """Test the decay of horizontal emittance with expected value.
    """
    qx=26.299364685601024
    qy=26.249362950069248
    
    circumference=6911.5038
   
    voltage_rf=7*1e6
    frequency=201.8251348335775*1e6
    lag_rf=180
    momentum_compaction_factor=0.001901471346864482
    
   
    beta_x  =  54.614389 # m
    beta_y  =  44.332517 # m
    alpha_x = -1.535235
    alpha_y =  1.314101

    Dx  =  2.444732 # m
    Dpx =  0.097522

    Dy  =  0.0 # m
    Dpy =  0.0

    #index of gamma factory along SPS line: 16675

    arc = xt.LineSegmentMap(
            qx=qx, qy=qy,
            length=circumference,
            alfx=alpha_x,
            alfy=alpha_y,
            betx=beta_x,
            bety=beta_y,
            
            dx=Dx,
            dpx=Dpx,
            dy=Dy,
            dpy=Dpy,
                        
            voltage_rf=voltage_rf,
            lag_rf=lag_rf,
            frequency_rf=frequency,
            momentum_compaction_factor=momentum_compaction_factor,
            longitudinal_mode = 'nonlinear'
            )
    
    q0 = 17
    mass0 = 37261297096.799995

    gamma0= 205.0686404689884
    beta0= 0.9999881125888904
    p0c = mass0*gamma0*beta0 #eV/c

    bunch_intensity = 4000000000.0

    particle_ref = xp.Particles(p0c=p0c, mass0=mass0, q0=q0, gamma0=gamma0)

    nemitt = 1.5e-6 # m*rad (normalized emittance)
    sigma_z = 0.095
    
    num_particles=int(1e3)

    line_arc=xt.Line(
            elements=[arc])
    line_arc.build_tracker()

    particles = xp.generate_matched_gaussian_bunch(
            num_particles=num_particles,
            total_intensity_particles=bunch_intensity,
            nemitt_x=nemitt, nemitt_y=nemitt, sigma_z=sigma_z,
            particle_ref=particle_ref,
            line=line_arc,
            )
    particles._init_random_number_generator()
    
    particles0=particles.copy()
    
    sigma_dp=2e-4 
    ##################
    # Laser Cooler #
    ##################

    #laser-ion beam collision angle
    theta_l = 2.6*np.pi/180 # rad
    nx = 0; ny = -np.sin(theta_l); nz = -np.cos(theta_l)

    # Ion excitation energy:
    ion_excited_lifetime=4.2789999999999997e-13
    hw0 = 661.89 # eV
    clight=cst.c
    hc=cst.hbar*clight/cst.e # eV*m (ħc)
    lambda_0 = 2*np.pi*hc/hw0 # m -- ion excitation wavelength
    lambda_l = 7.680000000000001e-07

    # Shift laser wavelength for fast longitudinal cooling:
    #lambda_l = lambda_l*(1+1*sigma_dp) # m

    laser_frequency = clight/lambda_l # Hz
    sigma_w = 2*np.pi*laser_frequency*sigma_dp
    #sigma_w = 2*np.pi*laser_frequency*sigma_dp/2 # for fast longitudinal cooling
    sigma_t = 1/sigma_w # sec -- Fourier-limited laser pulse
    
    laser_waist_radius = 1.3e-3 #m
    laser_energy = 5e-3        
    
    laser_x = -0.0015771812080536912

    GF_IP = xt.PulsedLaser(
                    laser_x=laser_x,
                    laser_y=0,
                    laser_z=0,
                    
                    laser_direction_nx = 0,
                    laser_direction_ny = ny,
                    laser_direction_nz = nz,
                    laser_energy         = laser_energy, # J
                    laser_duration_sigma = sigma_t, # sec
                    laser_wavelength = lambda_l, # m
                    laser_waist_radius = laser_waist_radius, # m
                    laser_waist_shift = 0, # m
                    ion_excitation_energy = hw0, # eV
                    ion_excited_lifetime  = ion_excited_lifetime, # sec,
                    record_flag=1                   
                    )

    # simulation parameters: simulate 10 s of cooling, and take data once every 100 ms
    max_time_s = 1
    int_time_s = 0.01
    T_per_turn = circumference/(clight*beta0)
    num_turns = int(max_time_s/T_per_turn)
    save_interval = int(int_time_s/T_per_turn)

    # create a monitor object, to reduce holded data
    monitor = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=1,
                            n_repetitions=int(num_turns/save_interval),
                            repetition_period=save_interval,
                            num_particles=num_particles)
                                
    line = xt.Line(
            elements=[monitor,GF_IP,arc])
        
    line.build_tracker(_context=test_context)

    # Start internal logging for the electron cooler
    record = line.start_internal_logging_for_elements_of_type(
            xt.PulsedLaser, capacity=num_particles*100)

    line.track(particles, num_turns=num_turns,
            turn_by_turn_monitor=False,with_progress=True)

    # extract relevant values
    x = monitor.x[:,:,0]
    px = monitor.px[:,:,0]
    y = monitor.y[:,:,0]
    py = monitor.py[:,:,0]
    delta = monitor.delta[:,:,0]
    zeta = monitor.zeta[:,:,0]
    state = monitor.state[:,:,0]
    time = monitor.at_turn[:, 0, 0] * T_per_turn

    gamma_x=(1+alpha_x**2)/beta_x
    gamma_y=(1+alpha_y**2)/beta_y

    action_x = (gamma_x*(x-Dx*delta)**2 + 2*alpha_x*(x-Dx*delta)*(px-Dpx*delta)+ beta_x*(px-Dpx*delta)**2)
    action_y = (gamma_y*(y-Dy*delta)**2 + 2*alpha_y*(y-Dy*delta)*(py-Dpy*delta)+ beta_y*(py-Dpy*delta)**2)

    emittance_x_twiss=np.mean(action_x,axis=1)*beta0*gamma0/2
    rms_dp_p=np.std(delta,axis=1)

    # Define the exponential decay function
    def exp_decay(t, A, tau ):
            return A * np.exp(-t / tau)

    A0 = emittance_x_twiss[0]
    tau0 = (time[-1] - time[0]) / 5.0
    C0 = emittance_x_twiss[-1]
    initial_guess = (A0, tau0)

    popt, pcov = curve_fit(exp_decay, time, emittance_x_twiss, p0=initial_guess)

    time_fit = np.linspace(np.min(time), np.max(time), 100)
    emittance_fit = exp_decay(time_fit, *popt)
    tau=popt[1]
   
    xo.assert_allclose(tau, 7, rtol=0, atol=1)
   

@for_all_test_contexts
def test_cw_laser(test_context):
    """Test phase space acumulation for Lanzhou experiment
    https://www.sciencedirect.com/science/article/pii/S0168900222011445?ref=pdf_download&fr=RR-2&rr=80ca25af8a5ace93
    """
        # Ion properties:
    m_u = 931.49410242e6 # eV/c^2 -- atomic mass unit
    A = 16 # Weight of O
   
    m_p = 938.272088e6 # eV/c^2 -- proton mass
    clight = 299792458.0 # m/s

    q0=5

    mass0 = A*m_u #+ Ne*m_e # eV/c^2

    beta_rel = 0.64
    gamma_rel = 1.30

    p0c = mass0*beta_rel*gamma_rel #eV/c

   
    circumference =  128.80 #m
    T = circumference/(clight*beta_rel)
    s_per_turn = T
    
    beta_x = 6
    beta_y = 2

    disp_x = 0
    Q_x = 2.2
    Q_y = 2.4
    
    arc = xt.LineSegmentMap(
                qx=Q_x, qy=Q_y,
                length=circumference,
                betx=beta_x,
                bety=beta_y
                )

    emittance_x=10*1e-6 #inital emittance
    emittance_y=15*1e-6 #inital emittance
    num_particles = int(1e4)

    sigma_x = np.sqrt(beta_x*emittance_x)
    sigma_px = np.sqrt(emittance_x*1/beta_x)
    sigma_y = np.sqrt(beta_y*emittance_y)
    sigma_py = np.sqrt(emittance_y*1/beta_y)
    
    
    delta = np.linspace(0, 1e-5, num_particles)

    # delta = np.random.normal(loc=0, scale=sigma_p, size=num_particles)
    x = np.random.normal(loc=0.0, scale=sigma_x, size=num_particles) + disp_x * delta
    px = np.random.normal(loc=0.0, scale=sigma_px, size=num_particles)
    y = np.random.normal(loc=0.0, scale=sigma_y, size=num_particles)
    py = np.random.normal(loc=0.0, scale=sigma_py, size=num_particles)

    particles = xp.Particles(
        mass0=mass0,
        p0c=p0c,
        q0=q0,
        x=x*0,
        px=px*0,
        y=y*0,
        py=py*0,
        delta=delta,
        zeta=0
        )
    particles._init_random_number_generator()

    ##################
    # Laser Cooler #
    ##################

    theta_l = 0
    nx = 0; ny = -np.sin(theta_l); nz = -np.cos(theta_l)

    # Ion excitation energy:
    # hw0 = 230.823 # eV
    hc=cst.hbar*clight/cst.e # eV*m (ħc)
    lambda_0 = 103.76*1e-9 # m -- ion excitation wavelength
    hw0 = 2*np.pi*hc/lambda_0 # eV -- ion excitation energy
    ion_excited_lifetime=2.44e-9


    lambda_l = 2.2130563056305633e-07 
    #lambda_l = lambda_l*(1+1*sigma_p) # m

    laser_frequency = clight/lambda_l # Hz
    
    laser_power=40*1e-3 #W
    laser_waist_radius = 5*1e-3
    laser_area=np.pi*(laser_waist_radius*laser_waist_radius)

    laser_intensity=laser_power/laser_area

    cooling_section_length=25
    CW_laser = xt.CWLaser(
                        laser_x=0,
                        laser_y=0,
                        laser_z=0,
                        
                        laser_direction_nx = 0,
                        laser_direction_ny = 0,
                        laser_direction_nz = -1,
                        laser_wavelength = lambda_l, # m
                        laser_waist_radius = laser_waist_radius, # m
                        laser_intensity=laser_intensity,
                        ion_excitation_energy = hw0, # eV
                        ion_excited_lifetime  = ion_excited_lifetime, # sec
                        cooling_section_length=cooling_section_length
                                
        )

     # ##################
     # # Tracking #
     # ##################

     # simulation parameters: simulate 10 s of cooling, and take data once every 100 ms
    T_per_turn = circumference/(clight*beta_rel)


    num_turns = int(1e5) # 0 emittance
    save_interval=num_turns/100

    # create a monitor object, to reduce holded data
    monitor = xt.ParticlesMonitor(start_at_turn=0, stop_at_turn=1,
                                n_repetitions=int(num_turns/save_interval),
                                repetition_period=save_interval,
                                num_particles=num_particles)

    line = xt.Line(
                elements=[monitor,CW_laser, arc])

    particle_ref = xp.Particles(mass0=mass0, q0=q0, p0c=p0c)

    line.particle_ref = particle_ref
    line.build_tracker(_context=test_context)

    line.track(particles, num_turns=num_turns,
                turn_by_turn_monitor=False,with_progress=True)

    # extract relevant values
    x = monitor.x[:,:,0]
    px = monitor.px[:,:,0]
    y = monitor.y[:,:,0]
    py = monitor.py[:,:,0]
    delta = monitor.delta[:,:,0]
    zeta = monitor.zeta[:,:,0]
    accumulated_length=monitor.s[:,:,0]
    state = monitor.state[:,:,0]
    time = monitor.at_turn[:, 0, 0] * T_per_turn

    excited=particles.state==2
    excited = excited.astype(int)
    fraction_excitation = sum(excited)/len(excited)
    time = np.arange(0, num_turns, save_interval) * s_per_turn

    delta_first_turn = delta[0, :]
    delta_final_turn = delta[-1, :]

    bins = np.linspace(0, 1e-5, 51)  # 50 bins between 0 and 1e-5

    hist_before, bin_edges = np.histogram(delta_first_turn, bins=bins)
    hist_after, _ = np.histogram(delta_final_turn, bins=bins)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    max_bin_height_before = np.max(hist_before)
    max_bin_height_after = np.max(hist_after)
    
    min_index_after = np.argmin(hist_after)
    min_bin_center_after = bin_centers[min_index_after]
    min_bin_count_after = hist_after[min_index_after]


    assert max_bin_height_after > 1.5* max_bin_height_before

    xo.assert_allclose(min_bin_center_after, 4.10e-06, rtol=0.1, atol=0)
        
