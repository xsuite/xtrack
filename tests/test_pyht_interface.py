# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import pathlib

from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts(excluding='ContextPyopencl')
def test_instability_cpu_gpu(test_context):

    import time
    import numpy as np
    from scipy import constants

    protonMass = constants.value("proton mass energy equivalent in MeV") * 1e6
    from scipy.stats import linregress
    from scipy.signal import hilbert

    import xtrack as xt
    import xpart as xp
    xp.enable_pyheadtail_interface()

    from PyHEADTAIL.particles.generators import generate_Gaussian6DTwiss
    from PyHEADTAIL.particles.slicing import UniformBinSlicer
    from PyHEADTAIL.impedances.wakes import WakeTable, WakeField
    from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
    from PyHEADTAIL.trackers.transverse_tracking import TransverseSegmentMap
    from PyHEADTAIL.trackers.longitudinal_tracking import LinearMap
    from PyHEADTAIL.trackers.detuners import ChromaticitySegment, AmplitudeDetuningSegment

    nTurn = 5000  # int(1E4)
    bunch_intensity = 1.8e11
    n_macroparticles = int(1e4)
    energy = 7e3  # [GeV]
    gamma = energy * 1e9 / protonMass
    betar = np.sqrt(1 - 1 / gamma ** 2)
    normemit = 1.8e-6
    beta_x = 68.9
    beta_y = 70.34
    Q_x = 0.31
    Q_y = 0.32
    chroma = -5.0  # 10.0
    sigma_4t = 1.2e-9
    sigma_z = sigma_4t / 4.0 * constants.c
    momentumCompaction = 3.483575072011584e-04
    eta = momentumCompaction - 1.0 / gamma ** 2
    voltage = 12.0e6
    h = 35640
    p0 = constants.m_p * betar * gamma * constants.c
    Q_s = np.sqrt(constants.e * voltage * eta * h / (2 * np.pi * betar * constants.c * p0))
    circumference = 26658.883199999
    averageRadius = circumference / (2 * np.pi)
    sigma_delta = Q_s * sigma_z / (averageRadius * eta)
    beta_s = sigma_z / sigma_delta
    emit_s = 4 * np.pi * sigma_z * sigma_delta * p0 / constants.e  # eVs for PyHEADTAIL

    n_slices_wakes = 200
    limit_z = 3 * sigma_z
    wake_folder = pathlib.Path(
        __file__).parent.joinpath('../examples/pyheadtail_interface/wakes').absolute()
    wakefile = wake_folder.joinpath(
        "wakeforhdtl_PyZbase_Allthemachine_7000GeV_B1_2021_TeleIndex1_wake.dat")
    slicer_for_wakefields = UniformBinSlicer(n_slices_wakes, z_cuts=(-limit_z, limit_z))
    waketable = WakeTable(
        wakefile, ["time", "dipole_x", "dipole_y", "quadrupole_x", "quadrupole_y"]
    )
    wake_field = WakeField(slicer_for_wakefields, waketable)
    wake_field.needs_cpu = True
    wake_field.needs_hidden_lost_particles = True

    damping_time = 7000  # 33.
    damper = TransverseDamper(dampingrate_x=damping_time, dampingrate_y=damping_time)
    damper.needs_cpu = True
    damper.needs_hidden_lost_particles = True
    i_oct = 15.
    detx_x = 1.4e5 * i_oct / 550.0  # from PTC with ATS optics, telescopic factor 1.0
    detx_y = -1.0e5 * i_oct / 550.0

    # expected octupole threshold with damper is 273A according to https://indico.cern.ch/event/902528/contributions/3798807/attachments/2010534/3359300/20200327_RunIII_stability_NMounet.pdf
    # expected growth rate with damper but without octupole is ~0.3 [$10^{-4}$/turn] (also according to Nicolas' presentation)

    checkTurn = 1000

    ########### PyHEADTAIL part ##############

    particles = generate_Gaussian6DTwiss(
        macroparticlenumber=n_macroparticles,
        intensity=bunch_intensity,
        charge=constants.e,
        mass=constants.m_p,
        circumference=circumference,
        gamma=gamma,
        alpha_x=0.0,
        alpha_y=0.0,
        beta_x=beta_x,
        beta_y=beta_y,
        beta_z=beta_s,
        epsn_x=normemit,
        epsn_y=normemit,
        epsn_z=emit_s,
    )

    x0 = np.copy(particles.x)
    px0 = np.copy(particles.xp)
    y0 = np.copy(particles.y)
    py0 = np.copy(particles.yp)
    zeta0 = np.copy(particles.z)
    delta0 = np.copy(particles.dp)

    chromatic_detuner = ChromaticitySegment(dQp_x=chroma, dQp_y=0.0)
    transverse_detuner = AmplitudeDetuningSegment(
        dapp_x=detx_x * p0,
        dapp_y=detx_x * p0,
        dapp_xy=detx_y * p0,
        dapp_yx=detx_y * p0,
        alpha_x=0.0,
        beta_x=beta_x,
        alpha_y=0.0,
        beta_y=beta_y,
    )
    arc_transverse = TransverseSegmentMap(
        alpha_x_s0=0.0,
        beta_x_s0=beta_x,
        D_x_s0=0.0,
        alpha_x_s1=0.0,
        beta_x_s1=beta_x,
        D_x_s1=0.0,
        alpha_y_s0=0.0,
        beta_y_s0=beta_y,
        D_y_s0=0.0,
        alpha_y_s1=0.0,
        beta_y_s1=beta_y,
        D_y_s1=0.0,
        dQ_x=Q_x,
        dQ_y=Q_y,
        segment_detuners=[chromatic_detuner, transverse_detuner],
    )
    arc_longitudinal = LinearMap(
        alpha_array=[momentumCompaction], circumference=circumference, Q_s=Q_s
    )

    turns = np.arange(nTurn)
    x = np.zeros(nTurn, dtype=float)
    t_pyht_start = time.time()
    for turn in range(nTurn):
        time0 = time.time()
        arc_transverse.track(particles)
        arc_longitudinal.track(particles)
        time1 = time.time()
        wake_field.track(particles)
        time2 = time.time()
        damper.track(particles)
        time3 = time.time()
        x[turn] = np.mean(particles.x)
        if turn % 1000 == 0:
            print(
                f"PyHt - turn {turn}" )
    t_pyht_end = time.time()
    print(f'PyHT full loop: {t_pyht_end - t_pyht_start}s')

    x /= np.sqrt(normemit * beta_x / gamma / betar)
    iMin = 1000
    iMax = nTurn - 1000
    if iMin >= iMax:
        iMin = 0
        iMax = nTurn
    ampl = np.abs(hilbert(x))
    b, a, r, p, stderr = linregress(turns[iMin:iMax], np.log(ampl[iMin:iMax]))
    gr_pyht = b

    p_pht = particles

    ############ xsuite-PyHEADTAIL part (the WakeField instance is shared) ########################

    particles = xp.Particles(
        _context=test_context,
        circumference=circumference,
        particlenumber_per_mp=bunch_intensity / n_macroparticles,
        q0=1,
        mass0=protonMass,
        gamma0=gamma,
        x=np.zeros(2*len(x0))
    )

    np2ctx = test_context.nparray_to_context_array
    particles.x[::2] = np2ctx(x0)
    particles.px[::2] = np2ctx(px0)
    particles.y[::2] = np2ctx(y0)
    particles.py[::2] = np2ctx(py0)
    particles.zeta[::2] = np2ctx(zeta0)
    particles.delta[::2] = np2ctx(delta0)
    particles.state[1::2] = 0

    arc = xt.LinearTransferMatrix(
        _context=test_context,
        alpha_x_0=0.0,
        beta_x_0=beta_x,
        disp_x_0=0.0,
        alpha_x_1=0.0,
        beta_x_1=beta_x,
        disp_x_1=0.0,
        alpha_y_0=0.0,
        beta_y_0=beta_y,
        disp_y_0=0.0,
        alpha_y_1=0.0,
        beta_y_1=beta_y,
        disp_y_1=0.0,
        Q_x=Q_x,
        Q_y=Q_y,
        beta_s=beta_s,
        Q_s=-Q_s,
        chroma_x=chroma,
        chroma_y=0.0,
        detx_x=detx_x,
        detx_y=detx_y,
        dety_y=detx_x,
        dety_x=detx_y,
        energy_ref_increment=0.0,
        energy_increment=0,
    )


    line = xt.Line(elements=[arc, wake_field, damper],
                       element_names=['arc', 'wake_field', 'damper'])
    line.build_tracker(_context=test_context)

    t_xt_start = time.time()
    turns = np.arange(nTurn)
    x = np.zeros(nTurn, dtype=float)
    for turn in range(nTurn):

        line.track(particles)

        x[turn] = np.average(particles.x[particles.state>0])
        if turn % 1000 == 0:
            print(
                f"PyHtXt - turn {turn}"
            )
    t_xt_end = time.time()

    print(f'Xt full loop: {t_xt_end - t_xt_start}s')

    x /= np.sqrt(normemit * beta_x / gamma / betar)
    iMin = 1000
    iMax = nTurn - 1000
    if iMin >= iMax:
        iMin = 0
        iMax = nTurn
    ampl = np.abs(hilbert(x))
    b, a, r, p, stderr = linregress(turns[iMin:iMax], np.log(ampl[iMin:iMax]))
    gr_xtpyht = b
    print(f"Growth rate {b*1E4} [$10^{-4}$/turn]")

    print(f'{gr_pyht=}, {gr_xtpyht=} {gr_pyht-gr_xtpyht=}')
    assert np.isclose(gr_xtpyht, gr_pyht, rtol=1e-3, atol=1e-100)

    xp.disable_pyheadtail_interface() # would stay enabled for following tests
                                      # called by pytest
