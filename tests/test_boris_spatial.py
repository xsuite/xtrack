import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe

import xobjects as xo
import xtrack as xt
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField


def test_boris_spatial():
    delta=np.array([0, 4])
    p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                    energy0=45.6e9/1000,
                    x=[-1e-3, -1e-3],
                    px=-1e-3*(1+delta),
                    y=1e-3,
                    delta=delta)

    p = p0.copy()

    sf = SolenoidField(L=4, a=0.3, B0=1.5, z0=20)

    integrator = xt.BorisSpatialIntegrator(fieldmap_callable=sf.get_field,
                                            s_start=0,
                                            s_end=30,
                                            n_steps=15000)
    p_boris = p.copy()
    integrator.track(p_boris)


    x_log = np.array(integrator.x_log)
    y_log = np.array(integrator.y_log)
    z_log = np.array(integrator.z_log)

    b_seen = sf.get_field(x_log, y_log, z_log)


    z_axis = np.linspace(0, 30, 1001)
    Bz_axis = sf.get_field(0 * z_axis, 0 * z_axis, z_axis)[2]

    P0_J = p.p0c[0] * qe / clight
    brho = P0_J / qe / p.q0

    # ks = 0.5 * (Bz_axis[:-1] + Bz_axis[1:]) / brho
    ks = Bz_axis / brho
    ks_entry = ks[:-1]
    ks_exit = ks_entry*0
    ks_exit = ks[1:]

    dz = z_axis[1]-z_axis[0]

    line = xt.Line(elements=[xt.VariableSolenoid(length=dz,
                                        ks_profile=[ks_entry[ii], ks_exit[ii]])
                                for ii in range(len(z_axis)-1)])
    line.build_tracker()

    p_xt = p0.copy()
    line.track(p_xt, turn_by_turn_monitor='ONE_TURN_EBE')
    mon = line.record_last_track

    p_xt = p0.copy()
    line.configure_radiation(model=None)
    line.track(p_xt, turn_by_turn_monitor='ONE_TURN_EBE')
    mon_no_rad = line.record_last_track

    Bz_mid = 0.5 * (Bz_axis[:-1] + Bz_axis[1:])
    Bz_mon = 0 * Bz_axis
    Bz_mon[1:] = Bz_mid

    # Wolsky Eq. 3.114
    Ax = -0.5 * Bz_mon * mon.y
    Ay =  0.5 * Bz_mon * mon.x

    # Wolsky Eq. 2.74
    ax_ref = Ax * p0.q0 * qe / P0_J
    ay_ref = Ay * p0.q0 * qe / P0_J

    px_mech = mon.px - ax_ref
    py_mech = mon.py - ay_ref
    pz_mech = np.sqrt((1 + mon.delta)**2 - px_mech**2 - py_mech**2)

    xp = px_mech / pz_mech
    yp = py_mech / pz_mech

    dx_ds = np.diff(mon.x, axis=1) / np.diff(mon.s, axis=1)
    dy_ds = np.diff(mon.y, axis=1) / np.diff(mon.s, axis=1)

    z_check = sf.z0 + sf.L * np.linspace(-2, 2, 1001)

    for i_part in range(z_log.shape[1]):

        this_s_boris = 0.5 * (z_log[:-1, i_part] + z_log[1:, i_part])
        dx_ds_boris = np.diff(x_log[:, i_part]) / np.diff(z_log[:, i_part])
        dy_ds_boris = np.diff(y_log[:, i_part]) / np.diff(z_log[:, i_part])

        s_xsuite = 0.5 * (mon.s[i_part, :-1] + mon.s[i_part, 1:])
        dx_ds_xsuite = np.diff(mon.x[i_part, :]) / np.diff(mon.s[i_part, :])
        dy_ds_xsuite = np.diff(mon.y[i_part, :]) / np.diff(mon.s[i_part, :])

        dx_ds_xsuite_check = np.interp(z_check, s_xsuite, dx_ds_xsuite)
        dy_ds_xsuite_check = np.interp(z_check, s_xsuite, dy_ds_xsuite)

        dx_ds_boris_check = np.interp(z_check, this_s_boris, dx_ds_boris)
        dy_ds_boris_check = np.interp(z_check, this_s_boris, dy_ds_boris)

        this_dx_ds = dx_ds[i_part, :]
        this_dy_ds = dy_ds[i_part, :]

        xo.assert_allclose(dx_ds_xsuite_check, dx_ds_boris_check, rtol=0,
                atol=2.8e-2 * (np.max(dx_ds_boris_check) - np.min(dx_ds_boris_check)))
        xo.assert_allclose(dy_ds_xsuite_check, dy_ds_boris_check, rtol=0,
                atol=2.8e-2 * (np.max(dy_ds_boris_check) - np.min(dy_ds_boris_check)))

        xo.assert_allclose(ax_ref[i_part, :], mon.ax[i_part, :],
                        rtol=0, atol=np.max(np.abs(ax_ref)*3e-2))
        xo.assert_allclose(ay_ref[i_part, :], mon.ay[i_part, :],
                        rtol=0, atol=np.max(np.abs(ay_ref)*3e-2))