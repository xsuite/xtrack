import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe
from scipy.constants import epsilon_0 as eps0

import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField


def test_solenoid_bz_map_vs_boris():
    ctx = xo.ContextCpu()

    boris_knl_description = xo.Kernel(
        c_name='boris_step',
        args=[
            xo.Arg(xo.Int64,   name='N_sub_steps'),
            xo.Arg(xo.Float64, name='Dtt'),
            xo.Arg(xo.Float64, name='B_field', pointer=True),
            xo.Arg(xo.Float64, name='B_skew', pointer=True),
            xo.Arg(xo.Float64, name='xn1', pointer=True),
            xo.Arg(xo.Float64, name='yn1', pointer=True),
            xo.Arg(xo.Float64, name='zn1', pointer=True),
            xo.Arg(xo.Float64, name='vxn1', pointer=True),
            xo.Arg(xo.Float64, name='vyn1', pointer=True),
            xo.Arg(xo.Float64, name='vzn1', pointer=True),
            xo.Arg(xo.Float64, name='Ex_n', pointer=True),
            xo.Arg(xo.Float64, name='Ey_n', pointer=True),
            xo.Arg(xo.Float64, name='Bx_n_custom', pointer=True),
            xo.Arg(xo.Float64, name='By_n_custom', pointer=True),
            xo.Arg(xo.Float64, name='Bz_n_custom', pointer=True),
            xo.Arg(xo.Int64,   name='custom_B'),
            xo.Arg(xo.Int64,   name='N_mp'),
            xo.Arg(xo.Int64,   name='N_multipoles'),
            xo.Arg(xo.Float64, name='charge'),
            xo.Arg(xo.Float64, name='mass', pointer=True),
        ],
    )

    ctx.add_kernels(
        kernels={'boris': boris_knl_description},
        sources=[xt._pkg_root / '_temp/boris_and_solenoid_map/boris.h'],
    )

    delta=np.array([0, 4])
    p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                    energy0=45.6e9,
                    x=[-1e-3, -1e-3], px=-1e-3*(1+delta), y=1e-3,
                    delta=delta)

    p = p0.copy()

    sf = SolenoidField(L=4, a=0.3, B0=1.5, z0=20)

    dt = 1e-10
    n_steps = 1500

    x_log = []
    y_log = []
    z_log = []
    px_log = []
    py_log = []
    pp_log = []
    beta_x_log = []
    beta_y_log = []
    beta_z_log = []

    for ii in range(n_steps):

        x = p.x.copy()
        y = p.y.copy()
        z = p.s.copy()



        gamma = p.energy / p.mass0
        mass0_kg = p.mass0 * qe / clight**2
        charge0_coulomb = p.q0 * qe

        p0c_J = p.p0c * qe

        Pxc_J = p.px * p0c_J
        Pyc_J = p.py * p0c_J
        Pzc_J = np.sqrt((p0c_J*(1 + p.delta))**2 - Pxc_J**2 - Pyc_J**2)

        vx = Pxc_J / clight / (gamma * mass0_kg) # m/s
        vy = Pyc_J / clight / (gamma * mass0_kg) # m/s
        vz = Pzc_J / clight / (gamma * mass0_kg) # m/s

        Bx, By, Bz = sf.get_field(x + vx * dt / 2,
                                    y + vy * dt / 2,
                                    z + vz * dt / 2)

        ctx.kernels.boris(
                N_sub_steps=1,
                Dtt=dt,
                B_field=np.array([0.]),
                B_skew=np.array([0.]),
                xn1=x,
                yn1=y,
                zn1=z,
                vxn1=vx,
                vyn1=vy,
                vzn1=vz,
                Ex_n=0 * x,
                Ey_n=0 * x,
                Bx_n_custom=Bx,
                By_n_custom=By,
                Bz_n_custom=Bz,
                custom_B=1,
                N_mp=len(x),
                N_multipoles=0,
                charge=charge0_coulomb,
                mass=mass0_kg * gamma,
        )

        p.x = x
        p.y = y
        p.s = z
        p.px = mass0_kg * gamma * vx * clight / p0c_J
        p.py = mass0_kg * gamma * vy * clight / p0c_J
        pz = mass0_kg * gamma * vz * clight / p0c_J
        pp = np.sqrt(p.px**2 + p.py**2 + pz**2)

        beta_x_after = vx / clight
        beta_y_after = vy / clight
        beta_z_after = vz / clight


        x_log.append(p.x.copy())
        y_log.append(p.y.copy())
        z_log.append(p.s.copy())
        px_log.append(p.px.copy())
        py_log.append(p.py.copy())
        pp_log.append(pp)
        beta_x_log.append(beta_x_after)
        beta_y_log.append(beta_y_after)
        beta_z_log.append(beta_z_after)

    x_log = np.array(x_log)
    y_log = np.array(y_log)
    z_log = np.array(z_log)
    px_log = np.array(px_log)
    py_log = np.array(py_log)
    pp_log = np.array(pp_log)
    beta_x_log = np.array(beta_x_log)
    beta_y_log = np.array(beta_y_log)
    beta_z_log = np.array(beta_z_log)

    # Compute beta dot
    beta_x_dot = 0 * beta_x_log
    beta_y_dot = 0 * beta_y_log
    beta_z_dot = 0 * beta_z_log
    beta_x_dot[1:-1] = (beta_x_log[2:] - beta_x_log[:-2]) / 2 / dt
    beta_y_dot[1:-1] = (beta_y_log[2:] - beta_y_log[:-2]) / 2 / dt
    beta_z_dot[1:-1] = (beta_z_log[2:] - beta_z_log[:-2]) / 2 / dt
    beta_dot_square = beta_x_dot**2 + beta_y_dot**2 + beta_z_dot**2

    # From Hofmann, "The physics of synchrontron radiation" Eq 3.7 and below
    pow_log = 2 * qe**2 * beta_dot_square * gamma**4 / (12 * np.pi * eps0 * clight)

    dE_ds_boris_J = 0 * pow_log

    dE_ds_boris_J[:-1]= pow_log[:-1] * dt / np.diff(z_log, axis=0)
    dE_ds_boris_eV = dE_ds_boris_J / qe

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
    line.configure_radiation(model='mean')

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

    dE_ds = 0*mon.ptau
    # Central differences
    dE_ds[:, 1:-1] = -((mon.ptau[:, 2:] - mon.ptau[:, :-2]) / (mon.s[:, 2:] - mon.s[:, :-2])
                            * p_xt.energy0[0])

    emitted_dpx = -(np.diff(mon.kin_px, axis=1) - np.diff(mon_no_rad.kin_px, axis=1))
    emitted_dpy = -(np.diff(mon.kin_py, axis=1) - np.diff(mon_no_rad.kin_py, axis=1))
    emitted_dp = -(np.diff(mon.delta, axis=1) - np.diff(mon_no_rad.delta, axis=1))

    z_check = sf.z0 + sf.L * np.linspace(-2, 2, 1001)

    for i_part in range(z_log.shape[1]):

        this_s_boris = 0.5 * (z_log[:-1, i_part] + z_log[1:, i_part])
        dx_ds_boris = np.diff(x_log[:, i_part]) / np.diff(z_log[:, i_part])
        dy_ds_boris = np.diff(y_log[:, i_part]) / np.diff(z_log[:, i_part])

        s_xsuite = 0.5 * (mon.s[i_part, :-1] + mon.s[i_part, 1:])
        dx_ds_xsuite = np.diff(mon.x[i_part, :]) / np.diff(mon.s[i_part, :])
        dy_ds_xsuite = np.diff(mon.y[i_part, :]) / np.diff(mon.s[i_part, :])
        dE_ds_xsuite = dE_ds[i_part, :]

        dx_ds_xsuite_check = np.interp(z_check, s_xsuite, dx_ds_xsuite)
        dy_ds_xsuite_check = np.interp(z_check, s_xsuite, dy_ds_xsuite)
        dE_ds_xsuite_check = np.interp(z_check, mon.s[i_part, :], dE_ds_xsuite)

        dx_ds_boris_check = np.interp(z_check, this_s_boris, dx_ds_boris)
        dy_ds_boris_check = np.interp(z_check, this_s_boris, dy_ds_boris)
        dE_ds_boris_check = np.interp(z_check, z_log[:, i_part], dE_ds_boris_eV[:, i_part])

        this_emitted_dpx = emitted_dpx[i_part, :]
        this_emitted_dpy = emitted_dpy[i_part, :]
        this_dE_ds = dE_ds[i_part, :]
        this_dx_ds = dx_ds[i_part, :]
        this_dy_ds = dy_ds[i_part, :]

        xo.assert_allclose(dx_ds_xsuite_check, dx_ds_boris_check, rtol=0,
                atol=2.8e-2 * (np.max(dx_ds_boris_check) - np.min(dx_ds_boris_check)))
        xo.assert_allclose(dy_ds_xsuite_check, dy_ds_boris_check, rtol=0,
                atol=2.8e-2 * (np.max(dy_ds_boris_check) - np.min(dy_ds_boris_check)))
        xo.assert_allclose(dE_ds_xsuite_check, dE_ds_boris_check, rtol=0,
                atol=1.5e-2 * (np.max(dE_ds_boris_check) - np.min(dE_ds_boris_check)))

        xo.assert_allclose(ax_ref[i_part, :], mon.ax[i_part, :],
                        rtol=0, atol=np.max(np.abs(ax_ref)*3e-2))
        xo.assert_allclose(ay_ref[i_part, :], mon.ay[i_part, :],
                        rtol=0, atol=np.max(np.abs(ay_ref)*3e-2))

        xo.assert_allclose(this_emitted_dpx,
                0.5 * (this_dE_ds[:-1] + this_dE_ds[1:]) * this_dx_ds * np.diff(mon.s[i_part, :])/p.p0c[0],
                rtol=0, atol=2e-2 * (np.max(this_emitted_dpx) - np.min(this_emitted_dpx)))
        xo.assert_allclose(this_emitted_dpy,
                0.5 * (this_dE_ds[:-1] + this_dE_ds[1:]) * this_dy_ds * np.diff(mon.s[i_part, :])/p.p0c[0],
                rtol=0, atol=5e-2 * (np.max(this_emitted_dpy) - np.min(this_emitted_dpy)))
