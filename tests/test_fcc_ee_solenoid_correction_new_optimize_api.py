import numpy as np
import xtrack as xt
from scipy.constants import c as clight
from scipy.constants import e as qe
import pathlib

from cpymad.madx import Madx


test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()


def test_fcc_ee_solenoid_correction_new_optimizer_api():
    fname = 'fccee_t'; pc_gev = 182.5

    mad = Madx(stdout=False)
    mad.call(str(test_data_folder) + '/fcc_ee/' + fname + '.seq')
    mad.beam(particle='positron', pc=pc_gev)
    mad.use('fccee_p_ring')

    line = xt.Line.from_madx_sequence(mad.sequence.fccee_p_ring, allow_thick=True,
                                    deferred_expressions=True)
    line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV,
                                    gamma0=mad.sequence.fccee_p_ring.beam.gamma)
    line.cycle('ip.4', inplace=True)
    line.append_element(element=xt.Marker(), name='ip.4.l')

    tt = line.get_table()
    bz_data_file = test_data_folder / 'fcc_ee/Bz_closed_before_quads.dat'

    line.vars['voltca1_ref'] = line.vv['voltca1']
    if 'voltca2' in line.vars.keys():
        line.vars['voltca2_ref'] = line.vv['voltca2']
    else:
        line.vars['voltca2_ref'] = 0

    line.vars['voltca1'] = 0
    line.vars['voltca2'] = 0

    import pandas as pd
    bz_df = pd.read_csv(bz_data_file, sep=r'\s+', skiprows=1, names=['z', 'Bz'])

    l_solenoid = 4.4
    ds_sol_start = -l_solenoid / 2 * np.cos(15e-3)
    ds_sol_end = +l_solenoid / 2 * np.cos(15e-3)
    ip_sol = 'ip.1'

    theta_tilt = 15e-3 # rad
    l_beam = 4.4
    l_solenoid = l_beam * np.cos(theta_tilt)
    ds_sol_start = -l_beam / 2
    ds_sol_end = +l_beam / 2
    ip_sol = 'ip.1'

    s_sol_slices = np.linspace(-l_solenoid/2, l_solenoid/2, 1001)
    bz_sol_slices = np.interp(s_sol_slices, bz_df.z, bz_df.Bz)
    bz_sol_slices[0] = 0
    bz_sol_slices[-1] = 0

    P0_J = line.particle_ref.p0c[0] * qe / clight
    brho = P0_J / qe / line.particle_ref.q0
    ks_entry = bz_sol_slices[:-1] / brho
    ks_exit = bz_sol_slices[1:] / brho
    l_sol_slices = np.diff(s_sol_slices)
    s_sol_slices_entry = s_sol_slices[:-1]

    sol_slices = []
    for ii in range(len(s_sol_slices_entry)):
        sol_slices.append(xt.VariableSolenoid(length=l_sol_slices[ii], ks_profile=[0, 0])) # Off for now

    s_ip = tt['s', ip_sol]

    line.discard_tracker()
    line.insert_element(name='sol_start_'+ip_sol, element=xt.Marker(),
                        at_s=s_ip + ds_sol_start)
    line.insert_element(name='sol_end_'+ip_sol, element=xt.Marker(),
                        at_s=s_ip + ds_sol_end)

    sol_start_tilt = xt.YRotation(angle=-theta_tilt * 180 / np.pi)
    sol_end_tilt = xt.YRotation(angle=+theta_tilt * 180 / np.pi)
    sol_start_shift = xt.XYShift(dx=l_solenoid/2 * np.tan(theta_tilt))
    sol_end_shift = xt.XYShift(dx=l_solenoid/2 * np.tan(theta_tilt))

    line.element_dict['sol_start_tilt_'+ip_sol] = sol_start_tilt
    line.element_dict['sol_end_tilt_'+ip_sol] = sol_end_tilt
    line.element_dict['sol_start_shift_'+ip_sol] = sol_start_shift
    line.element_dict['sol_end_shift_'+ip_sol] = sol_end_shift

    line.element_dict['sol_entry_'+ip_sol] = xt.Marker()
    line.element_dict['sol_exit_'+ip_sol] = xt.Marker()

    sol_slice_names = []
    sol_slice_names.append('sol_entry_'+ip_sol)
    for ii in range(len(s_sol_slices_entry)):
        nn = f'sol_slice_{ii}_{ip_sol}'
        line.element_dict[nn] = sol_slices[ii]
        sol_slice_names.append(nn)
    sol_slice_names.append('sol_exit_'+ip_sol)

    tt = line.get_table()
    names_upstream = list(tt.rows[:'sol_start_'+ip_sol].name)
    names_downstream = list(tt.rows['sol_end_'+ip_sol:].name[:-1]) # -1 to exclude '_end_point' added by the table

    element_names = (names_upstream
                    + ['sol_start_tilt_'+ip_sol, 'sol_start_shift_'+ip_sol]
                    + sol_slice_names
                    + ['sol_end_shift_'+ip_sol, 'sol_end_tilt_'+ip_sol]
                    + names_downstream)

    line.element_names = element_names

    # re-insert the ip
    line.element_dict.pop(ip_sol)
    tt = line.get_table()
    line.insert_element(name=ip_sol, element=xt.Marker(),
            at_s = 0.5 * (tt['s', 'sol_start_'+ip_sol] + tt['s', 'sol_end_'+ip_sol]))

    line.vars['on_corr_ip.1'] = 0

    line.build_tracker()

    # Set strength
    line.vars['on_sol_'+ip_sol] = 0
    for ii in range(len(s_sol_slices_entry)):
        nn = f'sol_slice_{ii}_{ip_sol}'
        line.element_refs[nn].ks_profile[0] = ks_entry[ii] * line.vars['on_sol_'+ip_sol]
        line.element_refs[nn].ks_profile[1] = ks_exit[ii] * line.vars['on_sol_'+ip_sol]


    tt = line.get_table()

    tt.rows['sol_start_ip.1':'sol_end_ip.1'].show()

    line.vars['on_corr_ip.1'] = 1
    line.vars['ks0.r1'] = 0
    line.vars['ks1.r1'] = 0
    line.vars['ks2.r1'] = 0
    line.vars['ks3.r1'] = 0
    line.vars['ks4.r1'] = 0
    line.vars['ks0.l1'] = 0
    line.vars['ks1.l1'] = 0
    line.vars['ks2.l1'] = 0
    line.vars['ks3.l1'] = 0
    line.vars['ks4.l1'] = 0

    line.element_refs['qc1r1.1'].k1s = line.vars['on_corr_ip.1'] * line.vars['ks0.r1']
    line.element_refs['qc2r1.1'].k1s = line.vars['on_corr_ip.1'] * line.vars['ks1.r1']
    line.element_refs['qc2r2.1'].k1s = line.vars['on_corr_ip.1'] * line.vars['ks2.r1']
    line.element_refs['qc1r2.1'].k1s = line.vars['on_corr_ip.1'] * line.vars['ks3.r1']
    line.element_refs['qc1l1.4'].k1s = line.vars['on_corr_ip.1'] * line.vars['ks0.l1']
    line.element_refs['qc2l1.4'].k1s = line.vars['on_corr_ip.1'] * line.vars['ks1.l1']
    line.element_refs['qc2l2.4'].k1s = line.vars['on_corr_ip.1'] * line.vars['ks2.l1']
    line.element_refs['qc1l2.4'].k1s = line.vars['on_corr_ip.1'] * line.vars['ks3.l1']

    line.vars['corr_k0.r1'] = 0
    line.vars['corr_k1.r1'] = 0
    line.vars['corr_k2.r1'] = 0
    line.vars['corr_k3.r1'] = 0
    line.vars['corr_k4.r1'] = 0
    line.vars['corr_k0.l1'] = 0
    line.vars['corr_k1.l1'] = 0
    line.vars['corr_k2.l1'] = 0
    line.vars['corr_k3.l1'] = 0
    line.vars['corr_k4.l1'] = 0

    line.element_refs['qc1r1.1'].k1 += line.vars['on_corr_ip.1'] * line.vars['corr_k0.r1']
    line.element_refs['qc2r1.1'].k1 += line.vars['on_corr_ip.1'] * line.vars['corr_k1.r1']
    line.element_refs['qc2r2.1'].k1 += line.vars['on_corr_ip.1'] * line.vars['corr_k2.r1']
    line.element_refs['qc1r2.1'].k1 += line.vars['on_corr_ip.1'] * line.vars['corr_k3.r1']
    line.element_refs['qc1l1.4'].k1 += line.vars['on_corr_ip.1'] * line.vars['corr_k0.l1']
    line.element_refs['qc2l1.4'].k1 += line.vars['on_corr_ip.1'] * line.vars['corr_k1.l1']
    line.element_refs['qc2l2.4'].k1 += line.vars['on_corr_ip.1'] * line.vars['corr_k2.l1']
    line.element_refs['qc1l2.4'].k1 += line.vars['on_corr_ip.1'] * line.vars['corr_k3.l1']


    Strategy = xt.Strategy
    Teapot = xt.Teapot
    slicing_strategies = [
        Strategy(slicing=None),  # Default catch-all as in MAD-X
        Strategy(slicing=Teapot(3), element_type=xt.Bend),
        Strategy(slicing=Teapot(3), element_type=xt.CombinedFunctionMagnet),
        # Strategy(slicing=Teapot(50), element_type=xt.Quadrupole), # Starting point
        Strategy(slicing=Teapot(5), name=r'^qf.*'),
        Strategy(slicing=Teapot(5), name=r'^qd.*'),
        Strategy(slicing=Teapot(5), name=r'^qfg.*'),
        Strategy(slicing=Teapot(5), name=r'^qdg.*'),
        Strategy(slicing=Teapot(5), name=r'^ql.*'),
        Strategy(slicing=Teapot(5), name=r'^qs.*'),
        Strategy(slicing=Teapot(10), name=r'^qb.*'),
        Strategy(slicing=Teapot(10), name=r'^qg.*'),
        Strategy(slicing=Teapot(10), name=r'^qh.*'),
        Strategy(slicing=Teapot(10), name=r'^qi.*'),
        Strategy(slicing=Teapot(10), name=r'^qr.*'),
        Strategy(slicing=Teapot(10), name=r'^qu.*'),
        Strategy(slicing=Teapot(10), name=r'^qy.*'),
        Strategy(slicing=Teapot(50), name=r'^qa.*'),
        Strategy(slicing=Teapot(50), name=r'^qc.*'),
        Strategy(slicing=Teapot(20), name=r'^sy\..*'),
        Strategy(slicing=Teapot(30), name=r'^mwi\..*'),
    ]
    line.discard_tracker()
    line.slice_thick_elements(slicing_strategies=slicing_strategies)

    # Add dipole correctors
    line.insert_element(name='mcb1.r1', element=xt.Multipole(knl=[0]),
                        at='qc1r1.1_exit')
    line.insert_element(name='mcb2.r1', element=xt.Multipole(knl=[0]),
                        at='qc1r2.1_exit')
    line.insert_element(name='mcb1.l1', element=xt.Multipole(knl=[0]),
                        at='qc1l1.4_entry')
    line.insert_element(name='mcb2.l1', element=xt.Multipole(knl=[0]),
                        at='qc1l2.4_entry')

    line.vars['acb1h.r1'] = 0
    line.vars['acb1v.r1'] = 0
    line.vars['acb2h.r1'] = 0
    line.vars['acb2v.r1'] = 0
    line.vars['acb1h.l1'] = 0
    line.vars['acb1v.l1'] = 0
    line.vars['acb2h.l1'] = 0
    line.vars['acb2v.l1'] = 0

    line.element_refs['mcb1.r1'].knl[0] = line.vars['on_corr_ip.1']*line.vars['acb1h.r1']
    line.element_refs['mcb2.r1'].knl[0] = line.vars['on_corr_ip.1']*line.vars['acb2h.r1']
    line.element_refs['mcb1.r1'].ksl[0] = line.vars['on_corr_ip.1']*line.vars['acb1v.r1']
    line.element_refs['mcb2.r1'].ksl[0] = line.vars['on_corr_ip.1']*line.vars['acb2v.r1']
    line.element_refs['mcb1.l1'].knl[0] = line.vars['on_corr_ip.1']*line.vars['acb1h.l1']
    line.element_refs['mcb2.l1'].knl[0] = line.vars['on_corr_ip.1']*line.vars['acb2h.l1']
    line.element_refs['mcb1.l1'].ksl[0] = line.vars['on_corr_ip.1']*line.vars['acb1v.l1']
    line.element_refs['mcb2.l1'].ksl[0] = line.vars['on_corr_ip.1']*line.vars['acb2v.l1']

    tw_thick_no_rad = line.twiss(method='4d')

    assert line.element_names[-1] == 'ip.4.l'
    assert line.element_names[0] == 'ip.4'

    opt = line.match(
        method='4d',
        start='ip.4', end='ip.4.l',
        init=tw_thick_no_rad,
        vary=xt.VaryList(['k1qf4', 'k1qf2', 'k1qd3', 'k1qd1',], step=1e-8,
        ),
        targets=[
            xt.TargetSet(at=xt.END, mux=tw_thick_no_rad.qx, muy=tw_thick_no_rad.qy, tol=1e-5),
        ]
    )
    opt.solve()

    line.vars['on_sol_ip.1'] = 0
    tw_sol_off = line.twiss(method='4d')
    line.vars['on_sol_ip.1'] = 1
    tw_sol_on = line.twiss(method='4d')

    opt_l = line.match(
        solve=False,
        method='4d', n_steps_max=30,
        start='pqc2le.4', end='ip.1', init=tw_sol_off, init_at=xt.START,
        vary=[
            xt.VaryList(['acb1h.l1', 'acb2h.l1','acb1v.l1', 'acb2v.l1'], step=1e-8, tag='corr_l'),
            xt.VaryList(['ks1.l1', 'ks2.l1', 'ks3.l1', 'ks0.l1'], step=1e-7, tag='skew_l'),
            xt.VaryList(['corr_k1.l1', 'corr_k2.l1', 'corr_k3.l1', 'corr_k0.l1'], step=1e-6, tag='normal_l'),
        ],
        targets=[
            xt.TargetSet(['x', 'y'], value=tw_sol_off, tol=1e-7, at='ip.1', tag='orbit'),
            xt.TargetSet(['px', 'py'], value=tw_sol_off, tol=1e-10, at='ip.1', tag='orbit'),
            xt.TargetRmatrix(
                        r13=0, r14=0, r23=0, r24=0, # Y-X block
                        r31=0, r32=0, r41=0, r42=0, # X-Y block,
                        start='pqc2le.4', end='ip.1', tol=1e-6, tag='coupl'),
            xt.Target('mux', value=tw_sol_off, at='ip.1', tag='mu_ip', weight=0.1, tol=1e-6),
            xt.Target('muy', value=tw_sol_off, at='ip.1', tag='mu_ip', weight=0.1, tol=1e-6),
            xt.Target('betx', value=tw_sol_off, at='ip.1', tag='bet_ip', weight=1, tol=1e-5),
            xt.Target('bety', value=tw_sol_off, at='ip.1', tag='bet_ip', weight=10, tol=1e-6),
            xt.Target('alfx', value=tw_sol_off, at='ip.1', tag='alf_ip', weight=0.1, tol=1e-4),
            xt.Target('alfy', value=tw_sol_off, at='ip.1', tag='alf_ip', weight=0.1, tol=1e-4),

        ]
    )


    for iter in range(2):
        # Orbit alone
        opt_l.disable(target=True); opt_l.disable(vary=True)
        opt_l.enable(target='orbit'); opt_l.enable(vary='corr_l'); opt_l.solve()

        # Coupling alone
        opt_l.disable(target=True); opt_l.disable(vary=True)
        opt_l.enable(target='coupl'); opt_l.enable(vary='skew_l'); opt_l.solve()

        # phase, beta and alpha alone
        opt_l.disable(target=True); opt_l.disable(vary=True)
        opt_l.enable(vary='normal_l')
        opt_l.enable(target='mu_ip'); opt_l.solve()
        opt_l.enable(target='bet_ip'); opt_l.solve()
        opt_l.enable(target='alf_ip'); opt_l.solve()

    # All together
    opt_l.enable(target=True)
    opt_l.enable(vary=True)
    opt_l.solve()


    opt_r = line.match(
        solve=False,
        method='4d', n_steps_max=30,
        start='ip.1', end='pqc2re.1', init=tw_sol_off, init_at=xt.END,
        vary=[
            xt.VaryList(['acb1h.r1', 'acb2h.r1','acb1v.r1', 'acb2v.r1'], step=1e-8, tag='corr_r'),
            xt.VaryList(['ks1.r1', 'ks2.r1', 'ks3.r1', 'ks0.r1'], step=1e-7, tag='skew_r'),
            xt.VaryList(['corr_k1.r1', 'corr_k2.r1', 'corr_k3.r1', 'corr_k0.r1'], step=1e-6, tag='normal_r'),
        ],
        targets=[
            xt.TargetSet(['x', 'y'], value=tw_sol_off, tol=1e-7, at='ip.1', tag='orbit'),
            xt.TargetSet(['px', 'py'], value=tw_sol_off, tol=1e-10, at='ip.1', tag='orbit'),
            xt.TargetRmatrix(r13=0, r14=0, r23=0, r24=0, # Y-X block
                            r31=0, r32=0, r41=0, r42=0, # X-Y block,
                            start='ip.1', end='pqc2re.1', tol=1e-6, tag='coupl'),
            xt.Target('mux', value=tw_sol_off, at='ip.1', tag='mu_ip', weight=0.1, tol=1e-6),
            xt.Target('muy', value=tw_sol_off, at='ip.1', tag='mu_ip', weight=0.1, tol=1e-6),
            xt.Target('betx', value=tw_sol_off, at='ip.1', tag='bet_ip', weight=1, tol=1e-5),
            xt.Target('bety', value=tw_sol_off, at='ip.1', tag='bet_ip', weight=10, tol=1e-6),
            xt.Target('alfx', value=tw_sol_off, at='ip.1', tag='alf_ip', weight=0.1, tol=1e-4),
            xt.Target('alfy', value=tw_sol_off, at='ip.1', tag='alf_ip', weight=0.1, tol=1e-4),

        ]
    )

    for iter in range(2):
        # Orbit alone
        opt_r.disable(target=True); opt_r.disable(vary=True)
        opt_r.enable(target='orbit'); opt_r.enable(vary='corr_r'); opt_r.solve()

        # Coupling alone
        opt_r.disable(target=True); opt_r.disable(vary=True)
        opt_r.enable(target='coupl'); opt_r.enable(vary='skew_r'); opt_r.solve()

        # phase, beta and alpha alone
        opt_r.disable(target=True); opt_r.disable(vary=True)
        opt_r.enable(vary='normal_r')
        opt_r.enable(target='mu_ip'); opt_r.solve()
        opt_r.enable(target='bet_ip'); opt_r.solve()
        opt_r.enable(target='alf_ip'); opt_r.solve()

    # All together
    opt_r.enable(target=True)
    opt_r.enable(vary=True)
    opt_r.solve()

    line.to_json(fname + '_with_sol_corrected.json')

    tw_sol_on_corrected = line.twiss(method='4d')

    assert_allclose = np.testing.assert_allclose

    # Check that tilt is present
    assert_allclose(tw_sol_off['kin_xprime', 'ip.1'], np.tan(0.015), atol=1e-14, rtol=0)

    # Check that solenoid introduces coupling
    assert tw_sol_on.c_minus > 1e-4

    # Check correction
    tw_chk = tw_sol_on_corrected

    assert_allclose(tw_chk['x', 'ip.1'], 0, atol=1e-8, rtol=0)
    assert_allclose(tw_chk['y', 'ip.1'], 0, atol=1e-10, rtol=0)
    assert_allclose(tw_chk['kin_xprime', 'ip.1'], tw_sol_off['kin_xprime', 'ip.1'],  atol=1e-9, rtol=0)
    assert_allclose(tw_chk['kin_yprime', 'ip.1'], 0,  atol=1e-8, rtol=0)
    assert_allclose(tw_chk['x', 'pqc2re.1'], 0, atol=5e-8, rtol=0)
    assert_allclose(tw_chk['y', 'pqc2re.1'], 0, atol=5e-8, rtol=0)
    assert_allclose(tw_chk['kin_xprime', 'pqc2re.1'], 0, atol=1e-8, rtol=0)
    assert_allclose(tw_chk['kin_yprime', 'pqc2re.1'], 0, atol=1e-8, rtol=0)
    assert_allclose(tw_chk['x', 'pqc2le.4'], 0, atol=5e-8, rtol=0)
    assert_allclose(tw_chk['y', 'pqc2le.4'], 0, atol=5e-8, rtol=0)
    assert_allclose(tw_chk['kin_xprime', 'pqc2le.4'], 0, atol=1e-8, rtol=0)
    assert_allclose(tw_chk['kin_yprime', 'pqc2le.4'], 0, atol=1e-8, rtol=0)

    assert_allclose(tw_chk['betx', 'ip.1'], tw_sol_off['betx', 'ip.1'], atol=0, rtol=5e-5)
    assert_allclose(tw_chk['bety', 'ip.1'], tw_sol_off['bety', 'ip.1'], atol=0, rtol=5e-5)
    assert_allclose(tw_chk['alfx', 'ip.1'], tw_sol_off['alfx', 'ip.1'], atol=1e-5, rtol=0)
    assert_allclose(tw_chk['alfy', 'ip.1'], tw_sol_off['alfy', 'ip.1'], atol=1e-5, rtol=0)
    assert_allclose(tw_chk['mux', 'ip.1'], tw_sol_off['mux', 'ip.1'], atol=2e-6, rtol=0)
    assert_allclose(tw_chk['muy', 'ip.1'], tw_sol_off['muy', 'ip.1'], atol=2e-6, rtol=0)

    assert_allclose(tw_chk['betx', 'pqc2re.1'], tw_sol_off['betx', 'pqc2re.1'], atol=0, rtol=5e-5)
    assert_allclose(tw_chk['bety', 'pqc2re.1'], tw_sol_off['bety', 'pqc2re.1'], atol=0, rtol=5e-5)
    assert_allclose(tw_chk['alfx', 'pqc2re.1'], tw_sol_off['alfx', 'pqc2re.1'], atol=1e-5, rtol=5e-5)
    assert_allclose(tw_chk['alfy', 'pqc2re.1'], tw_sol_off['alfy', 'pqc2re.1'], atol=1e-5, rtol=5e-5)
    assert_allclose(tw_chk['mux', 'pqc2re.1'], tw_sol_off['mux', 'pqc2re.1'], atol=2e-6, rtol=5e-5)
    assert_allclose(tw_chk['muy', 'pqc2re.1'], tw_sol_off['muy', 'pqc2re.1'], atol=2e-6, rtol=5e-5)

    assert_allclose(tw_chk['betx', 'pqc2le.4'], tw_sol_off['betx', 'pqc2le.4'], atol=0, rtol=5e-5)
    assert_allclose(tw_chk['bety', 'pqc2le.4'], tw_sol_off['bety', 'pqc2le.4'], atol=0, rtol=5e-5)
    assert_allclose(tw_chk['alfx', 'pqc2le.4'], tw_sol_off['alfx', 'pqc2le.4'], atol=1e-5, rtol=5e-5)
    assert_allclose(tw_chk['alfy', 'pqc2le.4'], tw_sol_off['alfy', 'pqc2le.4'], atol=1e-5, rtol=5e-5)
    assert_allclose(tw_chk['mux', 'pqc2le.4'], tw_sol_off['mux', 'pqc2le.4'], atol=2e-6, rtol=5e-5)
    assert_allclose(tw_chk['muy', 'pqc2le.4'], tw_sol_off['muy', 'pqc2le.4'], atol=2e-6, rtol=5e-5)

    assert tw_chk.c_minus < 1e-6
    assert_allclose(tw_chk['betx2', 'ip.1'] / tw_chk['betx', 'ip.1'], 0, atol=1e-10)
    assert_allclose(tw_chk['bety1', 'ip.1'] / tw_chk['bety', 'ip.1'], 0, atol=1e-10)
    assert_allclose(tw_chk['betx2', 'pqc2re.1'] / tw_chk['betx', 'pqc2re.1'], 0, atol=1e-10)
    assert_allclose(tw_chk['bety1', 'pqc2re.1'] / tw_chk['bety', 'pqc2re.1'], 0, atol=1e-10)
    assert_allclose(tw_chk['betx2', 'pqc2le.4'] / tw_chk['betx', 'pqc2le.4'], 0, atol=1e-10)
    assert_allclose(tw_chk['bety1', 'pqc2le.4'] / tw_chk['bety', 'pqc2le.4'], 0, atol=1e-10)
