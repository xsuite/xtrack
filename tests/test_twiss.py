import json
import pathlib
from itertools import product

import numpy as np
import pytest
from cpymad.madx import Madx

import xobjects as xo
import xpart as xp
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()

@for_all_test_contexts
def test_twiss_4d_fodo_vs_beta_rel(test_context):
    ## Generate a simple line
    n = 6
    fodo = [
        xt.Multipole(length=0.2, knl=[0, +0.2], ksl=[0, 0]),
        xt.Drift(length=1.0),
        xt.Multipole(length=0.2, knl=[0, -0.2], ksl=[0, 0]),
        xt.Drift(length=1.0),
        xt.Multipole(length=1.0, knl=[2 * np.pi / n], hxl=[2 * np.pi / n]),
        xt.Drift(length=1.0),
    ]
    line = xt.Line(elements=n * fodo + [xt.Cavity(frequency=1e9, voltage=0, lag=180)])
    line.build_tracker(_context=test_context)

    ## Twiss
    p0c_list = [1e8, 1e9, 1e10, 1e11, 1e12]
    tw_4d_list = []
    for p0c in p0c_list:
        line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=p0c)
        tw = line.twiss(method="4d", at_s=np.linspace(0, line.get_length(), 500))
        tw_4d_list.append(tw)

    for tw in tw_4d_list:
        xo.assert_allclose(tw.betx, tw_4d_list[0].betx, atol=1e-7, rtol=0)
        xo.assert_allclose(tw.bety, tw_4d_list[0].bety, atol=1e-7, rtol=0)
        xo.assert_allclose(tw.alfx, tw_4d_list[0].alfx, atol=1e-8, rtol=0)
        xo.assert_allclose(tw.alfy, tw_4d_list[0].alfy, atol=1e-8, rtol=0)
        xo.assert_allclose(tw.dx, tw_4d_list[0].dx, atol=1e-8, rtol=0)
        xo.assert_allclose(tw.dy, tw_4d_list[0].dy, atol=1e-8, rtol=0)
        xo.assert_allclose(tw.dpx, tw_4d_list[0].dpx, atol=1e-8, rtol=0)
        xo.assert_allclose(tw.dpy, tw_4d_list[0].dpy, atol=1e-8, rtol=0)
        xo.assert_allclose(tw.qx, tw_4d_list[0].qx, atol=1e-7, rtol=0)
        xo.assert_allclose(tw.qy, tw_4d_list[0].qy, atol=1e-7, rtol=0)
        xo.assert_allclose(tw.dqx, tw_4d_list[0].dqx, atol=1e-4, rtol=0)
        xo.assert_allclose(tw.dqy, tw_4d_list[0].dqy, atol=1e-4, rtol=0)

@for_all_test_contexts
def test_coupled_beta(test_context):
    mad = Madx(stdout=False)
    mad.call(str(test_data_folder / 'hllhc15_noerrors_nobb/sequence.madx'))
    mad.use('lhcb1')

    # introduce coupling
    mad.sequence.lhcb1.expanded_elements[7].ksl = [0, 1e-4]
    mad.twiss() # I see to need to do it twice to get the right coupling in madx?!

    tw_mad_coupling = mad.twiss(ripken=True).dframe()
    tw_mad_coupling.set_index('name', inplace=True)

    line = xt.Line.from_madx_sequence(mad.sequence.lhcb1)
    line.particle_ref = xp.Particles(p0c=7000e9, mass0=xp.PROTON_MASS_EV)

    line.build_tracker(_context=test_context)

    tw6d = line.twiss()
    tw4d = line.twiss(method='4d')

    for tw in (tw6d, tw4d):

        twdf = tw.to_pandas()
        twdf.set_index('name', inplace=True)

        ips = ['ip1', 'ip2', 'ip3', 'ip4', 'ip5', 'ip6', 'ip7', 'ip8']
        betx2_at_ips = twdf.loc[ips, 'betx2'].values
        bety1_at_ips = twdf.loc[ips, 'bety1'].values

        beta12_mad_at_ips = tw_mad_coupling.loc[[ip + ':1' for ip in ips], 'beta12'].values
        beta21_mad_at_ips = tw_mad_coupling.loc[[ip + ':1' for ip in ips], 'beta21'].values

        xo.assert_allclose(betx2_at_ips, beta12_mad_at_ips, rtol=1e-4, atol=0)
        xo.assert_allclose(bety1_at_ips, beta21_mad_at_ips, rtol=1e-4, atol=0)

        #cmin_ref = mad.table.summ.dqmin[0] # dqmin is not calculated correctly in madx
                                            # (https://github.com/MethodicalAcceleratorDesign/MAD-X/issues/1152)
        cmin_ref = 0.001972093557# obtained with madx with trial and error

        xo.assert_allclose(tw.c_minus, cmin_ref, rtol=0, atol=1e-5)


@for_all_test_contexts
def test_twiss_zeta0_delta0(test_context):
    mad = Madx(stdout=False)
    mad.call(str(test_data_folder
                 / 'hllhc15_noerrors_nobb/sequence_with_crabs.madx'))
    mad.use('lhcb1')
    mad.globals.on_crab1 = -190
    mad.globals.on_crab5 = -190

    line = xt.Line.from_madx_sequence(mad.sequence.lhcb1)
    line.particle_ref = xp.Particles(p0c=7000e9, mass0=xp.PROTON_MASS_EV)

    line.build_tracker(_context=test_context)

    # Measure crabbing angle at IP1 and IP5
    z1 = 1e-4
    z2 = -1e-4

    tw1 = line.twiss(zeta0=z1).to_pandas()
    tw2 = line.twiss(zeta0=z2).to_pandas()

    tw1.set_index('name', inplace=True)
    tw2.set_index('name', inplace=True)

    phi_c_ip1 = ((tw1.loc['ip1', 'x'] - tw2.loc['ip1', 'x'])
                 / (tw1.loc['ip1', 'zeta'] - tw2.loc['ip1', 'zeta']))

    phi_c_ip5 = ((tw1.loc['ip5', 'y'] - tw2.loc['ip5', 'y'])
                 / (tw1.loc['ip5', 'zeta'] - tw2.loc['ip5', 'zeta']))

    xo.assert_allclose(phi_c_ip1, -190e-6, atol=1e-7, rtol=0)
    xo.assert_allclose(phi_c_ip5, -190e-6, atol=1e-7, rtol=0)

    # Check crab dispersion
    tw6d = line.twiss()
    xo.assert_allclose(tw6d['dx_zeta', 'ip1'], -190e-6, atol=1e-7, rtol=0)
    xo.assert_allclose(tw6d['dy_zeta', 'ip5'], -190e-6, atol=1e-7, rtol=0)

@for_all_test_contexts
def test_get_normalized_coordinates(test_context):

    path_line_particles = test_data_folder / 'hllhc15_noerrors_nobb/line_and_particle.json'

    with open(path_line_particles, 'r') as fid:
        input_data = json.load(fid)
    line = xt.Line.from_dict(input_data['line'])
    line.particle_ref = xp.Particles.from_dict(input_data['particle'])

    line.build_tracker(_context=test_context)

    particles = line.build_particles(
        nemitt_x=2.5e-6, nemitt_y=1e-6,
        x_norm=[-1, 0, 0.5], y_norm=[0.3, -0.2, 0.2],
        px_norm=[0.1, 0.2, 0.3], py_norm=[0.5, 0.6, 0.8],
        zeta=[0, 0.1, -0.1], delta=[1e-4, 0., -1e-4])

    tw = line.twiss()

    norm_coord = tw.get_normalized_coordinates(particles, nemitt_x=2.5e-6,
                                            nemitt_y=1e-6)

    xo.assert_allclose(norm_coord['x_norm'], [-1, 0, 0.5], atol=1e-10, rtol=0)
    xo.assert_allclose(norm_coord['y_norm'], [0.3, -0.2, 0.2], atol=1e-10, rtol=0)
    xo.assert_allclose(norm_coord['px_norm'], [0.1, 0.2, 0.3], atol=1e-10, rtol=0)
    xo.assert_allclose(norm_coord['py_norm'], [0.5, 0.6, 0.8], atol=1e-10, rtol=0)


    # Introduce a non-zero closed orbit
    line['mqwa.a4r3.b1..1'].knl[0] = 10e-6
    line['mqwa.a4r3.b1..1'].ksl[0] = 5e-6

    particles1 = line.build_particles(
        nemitt_x=2.5e-6, nemitt_y=1e-6,
        x_norm=[-1, 0, 0.5], y_norm=[0.3, -0.2, 0.2],
        px_norm=[0.1, 0.2, 0.3], py_norm=[0.5, 0.6, 0.8],
        zeta=[0, 0.1, -0.1], delta=[1e-4, 0., -1e-4])

    tw1 = line.twiss()
    norm_coord1 = tw1.get_normalized_coordinates(particles1, nemitt_x=2.5e-6,
                                                nemitt_y=1e-6)

    xo.assert_allclose(norm_coord1['x_norm'], [-1, 0, 0.5], atol=1e-10, rtol=0)
    xo.assert_allclose(norm_coord1['y_norm'], [0.3, -0.2, 0.2], atol=1e-10, rtol=0)
    xo.assert_allclose(norm_coord1['px_norm'], [0.1, 0.2, 0.3], atol=1e-10, rtol=0)
    xo.assert_allclose(norm_coord1['py_norm'], [0.5, 0.6, 0.8], atol=1e-10, rtol=0)

    # Check computation at different locations

    particles2 = line.build_particles(at_element='s.ds.r3.b1',
        _capacity=10,
        nemitt_x=2.5e-6, nemitt_y=1e-6,
        x_norm=[-1, 0, 0.5], y_norm=[0.3, -0.2, 0.2],
        px_norm=[0.1, 0.2, 0.3], py_norm=[0.5, 0.6, 0.8],
        zeta=[0, 0.1, -0.1], delta=[1e-4, 0., -1e-4])

    particles3 = line.build_particles(at_element='s.ds.r7.b1',
        _capacity=10,
        nemitt_x=2.5e-6, nemitt_y=1e-6,
        x_norm=[-1, 0, 0.5], y_norm=[0.3, -0.2, 0.2],
        px_norm=[0.1, 0.2, 0.3], py_norm=[0.5, 0.6, 0.8],
        zeta=[0, 0.1, -0.1], delta=[1e-4, 0., -1e-4])

    particles23 = xp.Particles.merge([particles2, particles3])

    norm_coord23 = tw1.get_normalized_coordinates(particles23, nemitt_x=2.5e-6,
                                                nemitt_y=1e-6)

    assert particles23._capacity == 20
    xo.assert_allclose(norm_coord23['x_norm'][:3], [-1, 0, 0.5], atol=1e-10, rtol=0)
    xo.assert_allclose(norm_coord23['x_norm'][3:6], [-1, 0, 0.5], atol=1e-10, rtol=0)
    xo.assert_allclose(norm_coord23['x_norm'][6:], xt.particles.LAST_INVALID_STATE)
    xo.assert_allclose(norm_coord23['y_norm'][:3], [0.3, -0.2, 0.2], atol=1e-10, rtol=0)
    xo.assert_allclose(norm_coord23['y_norm'][3:6], [0.3, -0.2, 0.2], atol=1e-10, rtol=0)
    xo.assert_allclose(norm_coord23['y_norm'][6:], xt.particles.LAST_INVALID_STATE)
    xo.assert_allclose(norm_coord23['px_norm'][:3], [0.1, 0.2, 0.3], atol=1e-10, rtol=0)
    xo.assert_allclose(norm_coord23['px_norm'][3:6], [0.1, 0.2, 0.3], atol=1e-10, rtol=0)
    xo.assert_allclose(norm_coord23['px_norm'][6:], xt.particles.LAST_INVALID_STATE)
    xo.assert_allclose(norm_coord23['py_norm'][:3], [0.5, 0.6, 0.8], atol=1e-10, rtol=0)
    xo.assert_allclose(norm_coord23['py_norm'][3:6], [0.5, 0.6, 0.8], atol=1e-10, rtol=0)
    xo.assert_allclose(norm_coord23['py_norm'][6:], xt.particles.LAST_INVALID_STATE)

    particles23.move(_context=xo.context_default)
    assert np.all(particles23.at_element[:3] == line.element_names.index('s.ds.r3.b1'))
    assert np.all(particles23.at_element[3:6] == line.element_names.index('s.ds.r7.b1'))
    assert np.all(particles23.at_element[6:] == xt.particles.LAST_INVALID_STATE)

@for_all_test_contexts
def test_get_normalized_coordinates_twiss_init(test_context):

    ctx2np = test_context.nparray_from_context_array
    path_line_particles = test_data_folder / 'hllhc15_noerrors_nobb/line_and_particle.json'

    with open(path_line_particles, 'r') as fid:
        input_data = json.load(fid)
    line = xt.Line.from_dict(input_data['line'])
    line.particle_ref = xp.Particles.from_dict(input_data['particle'])

    line.build_tracker(_context=test_context)


    # Introduce a non-zero closed orbit
    line['mqwa.a4r3.b1..1'].knl[0] = 10e-6
    line['mqwa.a4r3.b1..1'].ksl[0] = 5e-6

    particles1 = line.build_particles(
        nemitt_x=2.5e-6, nemitt_y=1e-6,
        x_norm=[-1, 0, 0.5], y_norm=[0.3, -0.2, 0.2],
        px_norm=[0.1, 0.2, 0.3], py_norm=[0.5, 0.6, 0.8],
        zeta=[0, 0.1, -0.1], delta=[1e-4, 0., -1e-4])
    
    particles2 = line.build_particles(at_element='s.ds.r3.b1',
        # _capacity=10,
        nemitt_x=2.5e-6, nemitt_y=1e-6,
        x_norm=[-1, 0, 0.5], y_norm=[0.3, -0.2, 0.2],
        px_norm=[0.1, 0.2, 0.3], py_norm=[0.5, 0.6, 0.8],
        zeta=[0, 0.1, -0.1], delta=[1e-4, 0., -1e-4])

    particles3 = line.build_particles(at_element='s.ds.r7.b1',
        # _capaci/ty=10,
        nemitt_x=2.5e-6, nemitt_y=1e-6,
        x_norm=[-1, 0, 0.5], y_norm=[0.3, -0.2, 0.2],
        px_norm=[0.1, 0.2, 0.3], py_norm=[0.5, 0.6, 0.8],
        zeta=[0, 0.1, -0.1], delta=[1e-4, 0., -1e-4])


    particles4 = line.build_particles(at_element='s.ds.r7.b1',
        # _capaci/ty=10,
        nemitt_x=2.5e-6, nemitt_y=1e-6,
        x_norm=[-1, 0, 0.5], y_norm=[0.3, -0.2, 0.2],
        px_norm=[0.1, 0.2, 0.3], py_norm=[0.5, 0.6, 0.8],
        zeta_norm=[0, 0.1, -0.1], pzeta_norm=[1e-4, 0., -1e-4])

    tw = line.twiss()

    for part in [particles1,particles2,particles3,particles4]:
        tw = line.twiss()
        tw_init = tw.get_twiss_init(at_element=line.element_names[ctx2np(part.at_element)[0]])

        norm_coord = tw_init.get_normalized_coordinates(part, nemitt_x=2.5e-6,
                                                nemitt_y=1e-6)

        assert np.allclose(norm_coord['x_norm'], [-1, 0, 0.5], atol=1e-10, rtol=0)
        assert np.allclose(norm_coord['y_norm'], [0.3, -0.2, 0.2], atol=1e-10, rtol=0)
        assert np.allclose(norm_coord['px_norm'], [0.1, 0.2, 0.3], atol=1e-10, rtol=0)
        assert np.allclose(norm_coord['py_norm'], [0.5, 0.6, 0.8], atol=1e-10, rtol=0)


@for_all_test_contexts
def test_twiss_does_not_affect_monitors(test_context):

    path_line_particles = test_data_folder / 'hllhc15_noerrors_nobb/line_and_particle.json'

    with open(path_line_particles, 'r') as fid:
        input_data = json.load(fid)
    line = xt.Line.from_dict(input_data['line'])
    line.particle_ref = xp.Particles.from_dict(input_data['particle'])

    n_part =1
    monitor = xt.ParticlesMonitor(_context=test_context,
                                    start_at_turn = 0,
                                    stop_at_turn = 1,
                                    n_repetitions=10,
                                    repetition_period=1,
                                    num_particles =n_part)
    line.insert_element(index=0, element=monitor, name='monitor_start')
    line.build_tracker(_context=test_context)

    particles = line.build_particles(x=123e-6)
    line.track(particles, num_turns=10)
    assert monitor.x[0,0] == 123e-6

    particles = line.build_particles(x=456e-6)
    particles.at_turn = -10 # the monitor is skipped in this way in the twiss
    line.track(particles, num_turns=10)
    assert monitor.x[0,0] == 123e-6

    line.twiss()
    assert monitor.x[0,0] == 123e-6


@for_all_test_contexts
def test_knl_ksl_in_twiss(test_context):

    path_line_particles = test_data_folder / 'hllhc15_noerrors_nobb/line_and_particle.json'

    with open(path_line_particles, 'r') as fid:
        input_data = json.load(fid)
    line = xt.Line.from_dict(input_data['line'])
    line.particle_ref = xp.Particles.from_dict(input_data['particle'])

    line.build_tracker(_context=test_context)

    tw = line.twiss()

    tw_with_knl_ksl = line.twiss(strengths=True)
    tw_with_knl_ksl_part = line.twiss(strengths=True,
                        start='bpm.31l5.b1',
                        end='bpm.31r5.b1',
                        init=tw.get_twiss_init(at_element='bpm.31l5.b1'))

    for tt in [tw_with_knl_ksl, tw_with_knl_ksl_part]:

        for kk in ['k0l', 'k0sl', 'k1l', 'k1sl', 'k2l', 'k2sl']:
            assert kk in tt.keys()
            assert kk not in tw.keys()

        assert tt['k2l', 'ms.30r5.b1'] == line['ms.30r5.b1'].knl[2]
        assert tt['k0sl', 'mcbrdv.4r5.b1'] == line['mcbrdv.4r5.b1'].ksl[0]

def test_get_R_matrix():
    fname_line_particles = test_data_folder / 'hllhc15_noerrors_nobb/line_and_particle.json'
    line = xt.load(fname_line_particles)
    line.particle_ref = xp.Particles(p0c=7e12, mass0=xp.PROTON_MASS_EV)
    line.build_tracker()

    tw = line.twiss()

    R_IP3_IP6 = tw.get_R_matrix(start=0, end='ip6')
    R_IP6_IP3 = tw.get_R_matrix(start='ip6', end=len(tw.name)-1)

    # # Checks
    R_prod = R_IP6_IP3 @ R_IP3_IP6

    from xtrack.linear_normal_form import compute_linear_normal_form
    eig = np.linalg.eig
    norm = np.linalg.norm

    R_matrix = tw.R_matrix

    W_ref, invW_ref, Rot_ref, _ = compute_linear_normal_form(R_matrix)
    W_prod, invW_prod, Rot_prod, _ = compute_linear_normal_form(R_prod)


    for i_mode in range(3):
        lam_ref = eig(Rot_ref[2*i_mode:2*i_mode+2, 2*i_mode:2*i_mode+2])[0][0]
        lam_prod = eig(Rot_prod[2*i_mode:2*i_mode+2, 2*i_mode:2*i_mode+2])[0][0]

        xo.assert_allclose(np.abs(np.angle(lam_ref)) / 2 / np.pi,
                        np.abs(np.angle(lam_prod)) / 2 / np.pi,
                        rtol=0, atol=1e-6)

        xo.assert_allclose(
            norm(W_prod[:, 2*i_mode] - W_ref[:, 2*i_mode], ord=2)
            / norm(W_ref[:, 2*i_mode], ord=2),
            0, rtol=0, atol=5e-4)
        xo.assert_allclose(
            norm(W_prod[:4, 2*i_mode] - W_ref[:4, 2*i_mode], ord=2)
            / norm(W_ref[:4, 2*i_mode], ord=2),
            0, rtol=0, atol=5e-4)

    # Check method=4d

    tw4d = line.twiss(method='4d', freeze_longitudinal=True)

    R_IP3_IP6_4d = tw4d.get_R_matrix(start=0, end='ip6')
    R_IP6_IP3_4d = tw4d.get_R_matrix(start='ip6', end=len(tw4d.name)-1)

    R_prod_4d = R_IP6_IP3_4d @ R_IP3_IP6_4d

    # Checks
    from xtrack.linear_normal_form import compute_linear_normal_form
    eig = np.linalg.eig
    norm = np.linalg.norm

    R_matrix_4d = tw4d.R_matrix

    W_ref_4d, invW_ref_4d, Rot_ref_4d, _ = compute_linear_normal_form(
        R_matrix_4d, only_4d_block=True)
    W_prod_4d, invW_prod_4d, Rot_prod_4d, _ = compute_linear_normal_form(
        R_prod_4d, only_4d_block=True)

    for i_mode in range(3):
        lam_ref_4d = eig(
            Rot_ref_4d[2*i_mode:2*i_mode+2, 2*i_mode:2*i_mode+2])[0][0]
        lam_prod_4d = eig(
            Rot_prod_4d[2*i_mode:2*i_mode+2, 2*i_mode:2*i_mode+2])[0][0]

        xo.assert_allclose(np.abs(np.angle(lam_ref_4d)) / 2 / np.pi,
                        np.abs(np.angle(lam_prod_4d)) / 2 / np.pi,
                        rtol=0, atol=1e-6)

        xo.assert_allclose(
            norm(W_prod_4d[:, 2*i_mode] - W_ref_4d[:, 2*i_mode], ord=2)
            / norm(W_ref_4d[:, 2*i_mode], ord=2),
            0, rtol=0, atol=5e-5)

def test_hide_thin_groups():

    line = xt.load(test_data_folder /
                                        'lhc_no_bb/line_and_particle.json')
    line.particle_ref = xp.Particles(
                        mass0=xp.PROTON_MASS_EV, q0=1, energy0=7e12)
    line.build_tracker()

    tw = line.twiss()
    tw_htg = line.twiss(hide_thin_groups=True)

    for nn in ('x y px py zeta delta ptau '
            'betx bety alfx alfy gamx gamy dx dy dpx dpy').split():
        assert np.isnan(tw_htg[nn]).sum() == 2281
        assert np.isnan(tw[nn]).sum() == 0

        # Check in presence of a srotation
        assert tw.name[11197] == 'mbxws.1r8_pretilt'
        assert tw.name[11198] == 'mbxws.1r8'
        assert tw.name[11199] == 'mbxws.1r8_posttilt'

        assert tw_htg[nn][11197] == tw[nn][11197]
        assert np.isnan(tw_htg[nn][11198])
        assert np.isnan(tw_htg[nn][11199])
        assert tw_htg[nn][11200] == tw[nn][11200]

@for_all_test_contexts
def test_periodic_cell_twiss(test_context):
    collider = xt.load(test_data_folder /
                    'hllhc15_collider/collider_00_from_mad.json')
    collider.build_trackers(_context=test_context)

    collider.lhcb1.twiss_default['method'] = '4d'
    collider.lhcb2.twiss_default['method'] = '4d'
    collider.lhcb2.twiss_default['reverse'] = True

    for beam_name in ['b1', 'b2']:

        line = collider['lhc' + beam_name]
        start_cell = 's.cell.67.' + beam_name
        end_cell = 'e.cell.67.' + beam_name
        start_arc = 'e.ds.r6.' + beam_name
        end_arc = 'e.ds.l7.' + beam_name

        tw = line.twiss()

        assert tw.method == '4d'
        assert tw.orientation == 'forward'
        assert tw.reference_frame == {'b1':'proper', 'b2':'reverse'}[beam_name]
        assert 'dqx' in tw.keys() # check that periodic twiss is used

        mux_arc_target = tw['mux', end_arc] - tw['mux', start_arc]
        muy_arc_target = tw['muy', end_arc] - tw['muy', start_arc]

        tw0 = line.twiss()
        tw_cell = line.twiss(
            start=start_cell,
            end=end_cell,
            init_at=xt.START,
            init=tw0)

        assert tw_cell.method == '4d'
        assert 'dqx' not in tw_cell.keys() # check that periodic twiss is not used
        assert tw_cell.name[0] == start_cell
        assert tw_cell.name[-2] == end_cell
        assert tw_cell.method == '4d'
        assert tw_cell.orientation == {'b1': 'forward', 'b2': 'backward'}[beam_name]
        assert tw_cell.reference_frame == {'b1':'proper', 'b2':'reverse'}[beam_name]

        tw_cell_periodic = line.twiss(
            method='4d',
            start=start_cell,
            end=end_cell,
            init='periodic')

        assert tw_cell_periodic.method == '4d'
        assert 'dqx' in tw_cell_periodic.keys() # check that periodic twiss is used
        assert tw_cell_periodic.name[0] == start_cell
        assert tw_cell_periodic.name[-2] == end_cell
        assert tw_cell_periodic.method == '4d'
        assert tw_cell_periodic.orientation == 'forward'
        assert tw_cell_periodic.reference_frame == {'b1':'proper', 'b2':'reverse'}[beam_name]

        xo.assert_allclose(tw_cell_periodic.betx, tw_cell.betx, atol=0, rtol=1e-6)
        xo.assert_allclose(tw_cell_periodic.bety, tw_cell.bety, atol=0, rtol=1e-6)
        xo.assert_allclose(tw_cell_periodic.dx, tw_cell.dx, atol=1e-4, rtol=0)

        assert tw_cell_periodic['mux', 0] == 0
        assert tw_cell_periodic['muy', 0] == 0
        xo.assert_allclose(tw_cell_periodic.mux[-1],
                tw['mux', end_cell] - tw['mux', start_cell], rtol=0, atol=1e-6)
        xo.assert_allclose(tw_cell_periodic.muy[-1],
                tw['muy', end_cell] - tw['muy', start_cell], rtol=0, atol=1e-6)

        twinit_start_cell = tw_cell_periodic.get_twiss_init(start_cell)

        tw_to_end_arc = line.twiss(
            start=start_cell,
            end=end_arc,
            init=twinit_start_cell)
        assert tw_to_end_arc.method == '4d'
        assert tw_to_end_arc.orientation == {'b1': 'forward', 'b2': 'backward'}[beam_name]
        assert tw_to_end_arc.reference_frame == {'b1':'proper', 'b2':'reverse'}[beam_name]

        tw_to_start_arc = line.twiss(
            start=start_arc,
            end=start_cell,
            init=twinit_start_cell)
        assert tw_to_start_arc.method == '4d'
        assert tw_to_start_arc.orientation == {'b1': 'backward', 'b2': 'forward'}[beam_name]
        assert tw_to_start_arc.reference_frame == {'b1':'proper', 'b2':'reverse'}[beam_name]

        mux_arc_from_cell = tw_to_end_arc['mux', end_arc] - tw_to_start_arc['mux', start_arc]
        muy_arc_from_cell = tw_to_end_arc['muy', end_arc] - tw_to_start_arc['muy', start_arc]

        xo.assert_allclose(mux_arc_from_cell, mux_arc_target, rtol=1e-6)
        xo.assert_allclose(muy_arc_from_cell, muy_arc_target, rtol=1e-6)


        dp_test = 1e-4
        tw_cell_periodic_plus = line.twiss(
            method='4d',
            delta0=dp_test,
            start=start_cell,
            end=end_cell,
            init='periodic')
        tw_cell_periodic_minus = line.twiss(
            method='4d',
            delta0=-dp_test,
            start=start_cell,
            end=end_cell,
            init='periodic')

        dqx_expected = (tw_cell_periodic_plus.mux[-1] - tw_cell_periodic_minus.mux[-1]
                       - tw_cell_periodic_plus.mux[0] + tw_cell_periodic_minus.mux[0]
                       ) / 2 / dp_test

        dqy_expected = (tw_cell_periodic_plus.muy[-1] - tw_cell_periodic_minus.muy[-1]
                          - tw_cell_periodic_plus.muy[0] + tw_cell_periodic_minus.muy[0]
                            ) / 2 / dp_test

        if beam_name == 'b1':
            xo.assert_allclose(dqx_expected, 0.22855, rtol=0, atol=1e-4) # to catch regressions
            xo.assert_allclose(dqy_expected, 0.26654, rtol=0, atol=1e-4) # to catch regressions
        else:
            xo.assert_allclose(dqx_expected, -0.479425, rtol=0, atol=1e-4)
            xo.assert_allclose(dqy_expected, 0.543953, rtol=0, atol=1e-4)

        xo.assert_allclose(tw_cell_periodic.dqx, dqx_expected, rtol=0, atol=1e-4)
        xo.assert_allclose(tw_cell_periodic.dqy, dqy_expected, rtol=0, atol=1e-4)

@pytest.fixture(scope='module')
def collider_for_test_twiss_range():

    collider = xt.load(test_data_folder /
                    'hllhc15_thick/hllhc15_collider_thick.json')
    collider.lhcb1.twiss_default['method'] = '4d'
    collider.lhcb2.twiss_default['method'] = '4d'
    collider.lhcb2.twiss_default['reverse'] = True
    collider.build_trackers(_context=xo.ContextCpu())
    collider.lhcb1.particle_ref.move(_buffer=xo.ContextCpu().new_buffer())
    collider.lhcb2.particle_ref.move(_buffer=xo.ContextCpu().new_buffer())
    return collider



@for_all_test_contexts
@pytest.mark.parametrize('line_name', ['lhcb1', 'lhcb2'])
@pytest.mark.parametrize('check', ['fw', 'bw', 'fw_kw', 'bw_kw', 'fw_table', 'bw_table'])
@pytest.mark.parametrize('init_at_edge', [True, False], ids=['init_at_edge', 'init_inside'])
@pytest.mark.parametrize('cycle_to',
                         [('ip3', 'ip3'), ('s.ds.l6.b1', 's.ds.l6.b2'), ('ip6', 'ip6'), ('ip5', 'ip5')],
                         ids=['no_cycle', 'cycle_arc', 'cycle_edge1', 'cycle_edge2'])
def test_twiss_range(test_context, cycle_to, line_name, check, init_at_edge, collider_for_test_twiss_range):

    if collider_for_test_twiss_range is not None:
        collider = collider_for_test_twiss_range
    else:
        raise ValueError('This should not happen')
        collider = xt.load(test_data_folder /
                        'hllhc15_thick/hllhc15_collider_thick.json')
        collider.lhcb1.twiss_default['method'] = '4d'
        collider.lhcb2.twiss_default['method'] = '4d'
        collider.lhcb2.twiss_default['reverse'] = True

    if collider.lhcb1.element_names[0] != cycle_to[0]:
        collider.lhcb1.cycle(cycle_to[0], inplace=True)
    if collider.lhcb2.element_names[0] != cycle_to[1]:
        collider.lhcb2.cycle(cycle_to[1], inplace=True)

    loop_around = collider.lhcb1.element_names[0] != 'ip3'

    collider.vars['on_x5hs'] = 200
    collider.vars['on_x5vs'] = 123
    collider.vars['on_sep5h'] = 1
    collider.vars['on_sep5v'] = 2

    atols = dict(
        s=2e-8,
        zeta=5e-5,
        alfx=1e-8, alfy=1e-8, alfx1=1e-8, alfy2=1e-8, alfx2=1e-6, alfy1=1e-6,
        dzeta=1e-4, dx=1e-4, dy=1e-4, dpx=1e-5, dpy=1e-5,
        nuzeta=1e-5, dx_zeta=1e-7, dy_zeta=1e-7, dpx_zeta=1e-8, dpy_zeta=1e-8,
        nux=1e-8, nuy=1e-8,
        betx2=1e-4, bety1=1e-4,
        gamy1=5e-7, gamx2=5e-7,
    )

    rtols = dict(
        alfx=5e-9, alfy=5e-8, alfx1=5e-9, alfy2=5e-8, alfx2=5e-8, alfy1=5e-8,
        betx=1e-8, bety=1e-8, betx1=1e-8, bety2=1e-8, betx2=1e-7, bety1=1e-7,
        gamx=5e-9, gamy=5e-9, gamx1=5e-9, gamy2=5e-9, gamx2=1e-7, gamy1=1e-7,
    )

    if loop_around or not init_at_edge:
        rtols['betx'] = 2e-5
        rtols['bety'] = 2e-5
        rtols['alfx'] = rtols['alfx1'] = 4e-5
        rtols['alfy'] = rtols['alfy2'] = 4e-5
        rtols['gamx'] = 2e-5
        rtols['gamy'] = 2e-5
        rtols['betx1'] = 2e-5
        rtols['bety2'] = 1e-5
        rtols['betx2'] = 1e-4
        rtols['bety1'] = 1e-4
        rtols['alfx2'] = 2e-4
        rtols['alfy1'] = 2e-4
        rtols['gamx1'] = 2e-5
        rtols['gamy2'] = 1e-5
        rtols['gamx2'] = 1e-4
        rtols['gamy1'] = 1e-4

        atols['alfy'] = atols['alfy2'] = 4e-5
        atols['alfx'] = atols['alfx1'] = 4e-5
        atols['mux'] = 1e-5
        atols['muy'] = 1e-5
        atols['nux'] = 1e-8
        atols['nuy'] = 1e-8
        atols['dx_zeta'] = 2e-5
        atols['dpx_zeta'] = 2e-6
        atols['dy_zeta'] = 2e-5
        atols['dpy_zeta'] = 2e-6

    atol_default = 1e-11
    rtol_default = 1e-9

    line = collider[line_name]

    if isinstance(test_context, xo.ContextCpu) and (
        test_context.omp_num_threads != line._context.omp_num_threads):
        buffer = test_context.new_buffer()
    elif isinstance(test_context, line._context.__class__):
        buffer = line._buffer
    else:
        buffer = test_context.new_buffer()

    line.build_tracker(_buffer=buffer)

    if not check.endswith('_kw'):
        # Coupling is supported --> we test it
        collider.vars['kqs.a23b1'] = 1e-4
        collider.vars['kqs.a23b2'] = -1e-4
        collider.vars['on_disp'] = 1
    else:
        # Coupling is not supported --> we skip it
        collider.vars['kqs.a23b1'] = 0
        collider.vars['kqs.a23b2'] = 0
        collider.vars['on_disp'] = 0 # avoid feeddown from sextupoles

    if line.element_names[0] == 'ip5':
        # Need to avoid the crossing bumps in closed orbit search (convergence issues)
        tw = line.twiss(co_search_at='ip3')
    else:
        tw = line.twiss()

    tw_init_ip5 = tw.get_twiss_init('ip5')
    tw_init_ip6 = tw.get_twiss_init('ip6')

    xo.assert_allclose(tw_init_ip5.betx, tw['betx', 'ip5'], atol=0, rtol=1e-7)
    xo.assert_allclose(tw_init_ip5.bety, tw['bety', 'ip5'], atol=0, rtol=1e-7)
    xo.assert_allclose(tw_init_ip5.alfx, tw['alfx', 'ip5'], atol=0, rtol=1e-7)
    xo.assert_allclose(tw_init_ip5.alfy, tw['alfy', 'ip5'], atol=0, rtol=1e-7)
    xo.assert_allclose(tw_init_ip5.dx,   tw['dx', 'ip5'],   atol=0, rtol=1e-7)
    xo.assert_allclose(tw_init_ip5.dy,   tw['dy', 'ip5'],   atol=0, rtol=1e-7)
    xo.assert_allclose(tw_init_ip5.dpx,  tw['dpx', 'ip5'],  atol=0, rtol=1e-7)
    xo.assert_allclose(tw_init_ip5.dpy,  tw['dpy', 'ip5'],  atol=0, rtol=1e-7)
    xo.assert_allclose(tw_init_ip5.ax_chrom, tw['ax_chrom', 'ip5'], atol=0, rtol=1e-7)
    xo.assert_allclose(tw_init_ip5.ay_chrom, tw['ay_chrom', 'ip5'], atol=0, rtol=1e-7)
    xo.assert_allclose(tw_init_ip5.bx_chrom, tw['bx_chrom', 'ip5'], atol=0, rtol=1e-7)
    xo.assert_allclose(tw_init_ip5.by_chrom, tw['by_chrom', 'ip5'], atol=0, rtol=1e-7)

    xo.assert_allclose(tw_init_ip6.betx, tw['betx', 'ip6'], atol=0, rtol=1e-7)
    xo.assert_allclose(tw_init_ip6.bety, tw['bety', 'ip6'], atol=0, rtol=1e-7)
    xo.assert_allclose(tw_init_ip6.alfx, tw['alfx', 'ip6'], atol=0, rtol=1e-7)
    xo.assert_allclose(tw_init_ip6.alfy, tw['alfy', 'ip6'], atol=0, rtol=1e-7)
    xo.assert_allclose(tw_init_ip6.dx,   tw['dx', 'ip6'],   atol=0, rtol=1e-7)
    xo.assert_allclose(tw_init_ip6.dy,   tw['dy', 'ip6'],   atol=0, rtol=1e-7)
    xo.assert_allclose(tw_init_ip6.dpx,  tw['dpx', 'ip6'],  atol=0, rtol=1e-7)
    xo.assert_allclose(tw_init_ip6.dpy,  tw['dpy', 'ip6'],  atol=0, rtol=1e-7)
    xo.assert_allclose(tw_init_ip6.ax_chrom, tw['ax_chrom', 'ip6'], atol=0, rtol=1e-7)
    xo.assert_allclose(tw_init_ip6.ay_chrom, tw['ay_chrom', 'ip6'], atol=0, rtol=1e-7)
    xo.assert_allclose(tw_init_ip6.bx_chrom, tw['bx_chrom', 'ip6'], atol=0, rtol=1e-7)
    xo.assert_allclose(tw_init_ip6.by_chrom, tw['by_chrom', 'ip6'], atol=0, rtol=1e-7)

    if init_at_edge:
        estart_user = 'ip5'
        estop_user = 'ip6'
    else:
        estart_user = 'ip4'
        estop_user = 'ip7'

    if check == 'fw':
        tw_test = line.twiss(start=estart_user, end=estop_user,
                                init=tw_init_ip5)
        name_init = 'ip5'
    elif check == 'bw':
        tw_test = line.twiss(start=estart_user, end=estop_user,
                                    init=tw_init_ip6)
        name_init = 'ip6'
    elif check == 'fw_table' and init_at_edge:
        tw_test = line.twiss(start=estart_user, end=estop_user,
                             init=tw) # init_at=xt.START is default
        name_init = 'ip5'
    elif check == 'bw_table' and init_at_edge:
        tw_test = line.twiss(start=estart_user, end=estop_user,
                             init=tw, init_at=xt.END)
        name_init = 'ip6'
    elif check == 'fw_table' and not init_at_edge:
        tw_test = line.twiss(start=estart_user, end=estop_user,
                             init=tw, init_at='ip5')
        name_init = 'ip5'
    elif check == 'bw_table' and not init_at_edge:
        tw_test = line.twiss(start=estart_user, end=estop_user,
                             init=tw, init_at='ip6')
        name_init = 'ip6'
    elif check == 'fw_kw':
        tw_test = line.twiss(start=estart_user, end=estop_user,
                            init_at='ip5',
                            x=tw['x', 'ip5'],
                            px=tw['px', 'ip5'],
                            y=tw['y', 'ip5'],
                            py=tw['py', 'ip5'],
                            zeta=tw['zeta', 'ip5'],
                            delta=tw['delta', 'ip5'],
                            betx=tw['betx', 'ip5'],
                            alfx=tw['alfx', 'ip5'],
                            bety=tw['bety', 'ip5'],
                            alfy=tw['alfy', 'ip5'],
                            dx=tw['dx', 'ip5'],
                            dpx=tw['dpx', 'ip5'],
                            dy=tw['dy', 'ip5'],
                            dpy=tw['dpy', 'ip5'],
                            dzeta=tw['dzeta', 'ip5'],
                            mux=tw['mux', 'ip5'],
                            muy=tw['muy', 'ip5'],
                            muzeta=tw['muzeta', 'ip5'],
                            ax_chrom=tw['ax_chrom', 'ip5'],
                            bx_chrom=tw['bx_chrom', 'ip5'],
                            ay_chrom=tw['ay_chrom', 'ip5'],
                            by_chrom=tw['by_chrom', 'ip5'],
                                )
        name_init = 'ip5'
    elif check == 'bw_kw':
        tw_test = line.twiss(start=estart_user, end=estop_user,
                            init_at='ip6',
                            x=tw['x', 'ip6'],
                            px=tw['px', 'ip6'],
                            y=tw['y', 'ip6'],
                            py=tw['py', 'ip6'],
                            zeta=tw['zeta', 'ip6'],
                            delta=tw['delta', 'ip6'],
                            betx=tw['betx', 'ip6'],
                            alfx=tw['alfx', 'ip6'],
                            bety=tw['bety', 'ip6'],
                            alfy=tw['alfy', 'ip6'],
                            dx=tw['dx', 'ip6'],
                            dpx=tw['dpx', 'ip6'],
                            dy=tw['dy', 'ip6'],
                            dpy=tw['dpy', 'ip6'],
                            dzeta=tw['dzeta', 'ip6'],
                            mux=tw['mux', 'ip6'],
                            muy=tw['muy', 'ip6'],
                            muzeta=tw['muzeta', 'ip6'],
                            ax_chrom=tw['ax_chrom', 'ip6'],
                            bx_chrom=tw['bx_chrom', 'ip6'],
                            ay_chrom=tw['ay_chrom', 'ip6'],
                            by_chrom=tw['by_chrom', 'ip6'],
                            )
        name_init = 'ip6'
    else:
        raise ValueError(f'Unknown config {check}')

    xo.assert_allclose(tw_test['s', name_init], tw['s', name_init], 1e-10)

    assert tw_init_ip5.reference_frame == (
        {'lhcb1': 'proper', 'lhcb2': 'reverse'}[line_name])
    assert tw_init_ip5.element_name == 'ip5'

    if (loop_around
        and not (line_name == 'lhcb2' and estop_user == 'ip6' and  cycle_to[1] == 'ip6')
        and not (line_name == 'lhcb1' and estart_user == 'ip5' and  cycle_to[1] == 'ip5')
    ):
        tw_part1 = tw.rows[estart_user:]
        tw_part2 = tw.rows[:estop_user]
        tw_part = xt.TwissTable.concatenate([tw_part1, tw_part2])
        tw_part.s += tw['s', name_init] - tw_part['s', name_init]
        tw_part.mux += tw['mux', name_init] - tw_part['mux', name_init]
        tw_part.muy += tw['muy', name_init] - tw_part['muy', name_init]
        tw_part.muzeta += tw['muzeta', name_init] - tw_part['muzeta', name_init]
        tw_part.dzeta += tw['dzeta', name_init] - tw_part['dzeta', name_init]
        tw_part._data['method'] = '4d'
        tw_part._data['radiation_method'] = None
        tw_part._data['orientation'] = (
            {'lhcb1': 'forward', 'lhcb2': 'backward'}[line_name])
    else:
        tw_part = tw.rows[estart_user:estop_user]
    assert tw_part.name[0] == estart_user
    assert tw_part.name[-1] == estop_user

    assert tw_test.name[-1] == '_end_point'

    tw_test = tw_test.rows[:-1]
    assert np.all(tw_test.name == tw_part.name)
    assert np.all(tw_test.name_env == tw_part.name_env)

    for kk in tw_test._data.keys():
        if kk in ['name', 'name_env', 'W_matrix', 'particle_on_co', 'values_at',
                    'method', 'radiation_method', 'reference_frame',
                    'orientation', 'steps_r_matrix', 'line_config',
                    'loop_around', '_action', 'completed_init',
                    'phix', 'phiy', 'phizeta', # are only relative (not unwrapped)
                    ]:
            continue # some tested separately
        atol = atols.get(kk, atol_default)
        rtol = rtols.get(kk, rtol_default)
        xo.assert_allclose(
            tw_test._data[kk], tw_part._data[kk], rtol=rtol, atol=atol)

    assert tw_test.values_at == tw_part.values_at == 'entry'
    assert tw_test.method == tw_part.method == '4d'
    assert tw_test.radiation_method == tw_part.radiation_method == None
    assert tw_test.reference_frame == tw_part.reference_frame == (
        {'lhcb1': 'proper', 'lhcb2': 'reverse'}[line_name])

    if not check.endswith('_kw') and not loop_around:

        W_matrix_part = tw_part.W_matrix
        W_matrix_test = tw_test.W_matrix

        rel_error = []
        for ss in range(W_matrix_part.shape[0]):
            this_part = W_matrix_part[ss, :, :]
            this_test = W_matrix_test[ss, :, :]

            for ii in range(this_part.shape[0]):
                rel_error.append((np.linalg.norm(this_part[ii, :] - this_test[ii, :])
                                /np.linalg.norm(this_part[ii, :])))
        assert np.max(rel_error) < 1e-3


@for_all_test_contexts
def test_twiss_against_matrix(test_context):
    x_co = [1e-3, 2e-3]
    px_co = [2e-6, -3e-6]
    y_co = [3e-3, 4e-3]
    py_co = [4e-6, -5e-6]
    betx = [1., 2.]
    bety = [3., 4.]
    alfx = [0, 0.1]
    alfy = [0.2, 0.]
    dx = [10, 0]
    dy = [0, 20]
    dpx = [0.7, -0.3]
    dpy = [0.4, -0.6]
    bets = 1e-3
    dnqx = [0.21, 2, 30, 400]
    dnqy = [0.32, 3, 40, 500]

    segm_1 = xt.LineSegmentMap(
        qx=0.4, qy=0.3, qs=0.0001,
        bets=bets, length=0.1,
        betx=[betx[0], betx[1]],
        bety=[bety[0], bety[1]],
        alfx=[alfx[0], alfx[1]],
        alfy=[alfy[0], alfy[1]],
        dx=[dx[0], dx[1]],
        dpx=[dpx[0], dpx[1]],
        dy=[dy[0], dy[1]],
        dpy=[dpy[0], dpy[1]],
        x_ref=[x_co[0], x_co[1]],
        px_ref=[px_co[0], px_co[1]],
        y_ref=[y_co[0], y_co[1]],
        py_ref=[py_co[0], py_co[1]])
    segm_2 = xt.LineSegmentMap(
        qs=0.0003,
        bets=bets, length=0.2,
        dnqx=dnqx, dnqy=dnqy,
        betx=[betx[1], betx[0]],
        bety=[bety[1], bety[0]],
        alfx=[alfx[1], alfx[0]],
        alfy=[alfy[1], alfy[0]],
        dx=[dx[1], dx[0]],
        dpx=[dpx[1], dpx[0]],
        dy=[dy[1], dy[0]],
        dpy=[dpy[1], dpy[0]],
        x_ref=[x_co[1], x_co[0]],
        px_ref=[px_co[1], px_co[0]],
        y_ref=[y_co[1], y_co[0]],
        py_ref=[py_co[1], py_co[0]])

    line = xt.Line(elements=[segm_1, segm_2],
                   particle_ref=xt.Particles(p0c=1e9))
    line.build_tracker(_context=test_context)

    tw4d = line.twiss(method='4d')
    tw6d = line.twiss()

    xo.assert_allclose(tw6d.qs, 0.0004, atol=1e-7, rtol=0)
    xo.assert_allclose(tw6d.bets0, 1e-3, atol=1e-7, rtol=0)

    for tw in [tw4d, tw6d]:
        xo.assert_allclose(tw.qx, 0.4 + 0.21, atol=1e-7, rtol=0)
        xo.assert_allclose(tw.qy, 0.3 + 0.32, atol=1e-7, rtol=0)

        xo.assert_allclose(tw.dqx, 2, atol=1e-5, rtol=0)
        xo.assert_allclose(tw.dqy, 3, atol=1e-5, rtol=0)

        xo.assert_allclose(tw.s, [0, 0.1, 0.1 + 0.2], atol=1e-7, rtol=0)
        xo.assert_allclose(tw.mux, [0, 0.4, 0.4 + 0.21], atol=1e-7, rtol=0)
        xo.assert_allclose(tw.muy, [0, 0.3, 0.3 + 0.32], atol=1e-7, rtol=0)

        xo.assert_allclose(tw.betx, [1, 2, 1], atol=1e-7, rtol=0)
        xo.assert_allclose(tw.bety, [3, 4, 3], atol=1e-7, rtol=0)

        xo.assert_allclose(tw.alfx, [0, 0.1, 0], atol=1e-7, rtol=0)
        xo.assert_allclose(tw.alfy, [0.2, 0, 0.2], atol=1e-7, rtol=0)

        xo.assert_allclose(tw.dx, [10, 0, 10], atol=1e-4, rtol=0)
        xo.assert_allclose(tw.dy, [0, 20, 0], atol=1e-4, rtol=0)
        xo.assert_allclose(tw.dpx, [0.7, -0.3, 0.7], atol=1e-5, rtol=0)
        xo.assert_allclose(tw.dpy, [0.4, -0.6, 0.4], atol=1e-5, rtol=0)

        xo.assert_allclose(tw.x, [1e-3, 2e-3, 1e-3], atol=1e-7, rtol=0)
        xo.assert_allclose(tw.px, [2e-6, -3e-6, 2e-6], atol=1e-12, rtol=0)
        xo.assert_allclose(tw.y, [3e-3, 4e-3, 3e-3], atol=1e-7, rtol=0)
        xo.assert_allclose(tw.py, [4e-6, -5e-6, 4e-6], atol=1e-12, rtol=0)

        assert np.allclose(tw.py, [4e-6, -5e-6, 4e-6], atol=1e-12, rtol=0)

        chroma_table = line.get_non_linear_chromaticity((-1e-2, 1e-2),
                                                        num_delta=25)
        xo.assert_allclose(chroma_table.dnqx[1:], dnqx[1:], atol=1e-5, rtol=1e-5)
        xo.assert_allclose(chroma_table.dnqy[1:], dnqy[1:], atol=1e-5, rtol=1e-5)

@for_all_test_contexts
@pytest.mark.parametrize('machine', ['sps', 'psb'])
def test_longitudinal_plane_against_matrix(machine, test_context):

    if machine == 'sps':
        line = xt.load(test_data_folder /
            'sps_w_spacecharge/line_no_spacecharge_and_particle.json')
        # I put the cavity at the end of the ring to get closer to the kick-drift model
        line.cycle('actb.31739_aper', inplace=True)
        configurations = ['above transition', 'below transition']
        num_turns = 250
        cavity_name = 'acta.31637'
        sigmaz=0.20
    elif machine == 'psb':
        line = xt.load(test_data_folder /
            'psb_injection/line_and_particle.json')
        configurations = ['below transition']
        num_turns = 1000
        cavity_name = 'br.c02'
        sigmaz = 22.
    else:
        raise ValueError(f'Unknown machine {machine}')

    line.build_tracker(_context=test_context)

    for i_case, (configuration, longitudinal_mode) in enumerate(
        product(configurations,
                ['linear_fixed_qs', 'linear_fixed_rf', 'nonlinear'])):

        print(f'Case {i_case}: {configuration}, {longitudinal_mode}')

        if machine == 'sps':
            if configuration == 'above transition':
                line[cavity_name].lag = 180.
                line.particle_ref = xp.Particles(p0c=450e9, q0=1.0)
            else:
                line[cavity_name].lag = 0.
                line.particle_ref = xp.Particles(p0c=16e9, q0=1.0)

        # Build corresponding matrix
        tw = line.twiss()
        circumference = tw.circumference

        if longitudinal_mode == 'nonlinear':
            matrix = xt.LineSegmentMap(
                qx=tw.qx, qy=tw.qy,
                dqx=tw.dqx, dqy=tw.dqy,
                betx=tw.betx[0], alfx=tw.alfx[0],
                bety=tw.bety[0], alfy=tw.alfy[0],
                dx=tw.dx[0], dpx=tw.dpx[0],
                dy=tw.dy[0], dpy=tw.dpy[0],
                voltage_rf=line[cavity_name].voltage,
                frequency_rf=line[cavity_name].frequency,
                lag_rf=line[cavity_name].lag,
                momentum_compaction_factor=tw.momentum_compaction_factor,
                length=circumference)
        elif longitudinal_mode == 'linear_fixed_rf':
            matrix = xt.LineSegmentMap(
                longitudinal_mode='linear_fixed_rf',
                qx=tw.qx, qy=tw.qy,
                dqx=tw.dqx, dqy=tw.dqy,
                betx=tw.betx[0], alfx=tw.alfx[0],
                bety=tw.bety[0], alfy=tw.alfy[0],
                dx=tw.dx[0], dpx=tw.dpx[0],
                dy=tw.dy[0], dpy=tw.dpy[0],
                voltage_rf=line[cavity_name].voltage,
                frequency_rf=line[cavity_name].frequency,
                lag_rf=line[cavity_name].lag,
                momentum_compaction_factor=tw.momentum_compaction_factor,
                length=circumference)
        elif longitudinal_mode == 'linear_fixed_qs':
            eta = tw.slip_factor # > 0 above transition
            qs = tw.qs
            circumference = line.get_length()
            bet_s = eta * circumference / (2 * np.pi * qs)
            matrix = xt.LineSegmentMap(
                qx=tw.qx, qy=tw.qy,
                dqx=tw.dqx, dqy=tw.dqy,
                betx=tw.betx[0], alfx=tw.alfx[0],
                bety=tw.bety[0], alfy=tw.alfy[0],
                dx=tw.dx[0], dpx=tw.dpx[0],
                dy=tw.dy[0], dpy=tw.dpy[0],
                bets=bet_s, qs=qs,
                length=circumference)

        line_matrix = xt.Line(elements=[matrix])
        line_matrix.particle_ref = line.particle_ref.copy()
        line_matrix.build_tracker()

        # Compare tracking longitudinal tracking on one particle
        particle0_line = line.build_particles(x_norm=0, y_norm=0, zeta=1e-3)
        line.track(particle0_line.copy(), num_turns=num_turns, turn_by_turn_monitor=True)
        mon = line.record_last_track
        particle0_matrix = line_matrix.build_particles(x_norm=0, y_norm=0, zeta=1e-3)
        line_matrix.track(particle0_matrix.copy(), num_turns=num_turns, turn_by_turn_monitor=True)
        mon_matrix = line_matrix.record_last_track

        xo.assert_allclose(np.max(mon.zeta), np.max(mon_matrix.zeta), rtol=1e-2, atol=0)
        xo.assert_allclose(np.max(mon.pzeta), np.max(mon_matrix.pzeta), rtol=1e-2, atol=0)
        xo.assert_allclose(np.max(mon.x), np.max(mon_matrix.x), rtol=1e-2, atol=0)

        xo.assert_allclose(mon.zeta, mon_matrix.zeta, rtol=0, atol=5e-2*np.max(mon.zeta.T))
        xo.assert_allclose(mon.pzeta, mon_matrix.pzeta, rtol=0, atol=5e-2*np.max(mon.pzeta[:]))
        xo.assert_allclose(mon.x, mon_matrix.x, rtol=0, atol=5e-2*np.max(mon.x.T)) # There is some phase difference...

        # Match Gaussian distributions
        p_line = xp.generate_matched_gaussian_bunch(num_particles=1000000,
            nemitt_x=1e-6, nemitt_y=1e-6, sigma_z=5e-2, line=line, engine='linear')
        p_matrix = xp.generate_matched_gaussian_bunch(num_particles=1000000,
            nemitt_x=1e-6, nemitt_y=1e-6, sigma_z=5e-2, line=line_matrix, engine='linear')

        p_line.move(_context=xo.context_default)
        p_matrix.move(_context=xo.context_default)

        xo.assert_allclose(np.std(p_line.zeta), np.std(p_matrix.zeta), rtol=1e-2)
        xo.assert_allclose(np.std(p_line.pzeta), np.std(p_matrix.pzeta), rtol=2e-2)
        xo.assert_allclose(np.std(p_line.x), np.std(p_matrix.x), rtol=1e-2)
        xo.assert_allclose(np.std(p_line.px), np.std(p_matrix.px), rtol=1e-2)
        xo.assert_allclose(np.std(p_line.y), np.std(p_matrix.y), rtol=1e-2)
        xo.assert_allclose(np.std(p_line.py), np.std(p_matrix.py), rtol=1e-2)

        # Compare twiss
        tw_line = line.twiss()
        tw_matrix = line_matrix.twiss()

        if configuration == 'above transition':
            assert tw_line.bets0 > 0
            assert tw_matrix.bets0 > 0
            assert tw_line.slip_factor > 0
            assert tw_matrix.slip_factor > 0
        elif configuration == 'below transition':
            assert tw_line.bets0 < 0
            assert tw_matrix.bets0 < 0
            assert tw_line.slip_factor < 0
            assert tw_matrix.slip_factor < 0
        else:
            raise ValueError('Unknown configuration')

        line_frac_qx = np.mod(tw_line.qx, 1)
        line_frac_qy = np.mod(tw_line.qy, 1)
        matrix_frac_qx = np.mod(tw_matrix.qx, 1)
        matrix_frac_qy = np.mod(tw_matrix.qy, 1)

        xo.assert_allclose(line_frac_qx, matrix_frac_qx, atol=1e-5, rtol=0)
        xo.assert_allclose(line_frac_qy, matrix_frac_qy, atol=1e-5, rtol=0)
        xo.assert_allclose(tw_line.betx[0], tw_matrix.betx[0], atol=1e-5, rtol=0)
        xo.assert_allclose(tw_line.alfx[0], tw_matrix.alfx[0], atol=1e-5, rtol=0)
        xo.assert_allclose(tw_line.bety[0], tw_matrix.bety[0], atol=1e-5, rtol=0)
        xo.assert_allclose(tw_line.alfy[0], tw_matrix.alfy[0], atol=1e-5, rtol=0)
        xo.assert_allclose(tw_line.dx[0], tw_matrix.dx[0], atol=1e-5, rtol=0)
        xo.assert_allclose(tw_line.dpx[0], tw_matrix.dpx[0], atol=1e-5, rtol=0)
        xo.assert_allclose(tw_line.dy[0], tw_matrix.dy[0], atol=1e-5, rtol=0)
        xo.assert_allclose(tw_line.dpy[0], tw_matrix.dpy[0], atol=1e-5, rtol=0)

        assert tw_matrix.s[0] == 0
        xo.assert_allclose(tw_matrix.s[-1], tw_line.circumference, rtol=0, atol=1e-6)
        xo.assert_allclose(tw_matrix.bets0, tw_line.bets0, rtol=1e-2, atol=0)

        xo.assert_allclose(np.squeeze(mon.zeta), np.squeeze(mon_matrix.zeta),
                        rtol=0, atol=2e-2*np.max(np.squeeze(mon.zeta)))
        xo.assert_allclose(np.squeeze(mon.pzeta), np.squeeze(mon_matrix.pzeta),
                            rtol=0, atol=3e-2*np.max(np.squeeze(mon.pzeta)))

        particles_matrix = xp.generate_matched_gaussian_bunch(num_particles=1000000,
            nemitt_x=1e-6, nemitt_y=1e-6, sigma_z=sigmaz, line=line_matrix)

        particles_line = xp.generate_matched_gaussian_bunch(num_particles=1000000,
            nemitt_x=1e-6, nemitt_y=1e-6, sigma_z=sigmaz, line=line)

        particles_matrix.move(_context=xo.context_default)
        particles_line.move(_context=xo.context_default)

        xo.assert_allclose(np.std(particles_matrix.zeta), np.std(particles_line.zeta),
                        atol=0, rtol=2e-2)
        xo.assert_allclose(np.std(particles_matrix.pzeta), np.std(particles_line.pzeta),
            atol=0, rtol=(25e-2 if longitudinal_mode.startswith('linear') else 2e-2))

@for_all_test_contexts
def test_custom_twiss_init(test_context):

    line = xt.load(test_data_folder /
            'hllhc15_noerrors_nobb/line_w_knobs_and_particle.json')
    line.particle_ref = xp.Particles(
                        mass0=xp.PROTON_MASS_EV, q0=1, energy0=7e12)
    line.build_tracker(_context=test_context)
    line.vars['on_disp'] = 1

    tw = line.twiss()

    init_at = 'e.cell.45.b1'

    x = tw['x', init_at]
    y = tw['y', init_at]
    px = tw['px', init_at]
    py = tw['py', init_at]
    zeta = tw['zeta', init_at]
    delta = tw['delta', init_at]
    betx = tw['betx', init_at]
    bety = tw['bety', init_at]
    alfx = tw['alfx', init_at]
    alfy = tw['alfy', init_at]
    dx = tw['dx', init_at]
    dy = tw['dy', init_at]
    dpx = tw['dpx', init_at]
    dpy = tw['dpy', init_at]
    mux = tw['mux', init_at]
    muy = tw['muy', init_at]
    muzeta = tw['muzeta', init_at]
    dzeta = tw['dzeta', init_at]
    bets = tw.bets0
    reference_frame = 'proper'

    tw_init = xt.TwissInit(element_name=init_at,
        x=x, px=px, y=y, py=py, zeta=zeta, delta=delta,
        betx=betx, bety=bety, alfx=alfx, alfy=alfy,
        dx=dx, dy=dy, dpx=dpx, dpy=dpy,
        mux=mux, muy=muy, muzeta=muzeta, dzeta=dzeta,
        bets=bets, reference_frame=reference_frame)

    tw_test = line.twiss(start=init_at, end='ip6', init=tw_init)

    assert tw_test.name[-1] == '_end_point'
    tw_part = tw.rows['e.cell.45.b1':'ip6']

    tw_test = tw_test.rows[:-1]
    assert np.all(tw_test.name == tw_part.name)

    atols = dict(
        alfx=1e-8, alfy=1e-8,
        dzeta=1e-3, dx=1e-4, dy=1e-4, dpx=1e-5, dpy=1e-5,
        nuzeta=1e-5, dx_zeta=1e-4, dy_zeta=1e-4, betx2=1e-3, bety1=1e-3,
        muzeta=1e-7,
    )

    rtols = dict(
        alfx=5e-9, alfy=5e-8,
        betx=1e-8, bety=1e-8, betx1=1e-8, bety2=1e-8,
        gamx=1e-8, gamy=1e-8,
    )

    atol_default = 1e-11
    rtol_default = 1e-9


    for kk in tw_test._data.keys():
        if kk in ['name', 'W_matrix', 'particle_on_co', 'values_at', 'method',
                'radiation_method', 'reference_frame', 'orientation']:
            continue # tested separately
        atol = atols.get(kk, atol_default)
        rtol = rtols.get(kk, rtol_default)
        xo.assert_allclose(
            tw_test._data[kk], tw_part._data[kk], rtol=rtol, atol=atol)

    assert tw_test.values_at == tw_part.values_at == 'entry'
    assert tw_test.radiation_method == tw_part.radiation_method == 'full'
    assert tw_test.reference_frame == tw_part.reference_frame == 'proper'

    W_matrix_part = tw_part.W_matrix
    W_matrix_test = tw_test.W_matrix

    for ss in range(W_matrix_part.shape[0]):
        this_part = W_matrix_part[ss, :, :]
        this_test = W_matrix_test[ss, :, :]

        for ii in range(4):
            xo.assert_allclose((np.linalg.norm(this_part[ii, :] - this_test[ii, :])
                            /np.linalg.norm(this_part[ii, :])), 0, atol=3e-4)

@for_all_test_contexts
def test_crab_dispersion(test_context):

    collider = xt.load(test_data_folder /
                        'hllhc15_collider/collider_00_from_mad.json')
    collider.build_trackers(_context=test_context)

    collider.vars['vrf400'] = 16
    collider.vars['on_crab1'] = -190
    collider.vars['on_crab5'] = -190

    line = collider.lhcb1

    tw6d_rf_on = line.twiss()
    tw4d_rf_on = line.twiss(method='4d')

    collider.vars['vrf400'] = 0
    tw4d_rf_off = line.twiss(method='4d')

    collider.vars['vrf400'] = 16
    collider.vars['on_crab1'] = 0
    collider.vars['on_crab5'] = 0

    line = collider.lhcb1

    tw6d_rf_on_crab_off = line.twiss()
    tw4d_rf_on_crab_off = line.twiss(method='4d')

    xo.assert_allclose(tw4d_rf_on['delta'], 0, rtol=0, atol=1e-12)
    xo.assert_allclose(tw4d_rf_off['delta'], 0, rtol=0, atol=1e-12)
    xo.assert_allclose(tw4d_rf_on_crab_off['delta'], 0, rtol=0, atol=1e-12)

    xo.assert_allclose(tw6d_rf_on['dx_zeta', 'ip1'], -190e-6, rtol=1e-4, atol=0)
    xo.assert_allclose(tw6d_rf_on['dy_zeta', 'ip5'], -190e-6, rtol=1e-4, atol=0)
    xo.assert_allclose(tw4d_rf_on['dx_zeta', 'ip1'], -190e-6, rtol=1e-4, atol=0)
    xo.assert_allclose(tw4d_rf_on['dy_zeta', 'ip5'], -190e-6, rtol=1e-4, atol=0)
    xo.assert_allclose(tw4d_rf_off['dx_zeta', 'ip1'], -190e-6, rtol=1e-4, atol=0)
    xo.assert_allclose(tw4d_rf_off['dy_zeta', 'ip5'], -190e-6, rtol=1e-4, atol=0)

    xo.assert_allclose(tw6d_rf_on_crab_off['dx_zeta'], 0, rtol=0, atol=1e-8)
    xo.assert_allclose(tw6d_rf_on_crab_off['dy_zeta'], 0, rtol=0, atol=1e-8)
    xo.assert_allclose(tw4d_rf_on_crab_off['dx_zeta'], 0, rtol=0, atol=1e-8)
    xo.assert_allclose(tw4d_rf_on_crab_off['dy_zeta'], 0, rtol=0, atol=1e-8)

    xo.assert_allclose(tw6d_rf_on['dx_zeta'], tw4d_rf_on['dx_zeta'], rtol=0, atol=1e-7)
    xo.assert_allclose(tw6d_rf_on['dy_zeta'], tw4d_rf_on['dy_zeta'], rtol=0, atol=1e-7)
    xo.assert_allclose(tw6d_rf_on['dx_zeta'], tw4d_rf_off['dx_zeta'], rtol=0, atol=1e-7)
    xo.assert_allclose(tw6d_rf_on['dy_zeta'], tw4d_rf_off['dy_zeta'], rtol=0, atol=1e-7)

@for_all_test_contexts
def test_momentum_crab_dispersion(test_context):

    collider = xt.load(test_data_folder /
                        'hllhc15_collider/collider_00_from_mad.json')
    collider.build_trackers(_context=test_context)

    collider.vars['vrf400'] = 16
    collider.vars['on_crab1'] = -190
    collider.vars['on_crab5'] = -190

    line = collider.lhcb1

    tw6d = line.twiss(method='6d')
    dz = 1e-3
    tw4d_plus = line.twiss(method='4d', zeta0=dz, freeze_longitudinal=True)
    tw4d_minus = line.twiss(method='4d', zeta0=-dz, freeze_longitudinal=True)
    dpx_dzeta_expected = (tw4d_plus.px -  tw4d_minus.px) / (2*dz)
    dpy_dzeta_expected = (tw4d_plus.py -  tw4d_minus.py) / (2*dz)

    xo.assert_allclose(tw6d['dpx_zeta'], dpx_dzeta_expected, rtol=0, atol=1e-7)
    xo.assert_allclose(tw6d['dpy_zeta'], dpy_dzeta_expected, rtol=0, atol=1e-7)


@for_all_test_contexts
def test_twiss_init_file(test_context):

    path_line_particles = test_data_folder / 'hllhc15_noerrors_nobb/line_and_particle.json'

    with open(path_line_particles, 'r') as fid:
        input_data = json.load(fid)
    line = xt.Line.from_dict(input_data['line'])
    line.particle_ref = xp.Particles.from_dict(input_data['particle'])

    line.build_tracker(_context=test_context)

    location = 'ip5'

    tw_full = line.twiss()

    twinit_file = pathlib.Path('twiss_init_save_test.json')
    tw_init_base = tw_full.get_twiss_init(location)
    tw_init_base.to_json(twinit_file)
    tw_init_loaded = xt.TwissInit.from_json(twinit_file)

    # Check that the saving and loading produce the same results
    particle_check_fields = [kk for kk in tw_init_base.particle_on_co._xofields
                              if not kk.startswith('_')]
    for key, val in tw_init_base.__dict__.items():
        if val is None:
            assert tw_init_loaded.__dict__[key] is None
        elif isinstance(val, str):
            assert tw_init_loaded.__dict__[key] == val
        elif key == 'particle_on_co':
            loaded_pco = getattr(tw_init_loaded, key)
            for field in particle_check_fields:
                xo.assert_allclose(getattr(val, field), getattr(loaded_pco, field),
                                  atol=1e-9, rtol=0)
        else:
            xo.assert_allclose(tw_init_loaded.__dict__[key], val,  atol=1e-9, rtol=0)

    tw = line.twiss(start=location, end='ip7', init=tw_init_loaded)

    check_vars = ['betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx', 'dy', 'dpy',
                    'mux', 'muy', 'x', 'y', 'px', 'py']

    # check at a location downsteam
    loc_check = line.element_names[line.element_names.index(location) + 300]
    for var in check_vars:
        # Check at starting point
        xo.assert_allclose(tw[var, location], tw_full[var, location], atol=1e-9, rtol=0)

        # Check at a point in a downstream arc
        xo.assert_allclose(tw[var, loc_check], tw_full[var, loc_check], atol=2e-7, rtol=0)

    twinit_file.unlink()


@for_all_test_contexts
def test_custom_twiss_init(test_context):

    collider = xt.load(
        test_data_folder / 'hllhc15_thick/hllhc15_collider_thick.json')

    collider.lhcb1.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Uniform(1, mode='thick'))])
    collider.lhcb2.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Uniform(1, mode='thick'))])
    collider.build_trackers(_context=test_context)

    # Switch on crossing angles to get some vertical dispersion
    collider.vars['on_x1'] = 500
    collider.vars['on_x5'] = 500
    collider.vars['on_disp'] = 0

    for bn in ['b1', 'b2']:

        line = collider[f'lhc{bn}']
        if bn == 'b2':
            assert line.twiss_default['reverse'] is True
        else:
            assert 'reverse' not in line.twiss_default

        location = f'mb.a28l5.{bn}_entry'

        tw_full = line.twiss()
        betx0 = tw_full['betx', location]
        bety0 = tw_full['bety', location]
        alfx0 = tw_full['alfx', location]
        alfy0 = tw_full['alfy', location]
        dx0 = tw_full['dx', location]
        dpx0 = tw_full['dpx', location]
        dy0 = tw_full['dy', location]
        dpy0 = tw_full['dpy', location]
        mux0 = tw_full['mux', location]
        muy0 = tw_full['muy', location]
        x0 = tw_full['x', location]
        px0 = tw_full['px', location]
        y0 = tw_full['y', location]
        py0 = tw_full['py', location]

        tw_init_custom = xt.TwissInit(
                                betx=betx0, bety=bety0, alfx=alfx0, alfy=alfy0,
                                dx=dx0, dpx=dpx0, dy=dy0, dpy=dpy0,
                                mux=mux0, muy=muy0, x=x0, px=px0, y=y0, py=py0)

        tw = line.twiss(start=location, end='ip7', init=tw_init_custom)

        # Check at starting point
        xo.assert_allclose(tw['betx', location], betx0, atol=1e-9, rtol=0)
        xo.assert_allclose(tw['bety', location], bety0, atol=1e-9, rtol=0)
        xo.assert_allclose(tw['alfx', location], alfx0, atol=1e-9, rtol=0)
        xo.assert_allclose(tw['alfy', location], alfy0, atol=1e-9, rtol=0)
        xo.assert_allclose(tw['dx', location], dx0, atol=1e-9, rtol=0)
        xo.assert_allclose(tw['dpx', location], dpx0, atol=1e-9, rtol=0)
        xo.assert_allclose(tw['dy', location], dy0, atol=1e-9, rtol=0)
        xo.assert_allclose(tw['dpy', location], dpy0, atol=1e-9, rtol=0)
        xo.assert_allclose(tw['mux', location], mux0, atol=1e-9, rtol=0)
        xo.assert_allclose(tw['muy', location], muy0, atol=1e-9, rtol=0)
        xo.assert_allclose(tw['x', location], x0, atol=1e-9, rtol=0)
        xo.assert_allclose(tw['px', location], px0, atol=1e-9, rtol=0)
        xo.assert_allclose(tw['y', location], y0, atol=1e-9, rtol=0)
        xo.assert_allclose(tw['py', location], py0, atol=1e-9, rtol=0)

        # Check at a point in a downstream arc
        loc_check = f'mb.a24l7.{bn}..0'
        xo.assert_allclose(tw['betx', loc_check], tw_full['betx', loc_check],
                            atol=5e-6, rtol=0)
        xo.assert_allclose(tw['bety', loc_check], tw_full['bety', loc_check],
                            atol=5e-6, rtol=0)
        xo.assert_allclose(tw['alfx', loc_check], tw_full['alfx', loc_check],
                            atol=1e-7, rtol=0)
        xo.assert_allclose(tw['alfy', loc_check], tw_full['alfy', loc_check],
                            atol=1e-7, rtol=0)
        xo.assert_allclose(tw['dx', loc_check], tw_full['dx', loc_check],
                            atol=5e-8, rtol=0)
        xo.assert_allclose(tw['dpx', loc_check], tw_full['dpx', loc_check],
                            atol=1e-8, rtol=0)
        xo.assert_allclose(tw['dy', loc_check], tw_full['dy', loc_check],
                            atol=5e-8, rtol=0)
        xo.assert_allclose(tw['dpy', loc_check], tw_full['dpy', loc_check],
                            atol=1e-8, rtol=0)
        xo.assert_allclose(tw['mux', loc_check], tw_full['mux', loc_check],
                            atol=3e-9, rtol=0)
        xo.assert_allclose(tw['muy', loc_check], tw_full['muy', loc_check],
                            atol=3e-9, rtol=0)
        xo.assert_allclose(tw['x', loc_check], tw_full['x', loc_check],
                            atol=1e-9, rtol=0)
        xo.assert_allclose(tw['px', loc_check], tw_full['px', loc_check],
                            atol=1e-9, rtol=0)
        xo.assert_allclose(tw['y', loc_check], tw_full['y', loc_check],
                            atol=1e-9, rtol=0)
        xo.assert_allclose(tw['py', loc_check], tw_full['py', loc_check],
                            atol=1e-9, rtol=0)

        # twiss with boundary confitions at the end of the range
        tw_init_custom = xt.TwissInit(element_name=location,
                                betx=betx0, bety=bety0, alfx=alfx0, alfy=alfy0,
                                dx=dx0, dpx=dpx0, dy=dy0, dpy=dpy0,
                                mux=mux0, muy=muy0, x=x0, px=px0, y=y0, py=py0
                                )

        tw = line.twiss(start='ip4', end=location, init=tw_init_custom)

        # Check at end point
        xo.assert_allclose(tw['betx', location], betx0, atol=1e-9, rtol=0)
        xo.assert_allclose(tw['bety', location], bety0, atol=1e-9, rtol=0)
        xo.assert_allclose(tw['alfx', location], alfx0, atol=1e-9, rtol=0)
        xo.assert_allclose(tw['alfy', location], alfy0, atol=1e-9, rtol=0)
        xo.assert_allclose(tw['dx', location], dx0, atol=1e-9, rtol=0)
        xo.assert_allclose(tw['dpx', location], dpx0, atol=1e-9, rtol=0)
        xo.assert_allclose(tw['dy', location], dy0, atol=1e-9, rtol=0)
        xo.assert_allclose(tw['dpy', location], dpy0, atol=1e-9, rtol=0)
        xo.assert_allclose(tw['mux', location], mux0, atol=1e-9, rtol=0)
        xo.assert_allclose(tw['muy', location], muy0, atol=1e-9, rtol=0)
        xo.assert_allclose(tw['x', location], x0, atol=1e-9, rtol=0)
        xo.assert_allclose(tw['px', location], px0, atol=1e-9, rtol=0)
        xo.assert_allclose(tw['y', location], y0, atol=1e-9, rtol=0)

        # Check at a point in an upstream arc
        loc_check = f'mb.a24r4.{bn}..0'
        xo.assert_allclose(tw['betx', loc_check], tw_full['betx', loc_check],
                            atol=5e-6, rtol=0)
        xo.assert_allclose(tw['bety', loc_check], tw_full['bety', loc_check],
                            atol=5e-6, rtol=0)
        xo.assert_allclose(tw['alfx', loc_check], tw_full['alfx', loc_check],
                            atol=5e-8, rtol=0)
        xo.assert_allclose(tw['alfy', loc_check], tw_full['alfy', loc_check],
                            atol=5e-8, rtol=0)
        xo.assert_allclose(tw['dx', loc_check], tw_full['dx', loc_check],
                            atol=1e-8, rtol=0)
        xo.assert_allclose(tw['dpx', loc_check], tw_full['dpx', loc_check],
                            atol=1e-8, rtol=0)
        xo.assert_allclose(tw['dy', loc_check], tw_full['dy', loc_check],
                            atol=1e-8, rtol=0)
        xo.assert_allclose(tw['dpy', loc_check], tw_full['dpy', loc_check],
                            atol=1e-8, rtol=0)
        xo.assert_allclose(tw['mux', loc_check], tw_full['mux', loc_check],
                            atol=5e-9, rtol=0)
        xo.assert_allclose(tw['muy', loc_check], tw_full['muy', loc_check],
                            atol=5e-9, rtol=0)
        xo.assert_allclose(tw['x', loc_check], tw_full['x', loc_check],
                            atol=1e-9, rtol=0)
        xo.assert_allclose(tw['px', loc_check], tw_full['px', loc_check],
                            atol=1e-9, rtol=0)
        xo.assert_allclose(tw['y', loc_check], tw_full['y', loc_check],
                            atol=1e-9, rtol=0)
        xo.assert_allclose(tw['py', loc_check], tw_full['py', loc_check],
                            atol=1e-9, rtol=0)

@for_all_test_contexts
def test_adaptive_steps_for_rmatrix(test_context):

    collider = xt.load(
        test_data_folder / 'hllhc15_thick/hllhc15_collider_thick.json')
    collider.build_trackers(_context=test_context)
    collider.lhcb1.twiss_default['method'] = '4d'
    collider.lhcb2.twiss_default['method'] = '4d'
    collider.lhcb2.twiss_default['reverse'] = True

    collider.lhcb1.twiss_default['nemitt_x'] = 1e-6
    collider.lhcb1.twiss_default['nemitt_y'] = 1e-6
    collider.lhcb2.twiss_default['nemitt_x'] = 1e-6
    collider.lhcb2.twiss_default['nemitt_y'] = 1e-6

    tw = collider.twiss()
    assert tw.lhcb1.steps_r_matrix['adapted'] == False
    assert tw.lhcb2.steps_r_matrix['adapted'] == False

    collider.lhcb1.twiss_default['nemitt_x'] = 1e-8
    tw = collider.twiss()
    assert tw.lhcb1.steps_r_matrix['adapted'] == True
    assert tw.lhcb2.steps_r_matrix['adapted'] == False

    collider.lhcb2.twiss_default['nemitt_y'] = 2e-8
    tw = collider.twiss()
    assert tw.lhcb1.steps_r_matrix['adapted'] == True
    assert tw.lhcb2.steps_r_matrix['adapted'] == True

    expected_dx_b1 = 0.01 * np.sqrt(1e-8 * 0.15 / collider.lhcb1.particle_ref._xobject.gamma0[0])
    expected_dy_b1 = 0.01 * np.sqrt(1e-6 * 0.15 / collider.lhcb1.particle_ref._xobject.gamma0[0])
    expected_dx_b2 = 0.01 * np.sqrt(1e-6 * 0.15 / collider.lhcb1.particle_ref._xobject.gamma0[0])
    expected_dy_b2 = 0.01 * np.sqrt(2e-8 * 0.15 / collider.lhcb2.particle_ref._xobject.gamma0[0])

    xo.assert_allclose(tw.lhcb1.steps_r_matrix['dx'], expected_dx_b1, atol=0, rtol=1e-4)
    xo.assert_allclose(tw.lhcb1.steps_r_matrix['dy'], expected_dy_b1, atol=0, rtol=1e-4)
    xo.assert_allclose(tw.lhcb2.steps_r_matrix['dx'], expected_dx_b2, atol=0, rtol=1e-4)
    xo.assert_allclose(tw.lhcb2.steps_r_matrix['dy'], expected_dy_b2, atol=0, rtol=1e-4)

@for_all_test_contexts
def test_longitudinal_beam_sizes(test_context):

    # Load a line and build tracker
    line = xt.load(test_data_folder /
        'hllhc15_noerrors_nobb/line_and_particle.json')
    line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, energy0=7e12)
    line.build_tracker(_context=test_context)

    tw = line.twiss()

    nemitt_x = 2.5e-6
    nemitt_y = 2.5e-6

    sigma_pzeta = 2e-4
    gemitt_zeta = sigma_pzeta**2 * tw.bets0

    beam_sizes = tw.get_beam_covariance(nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                                        gemitt_zeta=gemitt_zeta)

    xo.assert_allclose(beam_sizes.sigma_pzeta, 2e-4, atol=0, rtol=2e-5)
    xo.assert_allclose(
        beam_sizes.sigma_zeta / beam_sizes.sigma_pzeta, tw.bets0, atol=0, rtol=5e-5)

@for_all_test_contexts
def test_second_order_chromaticity_and_dispersion(test_context):

    line = xt.load(test_data_folder /
                             'hllhc15_thick/lhc_thick_with_knobs.json')
    line.vars['on_x5'] = 300
    line.build_tracker(_context=test_context)

    tw = line.twiss(method='4d')
    tw_fw = line.twiss(start='ip4', end='ip6', init_at='ip4',
                x=tw['x', 'ip4'], px=tw['px', 'ip4'],
                y=tw['y', 'ip4'], py=tw['py', 'ip4'],
                betx=tw['betx', 'ip4'], bety=tw['bety', 'ip4'],
                alfx=tw['alfx', 'ip4'], alfy=tw['alfy', 'ip4'],
                dx=tw['dx', 'ip4'], dpx=tw['dpx', 'ip4'],
                dy=tw['dy', 'ip4'], dpy=tw['dpy', 'ip4'],
                ddx=tw['ddx', 'ip4'], ddy=tw['ddy', 'ip4'],
                ddpx=tw['ddpx', 'ip4'], ddpy=tw['ddpy', 'ip4'],
                compute_chromatic_properties=True)

    tw_bw = line.twiss(start='ip4', end='ip6', init_at='ip6',
                x=tw['x', 'ip6'], px=tw['px', 'ip6'],
                y=tw['y', 'ip6'], py=tw['py', 'ip6'],
                betx=tw['betx', 'ip6'], bety=tw['bety', 'ip6'],
                alfx=tw['alfx', 'ip6'], alfy=tw['alfy', 'ip6'],
                dx=tw['dx', 'ip6'], dpx=tw['dpx', 'ip6'],
                dy=tw['dy', 'ip6'], dpy=tw['dpy', 'ip6'],
                ddx=tw['ddx', 'ip6'], ddy=tw['ddy', 'ip6'],
                ddpx=tw['ddpx', 'ip6'], ddpy=tw['ddpy', 'ip6'],
                compute_chromatic_properties=True)

    nlchr = line.get_non_linear_chromaticity(delta0_range=(-1e-4, 1e-4),
                                             num_delta=21, fit_order=2)

    location = 'ip3'

    x_xs = np.array([tt['x', location] for tt in nlchr.twiss])
    px_xs = np.array([tt['px', location] for tt in nlchr.twiss])
    y_xs = np.array([tt['y', location] for tt in nlchr.twiss])
    py_xs = np.array([tt['py', location] for tt in nlchr.twiss])
    qx_xs = np.array([tt['qx'] for tt in nlchr.twiss])
    qy_xs = np.array([tt['qy'] for tt in nlchr.twiss])
    delta = np.array([tt['delta', location] for tt in nlchr.twiss])

    pxs_x = np.polyfit(delta, x_xs, 3)
    pxs_px = np.polyfit(delta, px_xs, 3)
    pxs_y = np.polyfit(delta, y_xs, 3)
    pxs_py = np.polyfit(delta, py_xs, 3)
    pxs_qx = np.polyfit(delta, qx_xs, 3)
    pxs_qy = np.polyfit(delta, qy_xs, 3)

    xo.assert_allclose(delta, nlchr.delta0, atol=1e-6, rtol=0)
    xo.assert_allclose(tw['dx', location], pxs_x[-2], atol=0, rtol=1e-4)
    xo.assert_allclose(tw['dpx', location], pxs_px[-2], atol=0, rtol=1e-4)
    xo.assert_allclose(tw['dy', location], pxs_y[-2], atol=0, rtol=1e-4)
    xo.assert_allclose(tw['dpy', location], pxs_py[-2], atol=0, rtol=1e-4)
    xo.assert_allclose(tw['ddx', location], 2*pxs_x[-3], atol=0, rtol=1e-4)
    xo.assert_allclose(tw['ddpx', location], 2*pxs_px[-3], atol=0, rtol=1e-4)
    xo.assert_allclose(tw['ddy', location], 2*pxs_y[-3], atol=0, rtol=1e-4)
    xo.assert_allclose(tw['ddpy', location], 2*pxs_py[-3], atol=0, rtol=1e-4)
    xo.assert_allclose(tw['dqx'], pxs_qx[-2], atol=0, rtol=1e-3)
    xo.assert_allclose(tw['ddqx'], pxs_qx[-3]*2, atol=0, rtol=1e-2)
    xo.assert_allclose(tw['dqy'], pxs_qy[-2], atol=0, rtol=1e-3)
    xo.assert_allclose(tw['ddqy'], pxs_qy[-3]*2, atol=0, rtol=1e-2)

    xo.assert_allclose(nlchr['dqx'], pxs_qx[-2], atol=0, rtol=2e-3)
    xo.assert_allclose(nlchr['dqy'], pxs_qy[-2], atol=0, rtol=2e-3)
    xo.assert_allclose(nlchr['ddqx'], pxs_qx[-3]*2, atol=0, rtol=1e-4)
    xo.assert_allclose(nlchr['ddqy'], pxs_qy[-3]*2, atol=0, rtol=1e-4)

    tw_part = tw.rows['ip4':'ip6']
    xo.assert_allclose(tw_part['ddx'], tw_fw.rows[:-1]['ddx'], atol=1e-2, rtol=0)
    xo.assert_allclose(tw_part['ddy'], tw_fw.rows[:-1]['ddy'], atol=1e-2, rtol=0)
    xo.assert_allclose(tw_part['ddpx'], tw_fw.rows[:-1]['ddpx'], atol=1e-3, rtol=0)
    xo.assert_allclose(tw_part['ddpy'], tw_fw.rows[:-1]['ddpy'], atol=1e-3, rtol=0)
    xo.assert_allclose(tw_part['dx'], tw_bw.rows[:-1]['dx'], atol=1e-2, rtol=0)
    xo.assert_allclose(tw_part['dy'], tw_bw.rows[:-1]['dy'], atol=1e-2, rtol=0)
    xo.assert_allclose(tw_part['dpx'], tw_bw.rows[:-1]['dpx'], atol=1e-3, rtol=0)
    xo.assert_allclose(tw_part['dpy'], tw_bw.rows[:-1]['dpy'], atol=1e-3, rtol=0)

@for_all_test_contexts
def test_twiss_strength_reverse_vs_madx(test_context):

    test_data_folder_str = str(test_data_folder)

    mad1=Madx(stdout=False)
    mad1.call(test_data_folder_str + '/hllhc15_thick/lhc.seq')
    mad1.call(test_data_folder_str + '/hllhc15_thick/hllhc_sequence.madx')
    mad1.input('beam, sequence=lhcb1, particle=proton, energy=7000;')
    mad1.use('lhcb1')
    mad1.input('beam, sequence=lhcb2, particle=proton, energy=7000, bv=-1;')
    mad1.use('lhcb2')
    mad1.call(test_data_folder_str + '/hllhc15_thick/opt_round_150_1500.madx')

    mad1.globals['on_x1'] = 100 # Check kicker expressions
    mad1.globals['on_sep2'] = 2 # Check kicker expressions
    mad1.globals['on_x5'] = 123 # Check kicker expressions

    mad1.globals['kqs.a23b1'] = 1e-4 # Check skew quad expressions
    mad1.globals['kqs.a12b2'] = 1e-4 # Check skew quad expressions

    mad1.globals['ksf.b1'] = 1e-3  # Check sext expressions
    mad1.globals['ksf.b2'] = 1e-3  # Check sext expressions

    mad1.globals['kss.a45b1'] = 1e-4 # Check skew sext expressions
    mad1.globals['kss.a45b2'] = 1e-4 # Check skew sext expressions

    mad1.globals['kof.a34b1'] = 3 # Check oct expressions
    mad1.globals['kof.a34b2'] = 3 # Check oct expressions

    mad1.globals['kcosx3.l2'] = 5 # Check skew oct expressions

    mad1.globals['kcdx3.r1'] = 1e-4 # Check thin decapole expressions

    mad1.globals['kcdsx3.r1'] = 1e-4 # Check thin skew decapole expressions

    mad1.globals['kctx3.l1'] = 1e-5 # Check thin dodecapole expressions

    mad1.globals['kctsx3.r1'] = 1e-5 # Check thin skew dodecapole expressions

    mad1.input('twiss, sequence=lhcb1, table=twisslhcb1;')
    mad1.input('twiss, sequence=lhcb2, table=twisslhcb2;')

    twm1 = xt.Table(mad1.table.twisslhcb1)
    twm2 = xt.Table(mad1.table.twisslhcb2)

    collider = xt.Environment.from_madx(madx=mad1)

    collider.build_trackers(_context=test_context)
    tw = collider.twiss(strengths=True, method='4d')

    # Normal strengths
    assert_allclose = xo.assert_allclose
    assert_allclose(twm1['k0l', 'mb.a20r8.b1:1'], tw.lhcb1['k0l', 'mb.a20r8.b1'], rtol=0, atol=1e-14)
    assert_allclose(twm2['k0l', 'mb.a20r8.b2:1'], tw.lhcb2['k0l', 'mb.a20r8.b2'], rtol=0, atol=1e-14)

    assert_allclose(twm1['k1l', 'mq.22l3.b1:1'], tw.lhcb1['k1l', 'mq.22l3.b1'], rtol=0, atol=1e-14)
    assert_allclose(twm2['k1l', 'mq.22l3.b2:1'], tw.lhcb2['k1l', 'mq.22l3.b2'], rtol=0, atol=1e-14)

    assert_allclose(twm1['k2l', 'ms.16l1.b1:1'], tw.lhcb1['k2l', 'ms.16l1.b1'], rtol=0, atol=1e-14)
    assert_allclose(twm2['k2l', 'ms.16l1.b2:1'], tw.lhcb2['k2l', 'ms.16l1.b2'], rtol=0, atol=1e-14)

    assert_allclose(twm1['k3l', 'mo.25l4.b1:1'], tw.lhcb1['k3l', 'mo.25l4.b1'], rtol=0, atol=1e-14)
    assert_allclose(twm2['k3l', 'mo.24l4.b2:1'], tw.lhcb2['k3l', 'mo.24l4.b2'], rtol=0, atol=1e-14)

    assert_allclose(twm1['k4l', 'mcdxf.3r1:1'], tw.lhcb1['k4l', 'mcdxf.3r1/lhcb1'], rtol=0, atol=1e-14)
    assert_allclose(twm2['k4l', 'mcdxf.3r1:1'], tw.lhcb2['k4l', 'mcdxf.3r1/lhcb2'], rtol=0, atol=1e-14)

    assert_allclose(twm1['k5l', 'mctxf.3l1:1'], tw.lhcb1['k5l', 'mctxf.3l1/lhcb1'], rtol=0, atol=1e-14)
    assert_allclose(twm2['k5l', 'mctxf.3l1:1'], tw.lhcb2['k5l', 'mctxf.3l1/lhcb2'], rtol=0, atol=1e-14)

    # Skew strengths
    assert_allclose(twm1['k1sl', 'mqs.27l3.b1:1'], tw.lhcb1['k1sl', 'mqs.27l3.b1'], rtol=0, atol=1e-14)
    assert_allclose(twm2['k1sl', 'mqs.23l2.b2:1'], tw.lhcb2['k1sl', 'mqs.23l2.b2'], rtol=0, atol=1e-14)

    assert_allclose(twm1['k2sl', 'mss.28l5.b1:1'], tw.lhcb1['k2sl', 'mss.28l5.b1'], rtol=0, atol=1e-14)
    assert_allclose(twm2['k2sl', 'mss.33l5.b2:1'], tw.lhcb2['k2sl', 'mss.33l5.b2'], rtol=0, atol=1e-14)

    assert_allclose(twm1['k3sl', 'mcosx.3l2:1'], tw.lhcb1['k3sl', 'mcosx.3l2/lhcb1'], rtol=0, atol=1e-14)
    assert_allclose(twm2['k3sl', 'mcosx.3l2:1'], tw.lhcb2['k3sl', 'mcosx.3l2/lhcb2'], rtol=0, atol=1e-14)

    assert_allclose(twm1['k4sl', 'mcdsxf.3r1:1'], tw.lhcb1['k4sl', 'mcdsxf.3r1/lhcb1'], rtol=0, atol=1e-14)
    assert_allclose(twm2['k4sl', 'mcdsxf.3r1:1'], tw.lhcb2['k4sl', 'mcdsxf.3r1/lhcb2'], rtol=0, atol=1e-14)

    assert_allclose(twm1['k5sl', 'mctsxf.3r1:1'], tw.lhcb1['k5sl', 'mctsxf.3r1/lhcb1'], rtol=0, atol=1e-14)
    assert_allclose(twm2['k5sl', 'mctsxf.3r1:1'], tw.lhcb2['k5sl', 'mctsxf.3r1/lhcb2'], rtol=0, atol=1e-14)


@for_all_test_contexts
@pytest.mark.parametrize('line_name', ['lhcb1'])
@pytest.mark.parametrize('section', [
    (xt.START, xt.END),
    (xt.START, '_end_point'),
    (xt.START, 'ip6'),
    ('ip4', xt.END),
])
def test_twiss_range_start_end(test_context, line_name, section, collider_for_test_twiss_range):
    collider = collider_for_test_twiss_range
    init_at = 'ip5'

    collider.vars['on_x5hs'] = 200
    collider.vars['on_x5vs'] = 123
    collider.vars['on_sep5h'] = 1
    collider.vars['on_sep5v'] = 2

    line = collider[line_name]

    if collider.lhcb1.element_names[0] != 'ip3':
        collider.lhcb1.cycle('ip3', inplace=True)
    if collider.lhcb2.element_names[0] != 'ip3':
        collider.lhcb2.cycle('ip3', inplace=True)

    if isinstance(test_context, xo.ContextCpu) and (
        test_context.omp_num_threads != line._context.omp_num_threads):
        buffer = test_context.new_buffer()
    elif isinstance(test_context, line._context.__class__):
        buffer = line._buffer
    else:
        buffer = test_context.new_buffer()

    line.build_tracker(_buffer=buffer)

    reverse = {'lhcb1': False, 'lhcb2':True}[line_name]

    tw = line.twiss(reverse=reverse)
    tw_init = tw.get_twiss_init(init_at)

    start = section[0]
    end = section[1]
    tw_test = line.twiss(start=start, end=end, init=tw_init, reverse=reverse)

    start_el = (line.element_names[0] if start == xt.START else start)
    end_el = (line.element_names[-1] if (end == xt.END or end == '_end_point') else end)
    tw_ref = line.twiss(start=start_el, end=end_el, init=tw_init, reverse=reverse)

    for kk in tw_test._data.keys():
        if kk in ('particle_on_co', '_action', 'completed_init'):
            continue

        if kk in ('name', 'method', 'values_at', 'radiation_method',
                  'reference_frame', 'name_env'):
            assert np.all(tw_test._data[kk] == tw_ref._data[kk])
            continue

        xo.assert_allclose(tw_test._data[kk], tw_ref._data[kk], rtol=1e-12, atol=5e-13)

@for_all_test_contexts
def test_arbitrary_start(test_context, collider_for_test_twiss_range):

    collider = collider_for_test_twiss_range

    # No orbit
    for kk in collider.vars.get_table().rows['on_.*'].name:
        kk = str(kk) # avoid numpy.str_
        collider.vars[kk] = 0

    if collider.lhcb1.element_names[0] != 'ip1':
        collider.lhcb1.cycle('ip1', inplace=True)
    if collider.lhcb2.element_names[0] != 'ip1':
        collider.lhcb2.cycle('ip1', inplace=True)

    line = collider.lhcb2 # <- use lhcb2 to test the reverse option
    assert line.twiss_default['method'] == '4d'
    assert line.twiss_default['reverse']

    if isinstance(test_context, xo.ContextCpu) and (
        test_context.omp_num_threads != line._context.omp_num_threads):
        buffer = test_context.new_buffer()
    elif isinstance(test_context, line._context.__class__):
        buffer = line._buffer
    else:
        buffer = test_context.new_buffer()

    line.build_tracker(_buffer=buffer)

    tw8_closed = line.twiss(start='ip8')
    tw8_open = line.twiss(start='ip8', betx=1.5, bety=1.5)

    tw = line.twiss()

    for tw8 in [tw8_closed, tw8_open]:
        assert tw8.name[-1] == '_end_point'
        assert np.all(tw8.rows['ip.?'].name
            == np.array(['ip8', 'ip1', 'ip2', 'ip3', 'ip4', 'ip5', 'ip6', 'ip7']))

        for nn in ['s', 'mux', 'muy']:
            assert np.all(np.diff(tw8.rows['ip.?'][nn]) > 0)
            assert tw8[nn][0] == 0
            xo.assert_allclose(tw8[nn][-1], tw[nn][-1], rtol=1e-12, atol=5e-7)

        xo.assert_allclose(
                tw8['betx', ['ip8', 'ip1', 'ip2', 'ip3', 'ip4', 'ip5', 'ip6', 'ip7']],
                tw[ 'betx', ['ip8', 'ip1', 'ip2', 'ip3', 'ip4', 'ip5', 'ip6', 'ip7']],
                rtol=1e-5, atol=0)

    collider.to_json('ok.json')

@for_all_test_contexts
def test_part_from_full_periodic(test_context, collider_for_test_twiss_range):

    collider = collider_for_test_twiss_range

    if collider.lhcb1.element_names[0] != 'ip1':
        collider.lhcb1.cycle('ip1', inplace=True)
    if collider.lhcb2.element_names[0] != 'ip1':
        collider.lhcb2.cycle('ip1', inplace=True)

    line = collider.lhcb2 # <- use lhcb2 to test the reverse option
    assert line.twiss_default['method'] == '4d'
    assert line.twiss_default['reverse']

    if isinstance(test_context, xo.ContextCpu) and (
        test_context.omp_num_threads != line._context.omp_num_threads):
        buffer = test_context.new_buffer()
    elif isinstance(test_context, line._context.__class__):
        buffer = line._buffer
    else:
        buffer = test_context.new_buffer()

    line.build_tracker(_buffer=buffer)

    tw = line.twiss()

    tw_part1 = line.twiss(start='ip8', end='ip2', zero_at='ip1', init='full_periodic')

    assert tw_part1.name[0] == 'ip8'
    assert tw_part1.name[-2] == 'ip2'
    assert tw_part1.name[-1] == '_end_point'

    for kk in ['s', 'mux', 'muy']:
        tw_part1[kk, 'ip1'] == 0.
        assert np.all(np.diff(tw_part1[kk]) >= -1e-14)
        xo.assert_allclose(
            tw_part1[kk, 'ip8'], -(tw[kk, '_end_point'] - tw[kk, 'ip8']),
            rtol=1e-12, atol=5e-7)
        xo.assert_allclose(
            tw_part1[kk, 'ip2'], tw[kk, 'ip2'] - tw[kk, 0],
            rtol=1e-12, atol=5e-7)

    tw_part2 = line.twiss(start='ip8', end='ip2', init='full_periodic')

    assert tw_part2.name[0] == 'ip8'
    assert tw_part2.name[-2] == 'ip2'
    assert tw_part2.name[-1] == '_end_point'

    for kk in ['s', 'mux', 'muy']:
        tw_part2[kk, 'ip8'] == 0.
        assert np.all(np.diff(tw_part2[kk]) >= -1e14)
        xo.assert_allclose(
            tw_part2[kk, 'ip2'],
            tw[kk, 'ip2'] - tw[kk, 0] +(tw[kk, '_end_point'] - tw[kk, 'ip8']),
            rtol=1e-12, atol=5e-7)


@for_all_test_contexts
def test_twiss_add_strengths(test_context):
    ## Generate a simple line
    n = 6
    fodo = [
        xt.Multipole(length=0.2, knl=[0, +0.2], ksl=[0, 0]),
        xt.Drift(length=1.0),
        xt.Multipole(length=0.2, knl=[0, -0.2], ksl=[0, 0]),
        xt.Drift(length=1.0),
        xt.Multipole(length=1.0, knl=[2 * np.pi / n], hxl=[2 * np.pi / n]),
        xt.Drift(length=1.0),
    ]
    line = xt.Line(elements=n * fodo + [xt.Cavity(frequency=1e9, voltage=0, lag=180)])
    line.build_tracker(_context=test_context)

    ## Twiss
    line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=1e8)
    tw = line.twiss(method="4d")

    assert "length" not in tw.keys()
    tw.add_strengths()
    assert "length" in tw.keys()

def test_coupling_calculations():

    # Load a line and build tracker
    line = xt.load(test_data_folder /
        'hllhc14_no_errors_with_coupling_knobs/line_b1.json')
    line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, energy0=7e12)
    line.cycle('ip1', inplace=True)
    line.twiss_default['method'] = '4d'

    # Flat machine
    for nn in line.vars.get_table().rows['on_.*|corr_.*'].name:
        line.vars[nn] = 0

    line['cmrskew'] = 0
    line['cmiskew'] = 0
    tw0 = line.twiss()

    line['cmrskew'] = 0.5e-4
    line['cmiskew'] = -0.3e-4

    tw = line.twiss(strengths = True)

    c_min_from_k1s = (0+0j) * tw.s
    for ii in xt.progress_indicator.progress(range(len(tw.s))):
        c_min_from_k1s[ii] = 1 / (2*np.pi) * np.sum(tw.k1sl * np.sqrt(tw0.betx * tw0.bety)
                * np.exp(1j * 2 * np.pi * ((tw0.mux - tw0.mux[ii]) - (tw0.muy - tw0.muy[ii]))))

    xo.assert_allclose(tw.c_minus_re + 1j*tw.c_minus_im, c_min_from_k1s, rtol=5e-2, atol=0)
    # Check phi1
    xo.assert_allclose(tw.c_minus_re + 1j*tw.c_minus_im,
        tw.c_minus * np.exp(1j * tw.c_phi1), rtol=1e-10, atol=0)
    # Check phi2
    xo.assert_allclose(tw.c_minus_re + 1j*tw.c_minus_im,
        tw.c_minus * np.exp(1j * (np.pi - tw.c_phi2)), rtol=3e-2, atol=0)
    # Check r1
    xo.assert_allclose(tw.c_r1, np.sqrt(tw.bety1 / tw.betx1), rtol=1e-5, atol=0)
    # Check r2
    xo.assert_allclose(tw.c_r2, np.sqrt(tw.betx2 / tw.bety2), rtol=1e-5, atol=0)

    # Check c_minus
    xo.assert_allclose(np.abs(tw.c_minus_re + 1j*tw.c_minus_im), tw.c_minus, rtol=1e-10, atol=0)
    xo.assert_allclose(tw.c_minus_re_0, tw.c_minus_re[0], rtol=1e-10, atol=0)
    xo.assert_allclose(tw.c_minus_im_0, tw.c_minus_im[0], rtol=1e-10, atol=0)

def test_twiss_collective_end_is_len():

    d1=xt.Drift(length=1)
    d2=xt.Drift(length=1)
    d3=xt.Drift(length=1)

    d2.iscollective=True

    line=xt.Line([d1,d2,d3])
    line.particle_ref = xt.Particles(energy0=10e9, mass0=xt.PROTON_MASS_EV)
    t = line.twiss4d(betx=1,bety=1,include_collective=True)

    ddd1=xt.Drift(length=1)
    ddd2=xt.Drift(length=1)
    ddd3=xt.Drift(length=1)
    line2=xt.Line([ddd1,ddd2,ddd3])
    line2.particle_ref = xt.Particles(energy0=10e9, mass0=xt.PROTON_MASS_EV)
    t2 = line2.twiss4d(betx=1,bety=1,include_collective=True)

    xo.assert_allclose(t.betx, t2.betx, atol=1e-12, rtol=0)


def test_twiss_compute_coupling_elements_edwards_teng():
    env = xt.load(test_data_folder / 'lhc_2024/lhc.seq')
    env.vars.set_from_madx_file(test_data_folder / 'lhc_2024/injection_optics.madx')

    # Clean machine
    env['on_x1'] = 0
    env['on_x2h'] = 0
    env['on_x2v'] = 0
    env['on_x5'] = 0
    env['on_x8h'] = 0
    env['on_x8v'] = 0

    env['on_sep1'] = 0
    env['on_sep2h'] = 0
    env['on_sep2v'] = 0
    env['on_sep5'] = 0
    env['on_sep8h'] = 0
    env['on_sep8v'] = 0

    env['on_a2'] = 0
    env['on_a8'] = 0

    # Introduce some coupling by powering up a skew quadrupole
    env['kqs.a67b1'] = 1e-4

    line = env.lhcb1
    line.particle_ref = xt.Particles(p0c=450e9)
    tw = line.twiss4d(coupling_edw_teng=True)

    tw_ng = line.madng_twiss()

    # We will correct for the sign, as our result is smoother
    sgn_ng = np.sign(tw.r11_edw_teng / tw_ng.r11_ng)

    # Check against MAD-NG
    xo.assert_allclose(tw.r11_edw_teng, sgn_ng * tw_ng.r11_ng, atol=np.max(np.abs(tw.r11_edw_teng)) * 5e-3, rtol=0, max_outliers=1)
    xo.assert_allclose(tw.r12_edw_teng, sgn_ng * tw_ng.r12_ng, atol=np.max(np.abs(tw.r12_edw_teng)) * 5e-3, rtol=0, max_outliers=1)
    xo.assert_allclose(tw.r21_edw_teng, sgn_ng * tw_ng.r21_ng, atol=np.max(np.abs(tw.r21_edw_teng)) * 5e-3, rtol=0, max_outliers=1)
    xo.assert_allclose(tw.r22_edw_teng, sgn_ng * tw_ng.r22_ng, atol=np.max(np.abs(tw.r22_edw_teng)) * 5e-3, rtol=0, max_outliers=1)
    xo.assert_allclose(tw.f1010, sgn_ng * tw_ng.f1010_ng, atol=3e-5, rtol=0, max_outliers=1)
    xo.assert_allclose(tw.f1001, sgn_ng * tw_ng.f1001_ng, atol=1e-3, rtol=0, max_outliers=1)

    # Test that reverse twiss is disabled for now
    with pytest.raises(NotImplementedError):
        line.twiss4d(coupling_edw_teng=True, reverse=True)

    # Test that reversal removes the quantities for now
    tw_rev = tw.reverse()
    rev_fields = set(dir(tw_rev))
    fields_shouldnt_be_there = {
        'r11_edw_teng', 'r12_edw_teng', 'r21_edw_teng', 'r22_edw_teng',
        'f1010', 'f1001',
    }
    assert not (rev_fields & fields_shouldnt_be_there)
