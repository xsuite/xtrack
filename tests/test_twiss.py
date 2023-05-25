import pathlib
import json

import numpy as np
from cpymad.madx import Madx

import xpart as xp
import xtrack as xt
import xobjects as xo
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
        assert np.allclose(tw.betx, tw_4d_list[0].betx, atol=1e-12, rtol=0)
        assert np.allclose(tw.bety, tw_4d_list[0].bety, atol=1e-12, rtol=0)
        assert np.allclose(tw.alfx, tw_4d_list[0].alfx, atol=1e-12, rtol=0)
        assert np.allclose(tw.alfy, tw_4d_list[0].alfy, atol=1e-12, rtol=0)
        assert np.allclose(tw.dx, tw_4d_list[0].dx, atol=1e-8, rtol=0)
        assert np.allclose(tw.dy, tw_4d_list[0].dy, atol=1e-8, rtol=0)
        assert np.allclose(tw.dpx, tw_4d_list[0].dpx, atol=1e-8, rtol=0)
        assert np.allclose(tw.dpy, tw_4d_list[0].dpy, atol=1e-8, rtol=0)
        assert np.isclose(tw.qx, tw_4d_list[0].qx, atol=1e-7, rtol=0)
        assert np.isclose(tw.qy, tw_4d_list[0].qy, atol=1e-7, rtol=0)
        assert np.isclose(tw.dqx, tw_4d_list[0].dqx, atol=1e-4, rtol=0)
        assert np.isclose(tw.dqy, tw_4d_list[0].dqy, atol=1e-4, rtol=0)


@for_all_test_contexts
def test_coupled_beta(test_context):
    mad = Madx()
    mad.call(str(test_data_folder / 'hllhc15_noerrors_nobb/sequence.madx'))
    mad.use('lhcb1')

    # introduce coupling
    mad.sequence.lhcb1.expanded_elements[7].ksl = [0, 1e-4]
    mad.twiss() # I see to need to do it twice to get the right coupling in madx?!

    tw_mad_coupling = mad.twiss(ripken=True).dframe()

    line = xt.Line.from_madx_sequence(mad.sequence.lhcb1)
    line.particle_ref = xp.Particles(p0c=7000e9, mass0=xp.PROTON_MASS_EV)

    line.build_tracker(_context=test_context)

    tw = line.twiss()

    twdf = tw.to_pandas()
    twdf.set_index('name', inplace=True)

    ips = ['ip1', 'ip2', 'ip3', 'ip4', 'ip5', 'ip6', 'ip7', 'ip8']
    betx2_at_ips = twdf.loc[ips, 'betx2'].values
    bety1_at_ips = twdf.loc[ips, 'bety1'].values

    tw_mad_coupling.set_index('name', inplace=True)
    beta12_mad_at_ips = tw_mad_coupling.loc[[ip + ':1' for ip in ips], 'beta12'].values
    beta21_mad_at_ips = tw_mad_coupling.loc[[ip + ':1' for ip in ips], 'beta21'].values

    assert np.allclose(betx2_at_ips, beta12_mad_at_ips, rtol=1e-4, atol=0)
    assert np.allclose(bety1_at_ips, beta21_mad_at_ips, rtol=1e-4, atol=0)

    #cmin_ref = mad.table.summ.dqmin[0] # dqmin is not calculated correctly in madx
                                        # (https://github.com/MethodicalAcceleratorDesign/MAD-X/issues/1152)
    cmin_ref = 0.001972093557# obtained with madx with trial and error

    assert np.isclose(tw.c_minus, cmin_ref, rtol=0, atol=1e-5)


@for_all_test_contexts
def test_twiss_zeta0_delta0(test_context):
    mad = Madx()
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

    assert np.isclose(phi_c_ip1, -190e-6, atol=1e-7, rtol=0)
    assert np.isclose(phi_c_ip5, -190e-6, atol=1e-7, rtol=0)

    # Check crab dispersion
    tw6d = line.twiss()
    assert np.isclose(tw6d['dx_zeta', 'ip1'], -190e-6, atol=1e-7, rtol=0)
    assert np.isclose(tw6d['dy_zeta', 'ip5'], -190e-6, atol=1e-7, rtol=0)

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

    assert np.allclose(norm_coord['x_norm'], [-1, 0, 0.5], atol=1e-10, rtol=0)
    assert np.allclose(norm_coord['y_norm'], [0.3, -0.2, 0.2], atol=1e-10, rtol=0)
    assert np.allclose(norm_coord['px_norm'], [0.1, 0.2, 0.3], atol=1e-10, rtol=0)
    assert np.allclose(norm_coord['py_norm'], [0.5, 0.6, 0.8], atol=1e-10, rtol=0)


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

    assert np.allclose(norm_coord1['x_norm'], [-1, 0, 0.5], atol=1e-10, rtol=0)
    assert np.allclose(norm_coord1['y_norm'], [0.3, -0.2, 0.2], atol=1e-10, rtol=0)
    assert np.allclose(norm_coord1['px_norm'], [0.1, 0.2, 0.3], atol=1e-10, rtol=0)
    assert np.allclose(norm_coord1['py_norm'], [0.5, 0.6, 0.8], atol=1e-10, rtol=0)

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
    assert np.allclose(norm_coord23['x_norm'][:3], [-1, 0, 0.5], atol=1e-10, rtol=0)
    assert np.allclose(norm_coord23['x_norm'][3:6], [-1, 0, 0.5], atol=1e-10, rtol=0)
    assert np.allclose(norm_coord23['x_norm'][6:], xp.particles.LAST_INVALID_STATE)
    assert np.allclose(norm_coord23['y_norm'][:3], [0.3, -0.2, 0.2], atol=1e-10, rtol=0)
    assert np.allclose(norm_coord23['y_norm'][3:6], [0.3, -0.2, 0.2], atol=1e-10, rtol=0)
    assert np.allclose(norm_coord23['y_norm'][6:], xp.particles.LAST_INVALID_STATE)
    assert np.allclose(norm_coord23['px_norm'][:3], [0.1, 0.2, 0.3], atol=1e-10, rtol=0)
    assert np.allclose(norm_coord23['px_norm'][3:6], [0.1, 0.2, 0.3], atol=1e-10, rtol=0)
    assert np.allclose(norm_coord23['px_norm'][6:], xp.particles.LAST_INVALID_STATE)
    assert np.allclose(norm_coord23['py_norm'][:3], [0.5, 0.6, 0.8], atol=1e-10, rtol=0)
    assert np.allclose(norm_coord23['py_norm'][3:6], [0.5, 0.6, 0.8], atol=1e-10, rtol=0)
    assert np.allclose(norm_coord23['py_norm'][6:], xp.particles.LAST_INVALID_STATE)

    particles23.move(_context=xo.context_default)
    assert np.all(particles23.at_element[:3] == line.element_names.index('s.ds.r3.b1'))
    assert np.all(particles23.at_element[3:6] == line.element_names.index('s.ds.r7.b1'))
    assert np.all(particles23.at_element[6:] == xp.particles.LAST_INVALID_STATE)

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
                        ele_start='bpm.31l5.b1',
                        ele_stop='bpm.31r5.b1',
                        twiss_init=tw.get_twiss_init(at_element='bpm.31l5.b1'))

    for tt in [tw_with_knl_ksl, tw_with_knl_ksl_part]:

        for kk in ['k0nl', 'k0sl', 'k1nl', 'k1sl', 'k2nl', 'k2sl']:
            assert kk in tt.keys()
            assert kk not in tw.keys()

        assert tt['k2nl', 'ms.30r5.b1'] == line['ms.30r5.b1'].knl[2]
        assert tt['k0sl', 'mcbrdv.4r5.b1'] == line['mcbrdv.4r5.b1'].ksl[0]

def test_get_R_matrix():
    fname_line_particles = test_data_folder / 'hllhc15_noerrors_nobb/line_and_particle.json'
    line = xt.Line.from_json(fname_line_particles)
    line.particle_ref = xp.Particles(p0c=7e12, mass0=xp.PROTON_MASS_EV)
    line.build_tracker()

    tw = line.twiss()

    R_IP3_IP6 = tw.get_R_matrix(ele_start=0, ele_stop='ip6')
    R_IP6_IP3 = tw.get_R_matrix(ele_start='ip6', ele_stop=len(tw.name)-1)

    # # Checks
    R_prod = R_IP6_IP3 @ R_IP3_IP6

    from xtrack.linear_normal_form import compute_linear_normal_form
    eig = np.linalg.eig
    norm = np.linalg.norm

    R_matrix = tw.R_matrix

    W_ref, invW_ref, Rot_ref = compute_linear_normal_form(R_matrix)
    W_prod, invW_prod, Rot_prod = compute_linear_normal_form(R_prod)


    for i_mode in range(3):
        lam_ref = eig(Rot_ref[2*i_mode:2*i_mode+2, 2*i_mode:2*i_mode+2])[0][0]
        lam_prod = eig(Rot_prod[2*i_mode:2*i_mode+2, 2*i_mode:2*i_mode+2])[0][0]

        assert np.isclose(np.abs(np.angle(lam_ref)) / 2 / np.pi,
                        np.abs(np.angle(lam_prod)) / 2 / np.pi,
                        rtol=0, atol=1e-6)

        assert np.isclose(
            norm(W_prod[:, 2*i_mode] - W_ref[:, 2*i_mode], ord=2)
            / norm(W_ref[:, 2*i_mode], ord=2),
            0, rtol=0, atol=5e-4)
        assert np.isclose(
            norm(W_prod[:4, 2*i_mode] - W_ref[:4, 2*i_mode], ord=2)
            / norm(W_ref[:4, 2*i_mode], ord=2),
            0, rtol=0, atol=5e-5)

    # Check method=4d

    tw4d = line.twiss(method='4d', freeze_longitudinal=True)

    R_IP3_IP6_4d = tw4d.get_R_matrix(ele_start=0, ele_stop='ip6')
    R_IP6_IP3_4d = tw4d.get_R_matrix(ele_start='ip6', ele_stop=len(tw4d.name)-1)

    R_prod_4d = R_IP6_IP3_4d @ R_IP3_IP6_4d

    # Checks
    from xtrack.linear_normal_form import compute_linear_normal_form
    eig = np.linalg.eig
    norm = np.linalg.norm

    R_matrix_4d = tw4d.R_matrix

    W_ref_4d, invW_ref_4d, Rot_ref_4d = compute_linear_normal_form(
        R_matrix_4d, only_4d_block=True)
    W_prod_4d, invW_prod_4d, Rot_prod_4d = compute_linear_normal_form(
        R_prod_4d, only_4d_block=True)

    for i_mode in range(3):
        lam_ref_4d = eig(
            Rot_ref_4d[2*i_mode:2*i_mode+2, 2*i_mode:2*i_mode+2])[0][0]
        lam_prod_4d = eig(
            Rot_prod_4d[2*i_mode:2*i_mode+2, 2*i_mode:2*i_mode+2])[0][0]

        assert np.isclose(np.abs(np.angle(lam_ref_4d)) / 2 / np.pi,
                        np.abs(np.angle(lam_prod_4d)) / 2 / np.pi,
                        rtol=0, atol=1e-6)

        assert np.isclose(
            norm(W_prod_4d[:, 2*i_mode] - W_ref_4d[:, 2*i_mode], ord=2)
            / norm(W_ref_4d[:, 2*i_mode], ord=2),
            0, rtol=0, atol=5e-5)


def test_hide_thin_groups():

    line = xt.Line.from_json(test_data_folder /
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


def test_periodic_cell_twiss():
    collider = xt.Multiline.from_json(test_data_folder /
                    'hllhc15_collider/collider_00_from_mad.json')
    collider.build_trackers()

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

        tw_cell = line.twiss(
            ele_start=start_cell,
            ele_stop=end_cell,
            twiss_init='preserve')

        assert tw_cell.method == '4d'
        assert 'dqx' not in tw_cell.keys() # check that periodic twiss is not used
        assert tw_cell.name[0] == start_cell
        assert tw_cell.name[-2] == end_cell
        assert tw_cell.method == '4d'
        assert tw_cell.orientation == 'forward'
        assert tw_cell.reference_frame == {'b1':'proper', 'b2':'reverse'}[beam_name]

        tw_cell_periodic = line.twiss(
            method='4d',
            ele_start=start_cell,
            ele_stop=end_cell,
            twiss_init='periodic')

        assert tw_cell_periodic.method == '4d'
        assert 'dqx' in tw_cell_periodic.keys() # check that periodic twiss is used
        assert tw_cell_periodic.name[0] == start_cell
        assert tw_cell_periodic.name[-2] == end_cell
        assert tw_cell_periodic.method == '4d'
        assert tw_cell_periodic.orientation == 'forward'
        assert tw_cell_periodic.reference_frame == {'b1':'proper', 'b2':'reverse'}[beam_name]

        assert np.allclose(tw_cell_periodic.betx, tw_cell.betx, atol=0, rtol=1e-6)
        assert np.allclose(tw_cell_periodic.bety, tw_cell.bety, atol=0, rtol=1e-6)
        assert np.allclose(tw_cell_periodic.dx, tw_cell.dx, atol=1e-4, rtol=0)

        assert tw_cell_periodic['mux', 0] == 0
        assert tw_cell_periodic['muy', 0] == 0
        assert np.isclose(tw_cell_periodic.mux[-1],
                tw['mux', end_cell] - tw['mux', start_cell], rtol=0, atol=1e-6)
        assert np.isclose(tw_cell_periodic.muy[-1],
                tw['muy', end_cell] - tw['muy', start_cell], rtol=0, atol=1e-6)

        twinit_start_cell = tw_cell_periodic.get_twiss_init(start_cell)

        tw_to_end_arc = line.twiss(
            ele_start=start_cell,
            ele_stop=end_arc,
            twiss_init=twinit_start_cell)
        assert tw_to_end_arc.method == '4d'
        assert tw_to_end_arc.orientation == {'b1': 'forward', 'b2': 'backward'}[beam_name]
        assert tw_to_end_arc.reference_frame == {'b1':'proper', 'b2':'reverse'}[beam_name]

        tw_to_start_arc = line.twiss(
            ele_start=start_arc,
            ele_stop=start_cell,
            twiss_init=twinit_start_cell)
        assert tw_to_start_arc.method == '4d'
        assert tw_to_start_arc.orientation == {'b1': 'backward', 'b2': 'forward'}[beam_name]
        assert tw_to_start_arc.reference_frame == {'b1':'proper', 'b2':'reverse'}[beam_name]

        mux_arc_from_cell = tw_to_end_arc['mux', end_arc] - tw_to_start_arc['mux', start_arc]
        muy_arc_from_cell = tw_to_end_arc['muy', end_arc] - tw_to_start_arc['muy', start_arc]

        assert np.isclose(mux_arc_from_cell, mux_arc_target, rtol=1e-6)
        assert np.isclose(muy_arc_from_cell, muy_arc_target, rtol=1e-6)

