import numpy as np
import xtrack as xt

line = xt.Line.from_json('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.vars['vrf400'] = 16
line.vars['lagrf400.b1'] = 0.5

# line.vars['acbh22.l7b1'] = 10e-6
line.vars['acbv21.l7b1'] = 10e-6

def build_tailor_map(line, ele_start, ele_stop, twiss_table):

    if twiss_table is None:
        tw = line.twiss()
    else:
        tw = line.twiss(reverse=False)

    twinit = tw.get_twiss_init(ele_start)
    twinit_out = tw.get_twiss_init(ele_stop)

    RR = line.compute_one_turn_matrix_finite_differences(
        ele_start=ele_start, ele_stop=ele_stop, particle_on_co=twinit.particle_on_co
        )['R_matrix']
    TT = line.compute_T_matrix(ele_start=ele_start, ele_stop=ele_stop,
                                particle_on_co=twinit.particle_on_co)

    x_co_in = np.array([
        twinit.particle_on_co.x[0],
        twinit.particle_on_co.px[0],
        twinit.particle_on_co.y[0],
        twinit.particle_on_co.py[0],
        twinit.particle_on_co.zeta[0],
        twinit.particle_on_co.pzeta[0],
    ])

    x_co_out = np.array([
        twinit_out.particle_on_co.x[0],
        twinit_out.particle_on_co.px[0],
        twinit_out.particle_on_co.y[0],
        twinit_out.particle_on_co.py[0],
        twinit_out.particle_on_co.zeta[0],
        twinit_out.particle_on_co.pzeta[0],
    ])

    R_T_fd = np.einsum('ijk,k->ij', TT, x_co_in)
    K_T_fd = R_T_fd @ x_co_in

    K_hat = x_co_out - RR @ x_co_in + K_T_fd
    RR_hat = RR - 2 * R_T_fd

    smap = xt.SecondOrderTaylorMap(R=RR_hat, T=TT, k=K_hat,
                        length=tw['s', ele_stop] - tw['s', ele_start])

    return smap

ele_cut = ['ip1', 'ip2', 'ip3', 'ip4', 'ip5', 'ip6', 'ip7']

ele_cut_ext = ele_cut.copy()
if line.element_names[0] not in ele_cut_ext:
    ele_cut_ext.insert(0, line.element_names[0])
if line.element_names[-1] not in ele_cut_ext:
    ele_cut_ext.append(line.element_names[-1])

ele_cut_sorted = []
for ee in line.element_names:
    if ee in ele_cut_ext:
        ele_cut_sorted.append(ee)

elements_map_line = []
names_map_line = []
tw = line.twiss()

for ii in range(len(ele_cut_sorted)-1):
    names_map_line.append(ele_cut_sorted[ii])
    elements_map_line.append(line[ele_cut_sorted[ii]])

    smap = build_tailor_map(line, ele_start=ele_cut_sorted[ii],
                            ele_stop=ele_cut_sorted[ii+1],
                            twiss_table=tw)
    names_map_line.append(f'map_{ii}')
    elements_map_line.append(smap)

names_map_line.append(ele_cut_sorted[-1])
elements_map_line.append(line[ele_cut_sorted[-1]])

line_maps = xt.Line(elements=elements_map_line, element_names=names_map_line)
line_maps.particle_ref = line.particle_ref.copy()

tw = line.twiss()
tw_map = line_maps.twiss()

assert np.isclose(np.mod(tw_map.qx, 1), np.mod(tw.qx, 1), rtol=0, atol=1e-7)
assert np.isclose(np.mod(tw_map.qy, 1), np.mod(tw.qy, 1), rtol=0, atol=1e-7)
assert np.isclose(tw_map.dqx, tw.dqx, rtol=0, atol=1e-3)
assert np.isclose(tw_map.dqy, tw.dqy, rtol=0, atol=1e-3)
assert np.isclose(tw_map.c_minus, tw.c_minus, rtol=0, atol=1e-5)

assert np.allclose(tw_map.rows[ele_cut].s, tw.rows[ele_cut].s, rtol=0, atol=1e-12)

assert np.allclose(tw_map.rows[ele_cut].x, tw.rows[ele_cut].x, rtol=0, atol=1e-12)
assert np.allclose(tw_map.rows[ele_cut].px, tw.rows[ele_cut].px, rtol=0, atol=1e-12)
assert np.allclose(tw_map.rows[ele_cut].y, tw.rows[ele_cut].y, rtol=0, atol=1e-12)
assert np.allclose(tw_map.rows[ele_cut].py, tw.rows[ele_cut].py, rtol=0, atol=1e-12)
assert np.allclose(tw_map.rows[ele_cut].zeta, tw.rows[ele_cut].zeta, rtol=0, atol=1e-10)
assert np.allclose(tw_map.rows[ele_cut].delta, tw.rows[ele_cut].delta, rtol=0, atol=1e-12)

assert np.allclose(tw_map.rows[ele_cut].betx, tw.rows[ele_cut].betx, rtol=1e-5, atol=0)
assert np.allclose(tw_map.rows[ele_cut].alfx, tw.rows[ele_cut].alfx, rtol=1e-5, atol=1e-6)
assert np.allclose(tw_map.rows[ele_cut].bety, tw.rows[ele_cut].bety, rtol=1e-5, atol=0)
assert np.allclose(tw_map.rows[ele_cut].alfy, tw.rows[ele_cut].alfy, rtol=1e-5, atol=1e-6)