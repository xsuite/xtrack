import numpy as np
import xtrack as xt

line = xt.Line.from_json('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.vars['vrf400'] = 16
line.vars['lagrf400.b1'] = 0.5

# line.vars['acbh22.l7b1'] = 10e-6
line.vars['acbv21.l7b1'] = 10e-6

def build_tailor_map(line, ele_start, ele_stop):
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

smap1 = build_tailor_map(line, ele_start='ip7', ele_stop='ip1')
smap2 = build_tailor_map(line, ele_start='ip1', ele_stop='ip2')
smap3 = build_tailor_map(line, ele_start='ip2', ele_stop='lhcb1ip7_p_')

line_maps = xt.Line(elements=[smap1, smap2, smap3])
line_maps.particle_ref = line.particle_ref.copy()

tw_map = line_maps.twiss()
tw = line.twiss()

