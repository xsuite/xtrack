import numpy as np
import xtrack as xt

line = xt.Line.from_json('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.vars['vrf400'] = 16
line.vars['lagrf400.b1'] = 0.5

tw = line.twiss()

ele_start = 'ip3'
ele_stop = 'ip4'
twinit = tw.get_twiss_init(ele_start)
RR = line.compute_one_turn_matrix_finite_differences(
    ele_start=ele_start, ele_stop=ele_stop, particle_on_co=twinit.particle_on_co
    )['R_matrix']
TT = line.compute_T_matrix(ele_start=ele_start, ele_stop=ele_stop,
                            particle_on_co=twinit.particle_on_co)

smap = xt.SecondOrderTaylorMap(R=RR, T=TT)

p_ref = line.build_particles(x=1e-4, px=2e-6, y=3e-4, py=4e-6, zeta=1e-4, delta=1e-4)
p_map = p_ref.copy()
p0 = p_ref.copy()

line.track(p_ref, ele_start=ele_start, ele_stop=ele_stop)
smap.track(p_map)

print('x', p_ref.x, p_map.x)
print('px', p_ref.px, p_map.px)
print('y', p_ref.y, p_map.y)
print('py', p_ref.py, p_map.py)
print('zeta', p_ref.zeta, p_map.zeta)
print('delta', p_ref.delta, p_map.delta)
