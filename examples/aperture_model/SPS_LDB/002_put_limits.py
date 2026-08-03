import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

S_TOL = 3.1e-2

sps = xt.load('sps.json')
sps.twiss_default['method'] = '4d'
env = sps.env

aperture = xt.Aperture.from_json('sps_aperture.json', line=sps, s_tol=S_TOL)

sps_thin = sps.copy(shallow=True)
sps_thin.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(slicing=xt.Teapot(1)),
        xt.Strategy(element_type=xt.Quadrupole, slicing=xt.Teapot(4)),
        xt.Strategy(element_type=xt.RBend, slicing=xt.Teapot(2)),
    ]
)
tt = sps_thin.get_table()

s_positions = aperture.s_around_transitions()
limits_transitions = aperture.get_limit_elements(s_positions)
insertions_transitions = []

for ii, (s, limit) in enumerate(limits_transitions.items()):
    name = f'aperture_{ii}'
    env.elements[name] = limit
    insertions_transitions.append(env.place(name=name, at=s))

tt_active = tt.rows.match_not(element_type='Drift.*|Limit.*|Marker|')
limits_up_downstream_active = aperture.get_limit_elements(np.unique(np.concatenate([tt_active.s_start, tt_active.s_end])))
insertions_up_downstream = []

for row in tt_active.rows:
    name_upstream = f'aperture_start_{row.name}'
    assert name_upstream not in env.elements
    env.elements[name_upstream] = limits_up_downstream_active[row.s_start]

    name_downstream = f'aperture_end_{row.name}'
    assert name_downstream not in env.elements
    env.elements[name_downstream] = limits_up_downstream_active[row.s_end]

    insertions_up_downstream.append(env.place(name=name_upstream, at=0, from_=row.name, from_anchor='start'))
    insertions_up_downstream.append(env.place(name=name_downstream, at=0, from_=row.name, from_anchor='end'))

sps_with_limits = sps_thin.copy(shallow=True)
sps_with_limits.insert(insertions_up_downstream)
sps_with_limits.insert(insertions_transitions)


aperture_from_limits = xt.Aperture.from_line_with_limits(sps_with_limits)
aperture_from_limits.plot_extents(aperture_from_limits.s_around_transitions(resolution=10), include_aper_tols=True)
plt.show()

env.lines['sps_thin'] = sps_thin
env.lines['sps_with_limits'] = sps_with_limits
env.to_json('sps_env.json')
