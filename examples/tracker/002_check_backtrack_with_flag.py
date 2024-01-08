import xtrack as xt

line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json')
line.build_tracker()

p = xt.Particles(
    p0c=7000e9, x=1e-4, px=1e-6, y=2e-4, py=3e-6, zeta=0.01, delta=1e-4)
p.get_table().show(digits=3)
# prints:
#
# particle_id s      x    px      y    py zeta  delta chi ...
#           0 0 0.0001 1e-06 0.0002 3e-06 0.01 0.0001   1

# Track one turn
line.track(p)
p.get_table().show()
# prints:
#
# particle_id s        x        px        y        py    zeta  delta chi ...
#           0 0 0.000324 -1.07e-05 1.21e-05 -1.42e-06 0.00909 0.0001   1

# Track back one turn
line.track(p, backtrack=True)
# The particle is back to the initial coordinates
p.get_table().show(digits=3)
# prints:
#
# particle_id s      x    px      y    py zeta  delta chi ...
#           0 0 0.0001 1e-06 0.0002 3e-06 0.01 0.0001   1

# It is also possible to backtrack with a specified start/stop elements

# Track three elements
line.track(p, ele_start=0, ele_stop=3)
p.get_table().cols['x px y py zeta delta at_element'].show()
# prints:
#
# particle_id           x    px           y    py zeta  delta at_element
#           0 0.000121028 1e-06 0.000263084 3e-06 0.01 0.0001          3

# Track back three elements
line.track(p, ele_start=0, ele_stop=3, backtrack=True)
p.get_table().cols['x px y py zeta delta at_element'].show()
# prints:
#
# particle_id      x    px      y    py zeta  delta at_element
#           0 0.0001 1e-06 0.0002 3e-06 0.01 0.0001          0