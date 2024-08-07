import numpy as np
import xtrack as xt

# Get a line and build a tracker
line = xt.Line.from_json('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.build_tracker()

# Switch RF on
line.vars['vrf400'] = 16
line.vars['lagrf400.b1'] = 0.5

# Enable crossing angle orbit bumps
line.vars['on_x1'] = 10
line.vars['on_x2'] = 20
line.vars['on_x5'] = 10
line.vars['on_x8'] = 30

# Generate line made on maps (splitting at defined markers)
ele_cut = ['ip1', 'ip2', 'ip5', 'ip8'] # markers where to split the line
line_maps = line.get_line_with_second_order_maps(split_at=ele_cut)
line_maps.build_tracker()

line_maps.get_table().show()
# prints:
#
# name              s element_type         isthick
# ip7               0 Marker                 False
# map_0             0 SecondOrderTaylorMap    True
# ip8         3321.22 Marker                 False
# map_1       3321.22 SecondOrderTaylorMap    True
# ip1         6664.72 Marker                 False
# map_2       6664.72 SecondOrderTaylorMap    True
# ip2         9997.16 Marker                 False
# map_3       9997.16 SecondOrderTaylorMap    True
# ip5           19994 Marker                 False
# map_4         19994 SecondOrderTaylorMap    True
# lhcb1ip7_p_ 26658.9 Marker                 False
# _end_point  26658.9                        False

# Compare twiss of the two lines
tw = line.twiss()
tw_map = line_maps.twiss()

tw.qx       # is 62.3099999
tw_map.qx   # is  0.3099999

tw.dqx      # is 1.9135
tw_map.dqx  # is 1.9137

tw.rows[['ip1', 'ip2', 'ip5', 'ip8']].cols['x px y py betx bety'].show()
# prints
#
# name           x          px           y          py    betx    bety
# ip8  9.78388e-09 3.00018e-05 6.25324e-09 5.47624e-09 1.50001 1.49999
# ip1  1.92278e-10 1.00117e-05 2.64869e-09  1.2723e-08    0.15    0.15
# ip2  1.47983e-08 2.11146e-09 -2.1955e-08 2.00015e-05      10 9.99997
# ip5 -3.03593e-09 5.75413e-09 5.13184e-10  1.0012e-05    0.15    0.15

tw_map.rows[['ip1', 'ip2', 'ip5', 'ip8']].cols['x px y py betx bety']
# prints
#
# name             x          px           y          py    betx    bety
# ip8    9.78388e-09 3.00018e-05 6.25324e-09 5.47624e-09 1.50001 1.49999
# ip1    1.92278e-10 1.00117e-05 2.64869e-09  1.2723e-08    0.15    0.15
# ip2    1.47983e-08 2.11146e-09 -2.1955e-08 2.00015e-05      10 9.99997
# ip5   -3.03593e-09 5.75413e-09 5.13184e-10  1.0012e-05    0.15    0.15