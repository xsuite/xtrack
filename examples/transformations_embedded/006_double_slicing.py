import xtrack as xt
import numpy as np

bend = xt.Bend(k0=0.4, h=0.3, length=1,
            # k1=0.1,
            #    shift_x=1e-3, shift_y=2e-3, rot_s_rad=0.2
            )

line = xt.Line(elements=[bend])

line.configure_bend_model(edge='linear', core='expanded')

line.slice_thick_elements(
    slicing_strategies=[xt.Strategy(xt.Uniform(3))])

line.cut_at_s([0.3])

line._insert_thin_elements_at_s([
    (0.4, [('mymark1', xt.Marker()), ('mymark2', xt.Marker())])])
