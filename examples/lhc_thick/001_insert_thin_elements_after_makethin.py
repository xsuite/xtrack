import time
import numpy as np
import xtrack as xt

line = xt.Line.from_json('lhc_thin.json')

tt = line.get_table()

# Identify active elements
tt_active = tt.rows[
    (tt.element_type != 'Drift') & (tt.element_type != 'Marker')]

# Generate an aperture and a marker to be put in front of  each active element
insertions = []
for ii, (nn, ss) in enumerate(zip(tt_active.name, tt_active.s)):
    print(f'{ii}/{len(tt_active.name)}', end='\r', flush=True)

    aa = xt.LimitRect(min_x=-0.1, max_x=0.1, min_y=-0.1, max_y=0.1)
    mm = xt.Marker()
    insertions.append((ss, [(f'aper_{nn}', aa), (f'marker_{nn}', mm)]))

# Generate an aperture and a marker to be added every 100 m
for ss in np.arange(100, tt.s[-1], 100):
    aa = xt.LimitRect(min_x=-0.2, max_x=0.2, min_y=-0.2, max_y=0.2)
    mm = xt.Marker()
    insertions.append((ss, [(f'aper_{ss}', aa), (f'marker_{ss}', mm)]))

line.discard_tracker()

# Insert all
line._insert_thin_elements_at_s(insertions)

# Inspect:
tt_after = line.get_table()
tt_after.rows['marker_20000.0<<30': 'marker_20000.0>>30'].show()
