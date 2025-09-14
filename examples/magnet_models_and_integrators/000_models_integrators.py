import xtrack as xt

line = xt.load('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

# Get table with all elements
tt = line.get_table()

# Table with all bends and all quadrupoles
tt_bend = tt.rows[(tt.element_type == 'Bend') | (tt.element_type == 'RBend')]
tt_quad = tt.rows[tt.element_type == 'Quadrupole']

# Set model and integrators for all bends
line.set(tt_bend, model='rot-kick-rot', integrator='teapot', num_multipole_kicks=4)

# Set model and integrators for all quadrupoles
line.set(tt_quad, model='mat-kick-mat', integrator='yoshida4', num_multipole_kicks=7)

# Set model and integrator for a specific family of quadrupoles
tt_mqxf = tt_quad.rows['mqxf.*']
line.set(tt_mqxf, model='drift-kick-drift-exact', integrator='yoshida4',
         num_multipole_kicks=21)

# Inspect a single element
line['mqxfa.b1l5'].model # is 'drift-kick-drift-exact'
line['mqxfa.b1l5'].integrator # is 'yoshida4'
line['mqxfa.b1l5'].num_multipole_kicks # is 21

# Alter a single element
line['mqxfa.b1l5'].model = 'mat-kick-mat'
line['mqxfa.b1l5'].integrator = 'teapot'
line['mqxfa.b1l5'].num_multipole_kicks = 10
