import xtrack as xt

line_old = xt.load('temp_line_before.json')
line_new = xt.load('lhc_thick_with_knobs.json')

# Patch 11 T dipoles
tt_new = line_new.get_table()
tt_new_mbh = tt_new.rows['mbh.*'].rows.match('Bend', 'element_type')
for nn in tt_new_mbh.name:
    line_new[nn].k0 = line_new[nn].h

tw_old = line_old.twiss4d()
tw_new = line_new.twiss4d()

two_new = line_new.twiss(betx=1, bety=1)