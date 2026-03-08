import xtrack as xt

line = xt.load('../lhc_thick/lhc_thick_with_knobs.json')
line.build_tracker()

tt = line.get_table(attr=True)
for ii in range(20):
    tt2 = line.get_table(attr=True)