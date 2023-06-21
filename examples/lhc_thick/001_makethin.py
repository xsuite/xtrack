import xtrack as xt

line = xt.Line.from_json('lhc_thick_with_knobs.json')
line.build_tracker()

line_thick = line.copy()

