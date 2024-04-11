import xtrack as xt

line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

tw0 = line.twiss4d()

line['mq.14r7.b1'].shift_x=0.5e-3
line['mq.14r2.b1'].shift_y=0.5e-3

tw1 = line.twiss4d()