import xtrack as xt

line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

tw_closed = line.twiss4d()

tw = line.twiss(start='ip8', end='ip1', betx=1.5, bety=1.5)

source = 'mq.15r8.b1'

