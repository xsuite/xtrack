import xtrack as xt

line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.build_tracker()
line.twiss_default['method'] = '4d'

tw_init_ip5 = line.twiss().get_twiss_init('ip5')
