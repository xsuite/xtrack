import xtrack as xt
import numpy as np

collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_collider/collider_00_from_mad.json')
collider.build_trackers()

collider.vv['on_x1'] = 250e-6

collider.install_beambeam_interactions(
    clockwise_line='lhcb1',
    anticlockwise_line='lhcb2',
    ip_names=['ip1'], #, 'ip2', 'ip5', 'ip8'],
    delay_at_ips_slots=[0], #, 891, 0, 2670],
    num_long_range_encounters_per_side=[2], #,2,2,2],
    num_slices_head_on=1,
    harmonic_number=35640,
    bunch_spacing_buckets=10,
    sigmaz=0.075)

bunch_slots_b1 = np.arange(10 , 20)
bunch_slots_b2 = np.arange(10 , 20)

