import xtrack as xt
import numpy as np

collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_collider/collider_00_from_mad.json')
collider.build_trackers()
collider.lhcb1.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['method'] = '4d'
collider.lhcb1.twiss_default['matrix_stability_tol'] = 2e-2
collider.lhcb2.twiss_default['matrix_stability_tol'] = 2e-2

collider.vv['on_x1'] = 250e-6
collider.vv['on_x5'] = 250e-6
collider.vv['on_x2'] = 250e-6
collider.vv['on_x8'] = 250e-6

collider.vv['on_sep2'] = 1
collider.vv['on_sep8'] = 1
collider.vv['on_sep1'] = 1
collider.vv['on_sep5'] = 1


collider.install_beambeam_interactions(
    clockwise_line='lhcb1',
    anticlockwise_line='lhcb2',
    ip_names=['ip1', 'ip2', 'ip5', 'ip8'],
    delay_at_ips_slots=[0, 891, 0, 2670],
    num_long_range_encounters_per_side=[25, 25, 25, 25],
    num_slices_head_on=11,
    harmonic_number=35640,
    bunch_spacing_buckets=10,
    sigmaz=0.075)

collider.configure_beambeam_interactions(
    num_particles=1e11,
    nemitt_x=2.5e-6,
    nemitt_y=2.5e-6)

bunch_slots_b1 = np.arange(10 , 20)
bunch_slots_b2 = np.arange(10 , 20)

collider.vars['beambeam_scale'] = 0
twb1_no_bb = collider.lhcb1.twiss()
twb2_no_bb = collider.lhcb2.twiss()

collider.vars['beambeam_scale'] = 1
twb1_bb_nodip = collider.lhcb1.twiss()
twb2_bb_nodip = collider.lhcb2.twiss()

for ln in ['lhcb1', 'lhcb2']:
    for ee in collider[ln].elements:
        if hasattr(ee, 'post_subtract_px'):
            for kk in collider.lhcb1['bb_lr.l1b1_02']._xofields.keys():
                if kk.startswith('post_subtract'):
                    setattr(ee, kk, 0)

twb1_bb_dip = collider.lhcb1.twiss()
twb2_bb_dip = collider.lhcb2.twiss()

sigmas_b1 = twb1_no_bb.get_betatron_sigmas(nemitt_x=2.5e-6, nemitt_y=2.5e-6)