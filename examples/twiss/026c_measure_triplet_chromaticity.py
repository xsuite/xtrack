import xtrack as xt

collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider['lhcb1'].twiss_default['method'] = '4d'
collider['lhcb2'].twiss_default['method'] = '4d'

line = collider.lhcb1
line.cycle('ip3', inplace=True)

tw = line.twiss(only_markers=True)


# Measure chromaticities of ITL5
start_range = 'mbxf.4l5_entry'
end_range = 'ip5'
twiss_init = tw.get_twiss_init(start_range)
twiss_init.ax_chrom = 0
twiss_init.bx_chrom = 0
twiss_init.ay_chrom = 0
twiss_init.by_chrom = 0
tw_l5 = line.twiss(ele_start=start_range, ele_stop=end_range,
                   twiss_init=twiss_init,
                   compute_chromatic_properties=True)

# Measure chromaticities of ITR5
start_range = 'ip5'
end_range = 'mbxf.4r5_exit'
twiss_init = tw.get_twiss_init(end_range)
twiss_init.ax_chrom = 0
twiss_init.bx_chrom = 0
twiss_init.ay_chrom = 0
twiss_init.by_chrom = 0
tw_r5 = line.twiss(ele_start=start_range, ele_stop=end_range,
                   twiss_init=twiss_init,
                   compute_chromatic_properties=True)

print(f'ITL5: dqx = {tw_l5.dmux[-1]:.2f}    dqy = {tw_l5.dmuy[-1]:.2f}')
print(f'ITR5: dqx = {tw_r5.dmux[-1]:.2f}    dqy = {tw_r5.dmuy[-1]:.2f}')
