import xtrack as xt

collider = xt.Environment.from_json(
    '../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider['lhcb1'].twiss_default['method'] = '4d'
collider['lhcb2'].twiss_default['method'] = '4d'

line = collider.lhcb1
line.cycle('ip3', inplace=True)

tw = line.twiss()


# Measure chromaticities of ITL5
start_range = 'mbxf.4l5_entry'
end_range = 'ip5'
init = tw.get_twiss_init(start_range)
init.ax_chrom = 0
init.bx_chrom = 0
init.ay_chrom = 0
init.by_chrom = 0
tw_l5 = line.twiss(start=start_range, end=end_range,
                   init=init,
                   compute_chromatic_properties=True)

# Measure chromaticities of ITR5
start_range = 'ip5'
end_range = 'mbxf.4r5_exit'
init = tw.get_twiss_init(end_range)
init.ax_chrom = 0
init.bx_chrom = 0
init.ay_chrom = 0
init.by_chrom = 0
tw_r5 = line.twiss(start=start_range, end=end_range,
                   init=init,
                   compute_chromatic_properties=True)

print(f'ITL5: dqx = {tw_l5.dmux[-1]:.2f}    dqy = {tw_l5.dmuy[-1]:.2f}')
print(f'ITR5: dqx = {tw_r5.dmux[-1]:.2f}    dqy = {tw_r5.dmuy[-1]:.2f}')
