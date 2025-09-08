import xtrack as xt

collider = xt.load("../../test_data/hllhc15_collider/collider_00_from_mad.json")

collider['on_crab1'] = -190
collider['on_crab5'] = -190

tw_b1 = collider['lhcb1'].twiss4d()
tw_b4 = collider['lhcb2'].twiss4d()

tw_b2 = tw_b4.reverse()

print('IP1')
print(f'B1 - dx_zeta: {tw_b1["dx_zeta", "ip1"]}')
print(f'B2 - dx_zeta: {tw_b2["dx_zeta", "ip1"]}')
print(f'B4 - dx_zeta: {tw_b4["dx_zeta", "ip1"]}')
print('IP5')
print(f'B1 - dy_zeta: {tw_b1["dy_zeta", "ip5"]}')
print(f'B2 - dy_zeta: {tw_b2["dy_zeta", "ip5"]}')
print(f'B4 - dy_zeta: {tw_b4["dy_zeta", "ip5"]}')