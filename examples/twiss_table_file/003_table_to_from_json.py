import xtrack as xt

lhc= xt.load('../../test_data/lhc_2024/lhc.seq')
lhc.vars.load('../../test_data/lhc_2024/injection_optics.madx')
lhc.set_particle_ref('proton', energy0=450e9)

tw1 = lhc.lhcb1.twiss4d()

# Check json round trip
tw1.to_json('test.json')
tw2 = xt.TwissTable.from_json('test.json')


