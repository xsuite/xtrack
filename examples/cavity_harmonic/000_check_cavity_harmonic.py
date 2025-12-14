import xtrack as xt

# TODO:
# grep for frequency and generalize (tapering, particle generation, ...)
# forbid isolated element in harmonic mode
# forbid harmonic and frequency at the same time
# MAD loaders

line = xt.load('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

tt = line.get_table(attr=True)
tt_cav = tt.rows.match(element_type='Cavity')

line['vrf400'] = 16
tw1 = line.twiss6d()

for nn in tt_cav.name:
    line[nn].frequency=0
    line[nn].harmonic = 35640

tw2 = line.twiss6d()