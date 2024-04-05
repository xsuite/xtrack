import xtrack as xt

line = xt.Line.from_json('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.build_tracker()

tt_thick = line.get_table(attr=True)

line.discard_tracker()
line.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(slicing=None), # Default slicing
        # xt.Strategy(slicing=xt.Teapot(3, mode='thick'), name='mb.*'),
        xt.Strategy(slicing=xt.Teapot(3, mode='thin'), name='mb.*'),
    ])

line.build_tracker()
tt_thin = line.get_table(attr=True)