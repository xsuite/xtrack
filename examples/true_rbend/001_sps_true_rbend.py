import xtrack as xt

line = xt.load('../../test_data/sps_thick/sps.seq')['sps']
line.vars.load('../../test_data/sps_thick/lhc_q20.str')
line.particle_ref = xt.Particles(p0c=26e9, mass0=xt.PROTON_MASS_EV)

tt = line.get_table()
tt_rbend = tt.rows[tt.element_type == 'RBend']

line.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(slicing=xt.Uniform(10, mode='thick'), element_type=xt.RBend),
    ])

line.set(tt_rbend, edge_entry_model='full')
line.set(tt_rbend, edge_exit_model='full')

line.set(tt_rbend, rbend_model='straight-body')
tw_straight = line.twiss4d()

line.set(tt_rbend, rbend_model='curved-body')
tw_curved = line.twiss4d()


