import xtrack as xt

# --------- Load from python

# Load lattice from python files
env = xt.load('../lattices/z/fccee_z_lattice.py')
line = env['fccee_p_ring']

# Load env from json
env = xt.load('../../test_data/hllhc15_thick/hllhc15_collider_thick.json')

# Load line from json
env = xt.load('../../test_data/sps_ions/line_and_particle.json')

# Load from madx
env = xt.load('../../test_data/sps_thick/sps.seq', format='madx')

##### LOADING OPTICS
env.vars.load('vars.json') # Assumes dict of variables (including expressions)
env.vars.load('vars.madx')
env.vars.load('vars.str', format='madx')

env.vars.load(string="""
    a:=1;
    b:=2;
    c:=a+b;
    """, format='madx')



