import xtrack as xt
import xobjects as xo
import numpy as np

env = xt.Environment()
line = env.new_line(components=[
    env.new('rb', xt.RBend, length_straight=2, angle=np.pi/3, rbend_model='curved-body'),
    env.new('end_point', xt.Marker)
])

line_thin = line.copy(shallow=True)
line_thin.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(slicing=xt.Teapot(10, mode='thin'))
    ]
)

line_thick_cut = line.copy(shallow=True)
line_thick_cut.cut_at_s(line_thin.get_table().s)

tt_thin = line_thin.get_table()
tt_thick_cut = line_thick_cut.get_table()
tt_thick = line.get_table()

xo.assert_allclose(tt_thick.s[-1], env['rb'].length, atol=1e-13)
xo.assert_allclose(tt_thin.s[-1], env['rb'].length, atol=1e-13)
xo.assert_allclose(tt_thick_cut.s[-1], env['rb'].length, atol=1e-13)

p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=45.6e9)
p_thick = p0.copy()
p_thin = p0.copy()
p_thick_cut = p0.copy()

line.reset_s_at_end_turn = False
line_thin.reset_s_at_end_turn = False
line_thick_cut.reset_s_at_end_turn = False

line.track(p_thick)
line_thin.track(p_thin)
line_thick_cut.track(p_thick_cut)

assert env['rb'].length > env['rb'].length_straight * 1.01
xo.assert_allclose(p_thick.s, env['rb'].length, atol=1e-13)
xo.assert_allclose(p_thin.s, env['rb'].length, atol=1e-13)
xo.assert_allclose(p_thick_cut.s, env['rb'].length, atol=1e-13)

p_thick_back = p_thick.copy()
p_thin_back = p_thin.copy()
p_thick_cut_back = p_thick_cut.copy()

line.track(p_thick_back, backtrack=True)
line_thin.track(p_thin_back, backtrack=True)
line_thick_cut.track(p_thick_cut_back, backtrack=True)

xo.assert_allclose(p_thick_back.s, 0, atol=1e-13)
xo.assert_allclose(p_thin_back.s, 0, atol=1e-13)
xo.assert_allclose(p_thick_cut_back.s, 0, atol=1e-13)

line.set_particle_ref(p0.copy())
line_thin.set_particle_ref(p0.copy())
line_thick_cut.set_particle_ref(p0.copy())

tw_thick = line.twiss(betx=1, bety=1)
tw_thin = line_thin.twiss(betx=1, bety=1)
tw_thick_cut = line_thick_cut.twiss(betx=1, bety=1)

tw_thin_compare = tw_thin.rows.match(name=r'drift_rb\.\.0|rb.*|.*end.*')

xo.assert_allclose(tw_thick_cut.s, tw_thin_compare.s, atol=1e-13)
# not expected to be precise bu
xo.assert_allclose(tw_thick_cut.x, tw_thin_compare.x, atol=3e-10)

tw_thick_back = line.twiss(init_at='end_point', betx=1, bety=1)
tw_thin_back = line_thin.twiss(init_at='end_point', betx=1, bety=1)
tw_thick_cut_back = line_thick_cut.twiss(init_at='end_point', betx=1, bety=1)

xo.assert_allclose(tw_thick.s, tt_thick.s, atol=1e-13)
xo.assert_allclose(tw_thin.s, tt_thin.s, atol=1e-13)
xo.assert_allclose(tw_thick_cut.s, tt_thick_cut.s, atol=1e-13)

assert tw_thick_back._orientation == 'backward'
assert tw_thin_back._orientation == 'backward'
assert tw_thick_cut_back._orientation == 'backward'
xo.assert_allclose(tw_thick_back.s, tw_thick.s, atol=1e-13)
xo.assert_allclose(tw_thin_back.s, tw_thin.s, atol=1e-13)
xo.assert_allclose(tw_thick_cut_back.s, tw_thick_cut.s, atol=1e-13)
xo.assert_allclose(tw_thick_back.x, tw_thick.x, atol=3e-10)
xo.assert_allclose(tw_thin_back.x, tw_thin.x, atol=1e-13)
xo.assert_allclose(tw_thick_cut_back.x, tw_thick_cut.x, atol=3e-10)

