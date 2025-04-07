import numpy as np
import xtrack as xt
import xdeps as xd

from cpymad.madx import Madx

# fname = 'fccee_z'; pc_gev = 45.6
# fname = 'fccee_w'; pc_gev = 80.
fname = 'fccee_h'; pc_gev = 120.
# fname = 'fccee_t'; pc_gev = 182.5

mad = Madx()
mad.call(fname + '.seq')
mad.beam(particle='positron', pc=pc_gev)
mad.use('fccee_p_ring')

mad.call('install_wigglers.madx')
mad.input("exec, define_wigglers_as_multipoles()")
mad.input("exec, install_wigglers()")
mad.use('fccee_p_ring')

line_thick = xt.Line.from_madx_sequence(mad.sequence.fccee_p_ring, allow_thick=True,
                                  deferred_expressions=True)

# Introduce 90 degree tilt for wiggler
tt = line_thick.get_table()

# We seem to have a regression in the table
# wigs = tt.rows['mwi.*', tt.element_type=='Bend'].name

wigs = tt.rows[tt.rows.mask['mwi.*'] & (tt.element_type=='Bend')].name

for nn in wigs:
    line_thick.element_refs[nn].rot_s_rad = np.pi/2

line_thick.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV,
                                 gamma0=mad.sequence.fccee_p_ring.beam.gamma)
line_thick.build_tracker()
tw_thick_no_rad = line_thick.twiss(method='4d')

line_thick.to_json(fname + '_thick.json')

line = line_thick.copy()
Strategy = xt.slicing.Strategy
Teapot = xt.slicing.Teapot
slicing_strategies = [
    Strategy(slicing=Teapot(1)),  # Default catch-all as in MAD-X
    Strategy(slicing=Teapot(3), element_type=xt.Bend),
    Strategy(slicing=Teapot(3), element_type=xt.CombinedFunctionMagnet),
    # Strategy(slicing=Teapot(50), element_type=xt.Quadrupole), # Starting point
    Strategy(slicing=Teapot(5), name=r'^qf.*'),
    Strategy(slicing=Teapot(5), name=r'^qd.*'),
    Strategy(slicing=Teapot(5), name=r'^qfg.*'),
    Strategy(slicing=Teapot(5), name=r'^qdg.*'),
    Strategy(slicing=Teapot(5), name=r'^ql.*'),
    Strategy(slicing=Teapot(5), name=r'^qs.*'),
    Strategy(slicing=Teapot(10), name=r'^qb.*'),
    Strategy(slicing=Teapot(10), name=r'^qg.*'),
    Strategy(slicing=Teapot(10), name=r'^qh.*'),
    Strategy(slicing=Teapot(10), name=r'^qi.*'),
    Strategy(slicing=Teapot(10), name=r'^qr.*'),
    Strategy(slicing=Teapot(10), name=r'^qu.*'),
    Strategy(slicing=Teapot(10), name=r'^qy.*'),
    Strategy(slicing=Teapot(50), name=r'^qa.*'),
    Strategy(slicing=Teapot(50), name=r'^qc.*'),
    Strategy(slicing=Teapot(20), name=r'^sy\..*'),
    Strategy(slicing=Teapot(30), name=r'^mwi\..*'),
]

line.slice_thick_elements(slicing_strategies=slicing_strategies)

line.build_tracker()
tw_thin_before = line.twiss(start=line.element_names[0], end=line.element_names[-1],
                          method='4d',
                          init=tw_thick_no_rad.get_twiss_init(0))

# Compare tunes
print('Before rematching:')

print('Tunes thick model:')
print(tw_thick_no_rad.qx, tw_thick_no_rad.qy)
print('Tunes thin model:')
print(tw_thin_before.mux[-1], tw_thin_before.muy[-1])

print('Beta beating at ips:')
print('H:', np.max(np.abs(
    tw_thin_before.rows['ip.*'].betx / tw_thick_no_rad.rows['ip.*'].betx -1)))
print('V:', np.max(np.abs(
    tw_thin_before.rows['ip.*'].bety / tw_thick_no_rad.rows['ip.*'].bety -1)))

print('Number of elements: ', len(line))
print('\n')

opt = line.match(
    method='4d',
    start=line.element_names[0], end=line.element_names[-1],
    init=tw_thick_no_rad.get_twiss_init(0),
    vary=xt.VaryList(['k1qf4', 'k1qf2', 'k1qd3', 'k1qd1',], step=1e-8,
    ),
    targets=[
        xt.TargetSet(at=xt.END, mux=tw_thick_no_rad.qx, muy=tw_thick_no_rad.qy, tol=1e-5),
    ]
)
opt.solve()
tw_thin_no_rad = line.twiss(method='4d')

print('After rematching:')
print('Tunes thick model:')
print(tw_thick_no_rad.qx, tw_thick_no_rad.qy)
print('Tunes thin model:')
print(tw_thin_no_rad.qx, tw_thin_no_rad.qy)

print('Beta beating at ips:')
print('H:', np.max(np.abs(
    tw_thin_no_rad.rows['ip.*'].betx / tw_thick_no_rad.rows['ip.*'].betx -1)))
print('V:', np.max(np.abs(
    tw_thin_no_rad.rows['ip.*'].bety / tw_thick_no_rad.rows['ip.*'].bety -1)))

print('Number of elements: ', len(line))

print('Change on arc quadrupoles:')
print(opt.log().vary[-1]/opt.log().vary[0] - 1)

print('\n Beta at the IPs:')
tw_thin_no_rad.rows['ip.*'].cols['betx bety'].show()

line.to_json(fname + '_thin.json')
# line_thick.to_json(fname + '_thick.json')

