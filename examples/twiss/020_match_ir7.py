import xtrack as xt
import xpart as xp

# Load the line
line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json')
line.particle_ref = xp.Particles(p0c=7e12, mass=xp.PROTON_MASS_EV)
collider = xt.Multiline(lines={'lhcb1': line})
collider.build_trackers()

tw_ref = collider.lhcb1.twiss()

ele_start_match = 's.ds.l7.b1'
ele_end_match = 'e.ds.r7.b1'
tw_init = tw_ref.get_twiss_init(ele_start_match)

betx_end_match = tw_ref['betx', ele_end_match]
bety_end_match = tw_ref['bety', ele_end_match]
alfx_end_match = tw_ref['alfx', ele_end_match]
alfy_end_match = tw_ref['alfy', ele_end_match]
dx_end_match = tw_ref['dx', ele_end_match]
dpx_end_match = tw_ref['dpx', ele_end_match]
mux_end_match = tw_ref['mux', ele_end_match]
muy_end_match = tw_ref['muy', ele_end_match]

betx_at_ip7 = tw_ref['betx', 'ip7']
bety_at_ip7 = tw_ref['bety', 'ip7']
alfx_at_ip7 = tw_ref['alfx', 'ip7']
alfy_at_ip7 = tw_ref['alfy', 'ip7']
dx_at_ip7 = tw_ref['dx', 'ip7']
dpx_at_ip7 = tw_ref['dpx', 'ip7']

scale = 23348.89927
scmin = 0.03*7000./line.vars['nrj']._value
qtlimitx28 = 1.0*225.0/scale
qtlimitx15 = 1.0*205.0/scale
qtlimit2 = 1.0*160.0/scale
qtlimit3 = 1.0*200.0/scale
qtlimit4 = 1.0*125.0/scale
qtlimit5 = 1.0*120.0/scale
qtlimit6 = 1.0*90.0/scale

collider.match(
    ele_start=ele_start_match,
    ele_stop=ele_end_match,
    twiss_init=tw_init,
    targets=[
        xt.Target(line='lhcb1', at='ip7',        tar='dx',   value=dx_at_ip7),
        xt.Target(line='lhcb1', at='ip7',        tar='dpx',  value=dpx_at_ip7),
        xt.Target(line='lhcb1', at='ip7',        tar='betx', value=betx_at_ip7),
        xt.Target(line='lhcb1', at='ip7',        tar='bety', value=bety_at_ip7),
        xt.Target(line='lhcb1', at='ip7',        tar='alfx', value=alfx_at_ip7),
        xt.Target(line='lhcb1', at='ip7',        tar='alfy', value=alfy_at_ip7),
        xt.Target(line='lhcb1', at='e.ds.r7.b1', tar='alfx', value=alfx_end_match),
        xt.Target(line='lhcb1', at='e.ds.r7.b1', tar='alfy', value=alfy_end_match),
        xt.Target(line='lhcb1', at='e.ds.r7.b1', tar='betx', value=betx_end_match),
        xt.Target(line='lhcb1', at='e.ds.r7.b1', tar='bety', value=bety_end_match),
        xt.Target(line='lhcb1', at='e.ds.r7.b1', tar='dx',   value=dx_end_match),
        xt.Target(line='lhcb1', at='e.ds.r7.b1', tar='dpx',  value=dpx_end_match),
        xt.Target(line='lhcb1', at='e.ds.r7.b1', tar='mux',  value=mux_end_match),
        xt.Target(line='lhcb1', at='e.ds.r7.b1', tar='muy ', value=muy_end_match),
    ],
    vary=[
        xt.Vary('kqt13.l7b1',  step=1.0E-9, limits=(-qtlimit5, qtlimit5)),
        xt.Vary('kqt12.l7b1',  step=1.0E-9, limits=(-qtlimit5, qtlimit5)),
        xt.Vary('kqtl11.l7b1', step=1.0E-9, limits=(-qtlimit4*300./550., qtlimit4*300./550.)),
        xt.Vary('kqtl10.l7b1', step=1.0E-9, limits=(-qtlimit4*500./550., qtlimit4*500./550.)),
        xt.Vary('kqtl9.l7b1',  step=1.0E-9, limits=(-qtlimit4*400./550., qtlimit4*400./550.)),
        xt.Vary('kqtl8.l7b1',  step=1.0E-9, limits=(-qtlimit4*300./550., qtlimit4*300./550.)),
        xt.Vary('kqtl7.l7b1',  step=1.0E-9, limits=(-qtlimit4, qtlimit4)),
        xt.Vary('kq6.l7b1',    step=1.0E-9, limits=(-qtlimit6, qtlimit6)),
        xt.Vary('kq6.r7b1',    step=1.0E-9, limits=(-qtlimit6, qtlimit6)),
        xt.Vary('kqtl7.r7b1',  step=1.0E-9, limits=(-qtlimit4, qtlimit4)),
        xt.Vary('kqtl8.r7b1',  step=1.0E-9, limits=(-qtlimit4*550./550., qtlimit4*550./550.)),
        xt.Vary('kqtl9.r7b1',  step=1.0E-9, limits=(-qtlimit4*500./550., qtlimit4*500./550.)),
        xt.Vary('kqtl10.r7b1', step=1.0E-9, limits=(-qtlimit4, qtlimit4)),
        xt.Vary('kqtl11.r7b1', step=1.0E-9, limits=(-qtlimit4, qtlimit4)),
        xt.Vary('kqt12.r7b1',  step=1.0E-9, limits=(-qtlimit5, qtlimit5)),
        xt.Vary('kqt13.r7b1',  step=1.0E-9, limits=(-qtlimit5, qtlimit5)),
    ]
)

tw_after = collider.lhcb1.twiss()
