import numpy as np
import xtrack as xt

line = xt.Line.from_json(
    'LHCB1_qx62.275_qy60.293_qp10_mo0.0_cmr0.000_phasechange0.0.json')
line.cycle('ip1', inplace=True)

line.build_tracker()

line.twiss_default['method'] = '4d'

line.vars['vrf400'] = 0.0

tw = line.twiss()

tw_mo = tw.rows['mo.*.b1']

def get_rdt(tmux, tmuy, betx, bety, mux, muy, j=2, k=0, l=0, m=2, q1=62.27, q2=60.295):
    dphix = mux-tmux
    dphix[np.where(dphix <= 0 )] += q1
    dphiy = muy-tmuy
    dphiy[np.where(dphiy <= 0 )] += q2
    phasor = np.exp(np.pi*2.j*((j-k)*(dphix) + (l-m)*(dphiy)))
    amp = np.sqrt(betx)**(j+k)*np.sqrt(bety)**(l+m)
    return np.sum(amp*phasor)/(1-np.exp(2*np.pi*1.j*((j-k)*q1 + (l-m)*2)))



rdt = get_rdt(tmux=0, tmuy=0, betx=tw_mo['betx'], bety=tw_mo['bety'],
        mux=tw_mo['mux'], muy=tw_mo['muy'], q1=tw.qx, q2=tw.qy)

class RDTarget(xt.Target):
    def __init__(self):
        xt.Target.__init__(
            self, tag='rdt', tar=self.compute_abs_target, value=0, tol=500, weight=1e-6)

    def compute_abs_target(self, tw):
        tw_mo = tw.rows['mo.*.b1']
        rdt = get_rdt(tmux=0, tmuy=0, betx=tw_mo['betx'], bety=tw_mo['bety'],
            mux=tw_mo['mux'], muy=tw_mo['muy'], q1=tw.qx, q2=tw.qy)
        return np.abs(rdt)

    def __repr__(self):
        return 'RDTarget'

opt = line.match_knob(
    solve=False,
    assert_within_tol=False,
    targets=[
        RDTarget(),
        xt.TargetSet(qx='preserve', qy='preserve'),
    ],
    vary=[
        xt.VaryList(
            ['kqtf.a12b1', 'kqtf.a23b1', 'kqtf.a34b1', 'kqtf.a45b1',
            'kqtf.a56b1', 'kqtf.a67b1', 'kqtf.a78b1', 'kqtf.a81b1'], step=1e-5),
        xt.VaryList(
            ['kqtd.a12b1', 'kqtd.a23b1', 'kqtd.a34b1', 'kqtd.a45b1',
            'kqtd.a56b1', 'kqtd.a67b1', 'kqtd.a78b1', 'kqtd.a81b1'], step=1e-5),
    ]
)




