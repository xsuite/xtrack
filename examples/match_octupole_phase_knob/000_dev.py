import time
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
    return np.sum(amp*phasor)#/(1-np.exp(2*np.pi*1.j*((j-k)*q1 + (l-m)*q2)))

rdt = get_rdt(tmux=0, tmuy=0, betx=tw_mo['betx'], bety=tw_mo['bety'],
        mux=tw_mo['mux'], muy=tw_mo['muy'], q1=tw.qx, q2=tw.qy)

class RDTarget(xt.Target):
    def __init__(self, j=2, k=0, l=0, m=2):
        self.j = j
        self.k = k
        self.l = l
        self.m = m

        xt.Target.__init__(
            self, tag='rdt', tar=self.compute_abs_target, value=0, tol=500, weight=1e-6)

    def compute_target(self, tw):
        tw_mo = tw.rows['mo.*.b1']
        rdt = get_rdt(tmux=0, tmuy=0, betx=tw_mo['betx'], bety=tw_mo['bety'],
            mux=tw_mo['mux'], muy=tw_mo['muy'], q1=tw.mux[-1], q2=tw.muy[-1],
            j=self.j, k=self.k, l=self.l, m=self.m)
        return rdt

    # def __repr__(self):
    #     return f'RDTarget(j={self.j}, k={self.k}, l={self.l}, m={self.m})'

class RDTargetReal(RDTarget):
    def compute_abs_target(self, tw):
        return super().compute_target(tw).real

class RDTargetImag(RDTarget):
    def compute_abs_target(self, tw):
        return super().compute_target(tw).imag

lims = {}
varlist = ['kqtf.a12b1', 'kqtf.a23b1', 'kqtf.a34b1', 'kqtf.a45b1',
           'kqtf.a56b1', 'kqtf.a67b1', 'kqtf.a78b1', 'kqtf.a81b1',
           'kqtd.a12b1', 'kqtd.a23b1', 'kqtd.a34b1', 'kqtd.a45b1',
           'kqtd.a56b1', 'kqtd.a67b1', 'kqtd.a78b1', 'kqtd.a81b1']

for var in varlist:
    val = line.vars[var]._value
    #vals = [ 0.8*val, 1.2*val]
    #offset = 0.0001
    lims[var] = ( -0.001,  +0.001)
    #lims[var] = ( -offset + min(vals),  offset + max(vals))


tw0 = line.twiss()

opt = line.match_knob(
    knob_name='phase_knob',
    run=False,
    assert_within_tol=False,
    init=tw.get_twiss_init(0),
    start=0, end=len(line)-1,
    targets=[
        RDTargetReal(j=2, k=0, l=0, m=2),
        RDTargetReal(j=4, k=0, l=0, m=0),
        # RDTargetReal(j=0, k=0, l=4, m=0),
        RDTargetImag(j=2, k=0, l=0, m=2),
        RDTargetImag(j=4, k=0, l=0, m=0),
        # RDTargetImag(j=0, k=0, l=4, m=0),
        xt.TargetSet(tag="tune", tars=['mux', 'muy'], value=tw0, at='lhcb1$end')
    ],
    vary=[
        xt.Vary('kqtf.a12b1', step=1.e-8, limits=lims['kqtf.a12b1']),
        xt.Vary('kqtf.a23b1', step=1.e-8, limits=lims['kqtf.a23b1']),
        xt.Vary('kqtf.a34b1', step=1.e-8, limits=lims['kqtf.a34b1']),
        xt.Vary('kqtf.a45b1', step=1.e-8, limits=lims['kqtf.a45b1']),
        xt.Vary('kqtf.a56b1', step=1.e-8, limits=lims['kqtf.a56b1']),
        xt.Vary('kqtf.a67b1', step=1.e-8, limits=lims['kqtf.a67b1']),
        xt.Vary('kqtf.a78b1', step=1.e-8, limits=lims['kqtf.a78b1']),
        xt.Vary('kqtf.a81b1', step=1.e-8, limits=lims['kqtf.a81b1']),
        xt.Vary('kqtd.a12b1', step=1.e-8, limits=lims['kqtd.a12b1']),
        xt.Vary('kqtd.a23b1', step=1.e-8, limits=lims['kqtd.a23b1']),
        xt.Vary('kqtd.a34b1', step=1.e-8, limits=lims['kqtd.a34b1']),
        xt.Vary('kqtd.a45b1', step=1.e-8, limits=lims['kqtd.a45b1']),
        xt.Vary('kqtd.a56b1', step=1.e-8, limits=lims['kqtd.a56b1']),
        xt.Vary('kqtd.a67b1', step=1.e-8, limits=lims['kqtd.a67b1']),
        xt.Vary('kqtd.a78b1', step=1.e-8, limits=lims['kqtd.a78b1']),
        xt.Vary('kqtd.a81b1', step=1.e-8, limits=lims['kqtd.a81b1']),
        # xt.VaryList(
        #     ['kqtf.a12b1', 'kqtf.a23b1', 'kqtf.a34b1', 'kqtf.a45b1',
        #     'kqtf.a56b1', 'kqtf.a67b1', 'kqtf.a78b1', 'kqtf.a81b1'], step=1e-6),
        # xt.VaryList(
        #     ['kqtd.a12b1', 'kqtd.a23b1', 'kqtd.a34b1', 'kqtd.a45b1',
        #     'kqtd.a56b1', 'kqtd.a67b1', 'kqtd.a78b1', 'kqtd.a81b1'], step=1e-6),
    ]
)

t1 = time.time()
opt.step(30)

opt.disable_targets(tag='rdt')
opt.solve()
t2 = time.time()
print(f'Optimization time: {t2-t1:.2f} s')

knob_values_xs = opt.get_knob_values()
target_values_xs = opt.target_status(ret=True).current_val
tw_xs = line.twiss()

# for ii in range(len(opt.vary)):
#     opt.vary[ii].limits = [opt.vary[ii].limits[0] - offset, opt.vary[ii].limits[1] + offset]

knob_values_ng  = {
    'kqtf.a12b1_from_phase_knob': -0.0022477200000,
    'kqtf.a23b1_from_phase_knob': -0.0006109026670,
    'kqtf.a34b1_from_phase_knob': -0.0006740726670,
    'kqtf.a45b1_from_phase_knob': +0.0015222900000,
    'kqtf.a56b1_from_phase_knob': +0.0011189300000,
    'kqtf.a67b1_from_phase_knob': +0.0020387763940,
    'kqtf.a78b1_from_phase_knob': -0.0011010306070,
    'kqtf.a81b1_from_phase_knob': -0.0001300250000,
    'kqtd.a12b1_from_phase_knob': -0.0001437190000,
    'kqtd.a23b1_from_phase_knob': +0.0010619748420,
    'kqtd.a34b1_from_phase_knob': +0.0001529048423,
    'kqtd.a45b1_from_phase_knob': -0.0004891330000,
    'kqtd.a56b1_from_phase_knob': +0.0008419600000,
    'kqtd.a67b1_from_phase_knob': +0.0016072722540,
    'kqtd.a78b1_from_phase_knob': -0.0013696167460,
    'kqtd.a81b1_from_phase_knob': -0.0016425400000,
}

line.vars.update(knob_values_ng)
target_values_ng = opt.target_status(ret=True).current_val
tw_ng = line.twiss()

print(f'xs: |f2002| = \t{np.sqrt(target_values_xs[0]**2 + target_values_xs[1]**2):.2e} '
        f'\t|f4000| = {np.sqrt(target_values_xs[2]**2 + target_values_xs[3]**2):.2e}')
print(f'ng: |f2002| = \t{np.sqrt(target_values_ng[0]**2 + target_values_ng[1]**2):.2e} '
        f'\t|f4000| = {np.sqrt(target_values_ng[2]**2 + target_values_ng[3]**2):.2e}')

import matplotlib.pyplot as plt
plt.figure(1)
sp1 = plt.subplot(211)
plt.plot(tw.s, tw_ng.betx/tw.betx-1, label='ng')
plt.plot(tw.s, tw_xs.betx/tw.betx-1, label='xs')
plt.ylabel(r'$\Delta \beta_x / \beta_x$')
plt.legend()
plt.subplot(212, sharex=sp1)
plt.plot(tw.s, tw_ng.bety/tw.bety-1, label='ng')
plt.plot(tw.s, tw_xs.bety/tw.bety-1, label='xs')
plt.ylabel(r'$\Delta \beta_y / \beta_y$')
plt.xlabel('s [m]')
plt.legend()
plt.show()
