import xtrack as xt

collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider['lhcb1'].twiss_default['method'] = '4d'
collider['lhcb2'].twiss_default['method'] = '4d'

line = collider.lhcb1
line.cycle('ip3', inplace=True)

tw = line.twiss()

init = xt.TwissInit(betx=0.15, bety=0.15,
                           ax_chrom=42.7928, bx_chrom=-18.4181,
                           ay_chrom=-18.0191, by_chrom= 11.54862)
tw_open = line.twiss(start='ip5', end='ip7', init=init,
                        compute_chromatic_properties=True,
                        )

ksfs = ['ksf1.a81b1', 'ksf1.a12b1', 'ksf1.a45b1', 'ksf1.a56b1']
ksds = ['ksd2.a81b1', 'ksd2.a12b1', 'ksd2.a45b1', 'ksd2.a56b1']

for kk in ksfs:
    line.vv[kk] = 0.06
for kk in ksds:
    line.vv[kk] = -0.099

twl = line.twiss()

opt = line.match(solve=False,
        vary=[xt.VaryList(
            ('ksf1.a81b1 ksd2.a81b1 ksf1.a12b1 ksd2.a12b1 ksf1.a45b1 '
             'ksd2.a45b1 ksf1.a56b1 ksd2.a56b1').split(),
            limits=(-0.38, 0.38), step=1e-6)],
        targets=[
            xt.Target('ax_chrom', 0, at='ip3'),
            xt.Target('bx_chrom', 0, at='ip3'),
            xt.Target('ay_chrom', 0, at='ip3'),
            xt.Target('by_chrom', 0, at='ip3'),
            xt.Target(lambda tt: tt['wx_chrom', 'ip3'] - tt['wx_chrom', 'ip7'], 0),
            xt.Target(lambda tt: tt['wy_chrom', 'ip3'] - tt['wy_chrom', 'ip7'], 0),
            xt.Target('bx_chrom', 0, at='ip1'),
            xt.Target('by_chrom', 0, at='ip1'),
            xt.Target('bx_chrom', 0, at='ip5'),
            xt.Target('by_chrom', 0, at='ip5'),
        ])


import matplotlib.pyplot as plt
plt.close('all')

plt.plot(tw.s, tw.dmux)

for nn in tw.rows['mbxf.*'].name:
    plt.axvline(tw['s', nn], color='k', linestyle='--')

plt.show()