import numpy as np

import xtrack as xt

import matplotlib.pyplot as plt

line = xt.Line.from_json('psb_00_from_mad.json')
line.build_tracker()
line.twiss_default['method'] = '4d'

tw0 = line.twiss()

# inspect a bit...
tw0.qx, tw0.qy

# A few checks on the imported model
line.vars['k0bi1bsw1l11']._info() # Check that the knob controls k0 and the edges

line.element_refs['bi1.bsw1l1.1'].h._info() # Check no reference system curvature

# Build chicane knob (k0)
line.vars['bsw_k0l'] = 0
line.vars['k0bi1bsw1l11'] = (line.vars['bsw_k0l'] / line['bi1.bsw1l1.1'].length)
line.vars['k0bi1bsw1l12'] = (-line.vars['bsw_k0l'] / line['bi1.bsw1l1.2'].length)
line.vars['k0bi1bsw1l13'] = (-line.vars['bsw_k0l'] / line['bi1.bsw1l1.3'].length)
line.vars['k0bi1bsw1l14'] = (line.vars['bsw_k0l'] / line['bi1.bsw1l1.4'].length)

# Inspect:
line.vars['k0bi1bsw1l11']._info()

# Build knob to model eddy currents (k2)
line.vars['bsw_k2l'] = 0
line.element_refs['bi1.bsw1l1.1'].knl[2] = line.vars['bsw_k2l']
line.element_refs['bi1.bsw1l1.2'].knl[2] = -line.vars['bsw_k2l']
line.element_refs['bi1.bsw1l1.3'].knl[2] = -line.vars['bsw_k2l']
line.element_refs['bi1.bsw1l1.4'].knl[2] = line.vars['bsw_k2l']

# Save to file
line.to_json('psb_01_with_chicane.json')


## Checks:

# Match tunes (with chicane off)
line.match(
    targets=[
        xt.Target('qx',  4.4, tol=1e-6),
        xt.Target('qy',  4.45, tol=1e-6)],
    vary=[
        xt.Vary('kbrqf', step=1e-5),
        xt.Vary('kbrqd', step=1e-5)],
)

# Inspect bump and induced beta beating for different bump amplitudes (and no eddy currents)
plt.close('all')
bsw_k0l_ref = 6.6e-2 # Full bump amplitude

bsw_k0l_values = np.linspace(0, bsw_k0l_ref, 5)
fig1 = plt.figure(1)
sp1 = plt.subplot(3,1,1)
sp2 = plt.subplot(3,1,2, sharex=sp1)
sp3 = plt.subplot(3,1,3, sharex=sp1)

colors = plt.cm.rainbow(np.linspace(0,1,len(bsw_k0l_values)))

for ii, vv in enumerate(bsw_k0l_values[::-1]):
    line.vars['bsw_k0l'] = vv
    tw = line.twiss()

    sp1.plot(tw.s, tw.x, color=colors[ii])
    sp2.plot(tw.s, tw.betx, color=colors[ii])
    sp3.plot(tw.s, tw.bety, color=colors[ii])

sp1.set_ylabel('x [m]')
sp2.set_ylabel('betx [m]')
sp3.set_ylabel('bety [m]')
sp3.set_xlabel('s [m]')

# Inspect eddy currents effect (at start chicane ramp)
line.vars['bsw_k0l'] = bsw_k0l_ref * 0.95

bsw_k2l_ref = -9.7429e-2 # Maximum ramp rate

fig2 = plt.figure(2)
sp1 = plt.subplot(3,1,1)
sp2 = plt.subplot(3,1,2, sharex=sp1)
sp3 = plt.subplot(3,1,3, sharex=sp1)
for ii, vv in enumerate([bsw_k2l_ref, 0]):
    line.vars['bsw_k2l'] = vv
    tw = line.twiss()

    sp1.plot(tw.s, tw.x)
    sp2.plot(tw.s, tw.betx)
    sp3.plot(tw.s, tw.bety)

sp1.set_ylabel('x [m]')
sp2.set_ylabel('betx [m]')
sp3.set_ylabel('bety [m]')
sp3.set_xlabel('s [m]')


line.vars['bsw_k2l'] = bsw_k2l_ref
line.vars['bsw_k0l'] = bsw_k0l_ref
assert np.isclose(line['bi1.bsw1l1.1'].knl[2], bsw_k2l_ref, rtol=0, atol=1e-10)
assert np.isclose(line['bi1.bsw1l1.2'].knl[2], -bsw_k2l_ref, rtol=0, atol=1e-10)
assert np.isclose(line['bi1.bsw1l1.3'].knl[2], -bsw_k2l_ref, rtol=0, atol=1e-10)
assert np.isclose(line['bi1.bsw1l1.4'].knl[2], bsw_k2l_ref, rtol=0, atol=1e-10)
assert np.isclose(line['bi1.bsw1l1.1'].k0, bsw_k0l_ref / line['bi1.bsw1l1.1'].length, rtol=0, atol=1e-10)
assert np.isclose(line['bi1.bsw1l1.2'].k0, -bsw_k0l_ref / line['bi1.bsw1l1.2'].length, rtol=0, atol=1e-10)
assert np.isclose(line['bi1.bsw1l1.3'].k0, -bsw_k0l_ref / line['bi1.bsw1l1.3'].length, rtol=0, atol=1e-10)
assert np.isclose(line['bi1.bsw1l1.4'].k0, bsw_k0l_ref / line['bi1.bsw1l1.4'].length, rtol=0, atol=1e-10)

tw = line.twiss()
assert np.isclose(tw['x', 'bi1.tstr1l1'], -0.045716, rtol=0, atol=1e-5)
assert np.isclose(tw['y', 'bi1.tstr1l1'], 0.0000000, rtol=0, atol=1e-5)
assert np.isclose(tw['betx', 'bi1.tstr1l1'], 5.203667, rtol=0, atol=1e-4)
assert np.isclose(tw['bety', 'bi1.tstr1l1'], 6.902887, rtol=0, atol=1e-4)
assert np.isclose(tw.qy, 4.474414126093382, rtol=0, atol=1e-6) # verify that it does not change from one version to the other
assert np.isclose(tw.qx, 4.396717774779403, rtol=0, atol=1e-6)
assert np.isclose(tw.dqy, -8.625637734560598, rtol=0, atol=1e-3)
assert np.isclose(tw.dqx, -3.5604677592626643, rtol=0, atol=1e-3)

line.vars['bsw_k2l'] = bsw_k2l_ref / 3
assert np.isclose(line['bi1.bsw1l1.1'].knl[2], bsw_k2l_ref / 3, rtol=0, atol=1e-10)
assert np.isclose(line['bi1.bsw1l1.2'].knl[2], -bsw_k2l_ref / 3, rtol=0, atol=1e-10)
assert np.isclose(line['bi1.bsw1l1.3'].knl[2], -bsw_k2l_ref / 3, rtol=0, atol=1e-10)
assert np.isclose(line['bi1.bsw1l1.4'].knl[2], bsw_k2l_ref / 3, rtol=0, atol=1e-10)
assert np.isclose(line['bi1.bsw1l1.1'].k0, bsw_k0l_ref / line['bi1.bsw1l1.1'].length, rtol=0, atol=1e-10)
assert np.isclose(line['bi1.bsw1l1.2'].k0, -bsw_k0l_ref / line['bi1.bsw1l1.2'].length, rtol=0, atol=1e-10)
assert np.isclose(line['bi1.bsw1l1.3'].k0, -bsw_k0l_ref / line['bi1.bsw1l1.3'].length, rtol=0, atol=1e-10)
assert np.isclose(line['bi1.bsw1l1.4'].k0, bsw_k0l_ref / line['bi1.bsw1l1.4'].length, rtol=0, atol=1e-10)

tw = line.twiss()
assert np.isclose(tw['x', 'bi1.tstr1l1'], -0.0458633, rtol=0, atol=1e-5)
assert np.isclose(tw['y', 'bi1.tstr1l1'], 0.0000000, rtol=0, atol=1e-5)
assert np.isclose(tw['betx', 'bi1.tstr1l1'], 5.266456, rtol=0, atol=1e-4)
assert np.isclose(tw['bety', 'bi1.tstr1l1'], 6.320286, rtol=0, atol=1e-4)
assert np.isclose(tw.qy, 4.471766776419623, rtol=0, atol=1e-6)
assert np.isclose(tw.qx, 4.398899960718224, rtol=0, atol=1e-6)
assert np.isclose(tw.dqy, -8.2058757683523, rtol=0, atol=1e-3)
assert np.isclose(tw.dqx, -3.563488925077962, rtol=0, atol=1e-3)

# Switch off bsws
line.vars['bsw_k0l'] = 0
line.vars['bsw_k2l'] = 0
assert np.isclose(line['bi1.bsw1l1.1'].knl[2], 0, rtol=0, atol=1e-10)
assert np.isclose(line['bi1.bsw1l1.2'].knl[2], 0, rtol=0, atol=1e-10)
assert np.isclose(line['bi1.bsw1l1.3'].knl[2], 0, rtol=0, atol=1e-10)
assert np.isclose(line['bi1.bsw1l1.4'].knl[2], 0, rtol=0, atol=1e-10)
assert np.isclose(line['bi1.bsw1l1.1'].k0, 0, rtol=0, atol=1e-10)
assert np.isclose(line['bi1.bsw1l1.2'].k0, 0, rtol=0, atol=1e-10)
assert np.isclose(line['bi1.bsw1l1.3'].k0, 0, rtol=0, atol=1e-10)
assert np.isclose(line['bi1.bsw1l1.4'].k0, 0, rtol=0, atol=1e-10)

tw = line.twiss()
assert np.isclose(tw['x', 'bi1.tstr1l1'], 0, rtol=0, atol=1e-5)
assert np.isclose(tw['y', 'bi1.tstr1l1'], 0, rtol=0, atol=1e-5)
assert np.isclose(tw['betx', 'bi1.tstr1l1'], 5.2996347, rtol=0, atol=1e-4)
assert np.isclose(tw['bety', 'bi1.tstr1l1'], 3.838857, rtol=0, atol=1e-4)
assert np.isclose(tw.qy, 4.45, rtol=0, atol=1e-6)
assert np.isclose(tw.qx, 4.4, rtol=0, atol=1e-6)
assert np.isclose(tw.dqy, -7.149781341846406, rtol=0, atol=1e-4)
assert np.isclose(tw.dqx, -3.5655757511587893, rtol=0, atol=1e-4)

plt.show()