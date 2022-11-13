from cpymad.madx import Madx
import xtrack as xt
import xpart as xp

# Import a thick sequence
mad = Madx()
mad.call('../../test_data/clic_dr/sequence.madx')
mad.use('ring')

# Makethin
mad.input(f'''
select, flag=MAKETHIN, SLICE=4, thick=false;
select, flag=MAKETHIN, pattern=wig, slice=1;
MAKETHIN, SEQUENCE=ring, MAKEDIPEDGE=true;
use, sequence=RING;
''')
mad.use('ring')

# Build xtrack line
print('Build xtrack line...')
line = xt.Line.from_madx_sequence(mad.sequence['RING'])
line.particle_ref = xp.Particles(
        mass0=xp.ELECTRON_MASS_EV,
        q0=-1,
        gamma0=15000 # I push up the energy loss
        )

#line0 = line.copy()

line['rf'].voltage *= 100 # I push up the voltage

c0 = line['rf']
v0 = c0.voltage
s0 = line.get_s_position('rf')

line.insert_element(at_s=line.get_length()/2-s0, element=c0.copy(), name='rf1')
line.insert_element(at_s=line.get_length()/2+s0, element=c0.copy(), name='rf2')
line.insert_element(at_s=line.get_length()-s0, element=c0.copy(), name='rf3')

line = line.cycle('qdw1..1:38')

#line.insert_element(at_s=0., element=c0.copy(), name='rf1')
#line.insert_element(at_s=40., element=c0.copy(), name='rf11')
#line.insert_element(at_s=211., element=c0.copy(), name='rf2')
#line.insert_element(at_s=250., element=c0.copy(), name='rf21')
#line.insert_element(at_s=295., element=c0.copy(), name='rf3')
#line.insert_element(at_s=425., element=c0.copy(), name='rf4')
#
#line['rf2'].voltage *= 2
#line['rf'].voltage *= 2
#line['rf3'].voltage *= 2
#line['rf1'].voltage *= 1
#line['rf4'].voltage *= 1


tracker = line.build_tracker()

## Introduce some closed orbit
#line['qc1l1.1..1'].knl[0] += 1e-6
#line['qc1l1.1..1'].ksl[0] += 1e-6

# Initial twiss (no radiation)
tracker.configure_radiation(model=None)
tw_no_rad = tracker.twiss(mode='4d', freeze_longitudinal=True)

# Enable radiation
tracker.configure_radiation(model='mean')
# - Set cavity lags to compensate energy loss
# - Taper magnet strengths
tracker.compensate_radiation_energy_loss()

# Twiss(es) with radiation
tw_real_tracking = tracker.twiss(mode='6d', matrix_stability_tol=3.,
                    eneloss_and_damping=True)
tw_sympl = tracker.twiss(radiation_treatment='kick_as_co', mode='6d')
tw_preserve_angles = tracker.twiss(
                        radiation_treatment='preserve_angles',
                        mode='6d',
                        matrix_stability_tol=0.5)

import matplotlib.pyplot as plt
plt.close('all')

print('Non sympltectic tracker:')
print(f'Tune error =  error_qx: {abs(tw_real_tracking.qx - tw_no_rad.qx):.3e} error_qy: {abs(tw_real_tracking.qy - tw_no_rad.qy):.3e}')
print('Sympltectic tracker:')
print(f'Tune error =  error_qx: {abs(tw_sympl.qx - tw_no_rad.qx):.3e} error_qy: {abs(tw_sympl.qy - tw_no_rad.qy):.3e}')
print ('Preserve angles:')
print(f'Tune error =  error_qx: {abs(tw_preserve_angles.qx - tw_no_rad.qx):.3e} error_qy: {abs(tw_preserve_angles.qy - tw_no_rad.qy):.3e}')
plt.figure(2)

plt.subplot(2,1,1)
plt.plot(tw_no_rad.s, tw_sympl.betx/tw_no_rad.betx - 1)
plt.plot(tw_no_rad.s, tw_preserve_angles.betx/tw_no_rad.betx - 1)
#tw.betx *= (1 + delta_beta_corr)
#plt.plot(tw_no_rad.s, tw.betx/tw_no_rad.betx - 1)
plt.ylabel(r'$\Delta \beta_x / \beta_x$')

plt.subplot(2,1,2)
plt.plot(tw_no_rad.s, tw_sympl.bety/tw_no_rad.bety - 1)
plt.plot(tw_no_rad.s, tw_preserve_angles.bety/tw_no_rad.bety - 1)
#tw.bety *= (1 + delta_beta_corr)
#plt.plot(tw_no_rad.s, tw.bety/tw_no_rad.bety - 1)
plt.ylabel(r'$\Delta \beta_y / \beta_y$')

plt.figure(10)
plt.subplot(2,1,1)
plt.plot(tw_no_rad.s, tw_no_rad.x, 'k')
#plt.plot(tw_no_rad.s, tw_real_tracking.x, 'b')
plt.plot(tw_no_rad.s, tw_sympl.x, 'r')
plt.plot(tw_no_rad.s, tw_preserve_angles.x, 'g')

plt.subplot(2,1,2)
plt.plot(tw_no_rad.s, tw_no_rad.y, 'k')
#plt.plot(tw_no_rad.s, tw_real_tracking.y, 'b')
plt.plot(tw_no_rad.s, tw_sympl.y, 'r')
plt.plot(tw_no_rad.s, tw_preserve_angles.y, 'g')

plt.figure(3)
plt.subplot()
plt.plot(tw_no_rad.s, tracker.delta_taper)
plt.plot(tw_real_tracking.s, tw_real_tracking.delta)


plt.show()