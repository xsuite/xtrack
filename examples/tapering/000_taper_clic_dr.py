import json
import numpy as np
import xtrack as xt

case_name = 'clic_dr'
filename = '../../test_data/clic_dr/line_for_taper.json'
configs = [
    {'radiation_method': 'full', 'p0_correction': False, 'cavity_preserve_angle': False, 'beta_rtol': 2e-2},
    {'radiation_method': 'full', 'p0_correction': True, 'cavity_preserve_angle': False, 'beta_rtol': 2e-2},
    {'radiation_method': 'full', 'p0_correction': True, 'cavity_preserve_angle': True, 'beta_rtol': 2e-5},
    {'radiation_method': 'kick_as_co', 'p0_correction': True, 'cavity_preserve_angle': True, 'beta_rtol': 1e-3},
    {'radiation_method': 'scale_as_co', 'p0_correction': True, 'cavity_preserve_angle': True, 'beta_rtol': 1e-5},
]

# case_name = 'fcc-ee'
# filename = 'line_no_radiation.json'


with open(filename, 'r') as f:
    line = xt.Line.from_dict(json.load(f))

tracker = line.build_tracker()


# Initial twiss (no radiation)
tracker.configure_radiation(model=None)
tw_no_rad = tracker.twiss(method='4d', freeze_longitudinal=True)

# Enable radiation
tracker.configure_radiation(model='mean')
# - Set cavity lags to compensate energy loss
# - Taper magnet strengths
tracker.compensate_radiation_energy_loss()




import matplotlib.pyplot as plt
plt.close('all')
ifig = 0
for conf in configs:

    ifig += 1

    # Twiss(es) with radiation
    tracker.config.XTRACK_CAVITY_PRESERVE_ANGLE = conf['cavity_preserve_angle']
    tw = tracker.twiss(radiation_method=conf['radiation_method'],
                       eneloss_and_damping=(conf['radiation_method'] != 'kick_as_co'))
    tracker.config.XTRACK_CAVITY_PRESERVE_ANGLE = False

    if conf['p0_correction']:
        p0corr = 1 + tracker.delta_taper
    else:
        p0corr = 1

    plt.figure(ifig, figsize=(6.4*1.3, 4.8))
    plt.suptitle(f"Radiation method: {conf['radiation_method']}, "
                    f"p0 correction: {conf['p0_correction']}, "
                    f"cavity preserve angle: {conf['cavity_preserve_angle']}")

    plt.subplot(2,1,1)
    plt.title(f'error on Qx: {abs(tw.qx - tw_no_rad.qx):.2e}     '
                r'$(\Delta \beta_x / \beta_x)_{max}$ = '
                f'{np.max(tw.betx*p0corr/tw_no_rad.betx-1):.2e}')
    plt.plot(tw.s, tw.betx*p0corr/tw_no_rad.betx-1)
    plt.ylabel(r'$\Delta \beta_x / \beta_x$')

    plt.subplot(2,1,2)
    plt.title(f'error on Qy: {abs(tw.qy - tw_no_rad.qy):.2e}     '
                r'$(\Delta \beta_y / \beta_y)_{max}$ = '
                f'{np.max(tw.bety*p0corr/tw_no_rad.bety-1):.2e}')
    plt.plot(tw.s, tw.bety*p0corr/tw_no_rad.bety-1)
    plt.ylabel(r'$\Delta \beta_y / \beta_y$')

    plt.subplots_adjust(hspace=0.35, top=.85)

    assert np.isclose(tracker.delta_taper[0], 0, rtol=0, atol=1e-10)
    assert np.isclose(tracker.delta_taper[-1], 0, rtol=0, atol=1e-10)

    assert np.allclose(tw.delta, tracker.delta_taper, rtol=0, atol=1e-6)

    assert np.isclose(tw.qx, tw_no_rad.qx, rtol=0, atol=5e-4)
    assert np.isclose(tw.qy, tw_no_rad.qy, rtol=0, atol=5e-4)

    assert np.isclose(tw.dqx, tw_no_rad.dqx, rtol=0, atol=0.2)
    assert np.isclose(tw.dqy, tw_no_rad.dqy, rtol=0, atol=0.2)

    assert np.allclose(tw.x, tw_no_rad.x, rtol=0, atol=1e-7)
    assert np.allclose(tw.y, tw_no_rad.y, rtol=0, atol=1e-7)

    assert np.allclose(tw.betx*p0corr, tw_no_rad.betx, rtol=conf['beta_rtol'], atol=0)
    assert np.allclose(tw.bety*p0corr, tw_no_rad.bety, rtol=conf['beta_rtol'], atol=0)

    assert np.allclose(tw.dx, tw.dx, rtol=0.00, atol=0.1e-3)

    assert np.allclose(tw.dy, tw.dy, rtol=0.00, atol=0.1e-3)

    if conf['radiation_method'] != 'kick_as_co':
        eneloss = tw.eneloss_turn
        assert eneloss/line.particle_ref.energy0 > 0.01
        assert np.isclose(line['rf'].voltage*np.sin(line['rf'].lag/180*np.pi), eneloss/4, rtol=1e-5)
        assert np.isclose(line['rf1'].voltage*np.sin(line['rf1'].lag/180*np.pi), eneloss/4, rtol=1e-5)
        assert np.isclose(line['rf2a'].voltage*np.sin(line['rf2a'].lag/180*np.pi), eneloss/4*0.6, rtol=1e-5)
        assert np.isclose(line['rf2b'].voltage*np.sin(line['rf2b'].lag/180*np.pi), eneloss/4*0.4, rtol=1e-5)
        assert np.isclose(line['rf3'].voltage*np.sin(line['rf3'].lag/180*np.pi), eneloss/4, rtol=1e-5)


plt.show()
