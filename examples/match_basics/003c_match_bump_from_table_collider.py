import xtrack as xt

# Load a line and build a tracker
collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.build_trackers()

tw0 = collider.twiss(method='4d')

opt = collider.match(
    lines=['lhcb1', 'lhcb2'],
    ele_start=['mq.32l8.b1', 'mq.32l8.b2'],
    ele_stop=['mq.22l8.b1', 'mq.22l8.b2'],
    ele_init=['mq.32l8.b1', 'mq.32l8.b2'],
    table_for_twiss_init=tw0,
    vary=xt.VaryList(['acbv30.l8b1', 'acbv28.l8b1', 'acbv26.l8b1', 'acbv24.l8b1',
                      'acbv31.l8b2', 'acbv29.l8b2', 'acbv27.l8b2', 'acbv25.l8b2', 'acbv23.l8b2'],
                    step=1e-10, limits=[-1e-3, 1e-3]),
    targets = [
        xt.TargetSet(y=3e-3, py=0, at='mb.b28l8.b1', line='lhcb1'),
        xt.TargetSet(y=-3e-3, py=0, at='mb.b28l8.b2', line='lhcb2'),
        xt.TargetSet(y=0, py=0, at=xt.END, line='lhcb1'),
        xt.TargetSet(y=0, py=0, at=xt.END, line='lhcb2'),
    ])

opt.target_status()
# prints:
#
# Target status:
# id state tag tol_met      residue  current_val target_val description
#  0 ON           True  1.30104e-18        0.003      0.003 ('y', 'mb.b28l8.b1'), val=0.003, tol=1e- ...
#  1 ON           True -3.38813e-20 -3.38813e-20          0 ('py', 'mb.b28l8.b1'), val=0, tol=1e-10, ...
#  2 ON           True   -4.127e-17   -4.127e-17          0 ('y', 'mq.23l8.b1'), val=0, tol=1e-10, w ...
#  3 ON           True  -6.1664e-19  -6.1664e-19          0 ('py', 'mq.23l8.b1'), val=0, tol=1e-10,  ...

#!end-doc-part

line = collider['lhcb1']

# import matplotlib.pyplot as plt
# plt.close('all')

# for beam in ['lhcb1', 'lhcb2']:

#     i_beam = 1 if beam == 'lhcb1' else 2

#     opt.add_point_to_log(tag='matched')
#     opt.reload(0)
#     tw_before = line.twiss(method='4d')
#     opt.reload(tag='matched')
#     tw = line.twiss(method='4d')

#     fig = plt.figure(i_beam, figsize=(6.4*1.2, 4.8*0.8))
#     ax = fig.add_subplot(111)
#     ax.plot(tw.s, tw.y*1000, label='y')

#     for nn in ['mcbv.30l8.b1', 'mcbv.28l8.b1', 'mcbv.26l8.b1', 'mcbv.24l8.b1']:
#     ax.axvline(x=line.get_s_position(nn), color='k', linestyle='--', alpha=0.5)
#     ax.text(line.get_s_position(nn), 10, nn, rotation=90,
#             horizontalalignment='left', verticalalignment='top')

#     ax.axvline(x=line.get_s_position('mb.b28l8.b1'), color='r', linestyle='--', alpha=0.5)
#     ax.text(line.get_s_position('mb.b28l8.b1'), 10, 'mb.b28l8.b1', rotation=90,
#             horizontalalignment='left', verticalalignment='top')

#     ax.axvline(x=line.get_s_position('mq.30l8.b1'), color='g', linestyle='--', alpha=0.5)
#     ax.axvline(x=line.get_s_position('mq.23l8.b1'), color='g', linestyle='--', alpha=0.5)
#     ax.text(line.get_s_position('mq.30l8.b1'), 10, 'mq.30l8.b1', rotation=90,
#             horizontalalignment='right', verticalalignment='top')
#     ax.text(line.get_s_position('mq.23l8.b1'), 10, 'mq.23l8.b1', rotation=90,
#             horizontalalignment='right', verticalalignment='top')

#     ax.set_xlim(line.get_s_position('mq.30l8.b1') - 10,
#             line.get_s_position('mq.23l8.b1') + 10)
#     ax.set_xlabel('s [m]')
#     ax.set_ylabel('y [mm]')
#     ax.set_ylim(-0.5, 10)
#     plt.subplots_adjust(bottom=.152, top=.9, left=.1, right=.95)
#     plt.show()

#     import numpy as np
#     if beam == 'lhcb1':
#         assert np.isclose(tw['y', 'mb.b28l8.b1'], 3e-3, atol=1e-4)
#         assert np.isclose(tw['py', 'mb.b28l8.b1'], 0, atol=1e-6)
#         assert np.isclose(tw['y', 'mq.23l8.b1'], tw_before['y', 'mq.23l8.b1'], atol=1e-6)
#         assert np.isclose(tw['py', 'mq.23l8.b1'], tw_before['py', 'mq.23l8.b1'], atol=1e-7)
#         assert np.isclose(tw['y', 'mq.33l8.b1'], tw_before['y', 'mq.33l8.b1'], atol=1e-6)
#         assert np.isclose(tw['py', 'mq.33l8.b1'], tw_before['py', 'mq.33l8.b1'], atol=1e-7)`