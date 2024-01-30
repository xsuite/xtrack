import pathlib
import json
import numpy as np

import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

@for_all_test_contexts
def test_match_orbit_bump(test_context):

    filename = test_data_folder.joinpath(
        'hllhc14_no_errors_with_coupling_knobs/line_b1.json')

    with open(filename, 'r') as fid:
        dct = json.load(fid)
    line = xt.Line.from_dict(dct)

    line.build_tracker(_context=test_context)

    tw_before = line.twiss()

    opt = line.match(
        start='mq.33l8.b1',
        end='mq.23l8.b1',
        init=tw_before.get_twiss_init(at_element='mq.33l8.b1'),
        vary=[
            xt.Vary(name='acbv30.l8b1', step=1e-10),
            xt.Vary(name='acbv28.l8b1', step=1e-10),
            xt.Vary(name='acbv26.l8b1', step=1e-10),
            xt.Vary(name='acbv24.l8b1', step=1e-10),
        ],
        targets=[
            # I want the vertical orbit to be at 3 mm at mq.28l8.b1 with zero angle
            xt.Target('y', at='mb.b28l8.b1', value=3e-3, tol=1e-4, scale=1),
            xt.Target('py', at='mb.b28l8.b1', value=0, tol=1e-6, scale=1000),
            # I want the bump to be closed
            xt.Target('y', at='mq.23l8.b1', value=tw_before['y', 'mq.23l8.b1'],
                      tol=1e-6, scale=1),
            xt.Target('py', at='mq.23l8.b1', value=tw_before['py', 'mq.23l8.b1'],
                      tol=1e-7, scale=1000),
        ]
    )
    assert len(opt.actions) == 1
    assert opt.actions[0].kwargs['_keep_initial_particles'] is True
    assert isinstance(opt.actions[0].kwargs['_initial_particles'], xt.Particles)

    tw = line.twiss()

    assert np.isclose(tw['y', 'mb.b28l8.b1'], 3e-3, atol=1e-4)
    assert np.isclose(tw['py', 'mb.b28l8.b1'], 0, atol=1e-6)
    assert np.isclose(tw['y', 'mq.23l8.b1'], tw_before['y', 'mq.23l8.b1'], atol=1e-6)
    assert np.isclose(tw['py', 'mq.23l8.b1'], tw_before['py', 'mq.23l8.b1'], atol=1e-7)
    assert np.isclose(tw['y', 'mq.33l8.b1'], tw_before['y', 'mq.33l8.b1'], atol=1e-6)
    assert np.isclose(tw['py', 'mq.33l8.b1'], tw_before['py', 'mq.33l8.b1'], atol=1e-7)

    # There is a bit of leakage in the horizontal plane (due to feed-down from sextupoles)
    assert np.isclose(tw['x', 'mb.b28l8.b1'], tw_before['x', 'mb.b28l8.b1'], atol=100e-6)
    assert np.isclose(tw['px', 'mb.b28l8.b1'], tw_before['px', 'mb.b28l8.b1'], atol=100e-7)
    assert np.isclose(tw['x', 'mq.23l8.b1'], tw_before['x', 'mq.23l8.b1'], atol=100e-6)
    assert np.isclose(tw['px', 'mq.23l8.b1'], tw_before['px', 'mq.23l8.b1'], atol=100e-7)
    assert np.isclose(tw['x', 'mq.33l8.b1'], tw_before['x', 'mq.33l8.b1'], atol=100e-6)
    assert np.isclose(tw['px', 'mq.33l8.b1'], tw_before['px', 'mq.33l8.b1'], atol=100e-7)

    # Now I match the bump including the horizontal plane
    # I start from scratch
    line.vars['acbv30.l8b1'] = 0
    line.vars['acbv28.l8b1'] = 0
    line.vars['acbv26.l8b1'] = 0
    line.vars['acbv24.l8b1'] = 0

    opt = line.match(
        start='mq.33l8.b1',
        end='mq.23l8.b1',
        init=tw_before.get_twiss_init(at_element='mq.33l8.b1'),
        vary=[
            xt.Vary(name='acbv30.l8b1', step=1e-10),
            xt.Vary(name='acbv28.l8b1', step=1e-10),
            xt.Vary(name='acbv26.l8b1', step=1e-10),
            xt.Vary(name='acbv24.l8b1', step=1e-10),
            xt.Vary(name='acbh27.l8b1', step=1e-10),
            xt.Vary(name='acbh25.l8b1', step=1e-10),
        ],
        targets=[
            # I want the vertical orbit to be at 3 mm at mq.28l8.b1 with zero angle
            xt.Target('y', at='mb.b28l8.b1', value=3e-3, tol=1e-4, scale=1),
            xt.Target('py', at='mb.b28l8.b1', value=0, tol=1e-6, scale=1000),
            # I want the bump to be closed
            xt.Target('y', at='mq.23l8.b1', value=tw_before['y', 'mq.23l8.b1'],
                    tol=1e-6, scale=1),
            xt.Target('py', at='mq.23l8.b1', value=tw_before['py', 'mq.23l8.b1'],
                    tol=1e-7, scale=1000),
            xt.Target('x', at='mq.23l8.b1', value=tw_before['x', 'mq.23l8.b1'],
                    tol=1e-6, scale=1),
            xt.Target('px', at='mq.23l8.b1', value=tw_before['px', 'mq.23l8.b1'],
                    tol=1e-7, scale=1000),
        ]
    )

    assert len(opt.actions) == 1
    assert opt.actions[0].kwargs['_keep_initial_particles'] is True
    assert isinstance(opt.actions[0].kwargs['_initial_particles'], xt.Particles)

    tw = line.twiss()
    assert np.isclose(tw['y', 'mb.b28l8.b1'], 3e-3, atol=1e-4)
    assert np.isclose(tw['py', 'mb.b28l8.b1'], 0, atol=1e-6)
    assert np.isclose(tw['y', 'mq.23l8.b1'], tw_before['y', 'mq.23l8.b1'], atol=1e-6)
    assert np.isclose(tw['py', 'mq.23l8.b1'], tw_before['py', 'mq.23l8.b1'], atol=1e-7)
    assert np.isclose(tw['y', 'mq.33l8.b1'], tw_before['y', 'mq.33l8.b1'], atol=1e-6)
    assert np.isclose(tw['py', 'mq.33l8.b1'], tw_before['py', 'mq.33l8.b1'], atol=1e-7)

    # There is a bit of leakage in the horizontal plane (due to feed-down from sextupoles)
    assert np.isclose(tw['x', 'mb.b28l8.b1'], tw_before['x', 'mb.b28l8.b1'], atol=50e-6)
    assert np.isclose(tw['px', 'mb.b28l8.b1'], tw_before['px', 'mb.b28l8.b1'], atol=2e-6)
    assert np.isclose(tw['x', 'mq.23l8.b1'], tw_before['x', 'mq.23l8.b1'], atol=1e-6)
    assert np.isclose(tw['px', 'mq.23l8.b1'], tw_before['px', 'mq.23l8.b1'], atol=1e-7)
    assert np.isclose(tw['x', 'mq.33l8.b1'], tw_before['x', 'mq.33l8.b1'], atol=1e-6)
    assert np.isclose(tw['px', 'mq.33l8.b1'], tw_before['px', 'mq.33l8.b1'], atol=1e-7)

    # Same match but with init provided through a kwargs
    # I start from scratch
    line.vars['acbv30.l8b1'] = 0
    line.vars['acbv28.l8b1'] = 0
    line.vars['acbv26.l8b1'] = 0
    line.vars['acbv24.l8b1'] = 0

    tini = tw_before.get_twiss_init(at_element='mq.33l8.b1')

    opt = line.match(
        start='mq.33l8.b1',
        end='mq.23l8.b1',
        betx=1, bety=1,
        x=tini.x, px=tini.px, y=tini.y, py=tini.py,
        zeta=tini.zeta, delta=tini.delta,
        vary=[
            xt.Vary(name='acbv30.l8b1', step=1e-10),
            xt.Vary(name='acbv28.l8b1', step=1e-10),
            xt.Vary(name='acbv26.l8b1', step=1e-10),
            xt.Vary(name='acbv24.l8b1', step=1e-10),
            xt.Vary(name='acbh27.l8b1', step=1e-10),
            xt.Vary(name='acbh25.l8b1', step=1e-10),
        ],
        targets=[
            # I want the vertical orbit to be at 3 mm at mq.28l8.b1 with zero angle
            xt.Target('y', at='mb.b28l8.b1', value=3e-3, tol=1e-4, scale=1),
            xt.Target('py', at='mb.b28l8.b1', value=0, tol=1e-6, scale=1000),
            # I want the bump to be closed
            xt.Target('y', at='mq.23l8.b1', value=tw_before['y', 'mq.23l8.b1'],
                    tol=1e-6, scale=1),
            xt.Target('py', at='mq.23l8.b1', value=tw_before['py', 'mq.23l8.b1'],
                    tol=1e-7, scale=1000),
            xt.Target('x', at='mq.23l8.b1', value=tw_before['x', 'mq.23l8.b1'],
                    tol=1e-6, scale=1),
            xt.Target('px', at='mq.23l8.b1', value=tw_before['px', 'mq.23l8.b1'],
                    tol=1e-7, scale=1000),
        ]
    )

    assert len(opt.actions) == 1
    assert opt.actions[0].kwargs['_keep_initial_particles'] is True
    assert isinstance(opt.actions[0].kwargs['_initial_particles'], xt.Particles)

    tw = line.twiss()
    assert np.isclose(tw['y', 'mb.b28l8.b1'], 3e-3, atol=1e-4)
    assert np.isclose(tw['py', 'mb.b28l8.b1'], 0, atol=1e-6)
    assert np.isclose(tw['y', 'mq.23l8.b1'], tw_before['y', 'mq.23l8.b1'], atol=1e-6)
    assert np.isclose(tw['py', 'mq.23l8.b1'], tw_before['py', 'mq.23l8.b1'], atol=1e-7)
    assert np.isclose(tw['y', 'mq.33l8.b1'], tw_before['y', 'mq.33l8.b1'], atol=1e-6)
    assert np.isclose(tw['py', 'mq.33l8.b1'], tw_before['py', 'mq.33l8.b1'], atol=1e-7)

    # There is a bit of leakage in the horizontal plane (due to feed-down from sextupoles)
    assert np.isclose(tw['x', 'mb.b28l8.b1'], tw_before['x', 'mb.b28l8.b1'], atol=50e-6)
    assert np.isclose(tw['px', 'mb.b28l8.b1'], tw_before['px', 'mb.b28l8.b1'], atol=2e-6)
    assert np.isclose(tw['x', 'mq.23l8.b1'], tw_before['x', 'mq.23l8.b1'], atol=1e-6)
    assert np.isclose(tw['px', 'mq.23l8.b1'], tw_before['px', 'mq.23l8.b1'], atol=1e-7)
    assert np.isclose(tw['x', 'mq.33l8.b1'], tw_before['x', 'mq.33l8.b1'], atol=1e-6)
    assert np.isclose(tw['px', 'mq.33l8.b1'], tw_before['px', 'mq.33l8.b1'], atol=1e-7)

def test_match_orbit_bump_with_weights():

    with open(test_data_folder /
              'hllhc14_no_errors_with_coupling_knobs/line_b1.json', 'r') as fid:
        dct = json.load(fid)
    line = xt.Line.from_dict(dct)

    line.build_tracker()

    for init_at in [xt.START, xt.END]:
        tw0 = line.twiss()
        opt = line.match(
            #verbose=True,
            solver='jacobian',
            # Portion of the beam line to be modified and initial conditions
            start='mq.33l8.b1',
            end='mq.17l8.b1',
            init_at=init_at,
            init=tw0,
            # Dipole corrector strengths to be varied
            vary=[
                xt.Vary(name='acbv32.l8b1', step=1e-10, weight=0.7),
                xt.Vary(name='acbv28.l8b1', step=1e-10, weight=0.3),
                xt.Vary(name='acbv26.l8b1', step=1e-10),
                xt.Vary(name='acbv24.l8b1', step=1e-10),
                xt.Vary(name='acbv22.l8b1', step=1e-10, limits=[-38e-6, 38e-6], weight=1000),
                xt.Vary(name='acbv18.l8b1', step=1e-10),
            ],
            targets=[
                # I want the vertical orbit to be at 3 mm at mq.28l8.b1 with zero angle
                xt.Target('y', at='mb.b26l8.b1', value=3e-3, tol=1e-4),
                xt.Target('py', at='mb.b26l8.b1', value=0, tol=1e-6),
                # I want the bump to be closed
                xt.Target('y', at='mq.17l8.b1', value=tw0, tol=1e-6),
                xt.Target('py', at='mq.17l8.b1', value=tw0, tol=1e-7, weight=1e3),
                xt.Target('y', at='mq.33l8.b1', value=tw0, tol=1e-6),
                xt.Target('py', at='mq.33l8.b1', value=tw0, tol=1e-7, weight=1e3),
                # I want to limit the negative excursion ot the bump
                xt.Target('y', xt.LessThan(-1e-3), at='mq.30l8.b1', tol=1e-6),
            ]
        )

        assert len(opt.actions) == 1
        assert opt.actions[0].kwargs['_keep_initial_particles'] is True
        assert isinstance(opt.actions[0].kwargs['_initial_particles'], xt.Particles)

        tw = line.twiss()

        assert np.isclose(tw['y', 'mq.33l8.b1'], 0, atol=1e-6, rtol=0)
        assert np.isclose(tw['y', 'mq.17l8.b1'], 0, atol=1e-6, rtol=0)
        assert np.isclose(tw['py', 'mq.17l8.b1'], 0, atol=1e-8, rtol=0)
        assert np.isclose(tw['py', 'mq.33l8.b1'], 0, atol=1e-6, rtol=0)

        assert np.isclose(tw['y', 'mb.b26l8.b1'], 3e-3, atol=1e-6, rtol=0)
        assert np.isclose(tw['py', 'mb.b26l8.b1'], 0, atol=1e-8, rtol=0)

        assert np.isclose(tw['y', 'mq.30l8.b1'], -1e-3, atol=1e-6, rtol=0)
        assert np.isclose(line.vars['acbv22.l8b1']._value, 38e-6, atol=0, rtol=0.02)

        # Extract last twiss done by optimizer
        last_data = opt._err._last_data
        action = list(last_data.keys())[0]
        last_twiss  = last_data[action]
        assert last_twiss.orientation == (
            'backward' if init_at == xt.END else 'forward')
        assert last_twiss.method == '6d'
        assert last_twiss.reference_frame == 'proper'

        targets = opt.targets
        assert targets[0].tar == ('y', 'mb.b26l8.b1')
        assert targets[0].weight == 10
        assert targets[1].tar == ('py', 'mb.b26l8.b1')
        assert targets[1].weight == 100
        assert targets[3].tar == ('py', 'mq.17l8.b1')
        assert targets[3].weight == 1000


@for_all_test_contexts
def test_match_orbit_bump_within_multiline(test_context):

    filename = test_data_folder.joinpath(
        'hllhc14_no_errors_with_coupling_knobs/line_b1.json')

    with open(filename, 'r') as fid:
        dct = json.load(fid)
    line = xt.Line.from_dict(dct)
    collider = xt.Multiline(
        lines={'lhcb1': line}
    )

    collider.build_trackers(_context=test_context)

    tw_before = collider.twiss().lhcb1

    tw0 = collider.twiss()

    opt = collider.match(
        lines=['lhcb1'],
        start=['mq.33l8.b1'],
        end=['mq.23l8.b1'],
        init_at=xt.START,
        init=tw0,
        vary=[
            xt.Vary(name='acbv30.l8b1', step=1e-10),
            xt.Vary(name='acbv28.l8b1', step=1e-10),
            xt.Vary(name='acbv26.l8b1', step=1e-10),
            xt.Vary(name='acbv24.l8b1', step=1e-10),
        ],
        targets=[
            # I want the vertical orbit to be at 3 mm at mq.28l8.b1 with zero angle
            xt.Target('y', line='lhcb1', at='mb.b28l8.b1', value=3e-3, tol=1e-4, scale=1),
            xt.Target('py', line='lhcb1', at='mb.b28l8.b1', value=0, tol=1e-6, scale=1000),
            # I want the bump to be closed
            xt.Target('y', line='lhcb1', at='mq.23l8.b1', value=tw_before['y', 'mq.23l8.b1'],
                      tol=1e-6, scale=1),
            xt.Target('py', line='lhcb1', at='mq.23l8.b1', value=tw_before['py', 'mq.23l8.b1'],
                      tol=1e-7, scale=1000),
        ]
    )

    assert len(opt.actions) == 1
    assert len(opt.actions[0].kwargs['_keep_initial_particles']) == 1
    assert opt.actions[0].kwargs['_keep_initial_particles'][0] is True
    assert isinstance(opt.actions[0].kwargs['_initial_particles'][0], xt.Particles)

    tw_after_collider = collider.twiss()
    tw = tw_after_collider.lhcb1

    assert np.isclose(tw['y', 'mb.b28l8.b1'], 3e-3, atol=1e-4)
    assert np.isclose(tw['py', 'mb.b28l8.b1'], 0, atol=1e-6)
    assert np.isclose(tw['y', 'mq.23l8.b1'], tw_before['y', 'mq.23l8.b1'], atol=1e-6)
    assert np.isclose(tw['py', 'mq.23l8.b1'], tw_before['py', 'mq.23l8.b1'], atol=1e-7)
    assert np.isclose(tw['y', 'mq.33l8.b1'], tw_before['y', 'mq.33l8.b1'], atol=1e-6)
    assert np.isclose(tw['py', 'mq.33l8.b1'], tw_before['py', 'mq.33l8.b1'], atol=1e-7)

@for_all_test_contexts
def test_bump_step_and_smooth_inequalities(test_context):

    filename = test_data_folder.joinpath(
        'hllhc14_no_errors_with_coupling_knobs/line_b1.json')

    with open(filename, 'r') as fid:
        dct = json.load(fid)
    line = xt.Line.from_dict(dct)

    line.build_tracker(test_context)

    tw_before = line.twiss()

    GreaterThan = xt.GreaterThan
    LessThan = xt.LessThan

    tw0=line.twiss()

    opt = line.match(
        solve=False,
        solver='jacobian',
        # Portion of the beam line to be modified and initial conditions
        start='mq.33l8.b1',
        end='mq.17l8.b1',
        init_at=xt.START,
        init=tw0,
        # Dipole corrector strengths to be varied
        vary=[
            xt.Vary(name='acbv28.l8b1', step=1e-10),
            xt.Vary(name='acbv26.l8b1', step=1e-10),
            xt.Vary(name='acbv24.l8b1', step=1e-10),
            xt.Vary(name='acbv22.l8b1', step=1e-10),
        ],
        targets=[
            xt.Target('y', GreaterThan(2.7e-3), at='mb.b26l8.b1'),
            xt.Target('y', GreaterThan(2.7e-3), at='mb.b25l8.b1'),
            xt.Target('y', at='mq.24l8.b1', value=xt.LessThan(3e-3)),
            xt.Target('y', at='mq.26l8.b1', value=xt.LessThan(6e-3)),
            xt.TargetSet(['y', 'py'], at='mq.17l8.b1', value=tw0),
        ]
    )

    assert len(opt.actions) == 1
    assert opt.actions[0].kwargs['_keep_initial_particles'] is True
    assert isinstance(opt.actions[0].kwargs['_initial_particles'], xt.Particles)

    # Check freeze
    opt.step(1)
    ts = opt.target_status(ret=True)
    assert ts.tol_met[2] == False
    assert ts.tol_met[4] == False
    assert ts.tol_met[5] == False

    opt.targets[2].freeze()
    opt.targets[5].freeze()

    ts = opt.target_status(ret=True)
    assert ts.tol_met[2] == True
    assert ts.tol_met[4] == False
    assert ts.tol_met[5] == True

    opt.targets[2].unfreeze()
    opt.targets[5].unfreeze()

    ts = opt.target_status(ret=True)
    assert ts.tol_met[2] == False
    assert ts.tol_met[4] == False
    assert ts.tol_met[5] == False

    opt.solve()

    tw = line.twiss()
    assert tw['y', 'mb.b26l8.b1'] > 2.7e-3
    assert tw['y', 'mb.b25l8.b1'] > 2.7e-3
    assert tw['y', 'mq.24l8.b1'] < 3e-3 + 1e-6
    assert np.isclose(tw['y', 'mq.17l8.b1'], tw_before['y', 'mq.17l8.b1'], rtol=0, atol=1e-7)
    assert np.isclose(tw['py', 'mq.17l8.b1'], tw_before['py', 'mq.17l8.b1'], rtol=0, atol=1e-9)

    assert isinstance(opt.targets[0].value, xt.GreaterThan)
    assert isinstance(opt.targets[1].value, xt.GreaterThan)
    assert isinstance(opt.targets[2].value, xt.LessThan)

    assert opt.targets[0].value._value == 0
    assert opt.targets[1].value._value == 0
    assert opt.targets[2].value._value == 0

    assert opt.targets[0].value.mode == 'step'
    assert opt.targets[1].value.mode == 'step'
    assert opt.targets[2].value.mode == 'step'

    assert opt.targets[0].value.lower == 2.7e-3
    assert opt.targets[1].value.lower == 2.7e-3
    assert opt.targets[2].value.upper == 3e-3


    # Test mode smooth
    # Remove the bump

    for kk in ['acbv28.l8b1', 'acbv26.l8b1', 'acbv24.l8b1', 'acbv22.l8b1']:
        line.vars[kk] = 0

    tw_before = line.twiss()
    assert tw_before['y', 'mb.b26l8.b1'] < 1e-7
    assert tw_before['y', 'mb.b25l8.b1'] < 1e-7

    tw0 = line.twiss()
    opt = line.match(
        solve=False,
        solver='jacobian',
        # Portion of the beam line to be modified and initial conditions
        start='mq.33l8.b1',
        end='mq.17l8.b1',
        init_at=xt.START,
        init=tw0,
        # Dipole corrector strengths to be varied
        vary=[
            xt.Vary(name='acbv28.l8b1', step=1e-10),
            xt.Vary(name='acbv26.l8b1', step=1e-10),
            xt.Vary(name='acbv24.l8b1', step=1e-10),
            xt.Vary(name='acbv22.l8b1', step=1e-10),
        ],
        targets=[
            xt.Target('y', GreaterThan(2.7e-3, mode='smooth', sigma_rel=0.05), at='mb.b26l8.b1'),
            xt.Target('y', GreaterThan(2.7e-3, mode='smooth'), at='mb.b25l8.b1'),
            xt.Target('y', at='mq.24l8.b1', value=xt.LessThan(3e-3, mode='smooth', sigma_rel=0.04)),
            xt.Target('y', at='mq.26l8.b1', value=xt.LessThan(6e-3, mode='smooth')),
            xt.TargetSet(['y', 'py'], at='mq.17l8.b1', value=tw0),
        ]
    )

    assert len(opt.actions) == 1
    assert opt.actions[0].kwargs['_keep_initial_particles'] is True
    assert isinstance(opt.actions[0].kwargs['_initial_particles'], xt.Particles)

    # Check freeze
    opt.step(1)
    ts = opt.target_status(ret=True)
    assert ts.tol_met[0] == False
    assert ts.tol_met[1] == False
    assert ts.tol_met[5] == False

    opt.targets[0].freeze()
    opt.targets[5].freeze()

    ts = opt.target_status(ret=True)
    assert ts.tol_met[0] == True
    assert ts.tol_met[1] == False
    assert ts.tol_met[5] == True

    opt.targets[0].unfreeze()
    opt.targets[5].unfreeze()

    ts = opt.target_status(ret=True)
    assert ts.tol_met[0] == False
    assert ts.tol_met[1] == False
    assert ts.tol_met[5] == False

    opt.solve()

    tw = line.twiss()

    assert tw['y', 'mb.b26l8.b1'] > 2.7e-3 - 3e-6
    assert tw['y', 'mb.b25l8.b1'] > 2.7e-3 - 3e-6
    assert tw['y', 'mq.24l8.b1'] < 3e-3 + 3e-6
    assert tw['y', 'mq.26l8.b1'] < 6e-3 + 3e-6
    assert np.isclose(tw['y', 'mq.17l8.b1'], tw_before['y', 'mq.17l8.b1'], rtol=0, atol=1e-7)
    assert np.isclose(tw['py', 'mq.17l8.b1'], tw_before['py', 'mq.17l8.b1'], rtol=0, atol=1e-9)

    assert isinstance(opt.targets[0].value, xt.GreaterThan)
    assert isinstance(opt.targets[1].value, xt.GreaterThan)
    assert isinstance(opt.targets[2].value, xt.LessThan)
    assert isinstance(opt.targets[3].value, xt.LessThan)

    assert opt.targets[0].value._value == 0
    assert opt.targets[1].value._value == 0
    assert opt.targets[2].value._value == 0
    assert opt.targets[3].value._value == 0

    assert opt.targets[0].value.mode == 'smooth'
    assert opt.targets[1].value.mode == 'smooth'
    assert opt.targets[2].value.mode == 'smooth'
    assert opt.targets[3].value.mode == 'smooth'

    assert opt.targets[0].value.lower == 2.7e-3
    assert opt.targets[1].value.lower == 2.7e-3
    assert opt.targets[2].value.upper == 3e-3
    assert opt.targets[3].value.upper == 6e-3

    assert np.isclose(opt.targets[0].value.sigma, 0.05 * 2.7e-3, atol=0, rtol=1e-10)
    assert np.isclose(opt.targets[1].value.sigma, 0.01 * 2.7e-3, atol=0, rtol=1e-10)
    assert np.isclose(opt.targets[2].value.sigma, 0.04 * 3e-3, atol=0, rtol=1e-10)
    assert np.isclose(opt.targets[3].value.sigma, 0.01 * 6e-3, atol=0, rtol=1e-10)

    x_cut_norm = 1/16 + np.sqrt(33)/16
    poly = lambda x: 3 * x**3 - 2 * x**4

    # Check smooth target (GreaterThan)
    i_tar_gt = 0
    tar_gt = opt.targets[i_tar_gt]
    sigma_gt = tar_gt.value.sigma
    x0_gt = tar_gt.runeval()

    edge_test_gt = np.linspace(x0_gt - 3 * sigma_gt, x0_gt + 3 * sigma_gt, 100)

    residue_gt = edge_test_gt * 0
    for ii, xx in enumerate(edge_test_gt):
        tar_gt.value.lower = xx
        residue_gt[ii] =  opt._err()[i_tar_gt] / tar_gt.weight

    x_minus_edge_gt = x0_gt - edge_test_gt
    x_cut_gt = -x_cut_norm * sigma_gt

    mask_zero_gt = x_minus_edge_gt > 0
    assert np.all(residue_gt[mask_zero_gt] == 0)
    mask_linear_gt = x_minus_edge_gt < x_cut_gt
    assert np.allclose(residue_gt[mask_linear_gt],
        -x_minus_edge_gt[mask_linear_gt] - x_cut_norm * sigma_gt + sigma_gt*poly(x_cut_norm),
        atol=0, rtol=1e-10)
    mask_poly_gt = (~mask_zero_gt) & (~mask_linear_gt)
    assert np.allclose(residue_gt[mask_poly_gt],
        sigma_gt * poly(-x_minus_edge_gt[mask_poly_gt]/sigma_gt),
        atol=0, rtol=1e-10)

    # Check smooth target (LessThan)
    i_tar_lt = 2
    tar_lt = opt.targets[i_tar_lt]
    sigma_lt = tar_lt.value.sigma
    x0_lt = tar_lt.runeval()

    edge_test_lt = np.linspace(x0_lt - 3 * sigma_lt, x0_lt + 3 * sigma_lt, 100)

    residue_lt = edge_test_lt * 0
    for ii, xx in enumerate(edge_test_lt):
        tar_lt.value.upper = xx
        residue_lt[ii] =  opt._err()[i_tar_lt] / tar_lt.weight

    x_minus_edge_lt = x0_lt - edge_test_lt
    x_cut_lt = x_cut_norm * sigma_lt

    mask_zero_lt = x_minus_edge_lt < 0
    assert np.all(residue_lt[mask_zero_lt] == 0)
    mask_linear_lt = x_minus_edge_lt > x_cut_lt
    assert np.allclose(residue_lt[mask_linear_lt],
        x_minus_edge_lt[mask_linear_lt] - x_cut_norm * sigma_lt + sigma_lt*poly(x_cut_norm),
        atol=0, rtol=1e-10)
    mask_poly_lt = (~mask_zero_lt) & (~mask_linear_lt)
    assert np.allclose(residue_lt[mask_poly_lt],
        sigma_lt * poly(x_minus_edge_lt[mask_poly_lt]/sigma_lt),
        atol=0, rtol=1e-10)

@for_all_test_contexts
def test_match_bump_sets_implicit_end(test_context):

    line = xt.Line.from_json(test_data_folder /
                             'hllhc15_thick/lhc_thick_with_knobs.json')
    line.build_tracker(test_context)

    opt = line.match(
        start='mq.30l8.b1', end='mq.23l8.b1',
        betx=1, bety=1, y=0, py=0, # <-- conditions at start
        vary=xt.VaryList(['acbv30.l8b1', 'acbv28.l8b1', 'acbv26.l8b1', 'acbv24.l8b1'],
                        step=1e-10, limits=[-1e-3, 1e-3]),
        targets = [
            xt.TargetSet(y=3e-3, py=0, at='mb.b28l8.b1'),
            xt.TargetSet(y=0, py=0, at=xt.END)
        ])

    assert len(opt.actions) == 1
    assert opt.actions[0].kwargs['_keep_initial_particles'] is True
    assert isinstance(opt.actions[0].kwargs['_initial_particles'], xt.Particles)

    opt.tag(tag='matched')
    opt.reload(0)
    tw_before = line.twiss(method='4d')
    assert np.isclose(tw_before['y', 'mb.b28l8.b1'], 0, atol=1e-4)
    opt.reload(tag='matched')
    tw = line.twiss(method='4d')

    assert np.isclose(tw['y', 'mb.b28l8.b1'], 3e-3, atol=1e-4)
    assert np.isclose(tw['py', 'mb.b28l8.b1'], 0, atol=1e-6)
    assert np.isclose(tw['y', 'mq.23l8.b1'], tw_before['y', 'mq.23l8.b1'], atol=1e-6)
    assert np.isclose(tw['py', 'mq.23l8.b1'], tw_before['py', 'mq.23l8.b1'], atol=1e-7)
    assert np.isclose(tw['y', 'mq.33l8.b1'], tw_before['y', 'mq.33l8.b1'], atol=1e-6)
    assert np.isclose(tw['py', 'mq.33l8.b1'], tw_before['py', 'mq.33l8.b1'], atol=1e-7)

@for_all_test_contexts
def test_match_bump_sets_init_end(test_context):

    line = xt.Line.from_json(test_data_folder /
                             'hllhc15_thick/lhc_thick_with_knobs.json')
    line.build_tracker(test_context)

    opt = line.match(
        start='mq.30l8.b1', end='mq.23l8.b1',
        init_at=xt.END, betx=1, bety=1, y=0, py=0, # <-- conditions at end
        vary=xt.VaryList(['acbv30.l8b1', 'acbv28.l8b1', 'acbv26.l8b1', 'acbv24.l8b1'],
                        step=1e-10, limits=[-1e-3, 1e-3]),
        targets = [
            xt.TargetSet(y=3e-3, py=0, at='mb.b28l8.b1'),
            xt.TargetSet(y=0, py=0, at=xt.START)
    ])

    assert len(opt.actions) == 1
    assert opt.actions[0].kwargs['_keep_initial_particles'] is True
    assert isinstance(opt.actions[0].kwargs['_initial_particles'], xt.Particles)

    opt.tag(tag='matched')
    opt.reload(0)
    tw_before = line.twiss(method='4d')
    assert np.isclose(tw_before['y', 'mb.b28l8.b1'], 0, atol=1e-4)
    opt.reload(tag='matched')
    tw = line.twiss(method='4d')

    assert np.isclose(tw['y', 'mb.b28l8.b1'], 3e-3, atol=1e-4)
    assert np.isclose(tw['py', 'mb.b28l8.b1'], 0, atol=1e-6)
    assert np.isclose(tw['y', 'mq.23l8.b1'], tw_before['y', 'mq.23l8.b1'], atol=1e-6)
    assert np.isclose(tw['py', 'mq.23l8.b1'], tw_before['py', 'mq.23l8.b1'], atol=1e-7)
    assert np.isclose(tw['y', 'mq.33l8.b1'], tw_before['y', 'mq.33l8.b1'], atol=1e-6)
    assert np.isclose(tw['py', 'mq.33l8.b1'], tw_before['py', 'mq.33l8.b1'], atol=1e-7)

@for_all_test_contexts
def test_match_bump_sets_init_middle(test_context):

    line = xt.Line.from_json(test_data_folder /
                             'hllhc15_thick/lhc_thick_with_knobs.json')
    line.build_tracker(test_context)

    opt = line.match(
        start='mq.30l8.b1', end='mq.23l8.b1',
        init_at='mb.b28l8.b1', betx=1, bety=1, y=3e-3, py=0, # <-- conditions at point inside the range
        vary=xt.VaryList(['acbv30.l8b1', 'acbv28.l8b1', 'acbv26.l8b1', 'acbv24.l8b1'],
                        step=1e-10, limits=[-1e-3, 1e-3]),
        targets = [
            xt.TargetSet(y=0, py=0, at=xt.START),
            xt.TargetSet(y=0, py=0, at=xt.END)
    ])

    assert len(opt.actions) == 1
    assert opt.actions[0].kwargs['_keep_initial_particles'] is True
    assert opt.actions[0].kwargs['_initial_particles'] is None # due to init in the middle

    opt.tag(tag='matched')
    opt.reload(0)
    tw_before = line.twiss(method='4d')
    assert np.isclose(tw_before['y', 'mb.b28l8.b1'], 0, atol=1e-4)
    opt.reload(tag='matched')
    tw = line.twiss(method='4d')

    assert np.isclose(tw['y', 'mb.b28l8.b1'], 3e-3, atol=1e-4)
    assert np.isclose(tw['py', 'mb.b28l8.b1'], 0, atol=1e-6)
    assert np.isclose(tw['y', 'mq.23l8.b1'], tw_before['y', 'mq.23l8.b1'], atol=1e-6)
    assert np.isclose(tw['py', 'mq.23l8.b1'], tw_before['py', 'mq.23l8.b1'], atol=1e-7)
    assert np.isclose(tw['y', 'mq.33l8.b1'], tw_before['y', 'mq.33l8.b1'], atol=1e-6)
    assert np.isclose(tw['py', 'mq.33l8.b1'], tw_before['py', 'mq.33l8.b1'], atol=1e-7)

@for_all_test_contexts
def test_match_bump_sets_init_table(test_context):

    line = xt.Line.from_json(test_data_folder /
                             'hllhc15_thick/lhc_thick_with_knobs.json')
    line.build_tracker(test_context)

    tw0 = line.twiss(method='4d')
    opt = line.match(
        start='mq.30l8.b1', end='mq.23l8.b1',
        init=tw0, init_at=xt.END, # <-- Boundary conditions from table
        vary=xt.VaryList(['acbv30.l8b1', 'acbv28.l8b1', 'acbv26.l8b1', 'acbv24.l8b1'],
                        step=1e-10, limits=[-1e-3, 1e-3]),
        targets = [
            xt.TargetSet(y=3e-3, py=0, at='mb.b28l8.b1'),
            xt.TargetSet(['y', 'py'], value=tw0, at=xt.START) # <-- Target from table
        ])

    assert len(opt.actions) == 1
    assert opt.actions[0].kwargs['_keep_initial_particles'] is True
    assert isinstance(opt.actions[0].kwargs['_initial_particles'], xt.Particles)

    opt.tag(tag='matched')
    opt.reload(0)
    tw_before = line.twiss(method='4d')
    assert np.isclose(tw_before['y', 'mb.b28l8.b1'], 0, atol=1e-4)
    opt.reload(tag='matched')
    tw = line.twiss(method='4d')

    assert np.isclose(tw['y', 'mb.b28l8.b1'], 3e-3, atol=1e-4)
    assert np.isclose(tw['py', 'mb.b28l8.b1'], 0, atol=1e-6)
    assert np.isclose(tw['y', 'mq.23l8.b1'], tw_before['y', 'mq.23l8.b1'], atol=1e-6)
    assert np.isclose(tw['py', 'mq.23l8.b1'], tw_before['py', 'mq.23l8.b1'], atol=1e-7)
    assert np.isclose(tw['y', 'mq.33l8.b1'], tw_before['y', 'mq.33l8.b1'], atol=1e-6)
    assert np.isclose(tw['py', 'mq.33l8.b1'], tw_before['py', 'mq.33l8.b1'], atol=1e-7)

@for_all_test_contexts
def test_match_bump_common_elements(test_context):
    # Load a line and build a tracker
    collider = xt.Multiline.from_json(test_data_folder /
                        'hllhc15_thick/hllhc15_collider_thick.json')
    collider.build_trackers(test_context)

    tw0 = collider.twiss(method='4d')

    opt = collider.match(
        lines=['lhcb1', 'lhcb2'],
        start=['e.ds.l5.b1', 'e.ds.l5.b2'],
        end=['s.ds.r5.b1', 's.ds.r5.b2'],
        init=tw0,
        vary=xt.VaryList([
            'acbxv1.r5', 'acbxv1.l5', # <-- common elements
            'acbyvs4.l5b1', 'acbrdv4.r5b1', 'acbcv5.l5b1', # <-- b1
            'acbyvs4.l5b2', 'acbrdv4.r5b2', 'acbcv5.r5b2', # <-- b2
            ],
            step=1e-10, limits=[-1e-3, 1e-3]),
        targets = [
            xt.TargetSet(y=0, py=10e-6, at='ip5', line='lhcb1'),
            xt.TargetSet(y=0, py=-10e-6, at='ip5', line='lhcb2'),
            xt.TargetSet(y=0, py=0, at=xt.END, line='lhcb1'),
            xt.TargetSet(y=0, py=0, at=xt.END, line='lhcb2')
        ])

    assert len(opt.actions) == 1
    assert len(opt.actions[0].kwargs['_keep_initial_particles']) == 2
    assert opt.actions[0].kwargs['_keep_initial_particles'][0] is True
    assert opt.actions[0].kwargs['_keep_initial_particles'][1] is True
    assert isinstance(opt.actions[0].kwargs['_initial_particles'][0], xt.Particles)
    assert isinstance(opt.actions[0].kwargs['_initial_particles'][1], xt.Particles)

    tw = collider.twiss()
    assert np.isclose(tw.lhcb1['y', 'ip5'], 0, rtol=0, atol=1e-9)
    assert np.isclose(tw.lhcb1['py', 'ip5'], 10e-6, rtol=0, atol=1e-9)
    assert np.isclose(tw.lhcb1['y', 's.ds.r5.b1'], 0, rtol=0, atol=1e-9)
    assert np.isclose(tw.lhcb1['py', 's.ds.r5.b1'], 0, rtol=0, atol=1e-9)
    assert np.isclose(tw.lhcb2['y', 'ip5'], 0, rtol=0, atol=1e-9)
    assert np.isclose(tw.lhcb2['py', 'ip5'], -10e-6, rtol=0, atol=1e-10)
    assert np.isclose(tw.lhcb2['y', 's.ds.r5.b2'], 0, rtol=0, atol=1e-9)
    assert np.isclose(tw.lhcb2['py', 's.ds.r5.b2'], 0, rtol=0, atol=1e-9)

@for_all_test_contexts
def test_match_bump_common_elements_callables_and_inequalities(test_context):
    # Load a line and build a tracker
    collider = xt.Multiline.from_json(test_data_folder /
                        'hllhc15_thick/hllhc15_collider_thick.json')
    collider.build_trackers(test_context)

    tw0 = collider.twiss(method='4d')

    opt = collider.match(
        lines=['lhcb1', 'lhcb2'],
        start=['e.ds.l5.b1', 'e.ds.l5.b2'],
        end=['s.ds.r5.b1', 's.ds.r5.b2'],
        init=tw0,
        vary=xt.VaryList([
            'acbxv1.r5', 'acbxv1.l5', # <-- common elements
            'acbyvs4.l5b1', 'acbrdv4.r5b1', 'acbcv5.l5b1', # <-- b1
            'acbyvs4.l5b2', 'acbrdv4.r5b2', 'acbcv5.r5b2', # <-- b2
            ],
            step=1e-10, limits=[-1e-3, 1e-3]),
        targets = [
            xt.Target(y=0, at='ip5', line='lhcb1'),
            xt.Target('py', xt.GreaterThan(9e-6), at='ip5', line='lhcb1'), # <-- inequality
            xt.Target('py', xt.LessThan(  11e-6), at='ip5', line='lhcb1'), # <-- inequality
            xt.Target(y=0, at='ip5', line='lhcb2'),
            xt.Target(
                lambda tw: tw.lhcb1['py', 'ip5'] + tw.lhcb2['py', 'ip5'], value=0), # <-- callable
            xt.TargetSet(y=0, py=0, at=xt.END, line='lhcb1'),
            xt.TargetSet(y=0, py=0, at=xt.END, line='lhcb2')
        ])

    assert len(opt.actions) == 1
    assert len(opt.actions[0].kwargs['_keep_initial_particles']) == 2
    assert opt.actions[0].kwargs['_keep_initial_particles'][0] is True
    assert opt.actions[0].kwargs['_keep_initial_particles'][1] is True
    assert isinstance(opt.actions[0].kwargs['_initial_particles'][0], xt.Particles)
    assert isinstance(opt.actions[0].kwargs['_initial_particles'][1], xt.Particles)

    tw = collider.twiss()

    assert np.isclose(tw.lhcb1['y', 'ip5'], 0, rtol=0, atol=1e-9)
    assert np.isclose(tw.lhcb1['py', 'ip5'], 10e-6, rtol=0, atol=1.1e-6)
    assert np.isclose(tw.lhcb1['y', 's.ds.r5.b1'], 0, rtol=0, atol=1e-9)
    assert np.isclose(tw.lhcb1['py', 's.ds.r5.b1'], 0, rtol=0, atol=1e-9)
    assert np.isclose(tw.lhcb2['y', 'ip5'], 0, rtol=0, atol=1e-9)
    assert np.isclose(tw.lhcb2['py', 'ip5'] + tw.lhcb1['py', 'ip5'], 0, rtol=0, atol=1e-10)
    assert np.isclose(tw.lhcb2['y', 's.ds.r5.b2'], 0, rtol=0, atol=1e-9)
    assert np.isclose(tw.lhcb2['py', 's.ds.r5.b2'], 0, rtol=0, atol=1e-9)

@for_all_test_contexts
def test_match_bump_common_elements_targets_from_tables(test_context):
    # Load a line and build a tracker
    collider = xt.Multiline.from_json(test_data_folder /
                        'hllhc15_thick/hllhc15_collider_thick.json')
    collider.build_trackers(test_context)

    tw0 = collider.twiss(method='4d')

    twb1 = collider.lhcb1.twiss(start='e.ds.l5.b1', end='s.ds.r5.b1', init=tw0.lhcb1)
    twb2 = collider.lhcb2.twiss(start='e.ds.l5.b2', end='s.ds.r5.b2', init=tw0.lhcb2)
    vars = collider.vars
    line_b1 = collider.lhcb1

    opt = collider.match(
        solve=False,
        vary=xt.VaryList([
            'acbxv1.r5', 'acbxv1.l5', # <-- common elements
            'acbyvs4.l5b1', 'acbrdv4.r5b1', 'acbcv5.l5b1', 'acbcv6.r5b1', # <-- b1
            'acbyvs4.l5b2', 'acbrdv4.r5b2', 'acbcv5.r5b2', 'acbcv6.l5b2'  # <-- b2
            ],
            step=1e-10, limits=[-1e-3, 1e-3]),
        targets = [
            # Targets from b1 twiss
            twb1.target(y=0, py=10e-6, at='ip5'),
            twb1.target(y=0, py=0, at=xt.END),
            # Targets from b2 twiss
            twb2.target(y=0, py=-10e-6, at='ip5'),
            twb2.target(['y', 'py'], at=xt.END), # <-- preserve
            # Targets from vars
            vars.target('acbxv1.l5', xt.LessThan(1e-3)),
            vars.target('acbxv1.l5', xt.GreaterThan(1e-6)),
            vars.target(lambda vv: vv['acbxv1.l5'] + vv['acbxv1.r5'], xt.LessThan(1e-9)),
            # Targets from line
            line_b1.target(lambda ll: ll['mcbrdv.4r5.b1']._xobject.ksl[0], xt.GreaterThan(1e-6)),
            line_b1.target(lambda ll: ll['mcbxfbv.a2r5']._xobject.ksl[0] + ll['mcbxfbv.a2l5']._xobject.ksl[0],
                                    xt.LessThan(1e-9)),
        ])
    opt.solve()

    assert len(opt.actions) == 7
    assert opt.actions[0].kwargs['_keep_initial_particles'] is True
    assert opt.actions[1].kwargs['_keep_initial_particles'] is True
    assert isinstance(opt.actions[0].kwargs['_initial_particles'], xt.Particles)
    assert isinstance(opt.actions[1].kwargs['_initial_particles'], xt.Particles)

    tw = collider.twiss()

    assert np.isclose(tw.lhcb1['y', 'ip5'], 0, rtol=0, atol=1e-9)
    assert np.isclose(tw.lhcb1['py', 'ip5'], 10e-6, rtol=0, atol=1.1e-6)
    assert np.isclose(tw.lhcb1['y', 's.ds.r5.b1'], 0, rtol=0, atol=1e-9)
    assert np.isclose(tw.lhcb1['py', 's.ds.r5.b1'], 0, rtol=0, atol=1e-9)
    assert np.isclose(tw.lhcb2['y', 'ip5'], 0, rtol=0, atol=1e-9)
    assert np.isclose(tw.lhcb2['py', 'ip5'] + tw.lhcb1['py', 'ip5'], 0, rtol=0, atol=1e-10)
    assert np.isclose(tw.lhcb2['y', 's.ds.r5.b2'], 0, rtol=0, atol=1e-9)
    assert np.isclose(tw.lhcb2['py', 's.ds.r5.b2'], 0, rtol=0, atol=1e-9)
