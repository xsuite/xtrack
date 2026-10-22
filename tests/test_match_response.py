import numpy as np
import pytest
import xtrack as xt


def test_match_supplied_response_with_action():
    line = xt.Line(elements=[xt.Marker()])
    line['a'], line['b'] = 0.0, 0.0
    action = xt.Action(
        lambda: {'u': 2 * line['a'] + line['b'], 'v': line['a'] + 3 * line['b']}
    )
    opt = line.match(
        solve=False,
        vary=[xt.Vary('a', step=1e-6, weight=0.1), xt.Vary('b', step=1e-6, weight=3.0)],
        targets=[
            action.target('u', value=1.0, tol=1e-10, weight=2.0),
            action.target('v', value=2.0, tol=1e-10, weight=4.0),
        ],
    )
    response = opt.build_response()
    np.testing.assert_allclose(response.matrix, [[2, 1], [1, 3]], atol=1e-10)
    opt.set_response(response)
    svd = response.svd
    opt.step(rcond=1e-10)
    np.testing.assert_allclose([line['a'], line['b']], [0.2, 0.6], atol=1e-10)
    assert response.svd is svd
    assert opt.solver._last_jac_svd is svd
    # A clone's copied targets must not accidentally inherit the old response.
    assert opt.clone().response is None


def test_match_twiss_fixed_finite_difference_response():
    line = xt.Line(elements={'ring': xt.LineSegmentMap(qx=0.31, qy=0.32)})
    line.particle_ref = xt.Particles(p0c=1e9)
    line['qx'], line['qy'] = 0.31, 0.32
    line.ref['ring'].qx = line.ref['qx']
    line.ref['ring'].qy = line.ref['qy']
    opt = line.match(
        solve=False,
        method='4d',
        vary=[xt.Vary('qx', step=1e-5), xt.Vary('qy', step=1e-5)],
        targets=[xt.Target('qx', 0.33, tol=1e-9), xt.Target('qy', 0.34, tol=1e-9)],
    )
    response = opt.build_response()
    np.testing.assert_allclose(response.matrix, np.eye(2), atol=1e-7)
    opt.set_response(response)
    opt.solve(rcond=1e-8)
    tw = line.twiss(method='4d')
    np.testing.assert_allclose([tw.qx, tw.qy], [0.33, 0.34], atol=1e-9)
    opt._err.use_tpsa = True
    with pytest.raises(ValueError, match='use_tpsa'):
        opt.set_response(response)


def test_match_jacobian_report_and_env_snapshot():
    line = xt.Line(elements=[xt.Marker()])
    env = line.env
    line['a'] = 0.
    action = xt.Action(lambda: {'value': line['a']})
    opt = line.match(
        solve=False,
        vary=[xt.Vary('a', step=1e-6)],
        targets=[action.target('value', value=1., tol=1e-10)],
        solver_options={'n_bisections': 0},
    )
    opt.set_response([[-1.]])
    before = opt.get_knob_values()
    info = opt.run_jacobian(take_best=False)
    assert info.degraded
    env.vars.update(before)
    assert opt.get_knob_values() == before
    assert opt.get_knob_values(iteration=-1) != before
    opt.set_response([[1.]])
    info = opt.run_jacobian(take_best=False)
    assert info.improved and info.within_tol
    assert line['a'] == pytest.approx(1.)


def test_block_response_shared_twiss_and_activation():
    line = xt.Line(elements={'ring': xt.LineSegmentMap(qx=0.31, qy=0.32)})
    line.particle_ref = xt.Particles(p0c=1e9)
    line['qx'], line['qy'] = 0.31, 0.32
    line.ref['ring'].qx = line.ref['qx']
    line.ref['ring'].qy = line.ref['qy']
    calls = []

    def measure():
        calls.append(1)
        return line.twiss(method='4d')

    action = xt.Action(measure)
    opt = line.match(
        solve=False,
        vary=[xt.Vary('qx', step=1e-5, tag='x'), xt.Vary('qy', step=1e-5, tag='y')],
        targets=[action.target('qx', 0.33, tol=1e-9, tag='x'),
                 action.target('qy', 0.34, tol=1e-9, tag='y')],
        show_call_counter=False,
    )
    opt.set_response(blocks={
        'x': dict(matrix=[[1.]], targets=[0], vary=[0], rcond=1e-8),
        'y': dict(matrix=[[1.]], targets=[1], vary=[1], rcond=1e-6),
    })
    assert opt.actions == [action]
    calls.clear()
    np.testing.assert_allclose(opt.get_correction(), [0.02, 0.02], atol=1e-12)
    assert len(calls) == 1
    opt.disable(target=True, vary=True).enable(target='x', vary='x')
    svd = opt.response.svd
    np.testing.assert_allclose(opt.get_correction(), [0.02, 0.], atol=1e-12)
    assert len(calls) == 2
    info = opt.run_jacobian(take_best=False)
    assert info.improved and info.within_tol
    np.testing.assert_allclose([line['qx'], line['qy']], [0.33, 0.32], atol=1e-9)
    opt.disable(target=True, vary=True).enable(target='y', vary='y')
    opt.run_jacobian(take_best=False)
    np.testing.assert_allclose([line['qx'], line['qy']], [0.33, 0.34], atol=1e-9)
    opt.disable(target=True, vary=True).enable(target='x', vary='x')
    assert opt.response.svd is svd
    assert opt.clone().response is None
