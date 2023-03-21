import json
import time
import pathlib

import numpy as np

import xtrack as xt
import xpart as xp
from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
            __file__).parent.joinpath('../test_data').absolute()
path_line = test_data_folder.joinpath(
                'hllhc15_noerrors_nobb/line_w_knobs_and_particle.json')

with open(path_line) as f:
    dct = json.load(f)


@for_all_test_contexts
def test_match_tune_chromaticity(test_context):
    line = xt.Line.from_dict(dct['line'])
    line.particle_ref = xp.Particles.from_dict(dct['particle'])

    line.build_tracker(_context=test_context)

    print('\nInitial twiss parameters')
    tw_before = line.twiss()
    print(f"Qx = {tw_before['qx']:.5f} Qy = {tw_before['qy']:.5f} "
        f"Q'x = {tw_before['dqx']:.5f} Q'y = {tw_before['dqy']:.5f}")

    print(f"kqtf.b1 = {line.vars['kqtf.b1']._value}")
    print(f"kqtd.b1 = {line.vars['kqtd.b1']._value}")
    print(f"ksf.b1 = {line.vars['ksf.b1']._value}")
    print(f"ksd.b1 = {line.vars['ksd.b1']._value}")

    t1 = time.time()
    line.match(
        vary=[
            xt.Vary('kqtf.b1', step=1e-8),
            xt.Vary('kqtd.b1', step=1e-8),
            xt.Vary('ksf.b1', step=1e-8),
            xt.Vary('ksd.b1', step=1e-8),
        ],
        targets = [
            xt.Target('qx', 62.315, tol=1e-4),
            xt.Target(lambda tw: tw['qx'] - tw['qy'], 1.99, tol=1e-4),
            xt.Target('dqx', 10.0, tol=0.05),
            xt.Target('dqy', 12.0, tol=0.05)])
    t2 = time.time()
    print('\nTime fsolve: ', t2-t1)

    tw_final = line.twiss()
    print('\nFinal twiss parameters')
    print(f"Qx = {tw_final['qx']:.5f} Qy = {tw_final['qy']:.5f} "
        f"Q'x = {tw_final['dqx']:.5f} Q'y = {tw_final['dqy']:.5f}")
    print(f"kqtf.b1 = {line.vars['kqtf.b1']._value}")
    print(f"kqtd.b1 = {line.vars['kqtd.b1']._value}")
    print(f"ksf.b1 = {line.vars['ksf.b1']._value}")
    print(f"ksd.b1 = {line.vars['ksd.b1']._value}")

    assert np.isclose(tw_final['qx'], 62.315, atol=1e-7)
    assert np.isclose(tw_final['qy'], 60.325, atol=1e-7)
    assert np.isclose(tw_final['dqx'], 10.0, atol=0.05)
    assert np.isclose(tw_final['dqy'], 12.0, atol=0.05)


    t1 = time.time()
    line.match(
        vary=[
            xt.Vary('kqtf.b1', step=1e-8),
            xt.Vary('kqtd.b1', step=1e-8),
            xt.Vary('ksf.b1', step=1e-8),
            xt.Vary('ksd.b1', step=1e-8),
        ],
        targets = [
            xt.Target('qx', 62.27, tol=1e-4),
            xt.Target('qy', 60.28, tol=1e-4),
            xt.Target('dqx', -5.0, tol=0.05),
            xt.Target('dqy', -7.0, tol=0.05),])
    t2 = time.time()
    print('\nTime fsolve: ', t2-t1)

    tw_final = line.twiss()
    print('\nFinal twiss parameters')
    print(f"Qx = {tw_final['qx']:.5f} Qy = {tw_final['qy']:.5f} "
        f"Q'x = {tw_final['dqx']:.5f} Q'y = {tw_final['dqy']:.5f}")
    print(f"kqtf.b1 = {line.vars['kqtf.b1']._value}")
    print(f"kqtd.b1 = {line.vars['kqtd.b1']._value}")
    print(f"ksf.b1 = {line.vars['ksf.b1']._value}")
    print(f"ksd.b1 = {line.vars['ksd.b1']._value}")

    assert np.isclose(tw_final['qx'], 62.27, atol=1e-4)
    assert np.isclose(tw_final['qy'], 60.28, atol=1e-4)
    assert np.isclose(tw_final['dqx'], -5.0, atol=0.05)
    assert np.isclose(tw_final['dqy'], -7.0, atol=0.05)

    # Trying 4d matching
    for ee in line.elements:
        if isinstance(ee, xt.Cavity):
            ee.voltage = 0.0
    line.match(method='4d', # <-- 4d matchin
        freeze_longitudinal=True,
        vary=[
            xt.Vary('kqtf.b1', step=1e-8),
            xt.Vary('kqtd.b1', step=1e-8),
            xt.Vary('ksf.b1', step=1e-8),
            xt.Vary('ksd.b1', step=1e-8),
        ],
        targets = [
            xt.Target('qx', 62.29, tol=1e-4),
            xt.Target('qy', 60.31, tol=1e-4),
            xt.Target('dqx', 6.0, tol=0.05),
            xt.Target('dqy', 4.0, tol=0.05),])
    t2 = time.time()
    print('\nTime fsolve: ', t2-t1)

    tw_final = line.twiss(method='4d')
    print('\nFinal twiss parameters')
    print(f"Qx = {tw_final['qx']:.5f} Qy = {tw_final['qy']:.5f} "
        f"Q'x = {tw_final['dqx']:.5f} Q'y = {tw_final['dqy']:.5f}")
    print(f"kqtf.b1 = {line.vars['kqtf.b1']._value}")
    print(f"kqtd.b1 = {line.vars['kqtd.b1']._value}")
    print(f"ksf.b1 = {line.vars['ksf.b1']._value}")
    print(f"ksd.b1 = {line.vars['ksd.b1']._value}")

    assert np.isclose(tw_final['qx'], 62.29, atol=1e-4)
    assert np.isclose(tw_final['qy'], 60.31, atol=1e-4)
    assert np.isclose(tw_final['dqx'],  6.0, atol=0.05)
    assert np.isclose(tw_final['dqy'],  4.0, atol=0.05)


@for_all_test_contexts
def test_match_coupling(test_context):
    with open(test_data_folder /
        'hllhc14_no_errors_with_coupling_knobs/line_b1.json', 'r') as fid:
        dct_b1 = json.load(fid)
    line = xt.Line.from_dict(dct_b1)

    line.build_tracker(_context=test_context)

    tw = line.twiss()

    assert tw.c_minus < 1e-4

    # Try to measure and match coupling
    line.vars['cmrskew'] = 1e-3
    line.vars['cmiskew'] = 1e-3

    tw = line.twiss()
    assert tw.c_minus > 2e-4

    # Match coupling
    line.match(verbose=True,
        vary=[
            xt.Vary(name='cmrskew', limits=[-1e-2, 1e-2], step=1e-5),
            xt.Vary(name='cmiskew', limits=[-1e-2, 1e-2], step=1e-5),
        ],
        targets=[
            xt.Target('c_minus', 0, tol=1e-4)])

    tw = line.twiss()
    assert tw.c_minus < 2e-4
