# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xpart as xp
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

@for_all_test_contexts
def test_henonmap(test_context):
    alpha_x = 0.0
    beta_x = 100.0
    alpha_y = 0.0
    beta_y = 10.0
    K2 = 0.1
    lmbd = K2 * beta_x**(3.0/2.0) / 2.0
    K3 = -5.0 * 3.0 * K2**2 * beta_x / 2.0

    N = 10
    x_n = np.linspace(0, 0.2, N)
    x = x_n / lmbd * np.sqrt(beta_x)

    p_n = xp.Particles(x=x_n, _context=test_context)
    p = xp.Particles(x=x, _context=test_context)

    henon_n = xt.Henonmap(omega_x = 2 * np.pi * 0.334,
                          omega_y = 2 * np.pi * 0.279,
                          n_turns = 1, 
                          twiss_params = [0., 1., 0., 1.],
                          multipole_coeffs = [2.0, -30.0],
                          norm = True)
    line_n = xt.Line(elements=[henon_n], element_names=["henon_n"])
    line_n.build_tracker(_context=test_context)
    henon = xt.Henonmap(omega_x = 2 * np.pi * 0.334,
                        omega_y = 2 * np.pi * 0.279,
                        n_turns = 1, 
                        twiss_params = [alpha_x, beta_x, alpha_y, beta_y],
                        multipole_coeffs = [K2, K3],
                        norm = False)
    line = xt.Line(elements=[henon], element_names=["henon"])
    line.build_tracker(_context=test_context)
    
    nTurns = 100
    x_n_all = np.zeros(N * nTurns)
    x_all = np.zeros(N * nTurns)
    for n in range(nTurns):
        line_n.track(p_n)
        line.track(p)
        x_n_all[n*N:(n+1)*N] = test_context.nparray_from_context_array(p_n.x)
        x_all[n*N:(n+1)*N] = test_context.nparray_from_context_array(p.x)
    x_all_n = x_all * lmbd / np.sqrt(beta_x)

    assert np.all(np.isclose(x_n_all, x_all_n, atol=0, rtol=1e-10))

    x_in_test  = np.asarray([1, 0, 0, 0, 2, 2, 2, 0, 0, 0, 3, 3, 3, 0, 4])
    px_in_test = np.asarray([0, 1, 0, 0, 2, 0, 0, 2, 2, 0, 3, 3, 0, 3, 4])
    y_in_test  = np.asarray([0, 0, 1, 0, 0, 2, 0, 2, 0, 2, 3, 0, 3, 3, 4])
    py_in_test = np.asarray([0, 0, 0, 1, 0, 0, 2, 0, 2, 2, 0, 3, 3, 3, 4])
    x_out_test  = -0.5 * x_in_test + np.sqrt(3.0) / 2.0 * (px_in_test + (x_in_test**2 - y_in_test**2 / 100.0) / 20.0)
    px_out_test = -np.sqrt(3.0) / 2.0 * x_in_test - 0.5 * (px_in_test + (x_in_test**2 - y_in_test**2 / 100.0) / 20.0)
    y_out_test  = np.sqrt(2.0) / 2.0 * y_in_test + np.sqrt(2.0) / 2.0 * (py_in_test - x_in_test * y_in_test / 1000.0)
    py_out_test = -np.sqrt(2.0) / 2.0 * y_in_test + np.sqrt(2.0) / 2.0 * (py_in_test - x_in_test * y_in_test / 1000.0)

    p_test = xp.Particles(x=x_in_test, px=px_in_test, y=y_in_test, py=py_in_test, _context=test_context)

    henon_test = xt.Henonmap(omega_x = 2 * np.pi / 3.0,
                             omega_y = 2 * np.pi / 8.0,
                             n_turns = 1, 
                             twiss_params = [0.0, 1.0, 0.0, 0.01],
                             multipole_coeffs = [0.1],
                             norm = False)
    line_test = xt.Line(elements=[henon_test], element_names=["henon"])
    line_test.build_tracker(_context=test_context)

    line_test.track(p_test)

    x_out = test_context.nparray_from_context_array(p_test.x)[np.argsort(test_context.nparray_from_context_array(p_test.particle_id))]
    px_out = test_context.nparray_from_context_array(p_test.px)[np.argsort(test_context.nparray_from_context_array(p_test.particle_id))]
    y_out = test_context.nparray_from_context_array(p_test.y)[np.argsort(test_context.nparray_from_context_array(p_test.particle_id))]
    py_out = test_context.nparray_from_context_array(p_test.py)[np.argsort(test_context.nparray_from_context_array(p_test.particle_id))]

    assert np.any(np.isclose(x_out, x_out_test, atol=0, rtol=1e-10))
    assert np.any(np.isclose(px_out, px_out_test, atol=0, rtol=1e-10))
    assert np.any(np.isclose(y_out, y_out_test, atol=0, rtol=1e-10))
    assert np.any(np.isclose(py_out, py_out_test, atol=0, rtol=1e-10))