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
    alpha_x = 1.0
    beta_x = 100.0
    alpha_y = 2.0
    beta_y = 10.0
    K2 = 0.1
    lmbd = K2 * beta_x**(3.0/2.0) / 2.0
    K3 = -5.0 * 3.0 * K2**2 * beta_x / 2.0

    N = 10
    x_n = np.linspace(0, 0.2, N)
    px_n = np.linspace(0, 0.2, N)
    x = x_n / lmbd * np.sqrt(beta_x)
    px = - alpha_x * x_n / np.sqrt(beta_x) / lmbd + px_n / np.sqrt(beta_x) / lmbd

    p_n = xp.Particles(x=x_n, px=px_n, _context=test_context)
    p = xp.Particles(x=x, px=px, _context=test_context)

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
    px_n_all = np.zeros(N * nTurns)
    px_all = np.zeros(N * nTurns)
    for n in range(nTurns):
        line_n.track(p_n)
        line.track(p)
        x_n_all[n*N:(n+1)*N] = test_context.nparray_from_context_array(p_n.x)
        x_all[n*N:(n+1)*N] = test_context.nparray_from_context_array(p.x)
        px_n_all[n*N:(n+1)*N] = test_context.nparray_from_context_array(p_n.px)
        px_all[n*N:(n+1)*N] = test_context.nparray_from_context_array(p.px)
    x_all_n = x_all * lmbd / np.sqrt(beta_x)
    px_all_n = alpha_x * x_all / np.sqrt(beta_x) * lmbd + px_all * np.sqrt(beta_x) * lmbd

    assert np.all(np.isclose(x_n_all, x_all_n, atol=1e-15, rtol=1e-10))
    assert np.all(np.isclose(px_n_all, px_all_n, atol=1e-15, rtol=1e-10))

    x_in_test  = np.asarray([1, 0, 0, 0, 2, 2, 2, 0, 0, 0, 3, 3, 3, 0, 4]) * 0.01
    px_in_test = np.asarray([0, 1, 0, 0, 2, 0, 0, 2, 2, 0, 3, 3, 0, 3, 4]) * 0.01
    y_in_test  = np.asarray([0, 0, 1, 0, 0, 2, 0, 2, 0, 2, 3, 0, 3, 3, 4]) * 0.01
    py_in_test = np.asarray([0, 0, 0, 1, 0, 0, 2, 0, 2, 2, 0, 3, 3, 3, 4]) * 0.01
    
    omega_x = 2 * np.pi / 3.0
    omega_y = 2 * np.pi / 8.0
    
    sin_omega_x = np.sin(omega_x)
    cos_omega_x = np.cos(omega_x)
    sin_omega_y = np.sin(omega_y)
    cos_omega_y = np.cos(omega_y)

    x_out_test = cos_omega_x * x_in_test + sin_omega_x * (px_in_test + (x_in_test**2 - y_in_test**2))
    px_out_test = -sin_omega_x * x_in_test + cos_omega_x * (px_in_test + (x_in_test**2 - y_in_test**2))
    y_out_test = cos_omega_y * y_in_test + sin_omega_y * (py_in_test - 2 * x_in_test * y_in_test)
    py_out_test = -sin_omega_y * y_in_test + cos_omega_y * (py_in_test - 2 * x_in_test * y_in_test)

    p_test = xp.Particles(x=x_in_test, px=px_in_test, y=y_in_test, py=py_in_test, _context=test_context)

    henon_test = xt.Henonmap(omega_x = omega_x,
                             omega_y = omega_y,
                             n_turns = 1, 
                             twiss_params = [0.0, 1.0, 0.0, 1.0],
                             multipole_coeffs = [2.0],
                             norm = True)
    line_test = xt.Line(elements=[henon_test], element_names=["henon"])
    line_test.build_tracker(_context=test_context)

    line_test.track(p_test)

    x_out = test_context.nparray_from_context_array(p_test.x)[np.argsort(test_context.nparray_from_context_array(p_test.particle_id))]
    px_out = test_context.nparray_from_context_array(p_test.px)[np.argsort(test_context.nparray_from_context_array(p_test.particle_id))]
    y_out = test_context.nparray_from_context_array(p_test.y)[np.argsort(test_context.nparray_from_context_array(p_test.particle_id))]
    py_out = test_context.nparray_from_context_array(p_test.py)[np.argsort(test_context.nparray_from_context_array(p_test.particle_id))]

    assert np.all(np.isclose(x_out, x_out_test, atol=1e-15, rtol=1e-10))
    assert np.all(np.isclose(px_out, px_out_test, atol=1e-15, rtol=1e-10))
    assert np.all(np.isclose(y_out, y_out_test, atol=1e-15, rtol=1e-10))
    assert np.all(np.isclose(py_out, py_out_test, atol=1e-15, rtol=1e-10))

    p_inv_test = xp.Particles(x=x_out, px=px_out, y=y_out, py=py_out, _context=test_context)

    line_test.track(p_inv_test, backtrack=True)

    x_in_test_inv = test_context.nparray_from_context_array(p_inv_test.x)[np.argsort(test_context.nparray_from_context_array(p_inv_test.particle_id))]
    px_in_test_inv = test_context.nparray_from_context_array(p_inv_test.px)[np.argsort(test_context.nparray_from_context_array(p_inv_test.particle_id))]
    y_in_test_inv = test_context.nparray_from_context_array(p_inv_test.y)[np.argsort(test_context.nparray_from_context_array(p_inv_test.particle_id))]
    py_in_test_inv = test_context.nparray_from_context_array(p_inv_test.py)[np.argsort(test_context.nparray_from_context_array(p_inv_test.particle_id))]

    assert np.all(np.isclose(x_in_test_inv, x_in_test, atol=1e-15, rtol=1e-10))
    assert np.all(np.isclose(px_in_test_inv, px_in_test, atol=1e-15, rtol=1e-10))
    assert np.all(np.isclose(y_in_test_inv, y_in_test, atol=1e-15, rtol=1e-10))
    assert np.all(np.isclose(py_in_test_inv, py_in_test, atol=1e-15, rtol=1e-10))