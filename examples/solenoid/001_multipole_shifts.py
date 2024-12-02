"""
Test implementation of the multipole shifts in solenoid slices
=============================================
Author(s): John P T Salvesen
Email:  john.salvesen@cern.ch
Date:   14-11-2024
"""
################################################################################
# Required Modules
################################################################################
import xtrack as xt
import numpy as np
import matplotlib.pyplot as plt

################################################################################
# User Parameters
################################################################################
N_SLICES    = int(1E3)

BETX        = 100E-3
BETY        = 1E-3
PX0         = 0

KS          = 0.00
K0          = 1E-3
K1          = 1E-3
K2          = 1E-3

X_SHIFT     = 1E-3
Y_SHIFT     = 1E-3

################################################################################
# Build Test Elements
################################################################################
drift0  = xt.Drift(length = 1)
drift1  = xt.Drift(length = 1)

bend    = xt.Bend(length = 1, k0 = K0)
quad    = xt.Quadrupole(length = 1, k1 = K1)
sext    = xt.Sextupole(length = 1, k2 = K2)

bend_sol    = xt.Solenoid(length = 1 / N_SLICES, ks = KS,
    knl = [K0 * (1/N_SLICES), 0, 0], num_multipole_kicks = 1)
quad_sol    = xt.Solenoid(length = 1 / N_SLICES, ks = KS,
    knl = [0, K1 * (1/N_SLICES), 0], num_multipole_kicks = 1)
sext_sol    = xt.Solenoid(length = 1 / N_SLICES, ks = KS,
    knl = [0, 0, K2 * (1/N_SLICES)], num_multipole_kicks = 1)

################################################################################
# Comparisons
################################################################################

for test_element, test_sol, title in zip(
    [bend, quad, sext], [bend_sol, quad_sol, sext_sol], ['Bend', 'Quadrupole', 'Sextupole']):
    ########################################
    # Build Lines
    ########################################
    line    = xt.Line(
        elements        = [drift0] + [test_element] + [drift0],
        particle_ref    = xt.Particles(p0c = 1E9, mass0 = xt.ELECTRON_MASS_EV))
    line.configure_bend_model(edge = 'suppressed')

    sol_line   = xt.Line(
        elements        = [drift1] + [test_sol] * N_SLICES + [drift1],
        particle_ref    = xt.Particles(p0c = 1E9, mass0 = xt.ELECTRON_MASS_EV))

    # Slice test line
    line.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(slicing = xt.Uniform(N_SLICES, mode='thin'), element_type = xt.Bend),
            xt.Strategy(slicing = xt.Uniform(N_SLICES, mode='thin'), element_type = xt.Quadrupole),
            xt.Strategy(slicing = xt.Uniform(N_SLICES, mode='thin'), element_type = xt.Sextupole)])

    ########################################
    # Initialise Plot
    ########################################
    fig, axs    = plt.subplots(2, 2, figsize = (10, 10))
    axs         = axs.flatten()
    fig.suptitle(title)

    ########################################
    # Test and plot with shifts
    ########################################
    for i, (shift_x, shift_y) in enumerate(
        zip([0, X_SHIFT, 0, X_SHIFT],[0, 0, Y_SHIFT, Y_SHIFT])):

        test_element.shift_x    = shift_x
        test_element.shift_y    = shift_y
        test_sol.mult_shift_x   = shift_x
        test_sol.mult_shift_y   = shift_y

        tw     = line.twiss(
            _continue_if_lost = True,
            start   = xt.START,
            end     = xt.END,
            betx    = BETX,
            bety    = BETY,
            px      = PX0)
        tw_sol  = sol_line.twiss(
            _continue_if_lost = True,
            start   = xt.START,
            end     = xt.END,
            betx    = BETX,
            bety    = BETY,
            px      = PX0)

        axs[i].plot(tw.s,       tw.x,       color = 'k', label = 'Element x')
        axs[i].plot(tw.s,       tw.y,       color = 'r', label = 'Element y')
        axs[i].plot(tw_sol.s,   tw_sol.x,   color = 'b', linestyle = ':', label = 'Sol x')
        axs[i].plot(tw_sol.s,   tw_sol.y,   color = 'g', linestyle = ':', label = 'Sol y')

        axs[i].set_title(f'Shift x = {shift_x * 1000} [mm], Shift y = {shift_y * 1000} [mm]')

        ########################################
        # Assertions
        ########################################
        assert np.isclose(tw.x[-1], tw_sol.x[-1], rtol = 1E-6)
        assert np.isclose(tw.y[-1], tw_sol.y[-1], rtol = 1E-6)

    ########################################
    # Figure adjustments
    ########################################
    fig.legend(
        labels = ['Element x', 'Element y', 'Sol x', 'Sol y'],
        loc = 'upper center',
        ncol = 4)

########################################
# Show Plots
########################################
plt.show()
