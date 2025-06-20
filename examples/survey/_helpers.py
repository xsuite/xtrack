"""
Helpers for testing the MadPoint class
"""
import numpy as np
from _madpoint import MadPoint

def zero_small_values(arr, tol):
    """
    Set values within a tolerance of 0, to 0
    """
    return np.where(np.abs(arr) < tol, 0, arr)

def madpoint_twiss_survey(line):
    """
    Produce twiss and survey for a line, including MadPoint
    """
    survey  = line.survey()
    twiss   = line.twiss4d(_continue_if_lost = True, betx = 1, bety = 1)

    madpoints = []
    xx = []
    yy = []
    zz = []
    for nn in twiss.name:
        madpoints.append(
            MadPoint(name = nn, xsuite_twiss = twiss, xsuite_survey = survey))
        xx.append(madpoints[-1].p[0])
        yy.append(madpoints[-1].p[1])
        zz.append(madpoints[-1].p[2])

    survey['xx'] = np.array(xx)
    survey['yy'] = np.array(yy)
    survey['zz'] = np.array(zz)

    return survey, twiss

def add_to_plot(axes, survey, twiss, index, tol = 1E-12, xlims = (-0.1, 2.1), ylims = (-2E-3, 2E-3)):
    """
    Add a line to the overall plot to show a specific case
    """
    axes[0, index].plot(
        zero_small_values(survey.Z, tol),
        zero_small_values(survey.X, tol),
        c = 'k')
    axes[0, index].plot(
        zero_small_values(survey.Z, tol),
        zero_small_values(survey.Y, tol),
        c = 'r')

    axes[1, index].plot(
        zero_small_values(twiss.s, tol),
        zero_small_values(twiss.x, tol),
        c = 'k')
    axes[1, index].plot(
        zero_small_values(twiss.s, tol),
        zero_small_values(twiss.y, tol),
        c = 'r')

    axes[2, index].plot(
        zero_small_values(survey.s, tol),
        zero_small_values(survey.xx, tol),
        c = 'k')
    axes[2, index].plot(
        zero_small_values(survey.s, tol),
        zero_small_values(survey.yy, tol),
        c = 'r')

    axes[0, index].set_xlim(xlims)
    axes[0, index].set_ylim(ylims)
    axes[1, index].set_xlim(xlims)
    axes[1, index].set_ylim(ylims)
    axes[2, index].set_xlim(xlims)
    axes[2, index].set_ylim(ylims)
