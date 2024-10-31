"""
Helpers for testing the MadPoint class
"""
import numpy as np
import matplotlib.pyplot as plt
from _madpoint import MadPoint

def survey_test(line):
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

def summary_plot(survey, twiss, title, zero_tol = 1E-12):
    """
    Summary plot comparing survey, twiss and MadPoint
    """
    def zero_small_values(arr, tol = zero_tol):
        return np.where(np.abs(arr) < tol, 0, arr)

    fix, axs = plt.subplots(3, 2, figsize=(16, 8))
    axs[0, 0].plot(
        zero_small_values(survey.Z),
        zero_small_values(survey.X))
    axs[1, 0].plot(
        zero_small_values(twiss.s),
        zero_small_values(twiss.x))
    axs[2, 0].plot(
        zero_small_values(survey.s),
        zero_small_values(survey.xx))

    axs[0, 1].plot(
        zero_small_values(survey.Z),
        zero_small_values(survey.Y))
    axs[1, 1].plot(
        zero_small_values(twiss.s),
        zero_small_values(twiss.y))
    axs[2, 1].plot(
        zero_small_values(survey.s),
        zero_small_values(survey.yy))

    axs[0, 0].set_xlabel('Z [m]')
    axs[0, 0].set_ylabel('X [m]')
    axs[1, 0].set_xlabel('s [m]')
    axs[1, 0].set_ylabel('x [m]')
    axs[2, 0].set_xlabel('s [m]')
    axs[2, 0].set_ylabel('xx [m]')

    axs[0, 1].set_xlabel('Z [m]')
    axs[0, 1].set_ylabel('Y [m]')
    axs[1, 1].set_xlabel('s [m]')
    axs[1, 1].set_ylabel('y [m]')
    axs[2, 1].set_xlabel('s [m]')
    axs[2, 1].set_ylabel('yy [m]')

    fix.suptitle(title)

    plt.tight_layout()
