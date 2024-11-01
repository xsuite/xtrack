"""
Helpers for testing the MadPoint class
"""
import numpy as np
import matplotlib.pyplot as plt
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

# def summary_plot(survey, twiss, title, zero_tol = 1E-12):
#     """
#     Summary plot comparing survey, twiss and MadPoint
#     """
#     def zero_small_values(arr, tol = zero_tol):
#         return np.where(np.abs(arr) < tol, 0, arr)

#     fig, axs = plt.subplots(3, 2, figsize=(16, 8))
#     axs[0, 0].plot(
#         zero_small_values(survey.Z),
#         zero_small_values(survey.X))
#     axs[1, 0].plot(
#         zero_small_values(twiss.s),
#         zero_small_values(twiss.x))
#     axs[2, 0].plot(
#         zero_small_values(survey.s),
#         zero_small_values(survey.xx))

#     axs[0, 1].plot(
#         zero_small_values(survey.Z),
#         zero_small_values(survey.Y))
#     axs[1, 1].plot(
#         zero_small_values(twiss.s),
#         zero_small_values(twiss.y))
#     axs[2, 1].plot(
#         zero_small_values(survey.s),
#         zero_small_values(survey.yy))

#     axs[0, 0].set_xlabel('Z [m]')
#     axs[0, 0].set_ylabel('X [m]')
#     axs[1, 0].set_xlabel('s [m]')
#     axs[1, 0].set_ylabel('x [m]')
#     axs[2, 0].set_xlabel('s [m]')
#     axs[2, 0].set_ylabel('xx [m]')

#     axs[0, 1].set_xlabel('Z [m]')
#     axs[0, 1].set_ylabel('Y [m]')
#     axs[1, 1].set_xlabel('s [m]')
#     axs[1, 1].set_ylabel('y [m]')
#     axs[2, 1].set_xlabel('s [m]')
#     axs[2, 1].set_ylabel('yy [m]')

#     fig.suptitle(title)

#     plt.tight_layout()

def summary_plot(survey, twiss, title, zero_tol = 1E-12):
    """
    Summary plot comparing survey, twiss and MadPoint
    """

    fig, axs = plt.subplots(3, 1, figsize=(16, 8))
    axs[0].plot(
        zero_small_values(survey.Z, zero_tol),
        zero_small_values(survey.X, zero_tol),
        c = 'k')
    axs[0].plot(
        zero_small_values(survey.Z, zero_tol),
        zero_small_values(survey.Y, zero_tol),
        c = 'r')
    
    axs[1].plot(
        zero_small_values(twiss.s, zero_tol),
        zero_small_values(twiss.x, zero_tol),
        c = 'k')
    axs[1].plot(
        zero_small_values(twiss.s, zero_tol),
        zero_small_values(twiss.y, zero_tol),
        c = 'r')
    
    axs[2].plot(
        zero_small_values(survey.s, zero_tol),
        zero_small_values(survey.xx, zero_tol),
        c = 'k')
    axs[2].plot(
        zero_small_values(survey.s, zero_tol),
        zero_small_values(survey.yy, zero_tol),
        c = 'r')

    axs[0].set_xlabel('Z [m]')
    axs[0].set_ylabel('X [m]')
    axs[0].set_title('Survey')

    axs[1].set_xlabel('s [m]')
    axs[1].set_ylabel('x [m]')
    axs[1].set_title('Twiss')

    axs[2].set_xlabel('s [m]')
    axs[2].set_ylabel('xx [m]')
    axs[2].set_title('MadPoint')

    axs[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, ncol=2)

    fig.suptitle(title)

    plt.tight_layout()
