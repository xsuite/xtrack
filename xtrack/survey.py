# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

# MADX Reference:
# https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/2dcd046b1f6ca2b44ef67c8d572ff74370deee25/src/survey.f90


import numpy as np

from .general import Table
from .twiss import TwissTable
import xtrack as xt

# Required functions
# ==================================================
def get_w_from_angles(theta, phi, psi, reverse_xs=False):
    """W matrix, see MAD-X manual"""
    costhe = np.cos(theta)
    cosphi = np.cos(phi)
    cospsi = np.cos(psi)
    sinthe = np.sin(theta)
    sinphi = np.sin(phi)
    sinpsi = np.sin(psi)
    w = np.zeros([3, 3])
    w[0, 0] = +costhe * cospsi - sinthe * sinphi * sinpsi
    w[0, 1] = -costhe * sinpsi - sinthe * sinphi * cospsi
    w[0, 2] = sinthe * cosphi
    w[1, 0] = cosphi * sinpsi
    w[1, 1] = cosphi * cospsi
    w[1, 2] = sinphi
    w[2, 0] = -sinthe * cospsi - costhe * sinphi * sinpsi
    w[2, 1] = +sinthe * sinpsi - costhe * sinphi * cospsi
    w[2, 2] = costhe * cosphi

    if reverse_xs:
        w[:, 0] *= -1
        w[:, 2] *= -1

    return w


def get_angles_from_w(w, reverse_xs=False):
    """Inverse function of get_w_from_angles()"""
    # w[0, 2]/w[2, 2] = (sinthe * cosphi)/(costhe * cosphi)
    # w[1, 0]/w[1, 1] = (cosphi * sinpsi)/(cosphi * cospsi)
    # w[1, 2]/w[1, 1] = (sinphi)/(cosphi * cospsi)

    if reverse_xs:
        w = w.copy()
        w[:, 0] *= -1
        w[:, 2] *= -1

    theta = np.arctan2(w[0, 2], w[2, 2])
    psi = np.arctan2(w[1, 0], w[1, 1])
    phi = np.arctan2(w[1, 2], w[1, 1] / np.cos(psi))

    # TODO: arctan2 returns angle between [-pi,pi]. Hence theta ends up not at 2pi after a full survey
    return theta, phi, psi


def advance_bend(v, w, R, S):
    """Advancing through bending element, see MAD-X manual:
    v2 = w1*R + v1  | w2 = w1*S"""
    return np.dot(w, R) + v, np.dot(w, S)


def advance_drift(v, w, R):
    """Advancing through drift element, see MAD-X manual:
    v2 = w1*R + v1  | w2 = w1*S -> S is unity"""
    return np.dot(w, R) + v, w


def advance_element(v, w, length=0, angle=0, tilt=0):
    """Computing the advance element-by-element. See MAD-X manual for generation of R and S"""
    if angle == 0:
        R = np.array([0, 0, length])
        return advance_drift(v, w, R)
    elif tilt == 0:
        # Relevant sine/cosine
        ca = np.cos(angle)
        sa = np.sin(angle)
        # ------
        rho = length / angle
        R = np.array([rho * (ca - 1), 0, rho * sa])
        S = np.array([[ca, 0, -sa], [0, 1, 0], [sa, 0, ca]])
        return advance_bend(v, w, R, S)

    else:
        # Relevant sine/cosine
        ca = np.cos(angle)
        sa = np.sin(angle)
        ct = np.cos(tilt)
        st = np.sin(tilt)
        # ------
        rho = length / angle
        R = np.array([rho * (ca - 1), 0, rho * sa])
        S = np.array([[ca, 0, -sa], [0, 1, 0], [sa, 0, ca]])

        # Orthogonal rotation matrix for tilt
        T = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])
        Tinv = np.array([[ct, st, 0], [-st, ct, 0], [0, 0, 1]])

        return advance_bend(v, w, np.dot(T, R), np.dot(T, np.dot(S, Tinv)))


class SurveyTable(Table):

    def reverse(self, X0=0, Y0=0, Z0=0, theta0=0, phi0=0, psi0=0, element0=None):

        if element0 is None:
            element0 = len(self.name) - self.element0 - 1

        # We cut away the last marker (added by survey) and reverse the order
        out_drift_length = list(self.drift_length[:-1][::-1])
        out_angle = list(-self.angle[:-1][::-1])
        out_tilt = list(-self.tilt[:-1][::-1])
        out_name = list(self.name[:-1][::-1])

        if type(element0) is str:
            element0 = out_name.index(element0)

        X, Y, Z, theta, phi, psi = compute_survey(
                                        X0, Y0, Z0, theta0, phi0, psi0,
                                        out_drift_length, out_angle, out_tilt,
                                        element0=element0)

        # Initializing dictionary
        out = SurveyTable()
        out["X"] = np.array(X)
        out["Y"] = np.array(Y)
        out["Z"] = np.array(Z)
        out["theta"] = np.unwrap(theta)
        out["phi"] = np.unwrap(phi)
        out["psi"] = np.unwrap(psi)

        out["name"] = list(out_name) + ["_end_point"]
        out["s"] = self.s[-1] - self.s[::-1]

        out['drift_length'] = np.array(out_drift_length + [0.])
        out['angle'] = np.array(out_angle + [0.])
        out['tilt'] = np.array(out_tilt + [0.])

        return out

SurveyTable.__getitem__ = TwissTable.__getitem__

def _get_s_increments(elements):
    lengths = []
    for ee in elements:
        if xt.line._is_thick(ee):
            lengths.append(ee.length)
        else:
            lengths.append(0.0)
    return lengths

# ==================================================

# Main function
# ==================================================
def survey_from_tracker(tracker, X0=0, Y0=0, Z0=0, theta0=0, phi0=0, psi0=0,
                        element0=0,
                        values_at_element_exit=False,
                        reverse=True):
    """Execute SURVEY command. Based on MADX equivalent.
    Attributes, must be given in this order in the dictionary:
    X0        (float)    Initial X position.
    Y0        (float)    Initial Y position.
    Z0        (float)    Initial Z position.
    theta0    (float)    Initial azimuthal angle.
    phi0      (float)    Initial elevation angle.
    psi0      (float)    Initial roll angle."""

    assert not values_at_element_exit, "Not implemented yet"

    line = tracker.line

    # Extract drift lengths
    drift_length = _get_s_increments(line.elements)

    # Extract angle and tilt from elements
    angle = []
    tilt = []
    for nn in line.element_names:
        ee = line[nn]
        hxl, hyl = (ee.hxl, ee.hyl) if hasattr(ee, "hxl") else (0, 0)
        assert hyl == 0, ("Survey of machines with tilt not yet implemented, "
                          f"{nn} has hyl={hyl} ")
        this_angle = hxl  # TODO: generalize for non-flat lines
        this_tilt = 0     # TODO: generalize for non-flat lines

        angle.append(this_angle)
        tilt.append(this_tilt)

    if type(element0) == str:
        element0 = line.element_names.index(element0)

    X, Y, Z, theta, phi, psi = compute_survey(
        X0, Y0, Z0, theta0, phi0, psi0, drift_length, angle, tilt,
        element0=element0)

    # Initializing dictionary
    out = SurveyTable()
    out["X"] = np.array(X)
    out["Y"] = np.array(Y)
    out["Z"] = np.array(Z)
    out["theta"] = np.unwrap(theta)
    out["phi"] = np.unwrap(phi)
    out["psi"] = np.unwrap(psi)

    out["name"] = list(line.element_names) + ["_end_point"]
    out["s"] = np.array(line.get_s_elements() + [line.get_length()])

    out['drift_length'] = np.array(drift_length + [0.])
    out['angle'] = np.array(angle + [0.])
    out['tilt'] = np.array(tilt + [0.])

    out['element0'] = element0

    if reverse:
        out = out.reverse(X0=X0, Y0=Y0, Z0=Z0,
                          theta0=theta0, phi0=phi0, psi0=psi0)

    return out


def compute_survey(X0, Y0, Z0, theta0, phi0, psi0, drift_length, angle, tilt,
                   element0=0, reverse_xs=False):

    if element0 != 0:
        assert not(reverse_xs), "Not implemented yet"
        drift_forward = drift_length[element0:]
        angle_forward = angle[element0:]
        tilt_forward = tilt[element0:]
        (X_forward, Y_forward, Z_forward, theta_forward, phi_forward,
            psi_forward) = compute_survey(X0, Y0, Z0, theta0, phi0, psi0,
                                    drift_forward, angle_forward, tilt_forward)

        drift_backward = drift_length[:element0][::-1]
        angle_backward = -np.array(angle[:element0][::-1])
        tilt_backward = -np.array(tilt[:element0][::-1])
        (X_backward, Y_backward, Z_backward, theta_backward, phi_backward,
            psi_backward) = compute_survey(X0, Y0, Z0, theta0, phi0, psi0,
                                    drift_backward, angle_backward, tilt_backward,
                                    reverse_xs=True)

        X = np.array(X_backward[::-1][:-1] + X_forward)
        Y = np.array(Y_backward[::-1][:-1] + Y_forward)
        Z = np.array(Z_backward[::-1][:-1] + Z_forward)
        theta = np.array(theta_backward[::-1][:-1] + theta_forward)
        phi = np.array(phi_backward[::-1][:-1] + phi_forward)
        psi = np.array(psi_backward[::-1][:-1]+ psi_forward)
        return X, Y, Z, theta, phi, psi

    X = []
    Y = []
    Z = []
    theta = []
    phi = []
    psi = []
    v = np.array([X0, Y0, Z0])
    w = get_w_from_angles(theta=theta0, phi=phi0, psi=psi0,
                          reverse_xs=reverse_xs)
    # Advancing element by element
    for ll, aa, tt in zip(drift_length, angle, tilt):

        th, ph, ps = get_angles_from_w(w, reverse_xs=reverse_xs)

        X.append(v[0])
        Y.append(v[1])
        Z.append(v[2])
        theta.append(th)
        phi.append(ph)
        psi.append(ps)

        # Advancing
        v, w = advance_element(v, w, length=ll, angle=aa, tilt=tt)

    # Last marker
    th, ph, ps = get_angles_from_w(w, reverse_xs=reverse_xs)
    X.append(v[0])
    Y.append(v[1])
    Z.append(v[2])
    theta.append(th)
    phi.append(ph)
    psi.append(ps)

    # Returns as SurveyTable object
    return X, Y, Z, theta, phi, psi
