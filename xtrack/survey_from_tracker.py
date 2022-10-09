# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

# MADX Reference:
# https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/2dcd046b1f6ca2b44ef67c8d572ff74370deee25/src/survey.f90


import numpy as np

from .general import Table
import xtrack as xt

# Required functions
# ==================================================
def get_w_from_angles(theta, phi, psi):
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
    return w


def get_angles_from_w(w):
    """Inverse function of get_w_from_angles()"""
    # w[0, 2]/w[2, 2] = (sinthe * cosphi)/(costhe * cosphi)
    # w[1, 0]/w[1, 1] = (cosphi * sinpsi)/(cosphi * cospsi)
    # w[1, 2]/w[1, 1] = (sinphi)/(cosphi * cospsi)

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

    def mirror(self):
        new = SurveyTable()
        for kk, vv in self.items():
            new[kk] = vv

        # inverting
        for kk in new.keys():
            new[kk] = new[kk][::-1]

        # Reflection about the starting point for all cartesian coordinates
        new.s = new.s[0] - new.s
        new.X = 2 * new.X[0] - new.X
        new.Y = 2 * new.Y[0] - new.Y
        new.Z = 2 * new.Z[0] - new.Z

        # Offsetting theta by 2pi because of the inversion
        new.theta = new.theta - 2 * np.pi

        return new

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
                        values_at_element_exit=False):
    """Execute SURVEY command. Based on MADX equivalent.
    Attributes, must be given in this order in the dictionary:
    X0        (float)    Initial X position.
    Y0        (float)    Initial Y position.
    Z0        (float)    Initial Z position.
    theta0    (float)    Initial azimuthal angle.
    phi0      (float)    Initial elevation angle.
    psi0      (float)    Initial roll angle."""

    assert not values_at_element_exit, "Not implemented yet"

    elements = tracker.line.elements

    # Initializing dictionary
    out = SurveyTable(
        {
            "drift_length": [],
            "tilt": [],
            "angle": [],
            "X": [],
            "Y": [],
            "Z": [],
            "theta": [],
            "phi": [],
            "psi": [],
        }
    )

    line = tracker.line

    # Extract drift lengths
    drift_length = np.array(_get_s_increments(elements))

    # Extract angle and tilt from elements
    angle = []
    tilt = []
    for nn in line.element_names:
        ee = line[nn]
        hxl, hyl = (ee.hxl, ee.hyl) if hasattr(ee, "hxl") else (0, 0)
        assert hyl == 0, ("Survey of machines with tilt not yet implemented, "
                          f"{name} has hyl={hyl} ")
        this_angle = hxl  # TODO: generalize for non-flat lines
        this_tilt = 0     # TODO: generalize for non-flat lines

        angle.append(this_angle)
        tilt.append(this_tilt)

    v = np.array([X0, Y0, Z0])
    w = get_w_from_angles(theta=theta0, phi=phi0, psi=psi0)
    # Advancing element by element
    for ll, aa, tt in zip(drift_length, angle, tilt):

        theta, phi, psi = get_angles_from_w(w)

        out.drift_length.append(ll)
        out.tilt.append(tt)
        out.angle.append(aa)
        out.X.append(v[0])
        out.Y.append(v[1])
        out.Z.append(v[2])
        out.theta.append(theta)
        out.phi.append(phi)
        out.psi.append(psi)

        # Advancing
        v, w = advance_element(v, w, length=ll, angle=aa, tilt=tt)

    # Last marker
    theta, phi, psi = get_angles_from_w(w)
    out.drift_length.append(0)
    out.tilt.append(0)
    out.angle.append(0)
    out.X.append(v[0])
    out.Y.append(v[1])
    out.Z.append(v[2])
    out.theta.append(theta)
    out.phi.append(phi)
    out.psi.append(psi)

    out["X"] = np.array(out["X"])
    out["Y"] = np.array(out["Y"])
    out["Z"] = np.array(out["Z"])
    out["theta"] = np.unwrap(np.array(out["theta"]))
    out["phi"] = np.unwrap(np.array(out["phi"]))
    out["psi"] = np.unwrap(np.array(out["psi"]))

    out["name"] = line.element_names + ("_end_point",)
    out["s"] = np.array(tracker.line.get_s_elements() + [tracker.line.get_length()])

    # Returns as SurveyTable object
    return out
