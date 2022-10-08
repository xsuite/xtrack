# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

# MADX Reference:
# https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/2dcd046b1f6ca2b44ef67c8d572ff74370deee25/src/survey.f90


import logging
import numpy as np


log = logging.getLogger(__name__)
# Example of log call:
# log.warning('Need second attempt on closed orbit search')


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


class SurveyTable(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self

    def to_pandas(self, index=None):
        import pandas as pd

        df = pd.DataFrame(self)
        if index is not None:
            df.set_index(index, inplace=True)
        return df

    def mirror(self):
        new = SurveyTable()
        for kk, vv in self.items():
            new[kk] = vv

        for kk in new.keys():
            new[kk] = new[kk][::-1]

        return new

# ==================================================

# Main function
# ==================================================
def survey_from_tracker(tracker, X0=0, Y0=0, Z0=0, theta0=0, phi0=0, psi0=0):
    """Execute SURVEY command. Based on MADX equivalent.
    Attributes, must be given in this order in the dictionary:
    X0        (real)    Initial X position.
    Y0        (real)    Initial Y position.
    Z0        (real)    Initial Z position.
    theta0    (real)    Initial azimuthal angle.
    phi0      (real)    Initial elevation angle.
    psi0      (real)    Initial roll angle."""

    # Initializing dictionary
    survey_el_by_el = SurveyTable(
        {
            "name": tracker.line.element_names + ("_end_point",),
            "s": np.array(tracker.line.get_s_elements() + [tracker.line.get_length()]),
            "l": np.array(tracker.line.get_length_elements() + [0.0]),
            "X": [X0],
            "Y": [Y0],
            "Z": [Z0],
            "theta": [theta0],
            "phi": [phi0],
            "psi": [psi0],
        }
    )

    v = np.array([X0, Y0, Z0])
    w = get_w_from_angles(theta=theta0, phi=phi0, psi=psi0)
    # Advancing element by element
    for ee, length, name in zip(
        tracker.line.elements[1:],
        survey_el_by_el["l"][1:-1],
        survey_el_by_el["name"][1:-1],
    ):

        hxl, hyl = (ee.hxl, ee.hyl) if hasattr(ee, "hxl") else (0, 0)

        ##############
        # TODO Generalize for non-flat machines
        assert (
            hyl == 0
        ), f"Survey of machines with tilt not yet implemented, {name} has hyl={hyl} "

        angle = hxl
        tilt = 0
        ##############

        # Advancing
        v, w = advance_element(v, w, length=length, angle=angle, tilt=tilt)

        # Unpacking results
        theta, phi, psi = get_angles_from_w(w)
        # ----
        survey_el_by_el["X"].append(v[0])
        survey_el_by_el["Y"].append(v[1])
        survey_el_by_el["Z"].append(v[2])
        # ----
        survey_el_by_el["theta"].append(theta)
        survey_el_by_el["phi"].append(phi)
        survey_el_by_el["psi"].append(psi)

    # Repeating for endpoint
    for _key in ["X", "Y", "Z", "theta", "phi", "psi"]:
        survey_el_by_el[_key].append(survey_el_by_el[_key][-1])

    survey_el_by_el["X"] = np.array(survey_el_by_el["X"])
    survey_el_by_el["Y"] = np.array(survey_el_by_el["Y"])
    survey_el_by_el["Z"] = np.array(survey_el_by_el["Z"])
    survey_el_by_el["theta"] = np.unwrap(np.array(survey_el_by_el["theta"]))
    survey_el_by_el["phi"] = np.unwrap(np.array(survey_el_by_el["phi"]))
    survey_el_by_el["psi"] = np.unwrap(np.array(survey_el_by_el["psi"]))

    # Returns as SurveyTable object
    return survey_el_by_el
