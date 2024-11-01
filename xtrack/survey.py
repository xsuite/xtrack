# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

# MADX Reference:
# https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/2dcd046b1f6ca2b44ef67c8d572ff74370deee25/src/survey.f90


import numpy as np

from xdeps import Table
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

def advance_rotation(v, w, S):
    """Advancing through rotation element:
    Rotate w matrix according to transformation matrix S"""
    return v, np.dot(w, S)

def advance_drift(v, w, R):
    """Advancing through drift element, see MAD-X manual:
    v2 = w1*R + v1  | w2 = w1*S -> S is unity"""
    return np.dot(w, R) + v, w


def advance_element(v, w, length=0, angle=0, tilt=0, dx=0, dy=0,
                    transf_x_rad=0, transf_y_rad=0, transf_s_rad=0):
    """Computing the advance element-by-element.
    See MAD-X manual for generation of R and S"""
    # XYShift Handling
    if dx != 0 or dy != 0:
        assert angle == 0,  "dx and dy are only supported for angle = 0"
        assert tilt == 0,   "dx and dy are only supported for tilt = 0"
        assert length == 0, "dx and dy are only supported for length = 0"

        R = np.array([dx, dy, 0])
        # XYShift tarnsforms as a drift
        return advance_drift(v, w, R)

    # XRotation Handling
    if transf_x_rad != 0:
        assert angle == 0,  "rot_x_rad is only supported for angle = 0"
        assert tilt == 0,   "rot_x_rad is only supported for tilt = 0"
        assert length == 0, "rot_x_rad is only supported for length = 0"

        # Relevant sine/cosine
        cr = np.cos(-transf_x_rad)
        sr = np.sin(-transf_x_rad)
        # ------
        S = np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]]) # x rotation matrix
        return advance_rotation(v, w, S)

    # YRotation Handling
    if transf_y_rad != 0:
        assert angle == 0,  "rot_y_rad is only supported for angle = 0"
        assert tilt == 0,   "rot_y_rad is only supported for tilt = 0"
        assert length == 0, "rot_y_rad is only supported for length = 0"

        # Relevant sine/cosine
        cr = np.cos(transf_y_rad)
        sr = np.sin(transf_y_rad)
        # ------
        S = np.array([[cr, 0, -sr], [0, 1, 0], [sr, 0, cr]]) # y rotation matrix
        return advance_rotation(v, w, S)

    # SRotation Handling
    if transf_s_rad != 0:
        assert angle == 0,  "rot_s_rad is only supported for angle = 0"
        assert tilt == 0,   "rot_s_rad is only supported for tilt = 0"
        assert length == 0, "rot_s_rad is only supported for length = 0"

        # Relevant sine/cosine
        cr = np.cos(transf_s_rad)
        sr = np.sin(transf_s_rad)
        # ------
        S = np.array([[cr, sr, 0], [-sr, cr, 0], [0, 0, 1]]) # z rotation matrix
        return advance_rotation(v, w, S)

    # Non bending elements
    if angle == 0:
        R = np.array([0, 0, length])
        return advance_drift(v, w, R)
    
    # Horizontal bends
    elif tilt == 0:
        # Relevant sine/cosine
        ca = np.cos(angle)
        sa = np.sin(angle)
        # ------
        rho = length / angle
        R = np.array([rho * (ca - 1), 0, rho * sa])
        S = np.array([[ca, 0, -sa], [0, 1, 0], [sa, 0, ca]])
        return advance_bend(v, w, R, S)

    # Tilted bends
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

    _error_on_row_not_found = True

    def reverse(self, X0=None, Y0=None, Z0=None, theta0=None,
                phi0=None, psi0=None, element0=None):

        if element0 is None:
            element0 = len(self.name) - self.element0 - 1

        if (X0 is not None or Y0 is not None or Z0 is not None
                or theta0 is not None or phi0 is not None or psi0 is not None):
            assert (X0 is not None and Y0 is not None and Z0 is not None
                and theta0 is not None and phi0 is not None and psi0 is not None
                    ), (
            "X0, Y0, Z0, theta0, phi0, psi0 must be all None or all not None")

        if X0 is None:
            X0 = self.X[self.element0]
            Y0 = self.Y[self.element0]
            Z0 = self.Z[self.element0]
            theta0 = self.theta[self.element0]
            phi0 = self.phi[self.element0]
            psi0 = self.psi[self.element0]

        # We cut away the last marker (added by survey) and reverse the order
        out_drift_length = list(self.drift_length[:-1][::-1])
        out_angle = list(-self.angle[:-1][::-1])
        out_tilt = list(-self.tilt[:-1][::-1])
        out_dx = list(-self.dx[:-1][::-1])
        out_dy = list(-self.dy[:-1][::-1])
        out_rot_x_rad = list(-self.rot_x_rad[:-1][::-1])
        out_rot_y_rad = list(-self.rot_y_rad[:-1][::-1])
        out_rot_s_rad = list(-self.rot_s_rad[:-1][::-1])
        out_name = list(self.name[:-1][::-1])

        if type(element0) is str:
            element0 = out_name.index(element0)

        X, Y, Z, theta, phi, psi = compute_survey(
                                        X0, Y0, Z0, theta0, phi0, psi0,
                                        out_drift_length, out_angle, out_tilt,
                                        out_dx, out_dy, out_rot_x_rad,
                                        out_rot_y_rad, out_rot_s_rad, element0=element0)

        # Initializing dictionary
        out_columns = {}
        out_columns["X"] = np.array(X)
        out_columns["Y"] = np.array(Y)
        out_columns["Z"] = np.array(Z)
        out_columns["theta"] = np.unwrap(theta)
        out_columns["phi"] = np.unwrap(phi)
        out_columns["psi"] = np.unwrap(psi)

        out_columns["name"] = np.array(list(out_name) + ["_end_point"])
        out_columns["s"] = self.s[-1] - self.s[::-1]

        out_columns['drift_length'] = np.array(out_drift_length + [0.])

        out_columns['angle'] = np.array(out_angle + [0.])
        out_columns['tilt'] = np.array(out_tilt + [0.])

        out_columns["dx"] = np.array(out_dx + [0.])
        out_columns["dy"] = np.array(out_dy + [0.])

        out_columns["rot_x_rad"] = np.array(out_rot_x_rad + [0.])
        out_columns["rot_y_rad"] = np.array(out_rot_y_rad + [0.])
        out_columns["rot_s_rad"] = np.array(out_rot_s_rad + [0.])

        out_scalars = {}
        out_scalars["element0"] = element0

        out = SurveyTable(data=(out_columns | out_scalars),
                          col_names=out_columns.keys())

        return out

    def plot(self, element_width=None, legend=True, **kwargs):
        # Shallow copy of self
        out_sv_table = SurveyTable.__new__(SurveyTable)
        out_sv_table.__dict__.update(self.__dict__)
        out_sv_table._data = self._data.copy()

        # Removing the count for repeated elements
        out_sv_table.name = np.array([nn.split('::')[0] for nn in out_sv_table.name])

        if element_width is None:
            x_range = max(self.X) - min(self.X)
            y_range = max(self.Y) - min(self.Y)
            z_range = max(self.Z) - min(self.Z)
            element_width = max([x_range, y_range, z_range]) * 0.03
        import xplt
        xplt.FloorPlot(out_sv_table, self.line, element_width=element_width, **kwargs)
        if legend:
            import matplotlib.pyplot as plt
            plt.legend()


# ==================================================

# Main function
# ==================================================
def survey_from_line(line, X0=0, Y0=0, Z0=0, theta0=0, phi0=0, psi0=0,
                        element0=0,
                        values_at_element_exit=False,
                        reverse=True):
    """Execute SURVEY command. Based on MADX equivalent.
    Attributes, must be given in this order in the dictionary:
    X0        (float)    Initial X position in meters.
    Y0        (float)    Initial Y position in meters.
    Z0        (float)    Initial Z position in meters.
    theta0    (float)    Initial azimuthal angle in radians.
    phi0      (float)    Initial elevation angle in radians.
    psi0      (float)    Initial roll angle in radians."""

    if reverse:
        raise ValueError('`survey(..., reverse=True)` not supported anymore. '
                         'Use `survey(...).reverse()` instead.')

    assert not values_at_element_exit, "Not implemented yet"

    # Extract angle and tilt from elements
    tt      = line.get_table(attr = True)
    angle   = tt.angle_rad
    tilt    = tt.rot_s_rad
    drift_length = tt.length
    drift_length[~tt.isthick] = 0

    # Extract xy shifts from elements
    dx = tt.dx
    dy = tt.dy

    # Handling of XYSRotation elements
    transf_angle_rad = tt.transform_angle_rad
    transf_x_rad = transf_angle_rad * np.array(tt.element_type == 'XRotation')
    transf_y_rad = transf_angle_rad * np.array(tt.element_type == 'YRotation')
    transf_s_rad = transf_angle_rad * np.array(tt.element_type == 'SRotation')

    if type(element0) == str:
        element0 = line.element_names.index(element0)

    X, Y, Z, theta, phi, psi = compute_survey(
        X0, Y0, Z0, theta0, phi0, psi0, drift_length[:-1], angle[:-1], tilt[:-1],
        dx[:-1], dy[:-1], transf_x_rad, transf_y_rad, transf_s_rad, element0=element0)

    # Initializing dictionary
    out_columns = {}
    out_scalars = {}
    out_columns["X"] = np.array(X)
    out_columns["Y"] = np.array(Y)
    out_columns["Z"] = np.array(Z)
    out_columns["theta"] = np.unwrap(theta)
    out_columns["phi"] = np.unwrap(phi)
    out_columns["psi"] = np.unwrap(psi)

    out_columns["name"] = tt.name
    out_columns["s"] = tt.s
    out_columns['drift_length'] = drift_length
    out_columns['angle'] = angle
    out_columns['tilt'] = tilt

    out_columns['dx'] = dx
    out_columns['dy'] = dy

    out_columns['transf_x_rad'] = transf_x_rad
    out_columns['transf_y_rad'] = transf_y_rad
    out_columns['transf_s_rad'] = transf_s_rad

    out_scalars['element0'] = element0

    out = SurveyTable(data={**out_columns, **out_scalars},  # this is a merge
                      col_names=out_columns.keys())
    out._data['line'] = line

    return out


def compute_survey(X0, Y0, Z0, theta0, phi0, psi0, drift_length, angle, tilt,
                   dx, dy, transf_x_rad, transf_y_rad, transf_s_rad, element0=0, reverse_xs=False):

    if element0 != 0:
        assert not(reverse_xs), "Not implemented yet"
        drift_forward = drift_length[element0:]
        angle_forward = angle[element0:]
        tilt_forward = tilt[element0:]
        dx_forward = dx[element0:]
        dy_forward = dy[element0:]
        transf_x_rad_forward = transf_x_rad[element0:]
        transf_y_rad_forward = transf_y_rad[element0:]
        transf_s_rad_forward = transf_s_rad[element0:]
        (X_forward, Y_forward, Z_forward, theta_forward, phi_forward,
            psi_forward) = compute_survey(X0, Y0, Z0, theta0, phi0, psi0,
                                    drift_forward, angle_forward, tilt_forward,
                                    dx_forward, dy_forward, transf_x_rad_forward,
                                    transf_y_rad_forward, transf_s_rad_forward)

        drift_backward = drift_length[:element0][::-1]
        angle_backward = -np.array(angle[:element0][::-1])
        tilt_backward = -np.array(tilt[:element0][::-1])
        dx_backward = -np.array(dx[:element0][::-1])
        dy_backward = -np.array(dy[:element0][::-1])
        transf_x_rad_backward = -np.array(transf_x_rad[:element0][::-1])
        transf_y_rad_backward = -np.array(transf_y_rad[:element0][::-1])
        transf_s_rad_backward = -np.array(transf_s_rad[:element0][::-1])
        (X_backward, Y_backward, Z_backward, theta_backward, phi_backward,
            psi_backward) = compute_survey(X0, Y0, Z0, theta0, phi0, psi0,
                                    drift_backward, angle_backward, tilt_backward,
                                    dx_backward, dy_backward, transf_x_rad_backward,
                                    transf_y_rad_backward, transf_s_rad_backward,
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
    for ll, aa, tt, xx, yy, tx, ty, ts, in zip(drift_length, angle, tilt, dx, dy,
                                  transf_x_rad, transf_x_rad, transf_x_rad):

        th, ph, ps = get_angles_from_w(w, reverse_xs=reverse_xs)

        X.append(v[0])
        Y.append(v[1])
        Z.append(v[2])
        theta.append(th)
        phi.append(ph)
        psi.append(ps)

        # Advancing
        v, w = advance_element(v, w, length=ll, angle=aa, tilt=tt, dx=xx, dy=yy,
                               transf_x_rad=tx, transf_y_rad=ty, transf_s_rad=ts)

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
