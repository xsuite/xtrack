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

    # TODO: arctan2 returns angle between [-pi,pi].
    # Hence theta ends up not at 2pi after a full survey
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


def advance_element(
        v, w, length = 0, angle = 0, tilt = 0,
        ref_shift_x = 0, ref_shift_y = 0, ref_rot_x_rad = 0, ref_rot_y_rad = 0, ref_rot_s_rad = 0):
    """Computing the advance element-by-element.
    See MAD-X manual for generation of R and S"""
    # XYShift Handling
    if ref_shift_x != 0 or ref_shift_y != 0:
        assert angle == 0,  "ref_shift_x and ref_shift_y are only supported for angle = 0"
        assert tilt == 0,   "ref_shift_x and ref_shift_y are only supported for tilt = 0"
        assert length == 0, "ref_shift_x and ref_shift_y are only supported for length = 0"

        R = np.array([ref_shift_x, ref_shift_y, 0])
        # XYShift transforms as a drift
        return advance_drift(v, w, R)

    # XRotation Handling
    if ref_rot_x_rad != 0:
        assert angle == 0,  "rot_x_rad is only supported for angle = 0"
        assert tilt == 0,   "rot_x_rad is only supported for tilt = 0"
        assert length == 0, "rot_x_rad is only supported for length = 0"

        # Rotation sine/cosine
        cr = np.cos(-ref_rot_x_rad)
        sr = np.sin(-ref_rot_x_rad)
        # ------
        S = np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]]) # x rotation matrix
        return advance_rotation(v, w, S)

    # YRotation Handling
    if ref_rot_y_rad != 0:
        assert angle == 0,  "rot_y_rad is only supported for angle = 0"
        assert tilt == 0,   "rot_y_rad is only supported for tilt = 0"
        assert length == 0, "rot_y_rad is only supported for length = 0"

        # Rotation sine/cosine
        cr = np.cos(ref_rot_y_rad)
        sr = np.sin(ref_rot_y_rad)
        # ------
        S = np.array([[cr, 0, -sr], [0, 1, 0], [sr, 0, cr]]) # y rotation matrix
        return advance_rotation(v, w, S)

    # SRotation Handling
    if ref_rot_s_rad != 0:
        assert angle == 0,  "rot_s_rad is only supported for angle = 0"
        assert tilt == 0,   "rot_s_rad is only supported for tilt = 0"
        assert length == 0, "rot_s_rad is only supported for length = 0"

        # Rotation sine/cosine
        cr = np.cos(ref_rot_s_rad)
        sr = np.sin(ref_rot_s_rad)
        # ------
        S = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]) # z rotation matrix
        return advance_rotation(v, w, S)

    # Non bending elements
    if angle == 0:
        R = np.array([0, 0, length])
        return advance_drift(v, w, R)

    # Horizontal bending elements
    elif tilt == 0:
        # Angle sine/cosine
        ca = np.cos(angle)
        sa = np.sin(angle)
        # ------
        rho = length / angle
        R = np.array([rho * (ca - 1), 0, rho * sa])
        S = np.array([[ca, 0, -sa], [0, 1, 0], [sa, 0, ca]])
        return advance_bend(v, w, R, S)

    # Tilted bending elements
    else:
        # Angle sine/cosine
        ca = np.cos(angle)
        sa = np.sin(angle)
        # Tilt sine/cosine
        ct = np.cos(tilt)
        st = np.sin(tilt)
        # ------
        rho = length / angle
        R = np.array([rho * (ca - 1), 0, rho * sa])
        S = np.array([[ca, 0, -sa], [0, 1, 0], [sa, 0, ca]])

        # Orthogonal rotation matrix for tilt
        T       = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])
        Tinv    = np.array([[ct, st, 0], [-st, ct, 0], [0, 0, 1]])

        return advance_bend(v, w, np.dot(T, R), np.dot(T, np.dot(S, Tinv)))


class SurveyTable(Table):
    """
    Table for survey data.
    """

    _error_on_row_not_found = True

    def reverse(self):

        new_cols = {}

        element_properties = ['name', 'element_type', 'drift_length',
                                'length', 'angle', 'rot_s_rad',
                                'ref_shift_x', 'ref_shift_y',
                                'ref_rot_x_rad', 'ref_rot_y_rad', 'ref_rot_s_rad']

        for kk in element_properties:
            new_cols[kk] = self[kk].copy()
            new_cols[kk][:-1] = new_cols[kk][:-1][::-1]
            new_cols[kk][-1] = self[kk][-1]

        itake = slice(1, None, None)

        # s vector
        new_cols['s'] = self['s'].copy()
        new_cols['s'][:-1] = new_cols['s'][itake][::-1]
        new_cols['s'][-1] = self['s'][0]

        new_cols['s'] = self['s'][-1] - new_cols['s']

        new_W = self.W.copy()
        new_W[:-1, :, :] = new_W[itake, :, :][::-1, :, :]
        new_W[-1, :, :] = self.W[0, :, :]

        new_V = self.p0.copy()
        new_V[:-1, :] = new_V[itake, :][::-1, :]
        new_V[-1, :] = self.p0[0, :]

        # Reverse X and Z in the global frame
        new_V[:, 0] *= -1
        new_V[:, 2] *= -1
        new_W[:, 0, :] *= -1
        new_W[:, 2, :] *= -1

        # Reverse ix and iy of the local frame
        new_W[:, :, 0] *= -1
        new_W[:, :, 2] *= -1

        derived_quantities = _compute_survey_quantities_from_v_w(new_V, new_W)
        new_cols.update(derived_quantities)

        out = SurveyTable(
            data        = (new_cols | {'element0': self.element0}),
            col_names   = new_cols.keys())

        return out

    def plot(self, element_width = None, legend = True, **kwargs):
        """
        Plot the survey using xplt.FloorPlot
        """
        # Import the xplt module here
        # (Not at the top as not default installation with xsuite)
        import xplt

        # Shallow copy of self
        out_sv_table = SurveyTable.__new__(SurveyTable)
        out_sv_table.__dict__.update(self.__dict__)
        out_sv_table._data = self._data.copy()

        # Removing the count for repeated elements
        out_sv_table.name = np.array([nn.split('::')[0] for nn in out_sv_table.name])

        # Setting element width for plotting
        if element_width is None:
            x_range = max(self.X) - min(self.X)
            y_range = max(self.Y) - min(self.Y)
            z_range = max(self.Z) - min(self.Z)
            element_width   = max([x_range, y_range, z_range]) * 0.03

        xplt.FloorPlot(
            survey          = out_sv_table,
            line            = self.line,
            element_width   = element_width,
            **kwargs)

        if legend:
            import matplotlib.pyplot as plt
            plt.legend()

    def to_pandas(self, index=None, columns=None):
        if columns is None:
            columns = self._col_names

        data = self._data.copy()
        for cc in columns:
            if len(data[cc]) > 1:
                data[cc] = [data[cc][ii] for ii in range(len(data[cc])) if cc in self._col_names]

        import pandas as pd
        df = pd.DataFrame(data, columns=self._col_names)
        if index is not None:
            df.set_index(index, inplace=True)
        return df


# ==================================================
# Main function
# ==================================================
def survey_from_line(
        line,
        X0 = 0, Y0 = 0, Z0 = 0, theta0 = 0, phi0 = 0, psi0 = 0,
        element0 = 0, values_at_element_exit = False, reverse = True):
    """Execute SURVEY command. Based on MAref_shift_x equivalent.
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

    # Get line table to extract attributes
    tt      = line.get_table(attr = True)

    # Extract angle and tilt from elements
    angle   = tt.angle_rad
    tilt    = tt.rot_s_rad

    # Extract drift lengths
    drift_length = tt.length
    drift_length[~tt.isthick] = 0

    # Extract xy shifts from elements
    ref_shift_x = tt.ref_shift_x
    ref_shift_y = tt.ref_shift_y

    # Handling of XRotation, YRotation and SRotation elements
    ref_rot_angle_rad   = tt.ref_rot_angle_rad
    ref_rot_x_rad    = ref_rot_angle_rad * np.array(tt.element_type == 'XRotation')
    ref_rot_y_rad    = ref_rot_angle_rad * np.array(tt.element_type == 'YRotation')
    ref_rot_s_rad    = ref_rot_angle_rad * np.array(tt.element_type == 'SRotation')

    if isinstance(element0, str):
        element0 = line.element_names.index(element0)

    V, W = compute_survey(
        elements        = line.elements,
        X0              = X0,
        Y0              = Y0,
        Z0              = Z0,
        theta0          = theta0,
        phi0            = phi0,
        psi0            = psi0,
        drift_length    = drift_length[:-1],
        angle           = angle[:-1],
        tilt            = tilt[:-1],
        ref_shift_x     = ref_shift_x[:-1],
        ref_shift_y     = ref_shift_y[:-1],
        ref_rot_x_rad   = ref_rot_x_rad[:-1],
        ref_rot_y_rad   = ref_rot_y_rad[:-1],
        ref_rot_s_rad   = ref_rot_s_rad[:-1],
        element0        = element0)

    derived_quantities = _compute_survey_quantities_from_v_w(V, W)

    out_columns = derived_quantities
    out_scalars = {}

    # element properties
    out_columns["name"]             = tt.name
    out_columns["element_type"]     = tt.element_type
    out_columns['drift_length']     = drift_length
    out_columns['length']           = tt.length
    out_columns['angle']            = angle
    out_columns['rot_s_rad']        = tt.rot_s_rad
    out_columns['ref_shift_x']      = ref_shift_x
    out_columns['ref_shift_y']      = ref_shift_y
    out_columns['ref_rot_x_rad']    = ref_rot_x_rad
    out_columns['ref_rot_y_rad']    = ref_rot_y_rad
    out_columns['ref_rot_s_rad']    = ref_rot_s_rad

    out_columns["s"]                = tt.s

    out_scalars['element0']     = element0

    out = SurveyTable(
        data        = {**out_columns, **out_scalars},  # this is a merge
        col_names   = out_columns.keys())
    out._data['line'] = line

    return out


def compute_survey(
        elements,
        X0, Y0, Z0, theta0, phi0, psi0,
        drift_length, angle, tilt,
        ref_shift_x, ref_shift_y, ref_rot_x_rad, ref_rot_y_rad, ref_rot_s_rad,
        element0 = 0, backtrack=False):
    """
    Compute survey from initial position and orientation.
    """

    # If element0 is not the first element, split the survey
    if element0 != 0:

        # Forward section of survey
        elements_forward        = elements[element0:]
        drift_forward           = drift_length[element0:]
        angle_forward           = angle[element0:]
        tilt_forward            = tilt[element0:]
        ref_shift_x_forward     = ref_shift_x[element0:]
        ref_shift_y_forward     = ref_shift_y[element0:]
        ref_rot_x_rad_forward   = ref_rot_x_rad[element0:]
        ref_rot_y_rad_forward   = ref_rot_y_rad[element0:]
        ref_rot_s_rad_forward   = ref_rot_s_rad[element0:]

        # Evaluate forward survey
        (V_forward, W_forward)    = compute_survey(
            elements        = elements_forward,
            X0              = X0,
            Y0              = Y0,
            Z0              = Z0,
            theta0          = theta0,
            phi0            = phi0,
            psi0            = psi0,
            drift_length    = drift_forward,
            angle           = angle_forward,
            tilt            = tilt_forward,
            ref_shift_x     = ref_shift_x_forward,
            ref_shift_y     = ref_shift_y_forward,
            ref_rot_x_rad   = ref_rot_x_rad_forward,
            ref_rot_y_rad   = ref_rot_y_rad_forward,
            ref_rot_s_rad   = ref_rot_s_rad_forward,
            element0        = 0,
            backtrack       = backtrack)

        # Backward section of survey
        elements_backward       = elements[:element0][::-1]
        drift_backward          = np.array(drift_length[:element0][::-1])
        angle_backward          = np.array(angle[:element0][::-1])
        tilt_backward           = np.array(tilt[:element0][::-1])
        ref_shift_x_backward    = np.array(ref_shift_x[:element0][::-1])
        ref_shift_y_backward    = np.array(ref_shift_y[:element0][::-1])
        ref_rot_x_rad_backward  = np.array(ref_rot_x_rad[:element0][::-1])
        ref_rot_y_rad_backward  = np.array(ref_rot_y_rad[:element0][::-1])
        ref_rot_s_rad_backward  = np.array(ref_rot_s_rad[:element0][::-1])

        # Evaluate backward survey
        (V_backward, W_backward)   = compute_survey(
            elements        = elements_backward,
            X0              = X0,
            Y0              = Y0,
            Z0              = Z0,
            theta0          = theta0,
            phi0            = phi0,
            psi0            = psi0,
            drift_length    = drift_backward,
            angle           = angle_backward,
            tilt            = tilt_backward,
            ref_shift_x     = ref_shift_x_backward,
            ref_shift_y     = ref_shift_y_backward,
            ref_rot_x_rad   = ref_rot_x_rad_backward,
            ref_rot_y_rad   = ref_rot_y_rad_backward,
            ref_rot_s_rad   = ref_rot_s_rad_backward,
            element0        = 0,
            backtrack       = not backtrack)

        # Concatenate forward and backward
        W       = np.array(W_backward[::-1][:-1] + W_forward)
        V       = np.array(V_backward[::-1][:-1] + V_forward)
        return V, W

    # Initialise lists for storing the survey
    W       = []
    V       = []

    # Initial position and orientation
    v   = np.array([X0, Y0, Z0])
    w   = get_w_from_angles(
        theta       = theta0,
        phi         = phi0,
        psi         = psi0)

    # Advancing element by element
    for ee, ll, aa, tt, xx, yy, rx, ry, rs, in zip(
        elements, drift_length, angle, tilt,
        ref_shift_x, ref_shift_y,
        ref_rot_x_rad, ref_rot_y_rad, ref_rot_s_rad):

        # Store position and orientation at element entrance
        W.append(w)
        V.append(v)

        if hasattr(ee, '_propagate_survey'):
            v, w = ee._propagate_survey(v=v, w=w, backtrack=backtrack)
        else:

            if backtrack:
                ll = -ll
                aa = -aa
                xx = -xx
                yy = -yy
                rx = -rx
                ry = -ry
                rs = -rs

            # Advancing
            v, w = advance_element(
                v               = v,
                w               = w,
                length          = ll,
                angle           = aa,
                tilt            = tt,
                ref_shift_x     = xx,
                ref_shift_y     = yy,
                ref_rot_x_rad   = rx,
                ref_rot_y_rad   = ry,
                ref_rot_s_rad   = rs)

    # Last marker
    W.append(w)
    V.append(v)

    # Return data for SurveyTable object
    return V, W


def _compute_survey_quantities_from_v_w(V, W):

    W = np.array(W)
    V = np.array(V)

    theta = np.arctan2(W[:, 0, 2], W[:, 2, 2])
    psi = np.arctan2(W[:, 1, 0], W[:, 1, 1])
    phi = np.arctan2(W[:, 1, 2], W[:, 1, 1] / np.cos(psi))

    ex = W[:, :, 0]
    ey = W[:, :, 1]
    ez = W[:, :, 2]
    p0 = V.copy()
    X = V[:, 0]
    Y = V[:, 1]
    Z = V[:, 2]

    return {
        'X': X,
        'Y': Y,
        'Z': Z,
        'theta': np.unwrap(theta),
        'phi': np.unwrap(phi),
        'psi': np.unwrap(psi),
        'ex': ex,
        'ey': ey,
        'ez': ez,
        'p0': p0,
        'W': W
    }

