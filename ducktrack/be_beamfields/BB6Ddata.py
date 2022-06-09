import numpy as np

from . import boost


class ParBoost(object):
    # it is practically a struct
    def __init__(self, phi, alpha):
        self.sphi = np.sin(phi)
        self.cphi = np.cos(phi)
        self.tphi = np.tan(phi)
        self.salpha = np.sin(alpha)
        self.calpha = np.cos(alpha)

    def tobuffer(self):
        buf = [self.sphi, self.cphi, self.tphi, self.salpha, self.calpha]
        return np.array(buf, dtype=np.float64)


class Sigmas(object):
    def __init__(
        self,
        Sig_11_0,
        Sig_12_0,
        Sig_13_0,
        Sig_14_0,
        Sig_22_0,
        Sig_23_0,
        Sig_24_0,
        Sig_33_0,
        Sig_34_0,
        Sig_44_0,
    ):

        self.Sig_11_0 = Sig_11_0
        self.Sig_12_0 = Sig_12_0
        self.Sig_13_0 = Sig_13_0
        self.Sig_14_0 = Sig_14_0
        self.Sig_22_0 = Sig_22_0
        self.Sig_23_0 = Sig_23_0
        self.Sig_24_0 = Sig_24_0
        self.Sig_33_0 = Sig_33_0
        self.Sig_34_0 = Sig_34_0
        self.Sig_44_0 = Sig_44_0

    def tobuffer(self):
        buf = [
            self.Sig_11_0,
            self.Sig_12_0,
            self.Sig_13_0,
            self.Sig_14_0,
            self.Sig_22_0,
            self.Sig_23_0,
            self.Sig_24_0,
            self.Sig_33_0,
            self.Sig_34_0,
            self.Sig_44_0,
        ]
        return np.array(buf, dtype=np.float64)


def boost_sigmas(Sigma_0, cphi):
    Sigma_0_boosted = Sigmas(
        Sigma_0.Sig_11_0,
        Sigma_0.Sig_12_0 / cphi,
        Sigma_0.Sig_13_0,
        Sigma_0.Sig_14_0 / cphi,
        Sigma_0.Sig_22_0 / cphi / cphi,
        Sigma_0.Sig_23_0 / cphi,
        Sigma_0.Sig_24_0 / cphi / cphi,
        Sigma_0.Sig_33_0,
        Sigma_0.Sig_34_0 / cphi,
        Sigma_0.Sig_44_0 / cphi / cphi,
    )
    return Sigma_0_boosted


def int_to_float64arr(val):
    temp = np.zeros(1, (np.float64, {"i64": ("i8", 0)}))
    temp["i64"][0] = val
    return temp


class BB6D_Data(object):
    def __init__(
        self,
        q_part,
        parboost,
        Sigmas_0_star,
        N_slices,
        N_part_per_slice,
        x_slices_star,
        y_slices_star,
        sigma_slices_star,
        min_sigma_diff,
        threshold_singular,
        delta_x,
        delta_y,
        x_CO,
        px_CO,
        y_CO,
        py_CO,
        sigma_CO,
        delta_CO,
        Dx_sub,
        Dpx_sub,
        Dy_sub,
        Dpy_sub,
        Dsigma_sub,
        Ddelta_sub,
        enabled,
    ):

        self.q_part = q_part
        self.parboost = parboost
        self.Sigmas_0_star = Sigmas_0_star
        self.min_sigma_diff = min_sigma_diff
        self.threshold_singular = threshold_singular
        self.N_slices = N_slices
        self.N_part_per_slice = N_part_per_slice
        self.x_slices_star = x_slices_star
        self.y_slices_star = y_slices_star
        self.sigma_slices_star = sigma_slices_star

        self.delta_x = delta_x
        self.delta_y = delta_y
        self.x_CO = x_CO
        self.px_CO = px_CO
        self.y_CO = y_CO
        self.py_CO = py_CO
        self.sigma_CO = sigma_CO
        self.delta_CO = delta_CO
        self.Dx_sub = Dx_sub
        self.Dpx_sub = Dpx_sub
        self.Dy_sub = Dy_sub
        self.Dpy_sub = Dpy_sub
        self.Dsigma_sub = Dsigma_sub
        self.Ddelta_sub = Ddelta_sub

        self.enabled = enabled

    def tobuffer(self):

        buffer_list = []
        # Buffers corresponding to BB6D struct
        buffer_list.append(np.array([self.q_part], dtype=np.float64))
        buffer_list.append(self.parboost.tobuffer())
        buffer_list.append(self.Sigmas_0_star.tobuffer())
        buffer_list.append(np.array([self.min_sigma_diff], dtype=np.float64))
        buffer_list.append(
            np.array([self.threshold_singular], dtype=np.float64)
        )
        buffer_list.append(int_to_float64arr(self.N_slices))

        buffer_list.append(np.array([self.delta_x], dtype=np.float64))
        buffer_list.append(np.array([self.delta_y], dtype=np.float64))
        buffer_list.append(np.array([self.x_CO], dtype=np.float64))
        buffer_list.append(np.array([self.px_CO], dtype=np.float64))
        buffer_list.append(np.array([self.y_CO], dtype=np.float64))
        buffer_list.append(np.array([self.py_CO], dtype=np.float64))
        buffer_list.append(np.array([self.sigma_CO], dtype=np.float64))
        buffer_list.append(np.array([self.delta_CO], dtype=np.float64))
        buffer_list.append(np.array([self.Dx_sub], dtype=np.float64))
        buffer_list.append(np.array([self.Dpx_sub], dtype=np.float64))
        buffer_list.append(np.array([self.Dy_sub], dtype=np.float64))
        buffer_list.append(np.array([self.Dpy_sub], dtype=np.float64))
        buffer_list.append(np.array([self.Dsigma_sub], dtype=np.float64))
        buffer_list.append(np.array([self.Ddelta_sub], dtype=np.float64))
        buffer_list.append(
            int_to_float64arr({True: 1, False: 0}[self.enabled])
        )

        buffer_list.append(int_to_float64arr(3))  # offset to N_part_per_slice
        # offset to x_slices_star
        buffer_list.append(int_to_float64arr(2 + self.N_slices))
        # offset to y_slices_star
        buffer_list.append(int_to_float64arr(1 + 2 * self.N_slices))
        # offset to sigma_slices_star
        buffer_list.append(int_to_float64arr(0 + 3 * self.N_slices))

        # Buffers corresponding to arrays
        buffer_list.append(np.array(self.N_part_per_slice, dtype=np.float64))
        buffer_list.append(np.array(self.x_slices_star, dtype=np.float64))
        buffer_list.append(np.array(self.y_slices_star, dtype=np.float64))
        buffer_list.append(np.array(self.sigma_slices_star, dtype=np.float64))

        buf = np.concatenate(buffer_list)

        return buf


def BB6D_init(
    q_part,
    phi,
    alpha,
    delta_x,
    delta_y,
    N_part_per_slice,
    z_slices,
    Sig_11_0,
    Sig_12_0,
    Sig_13_0,
    Sig_14_0,
    Sig_22_0,
    Sig_23_0,
    Sig_24_0,
    Sig_33_0,
    Sig_34_0,
    Sig_44_0,
    x_CO,
    px_CO,
    y_CO,
    py_CO,
    sigma_CO,
    delta_CO,
    min_sigma_diff,
    threshold_singular,
    Dx_sub,
    Dpx_sub,
    Dy_sub,
    Dpy_sub,
    Dsigma_sub,
    Ddelta_sub,
    enabled,
):

    assert len(N_part_per_slice) == len(z_slices)

    # Prepare data for Lorentz transformation
    parboost = ParBoost(phi=phi, alpha=alpha)

    # Prepare data with strong beam shape
    Sigmas_0 = Sigmas(
        Sig_11_0,
        Sig_12_0,
        Sig_13_0,
        Sig_14_0,
        Sig_22_0,
        Sig_23_0,
        Sig_24_0,
        Sig_33_0,
        Sig_34_0,
        Sig_44_0,
    )

    # Boost strong beam shape
    Sigmas_0_star = boost_sigmas(Sigmas_0, parboost.cphi)

    # Sort according to z, head at the first position in the arrays
    ind_sorted = np.argsort(z_slices)[::-1]
    z_slices = np.take(z_slices, ind_sorted)
    N_part_per_slice = np.take(N_part_per_slice, ind_sorted)

    # By boosting the strong z and all zeros, I get the transverse coordinates
    # of the strong beam in the ref system of the weak
    boost_vect = np.vectorize(boost.boost, excluded="parboost")
    (
        x_slices_star,
        px_slices_star,
        y_slices_star,
        py_slices_star,
        sigma_slices_star,
        delta_slices_star,
    ) = boost_vect(
        x=0 * z_slices,
        px=0 * z_slices,
        y=0 * z_slices,
        py=0 * z_slices,
        sigma=z_slices,
        delta=0 * z_slices,
        parboost=parboost,
    )

    N_slices = len(z_slices)

    bb6d_data = BB6D_Data(
        q_part,
        parboost,
        Sigmas_0_star,
        N_slices,
        N_part_per_slice,
        x_slices_star,
        y_slices_star,
        sigma_slices_star,
        min_sigma_diff,
        threshold_singular,
        delta_x,
        delta_y,
        x_CO,
        px_CO,
        y_CO,
        py_CO,
        sigma_CO,
        delta_CO,
        Dx_sub,
        Dpx_sub,
        Dy_sub,
        Dpy_sub,
        Dsigma_sub,
        Ddelta_sub,
        enabled,
    )

    return bb6d_data
