# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np


def mysign(u):
    return 2 * np.float64(u >= 0) - 1.0


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


def _propagate_Sigma_matrix(
    Sigmas_at_0, S, threshold_singular=1e-16, handle_singularities=True
):

    Sig_11_0 = Sigmas_at_0.Sig_11_0
    Sig_12_0 = Sigmas_at_0.Sig_12_0
    Sig_13_0 = Sigmas_at_0.Sig_13_0
    Sig_14_0 = Sigmas_at_0.Sig_14_0
    Sig_22_0 = Sigmas_at_0.Sig_22_0
    Sig_23_0 = Sigmas_at_0.Sig_23_0
    Sig_24_0 = Sigmas_at_0.Sig_24_0
    Sig_33_0 = Sigmas_at_0.Sig_33_0
    Sig_34_0 = Sigmas_at_0.Sig_34_0
    Sig_44_0 = Sigmas_at_0.Sig_44_0

    # ~ Sig_11 = Sig_11_0 + 2.*Sig_12_0*S+Sig_22_0*S*S
    # ~ Sig_33 = Sig_33_0 + 2.*Sig_34_0*S+Sig_44_0*S*S
    # ~ Sig_13 = Sig_13_0 + (Sig_14_0+Sig_23_0)*S+Sig_24_0*S*S

    (
        Sig_11,
        Sig_12,
        Sig_13,
        Sig_14,
        Sig_22,
        Sig_23,
        Sig_24,
        Sig_33,
        Sig_34,
        Sig_44,
    ) = propagate_full_Sigma_matrix_in_drift(
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
        S,
    )

    R = Sig_11 - Sig_33
    W = Sig_11 + Sig_33
    T = R * R + 4 * Sig_13 * Sig_13

    # evaluate derivatives
    dS_R = 2.0 * (Sig_12_0 - Sig_34_0) + 2 * S * (Sig_22_0 - Sig_44_0)
    dS_W = 2.0 * (Sig_12_0 + Sig_34_0) + 2 * S * (Sig_22_0 + Sig_44_0)
    dS_Sig_13 = Sig_14_0 + Sig_23_0 + 2 * Sig_24_0 * S
    dS_T = 2 * R * dS_R + 8.0 * Sig_13 * dS_Sig_13

    signR = mysign(R)

    if T < threshold_singular and handle_singularities:
        a = Sig_12 - Sig_34
        b = Sig_22 - Sig_44
        c = Sig_14 + Sig_23
        d = Sig_24

        sqrt_a2_c2 = np.sqrt(a * a + c * c)

        if sqrt_a2_c2 * sqrt_a2_c2 * sqrt_a2_c2 < threshold_singular:
            # equivalent to: if np.abs(c)<threshold_singular and
            #                   np.abs(a)<threshold_singular:

            if np.abs(d) > threshold_singular:
                cos2theta = np.abs(b) / np.sqrt(b * b + 4 * d * d)
            else:
                cos2theta = 1.0  # Decoupled beam

            costheta = np.sqrt(0.5 * (1.0 + cos2theta))
            sintheta = mysign(b) * mysign(d) * np.sqrt(0.5 * (1.0 - cos2theta))

            dS_costheta = 0.0
            dS_sintheta = 0.0

            Sig_11_hat = 0.5 * W
            Sig_33_hat = 0.5 * W

            dS_Sig_11_hat = 0.5 * dS_W
            dS_Sig_33_hat = 0.5 * dS_W

        else:
            sqrt_a2_c2 = np.sqrt(a * a + c * c)
            cos2theta = np.abs(2 * a) / (2 * sqrt_a2_c2)
            costheta = np.sqrt(0.5 * (1.0 + cos2theta))
            sintheta = mysign(a) * mysign(c) * np.sqrt(0.5 * (1.0 - cos2theta))

            dS_cos2theta = mysign(a) * (
                0.5 * b / sqrt_a2_c2
                - a
                * (a * b + 2 * c * d)
                / (2 * sqrt_a2_c2 * sqrt_a2_c2 * sqrt_a2_c2)
            )

            dS_costheta = 1 / (4 * costheta) * dS_cos2theta
            if np.abs(sintheta) > threshold_singular:
                # equivalent to: if np.abs(c)>threshold_singular:
                dS_sintheta = -1 / (4 * sintheta) * dS_cos2theta
            else:
                dS_sintheta = d / (2 * a)

            Sig_11_hat = 0.5 * W
            Sig_33_hat = 0.5 * W

            dS_Sig_11_hat = 0.5 * dS_W + mysign(a) * sqrt_a2_c2
            dS_Sig_33_hat = 0.5 * dS_W - mysign(a) * sqrt_a2_c2

    else:

        sqrtT = np.sqrt(T)
        cos2theta = signR * R / sqrtT
        costheta = np.sqrt(0.5 * (1.0 + cos2theta))
        sintheta = signR * mysign(Sig_13) * np.sqrt(0.5 * (1.0 - cos2theta))

        # in sixtrack this line seems to be different different
        # ~ sintheta = -mysign((Sig_11-Sig_33))*np.sqrt(0.5*(1.-cos2theta))

        Sig_11_hat = 0.5 * (W + signR * sqrtT)
        Sig_33_hat = 0.5 * (W - signR * sqrtT)

        dS_cos2theta = signR * (
            dS_R / sqrtT - R / (2 * sqrtT * sqrtT * sqrtT) * dS_T
        )
        dS_costheta = 1 / (4 * costheta) * dS_cos2theta

        if np.abs(sintheta) < threshold_singular and handle_singularities:
            # equivalent to to np.abs(Sig_13)<threshold_singular
            dS_sintheta = (Sig_14 + Sig_23) / R
        else:
            dS_sintheta = -1 / (4 * sintheta) * dS_cos2theta

        dS_Sig_11_hat = 0.5 * (dS_W + signR * 0.5 / sqrtT * dS_T)
        dS_Sig_33_hat = 0.5 * (dS_W - signR * 0.5 / sqrtT * dS_T)

    # This will not be exposed in C
    """
    extra_data = {}
    extra_data['Sig_11'] = Sig_11
    extra_data['Sig_13'] = Sig_13
    extra_data['Sig_33'] = Sig_33
    extra_data['cos2theta'] = cos2theta
    extra_data['T'] = T
    extra_data['R'] = R
    """
    extra_data = -1.0

    # ~ if dS_sintheta>1e10:
    # ~ import pdb; pdb.set_trace()

    return (
        Sig_11_hat,
        Sig_33_hat,
        costheta,
        sintheta,
        dS_Sig_11_hat,
        dS_Sig_33_hat,
        dS_costheta,
        dS_sintheta,
        extra_data,
    )


propagate_Sigma_matrix = np.vectorize(_propagate_Sigma_matrix)

# def propagate_Sigma_matrix(Sigmas_at_0,
#                                       S,
#                                       threshold_singular=1e-16,
#                                       handle_singularities=True):
#
#     Sig_11_hat, Sig_33_hat, costheta, sintheta,\
#         dS_Sig_11_hat, dS_Sig_33_hat, dS_costheta, dS_sintheta,\
#         extra_data_list = np.vectorize(_propagate_Sigma_matrix,
#                                        excluded=['Sigmas_at_0',
#                                                  'threshold_singular',
#                                                  'handle_singularities'])(
#             Sigmas_at_0, S, threshold_singular, handle_singularities)
#
#     extra_data = {}
#     for kk in extra_data_list[0].keys():
#         extra_data[kk] = []
#         for ele in extra_data_list:
#             extra_data[kk].append(ele[kk])
#
#     return Sig_11_hat, Sig_33_hat, costheta, sintheta,\
#         dS_Sig_11_hat, dS_Sig_33_hat, dS_costheta, dS_sintheta,\
#         extra_data
#


def propagate_full_Sigma_matrix_in_drift(
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
    S,
):

    # Can be found in matrix form in A. Wolsky,
    # "Beam dynamics in high energy particle accelerators"

    Sig_11 = Sig_11_0 + 2.0 * Sig_12_0 * S + Sig_22_0 * S * S
    Sig_33 = Sig_33_0 + 2.0 * Sig_34_0 * S + Sig_44_0 * S * S
    Sig_13 = Sig_13_0 + (Sig_14_0 + Sig_23_0) * S + Sig_24_0 * S * S

    Sig_12 = Sig_12_0 + Sig_22_0 * S
    Sig_14 = Sig_14_0 + Sig_24_0 * S
    Sig_22 = Sig_22_0 + 0.0 * S
    Sig_23 = Sig_23_0 + Sig_24_0 * S
    Sig_24 = Sig_24_0 + 0.0 * S
    Sig_34 = Sig_34_0 + Sig_44_0 * S
    Sig_44 = Sig_44_0 + 0.0 * S

    return (
        Sig_11,
        Sig_12,
        Sig_13,
        Sig_14,
        Sig_22,
        Sig_23,
        Sig_24,
        Sig_33,
        Sig_34,
        Sig_44,
    )
