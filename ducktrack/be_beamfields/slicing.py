import numpy as np
from scipy.special import erfinv


def constant_charge_slicing_gaussian(N_part_tot, sigmaz, N_slices):

    if N_slices > 1:
        # working with intensity 1. and rescling at the end
        Qi = (np.arange(N_slices) / float(N_slices))[1:]

        z_cuts = np.sqrt(2) * sigmaz * erfinv(2 * Qi - 1.0)

        # ~ import pdb; pdb.set_trace()

        z_centroids = []
        first_centroid = (
            -sigmaz
            / np.sqrt(2 * np.pi)
            * np.exp(-z_cuts[0] ** 2 / (2 * sigmaz * sigmaz))
            * float(N_slices)
        )
        z_centroids.append(first_centroid)
        for ii in range(N_slices - 2):
            this_centroid = (
                -sigmaz
                / np.sqrt(2 * np.pi)
                * (
                    np.exp(-z_cuts[ii + 1] ** 2 / (2 * sigmaz * sigmaz))
                    - np.exp(-z_cuts[ii] ** 2 / (2 * sigmaz * sigmaz))
                )
                * float(N_slices)
            )
            # the multiplication times n slices comes from the fact that
            # we have to divide by the slice charge, i.e. 1./N
            z_centroids.append(this_centroid)

        last_centroid = (
            sigmaz
            / np.sqrt(2 * np.pi)
            * np.exp(-z_cuts[-1] ** 2 / (2 * sigmaz * sigmaz))
            * float(N_slices)
        )
        z_centroids.append(last_centroid)

        z_centroids = np.array(z_centroids)

        N_part_per_slice = z_centroids * 0.0 + N_part_tot / float(N_slices)
    elif N_slices == 1:
        z_centroids = np.array([0.0])
        z_cuts = []
        N_part_per_slice = np.array([N_part_tot])

    else:
        raise ValueError("Invalid number of slices")

    return z_centroids, z_cuts, N_part_per_slice
