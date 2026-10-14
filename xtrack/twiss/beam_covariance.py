# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

from ..table import Table


def _build_sigma_table(Sigma, s=None, name=None):

    res_data = {}
    if s is not None:
        res_data['s'] = s.copy()
    if name is not None:
        res_data['name'] = name.copy()

    # Longitudinal plane is untested

    res_data['sigma_x'] = np.sqrt(Sigma[:, 0, 0])
    res_data['sigma_y'] = np.sqrt(Sigma[:, 2, 2])
    res_data['sigma_zeta'] = np.sqrt(Sigma[:, 4, 4])

    res_data['sigma_px'] = np.sqrt(Sigma[:, 1, 1])
    res_data['sigma_py'] = np.sqrt(Sigma[:, 3, 3])
    res_data['sigma_pzeta'] = np.sqrt(Sigma[:, 5, 5])


    res_data['Sigma'] = Sigma
    res_data['Sigma11'] = Sigma[:, 0, 0]
    res_data['Sigma12'] = Sigma[:, 0, 1]
    res_data['Sigma13'] = Sigma[:, 0, 2]
    res_data['Sigma14'] = Sigma[:, 0, 3]
    res_data['Sigma15'] = Sigma[:, 0, 4]
    res_data['Sigma16'] = Sigma[:, 0, 5]

    res_data['Sigma21'] = Sigma[:, 1, 0]
    res_data['Sigma22'] = Sigma[:, 1, 1]
    res_data['Sigma23'] = Sigma[:, 1, 2]
    res_data['Sigma24'] = Sigma[:, 1, 3]
    res_data['Sigma25'] = Sigma[:, 1, 4]
    res_data['Sigma26'] = Sigma[:, 1, 5]

    res_data['Sigma31'] = Sigma[:, 2, 0]
    res_data['Sigma32'] = Sigma[:, 2, 1]
    res_data['Sigma33'] = Sigma[:, 2, 2]
    res_data['Sigma34'] = Sigma[:, 2, 3]
    res_data['Sigma41'] = Sigma[:, 3, 0]
    res_data['Sigma42'] = Sigma[:, 3, 1]
    res_data['Sigma43'] = Sigma[:, 3, 2]
    res_data['Sigma44'] = Sigma[:, 3, 3]
    res_data['Sigma51'] = Sigma[:, 4, 0]
    res_data['Sigma52'] = Sigma[:, 4, 1]


    return Table(res_data)
