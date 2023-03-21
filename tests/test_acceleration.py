# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import pathlib
import json
import numpy as np
from scipy.constants import c as clight

import xpart as xp
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()


@for_all_test_contexts
def test_acceleration(test_context):
    Delta_p0c = 450e9/10*23e-6 # ramp rate 450GeV/10s

    fname_line = test_data_folder.joinpath(
            'sps_w_spacecharge/line_no_spacecharge_and_particle.json')

    with open(fname_line, 'r') as fid:
        input_data = json.load(fid)
    line = xt.Line.from_dict(input_data['line'])

    energy_increase = xt.ReferenceEnergyIncrease(Delta_p0c=Delta_p0c)
    line.append_element(energy_increase, 'energy_increase')

    line.build_tracker(_context=test_context)

    # Assume only first cavity is active
    frequency = line.get_elements_of_type(xt.Cavity)[0][0].frequency
    voltage = line.get_elements_of_type(xt.Cavity)[0][0].voltage
    #Assuming proton and beta=1
    stable_z = np.arcsin(Delta_p0c/voltage)/frequency/2/np.pi*clight

    p_co = line.find_closed_orbit(particle_ref=xp.Particles.from_dict(
                            input_data['particle']))

    assert np.isclose(p_co._xobject.zeta[0], stable_z, atol=0, rtol=1e-2)
