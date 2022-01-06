import pathlib
import json
import numpy as np
from scipy.constants import c as clight

import xobjects as xo
import xpart as xp
import xtrack as xt

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

def test_acceleration():

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        Delta_p0c = 450e9/10*23e-6 # ramp rate 450GeV/10s

        fname_line = test_data_folder.joinpath(
                'sps_w_spacecharge/line_no_spacecharge_and_particle.json')

        with open(fname_line, 'r') as fid:
            input_data = json.load(fid)
        line = xt.Line.from_dict(input_data['line'])

        energy_increase = xt.ReferenceEnergyIncrease(Delta_p0c=Delta_p0c)
        line.append_element(energy_increase, 'energy_increase')

        tracker = xt.Tracker(line=line, _context=context)

        # Assume only first cavity is active
        frequency = line.get_elements_of_type(xt.Cavity)[0][0].frequency
        voltage = line.get_elements_of_type(xt.Cavity)[0][0].voltage
        #Assuming proton and beta=1
        stable_z = np.arcsin(Delta_p0c/voltage)/frequency/2/np.pi*clight

        p_co = tracker.find_closed_orbit(particle_ref=xp.Particles.from_dict(
                                input_data['particle']))

        assert np.isclose(p_co._xobject.zeta[0], stable_z, atol=0, rtol=1e-2)

