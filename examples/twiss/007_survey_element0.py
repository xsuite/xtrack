# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp

###############
# Load a line #
###############

fname_line_particles = '../../test_data/hllhc15_noerrors_nobb/line_and_particle.json'
#fname_line_particles = '../../test_data/sps_w_spacecharge/line_no_spacecharge_and_particle.json' #!skip-doc

with open(fname_line_particles, 'r') as fid:
    input_data = json.load(fid)
line = xt.Line.from_dict(input_data['line'])
line.particle_ref = xp.Particles.from_dict(input_data['particle'])

tracker = line.build_tracker()

sv0 = tracker.survey(element0='ip5').to_pandas()

line_c = line.cycle('ip5')
tracker_c = line_c.build_tracker()
sv_c = tracker_c.survey().to_pandas()


for ename in ['ip8', 'ip1', 'mb.c12r8.b1..1',
              'mb.c12r1.b1..1', 'mb.c12l5.b1..1', 'drift_10', 'drift_10000']:
    svc_at_e = sv_c[sv_c.name==ename]
    sv0_at_e = sv0[sv0.name==ename]

    assert np.isclose(svc_at_e.X, sv0_at_e.X, rtol=0, atol=5e-4)
    assert np.isclose(svc_at_e.Y, sv0_at_e.Y, rtol=0, atol=5e-4)
    assert np.isclose(svc_at_e.Z, sv0_at_e.Z, rtol=0, atol=5e-4)
    assert np.isclose(np.mod(svc_at_e.theta, 2*np.pi), np.mod(sv0_at_e.theta, 2*np.pi),
                      rtol=0, atol=1e-14)
    assert np.isclose(svc_at_e.phi, sv0_at_e.phi, rtol=0, atol=1e-14)
    assert np.isclose(svc_at_e.psi, sv0_at_e.psi, rtol=0, atol=1e-14)
    assert np.isclose(svc_at_e.drift_length, sv0_at_e.drift_length, rtol=0, atol=1e-14)
    assert np.isclose(svc_at_e.angle, sv0_at_e.angle, rtol=0, atol=1e-14)
    assert np.isclose(svc_at_e.tilt, sv0_at_e.tilt, rtol=0, atol=1e-14)





