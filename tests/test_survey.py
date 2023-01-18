import json
import pathlib
import numpy as np

import xtrack as xt
import xpart as xp
from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()


@for_all_test_contexts
def test_survey_element0(test_context):
    fname_line_particles = test_data_folder.joinpath(
        'hllhc15_noerrors_nobb/line_and_particle.json')

    with open(fname_line_particles, 'r') as fid:
        input_data = json.load(fid)

    line = xt.Line.from_dict(input_data['line'])
    line.particle_ref = xp.Particles.from_dict(input_data['particle'])

    tracker = line.build_tracker(_context=test_context)

    starting = {
        "theta0": -np.pi / 9,
        "psi0": np.pi / 7,
        "phi0": np.pi / 11,
        "X0": -300,
        "Y0": 150,
        "Z0": -100,
    }

    line_c = line.cycle('ip5')
    tracker_c = line_c.build_tracker()

    for reverse in [False, True]:

        sv0 = tracker.survey(element0='ip5', **starting, reverse=reverse).to_pandas()
        sv_c = tracker_c.survey(**starting, reverse=reverse).to_pandas()


        for ename in ['ip5', 'ip8', 'ip1', 'mb.c12r8.b1..1',
                    'mb.c12r1.b1..1', 'mb.c12l5.b1..1', 'drift_10', 'drift_10000']:
            svc_at_e = sv_c[sv_c.name == ename]
            sv0_at_e = sv0[sv0.name == ename]

            assert np.isclose(svc_at_e.X, sv0_at_e.X, rtol=0, atol=5e-4)
            assert np.isclose(svc_at_e.Y, sv0_at_e.Y, rtol=0, atol=5e-4)
            assert np.isclose(svc_at_e.Z, sv0_at_e.Z, rtol=0, atol=5e-4)
            assert np.isclose(np.mod(svc_at_e.theta, 2*np.pi), np.mod(sv0_at_e.theta, 2*np.pi),
                            rtol=0, atol=1e-12)
            assert np.isclose(svc_at_e.phi, sv0_at_e.phi, rtol=0, atol=5e-12)
            assert np.isclose(svc_at_e.psi, sv0_at_e.psi, rtol=0, atol=5e-12)
            assert np.isclose(svc_at_e.drift_length, sv0_at_e.drift_length, rtol=0, atol=1e-12)
            assert np.isclose(svc_at_e.angle, sv0_at_e.angle, rtol=0, atol=1e-12)
            assert np.isclose(svc_at_e.tilt, sv0_at_e.tilt, rtol=0, atol=1e-12)
