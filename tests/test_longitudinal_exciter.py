import pathlib

import numpy as np
import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()

@for_all_test_contexts
def test_longitudinal_exciter(test_context):
    """
    This test checks that tracking with a LongitudinalExciter element produces the same effect
    as tracking with a cavity set to the same frequency, voltage, and phase, for off-momentum
    particles. This ensures that the LongitudinalExciter is correctly implemented and consistent
    with the established Cavity element for equivalent excitation.
    """

    line = xt.Line.from_json(test_data_folder /
                            'hllhc15_thick/lhc_thick_with_knobs.json')
    line.build_tracker(_context=test_context)
    
    tw = line.twiss(method="4d")
    line_exciter = line.copy()
    line_cavity = line.copy()

    num_turns = 1000
    harmonic = 35640
    voltage = 10e6
    lag = 180
    cav_dpp = 1e-3
    t0 = tw['T_rev0']
    f0 = harmonic/t0
    cav_df = - cav_dpp*tw['slip_factor']*f0
    cav_f = f0 + cav_df
    sampling_freq = 100_000*cav_f

    cavity = xt.Cavity(frequency=cav_f, voltage=voltage, lag=lag, absolute_time=True)

    tarray = np.arange(0, 1/cav_f, 1/sampling_freq)
    samples = np.sin(2*np.pi*cav_f*tarray + lag/180*np.pi)

    longi_exciter = xt.LongitudinalExciter(voltage=voltage,
                                        samples=samples,
                                        sampling_frequency=sampling_freq,
                                        frev=1/t0,
                                        duration=num_turns*t0,
                                        start_turn=0)

    line_exciter.insert("exciter", longi_exciter, at=0.)
    line_cavity.insert("cavity", cavity, at=0.)

    part_exciter = line.build_particles(
        method="4d",
        zeta=0,
        delta=np.linspace(-1e-3, 1e-3, 10),
        x_norm=0,
        px_norm=0,
        y_norm=0,
        py_norm=0,
        nemitt_x=0,
        nemitt_y=0,
    )

    part_cavity = part_exciter.copy()

    line_exciter.track(part_exciter, num_turns=100, turn_by_turn_monitor=True)
    line_cavity.track(part_cavity, num_turns=100, turn_by_turn_monitor=True)

    delta_exciter = line_exciter.record_last_track.delta
    zeta_exciter = line_exciter.record_last_track.zeta
    delta_cavity = line_cavity.record_last_track.delta
    zeta_cavity = line_cavity.record_last_track.zeta

    xo.assert_allclose(delta_exciter, delta_cavity, atol=1e-5)
    xo.assert_allclose(zeta_exciter, zeta_cavity, atol=1e-5)