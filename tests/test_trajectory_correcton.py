import pathlib
import json

import numpy as np
from cpymad.madx import Madx

import xpart as xp
import xtrack as xt
import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()

@for_all_test_contexts
def test_orbit_correction_basics(test_context):

    line = xt.Line.from_json(test_data_folder
                             / 'hllhc15_thick/lhc_thick_with_knobs.json')
    line.build_tracker(_context=test_context)
    tt = line.get_table()

    # Define elements to be used as monitors for orbit correction
    # (for LHC all element names starting by "bpm" and not ending by "_entry" or "_exit")
    #tt_monitors = tt.rows['bpm.*'].rows['.*(?<!_entry)$'].rows['.*(?<!_exit)$']
    tt_monitors = tt.rows['bpm.*','.*(?<!_entry)$','.*(?<!_exit)$']
    line.steering_monitors_x = tt_monitors.name
    line.steering_monitors_y = tt_monitors.name

    # Define elements to be used as correctors for orbit correction
    # (for LHC all element namesstarting by "mcb.", containing "h." or "v.")
    tt_h_correctors = tt.rows['mcb.*'].rows[r'.*h\..*']
    line.steering_correctors_x = tt_h_correctors.name
    tt_v_correctors = tt.rows['mcb.*'].rows[r'.*v\..*']
    line.steering_correctors_y = tt_v_correctors.name

    # Reference twiss (no misalignments)
    tw_ref = line.twiss4d()

    # Introduce misalignments on all quadrupoles
    tt = line.get_table()
    tt_quad = tt.rows[tt.element_type == 'Quadrupole']
    rgen = np.random.RandomState(1) # fix seed for random number generator
                                    # (to have reproducible results)
    shift_x = rgen.randn(len(tt_quad)) * 0.01e-3 # 0.01 mm rms shift on all quads
    shift_y = rgen.randn(len(tt_quad)) * 0.01e-3 # 0.01 mm rms shift on all quads
    for nn_quad, sx, sy in zip(tt_quad.name, shift_x, shift_y):
        line.element_refs[nn_quad].shift_x = sx
        line.element_refs[nn_quad].shift_y = sy

    # Twiss before correction
    tw_before = line.twiss4d()

    # Correct orbit
    orbit_correction = line.correct_trajectory(twiss_table=tw_ref)

    # Twiss after correction
    tw_after = line.twiss4d()

    # Extract correction strength
    kicks_x = orbit_correction.x_correction.get_kick_values()
    kicks_y = orbit_correction.y_correction.get_kick_values()

    assert tw_before.x.std() > 1e-3
    assert tw_before.y.std() > 1e-3
    assert tw_after.y.std() < 5e-6
    assert tw_after.x.std() < 5e-6

    assert kicks_x.std() < 5e-6
    assert kicks_y.std() < 5e-6

    xo.assert_allclose(np.abs(kicks_x).max(), 17e-6, atol=2e-6)
    xo.assert_allclose(np.abs(kicks_y).max(), 5e-6, atol=1e-6)

@for_all_test_contexts
def test_orbit_correction_micado(test_context):

    line = xt.Line.from_json(test_data_folder
        / 'hllhc15_thick/lhc_thick_with_knobs.json')
    line.build_tracker(_context=test_context)
    tt = line.get_table()

    # Define elements to be used as monitors for orbit correction
    # (for LHC all element names starting by "bpm" and not ending by "_entry" or "_exit")
    tt_monitors = tt.rows['bpm.*'].rows['.*(?<!_entry)$'].rows['.*(?<!_exit)$']
    line.steering_monitors_x = tt_monitors.name
    line.steering_monitors_y = tt_monitors.name

    # Define elements to be used as correctors for orbit correction
    # (for LHC all element namesstarting by "mcb.", containing "h." or "v.")
    tt_h_correctors = tt.rows['mcb.*'].rows[r'.*h\..*']
    line.steering_correctors_x = tt_h_correctors.name
    tt_v_correctors = tt.rows['mcb.*'].rows[r'.*v\..*']
    line.steering_correctors_y = tt_v_correctors.name

    # Reference twiss (no misalignments)
    tw_ref = line.twiss4d()

    # Introduce misalignments on all quadrupoles
    tt = line.get_table()
    tt_quad = tt.rows[tt.element_type == 'Quadrupole']
    rgen = np.random.RandomState(1) # fix seed for random number generator
    shift_x = rgen.randn(len(tt_quad)) * 0.01e-3 # 0.01 mm rms shift on all quads
    shift_y = rgen.randn(len(tt_quad)) * 0.01e-3 # 0.01 mm rms shift on all quads
    for nn_quad, sx, sy in zip(tt_quad.name, shift_x, shift_y):
        line.element_refs[nn_quad].shift_x = sx
        line.element_refs[nn_quad].shift_y = sy

    # Twiss before correction
    tw_before = line.twiss4d()

    # Correct orbit using 5 correctors in each plane
    orbit_correction = line.correct_trajectory(twiss_table=tw_ref, n_micado=5,
                                               n_iter=1)

    # Twiss after correction
    tw_after = line.twiss4d()

    # Extract correction strength
    s_x_correctors = orbit_correction.x_correction.s_correctors
    s_y_correctors = orbit_correction.y_correction.s_correctors
    kicks_x = orbit_correction.x_correction.get_kick_values()
    kicks_y = orbit_correction.y_correction.get_kick_values()

    assert tw_before.x.std() > 1e-3
    assert tw_before.y.std() > 1e-3
    assert tw_after.y.std() < 2e-4
    assert tw_after.x.std() < 2e-4

    assert np.sum(np.abs(kicks_x)>0) == 5
    assert np.sum(np.abs(kicks_y)>0) == 5

@for_all_test_contexts
def test_orbit_correction_customize(test_context):

    line = xt.Line.from_json(test_data_folder
         / 'hllhc15_thick/lhc_thick_with_knobs.json')
    line.build_tracker(_context=test_context)
    tt = line.get_table()

    # Define elements to be used as monitors for orbit correction
    # (for LHC all element names starting by "bpm" and not ending by "_entry" or "_exit")
    tt_monitors = tt.rows['bpm.*'].rows['.*(?<!_entry)$'].rows['.*(?<!_exit)$']
    line.steering_monitors_x = tt_monitors.name
    line.steering_monitors_y = tt_monitors.name

    # Define elements to be used as correctors for orbit correction
    # (for LHC all element namesstarting by "mcb.", containing "h." or "v.")
    tt_h_correctors = tt.rows['mcb.*'].rows[r'.*h\..*']
    line.steering_correctors_x = tt_h_correctors.name
    tt_v_correctors = tt.rows['mcb.*'].rows[r'.*v\..*']
    line.steering_correctors_y = tt_v_correctors.name

    # Reference twiss (no misalignments)
    tw_ref = line.twiss4d()

    # Introduce misalignments on all quadrupoles
    tt = line.get_table()
    tt_quad = tt.rows[tt.element_type == 'Quadrupole']
    rgen = np.random.RandomState(1) # fix seed for random number generator
    shift_x = rgen.randn(len(tt_quad)) * 0.01e-3 # 0.01 mm rms shift on all quads
    shift_y = rgen.randn(len(tt_quad)) * 0.01e-3 # 0.01 mm rms shift on all quads
    for nn_quad, sx, sy in zip(tt_quad.name, shift_x, shift_y):
        line.element_refs[nn_quad].shift_x = sx
        line.element_refs[nn_quad].shift_y = sy

    # Create orbit correction object without running the correction
    orbit_correction = line.correct_trajectory(twiss_table=tw_ref, run=False)

    # Twiss before correction
    tw_before = line.twiss4d()

    # Correct
    orbit_correction.correct()

    # Remove applied correction
    orbit_correction.clear_correction_knobs()

    # Correct with a customized number of singular values
    orbit_correction.correct(n_singular_values=(200, 210))

    # Twiss after correction
    tw_after = line.twiss4d()

    # Extract correction strength
    s_x_correctors = orbit_correction.x_correction.s_correctors
    s_y_correctors = orbit_correction.y_correction.s_correctors
    kicks_x = orbit_correction.x_correction.get_kick_values()
    kicks_y = orbit_correction.y_correction.get_kick_values()

    assert tw_before.x.std() > 1e-3
    assert tw_before.y.std() > 1e-3
    assert tw_after.y.std() < 10e-6
    assert tw_after.x.std() < 10e-6

    assert kicks_x.std() < 5e-7
    assert kicks_y.std() < 5e-7

    xo.assert_allclose(np.abs(kicks_x).max(), 2e-6, atol=2e-6)
    xo.assert_allclose(np.abs(kicks_y).max(), 2e-6, atol=1e-6)

@for_all_test_contexts
def test_orbit_correction_thread(test_context):

    line = xt.Line.from_json(test_data_folder
        / 'hllhc15_thick/lhc_thick_with_knobs.json')
    line.build_tracker(_context=test_context)
    tt = line.get_table()

    # Define elements to be used as monitors for orbit correction
    # (for LHC all element names starting by "bpm" and not ending by "_entry" or "_exit")
    tt_monitors = tt.rows['bpm.*'].rows['.*(?<!_entry)$'].rows['.*(?<!_exit)$']
    line.steering_monitors_x = tt_monitors.name
    line.steering_monitors_y = tt_monitors.name

    # Define elements to be used as correctors for orbit correction
    # (for LHC all element namesstarting by "mcb.", containing "h." or "v.")
    tt_h_correctors = tt.rows['mcb.*'].rows[r'.*h\..*']
    line.steering_correctors_x = tt_h_correctors.name
    tt_v_correctors = tt.rows['mcb.*'].rows[r'.*v\..*']
    line.steering_correctors_y = tt_v_correctors.name

    # Reference twiss (no misalignments)
    tw_ref = line.twiss4d()

    # Introduce misalignments on all quadrupoles
    tt = line.get_table()
    tt_quad = tt.rows[tt.element_type == 'Quadrupole']
    rgen = np.random.RandomState(2) # fix seed for random number generator
    shift_x = rgen.randn(len(tt_quad)) * 1e-3 # 1. mm rms shift on all quads
    shift_y = rgen.randn(len(tt_quad)) * 1e-3 # 1. mm rms shift on all quads
    for nn_quad, sx, sy in zip(tt_quad.name, shift_x, shift_y):
        line.element_refs[nn_quad].shift_x = sx
        line.element_refs[nn_quad].shift_y = sy

    # Closed twiss fails (closed orbit is not found)
    # line.twiss4d()

    # Create orbit correction object without running the correction
    orbit_correction = line.correct_trajectory(twiss_table=tw_ref, run=False)

    # Thread
    threader = orbit_correction.thread(ds_thread=500., # correct in sections of 500 m
                                    rcond_short=1e-4, rcond_long=1e-4)

    # Closed twiss after threading (closed orbit is found)
    tw_after_thread = line.twiss4d()

    # Correct (with custom number of singular values)
    orbit_correction.correct(n_singular_values=200)

    # Twiss after closed orbit correction
    tw_after = line.twiss4d()

    # Extract correction strength
    s_x_correctors = orbit_correction.x_correction.s_correctors
    s_y_correctors = orbit_correction.y_correction.s_correctors
    kicks_x = orbit_correction.x_correction.get_kick_values()
    kicks_y = orbit_correction.y_correction.get_kick_values()

    assert tw_after_thread.x.std() < 2e-3
    assert tw_after_thread.y.std() < 2e-3

    assert tw_after.x.std() < 5e-4
    assert tw_after.y.std() < 5e-4

    assert kicks_x.std() < 5e-5
    assert kicks_y.std() < 5e-5

@for_all_test_contexts
def test_correct_trajectory_transfer_line(test_context):

    mad_ti2 = Madx()
    mad_ti2.call(str(test_data_folder / 'sps_to_lhc_ti2/ti2.seq'))
    mad_ti2.call(str(test_data_folder / 'sps_to_lhc_ti2/ti2_liu.str'))
    mad_ti2.beam()
    mad_ti2.use('ti2')

    line = xt.Line.from_madx_sequence(mad_ti2.sequence['ti2'])
    line.particle_ref = xt.Particles(p0c=450e9, mass0=xt.PROTON_MASS_EV, q0=1)
    line.build_tracker(_context=test_context)
    tt = line.get_table()

    # Define elements to be used as monitors for orbit correction
    # (in this case all element names starting by "bpm" and not ending by "_entry" or "_exit")
    tt_monitors = tt.rows['bpm.*'].rows['.*(?<!_entry)$'].rows['.*(?<!_exit)$']
    line.steering_monitors_x = tt_monitors.name
    line.steering_monitors_y = tt_monitors.name

    # Define elements to be used as correctors for orbit correction
    # (in this case all element names starting by "mci.", containing "h." or "v.")
    tt_h_correctors = tt.rows['mci.*'].rows[r'.*h\..*']
    line.steering_correctors_x = tt_h_correctors.name
    tt_v_correctors = tt.rows['mci.*'].rows[r'.*v\..*']
    line.steering_correctors_y = tt_v_correctors.name

    # Initial conditions from upstream ring
    init = xt.TwissInit(betx=27.77906807, bety=120.39920690,
                        alfx=0.63611880, alfy=-2.70621900,
                        dx=-0.59866300, dpx=0.01603536)

    # Reference twiss (no misalignments)
    tw_ref = line.twiss4d(start='ti2$start', end='ti2$end', init=init)

    # Introduce misalignments on all quadrupoles
    tt = line.get_table()
    tt_quad = tt.rows['mqi.*']
    rgen = np.random.RandomState(2) # fix seed for random number generator
    shift_x = rgen.randn(len(tt_quad)) * 0.1e-3 # 0.1 mm rms shift on all quads
    shift_y = rgen.randn(len(tt_quad)) * 0.1e-3 # 0.1 mm rms shift on all quads
    for nn_quad, sx, sy in zip(tt_quad.name, shift_x, shift_y):
        line.element_refs[nn_quad].shift_x = sx
        line.element_refs[nn_quad].shift_y = sy

    # Twiss before correction
    tw_before = line.twiss4d(start='ti2$start', end='ti2$end', init=init)

    # Correct trajectory
    correction = line.correct_trajectory(twiss_table=tw_ref, start='ti2$start', end='ti2$end')

    # Twiss after correction
    tw_after = line.twiss4d(start='ti2$start', end='ti2$end', init=init)

    # Extract correction strength
    kicks_x = correction.x_correction.get_kick_values()
    kicks_y = correction.y_correction.get_kick_values()

    assert tw_before.x.std() > 1.5e-3
    assert tw_before.y.std() > 0.5e-3

    assert tw_after.x.std() < 0.3e-3
    assert tw_after.y.std() < 0.3e-3

    assert kicks_x.std() < 2e-5
    assert kicks_y.std() < 2e-5
