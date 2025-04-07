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
                                    rcond_short=1e-3, rcond_long=1e-3)

    kicks_x_thread = orbit_correction.x_correction.get_kick_values()
    kicks_x_thread = orbit_correction.x_correction.get_kick_values()

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

def test_orbit_correction_and_threading_shift_monitors():

    env = xt.Environment()
    env.particle_ref = xt.Particles(p0c=450e9, mass0=xt.PROTON_MASS_EV, q0=1)
    env.vars.default_to_zero = True

    env.new('mq', 'Quadrupole', length=0.8)
    env.new('bpm', 'Marker')
    env.new('corrector', 'Multipole', knl=[0])

    line = env.new_line(components=[

        env.new('line.start', 'Marker'),
        env.new('line.end', 'Marker', at=12.),

        env.new('mq1', 'mq', k1=0.2, at=3.),
        env.new('mq2', 'mq', k1=-0.2, at=5.),
        env.new('mq3', 'mq', k1=0.2, at=7.),
        env.new('mq4', 'mq', k1=-0.2, at=9.),

        env.new('bpm.s.1', 'bpm', at=0.1),
        env.new('bpm.s.2', 'bpm', at=0.5),
        env.new('bpm.q.1', 'bpm', at='mq1@start'),
        env.new('bpm.q.2', 'bpm', at='mq2@start'),
        env.new('bpm.q.3', 'bpm', at='mq3@start'),
        env.new('bpm.q.4', 'bpm', at='mq4@start'),
        env.new('bpm.e.1', 'bpm', at=11.5),
        env.new('bpm.e.2', 'bpm', at=11.9),

        env.new('corr1', 'corrector', at=1., knl=['k0l_corr1'], ksl=['k0sl_corr1']),
        env.new('corr2', 'corrector', at=2., knl=['k0l_corr2'], ksl=['k0sl_corr2']),
        env.new('corr3', 'corrector', at=10., knl=['k0l_corr3'], ksl=['k0sl_corr3']),
        env.new('corr4', 'corrector', at=11., knl=['k0l_corr4'], ksl=['k0sl_corr4']),
    ])

    # Define monitors and correctors for orbit steering
    line.steering_monitors_x = ['bpm.s.1', 'bpm.s.2',
                                'bpm.q.1', 'bpm.q.2', 'bpm.q.3', 'bpm.q.4',
                                'bpm.e.1', 'bpm.e.2']
    line.steering_correctors_x = ['corr1', 'corr2', 'corr3', 'corr4']
    line.steering_monitors_y = line.steering_monitors_x
    line.steering_correctors_y = line.steering_correctors_x

    # Twiss without misalignments
    tw0 = line.twiss4d()

    # Misalign all quadrupoles
    env.set(['mq1', 'mq2', 'mq3', 'mq4'], shift_x=1e-3, shift_y=2e-3, rot_s_rad=0.1)

    # Define BPM alignment
    bpm_alignment ={
        'bpm.q.1': {'shift_x': 1e-3, 'shift_y': 2e-3, 'rot_s_rad': 0.1},
        'bpm.q.2': {'shift_x': 1e-3, 'shift_y': 2e-3, 'rot_s_rad': 0.1},
        'bpm.q.3': {'shift_x': 1e-3, 'shift_y': 2e-3, 'rot_s_rad': 0.1},
        'bpm.q.4': {'shift_x': 1e-3, 'shift_y': 2e-3, 'rot_s_rad': 0.1},
    }

    # Correct orbit taking into account BPM alignment (centers the beam in all quadrupoles)
    correction = line.correct_trajectory(twiss_table=tw0,
                                        monitor_alignment=bpm_alignment, # <--BPM alignment
                                        run=False)
    correction.correct()

    #!end-doc-part

    tw_corr = line.twiss4d()

    xo.assert_allclose(correction.x_correction.shift_x_monitors,
                    [0., 0., 0.001, 0.001, 0.001, 0.001, 0., 0.], rtol=0, atol=1e-14)
    xo.assert_allclose(correction.x_correction.shift_y_monitors,
                    [0., 0., 0.002, 0.002, 0.002, 0.002, 0., 0.], rtol=0, atol=1e-14)
    xo.assert_allclose(correction.y_correction.shift_x_monitors,
                    [0., 0., 0.001, 0.001, 0.001, 0.001, 0., 0.], rtol=0, atol=1e-14)
    xo.assert_allclose(correction.y_correction.shift_y_monitors,
                    [0., 0., 0.002, 0.002, 0.002, 0.002, 0., 0.], rtol=0, atol=1e-14)

    # Data from previous step can be found in:
    correction.correct() # Some more steps to log the position
    xo.assert_allclose(correction.x_correction._position_before,0, rtol=0, atol=1e-10)
    xo.assert_allclose(correction.y_correction._position_before,0, rtol=0, atol=1e-10)

    correction.clear_correction_knobs()

    for nn in ['corr1', 'corr2', 'corr3', 'corr4']:
        assert line[nn].knl[0] == 0

    correction.thread(ds_thread = 10)
    tw_thread = line.twiss4d()

    xo.assert_allclose(correction.x_correction.shift_x_monitors,
                    [0., 0., 0.001, 0.001, 0.001, 0.001, 0., 0.], rtol=0, atol=1e-14)
    xo.assert_allclose(correction.x_correction.shift_y_monitors,
                    [0., 0., 0.002, 0.002, 0.002, 0.002, 0., 0.], rtol=0, atol=1e-14)
    xo.assert_allclose(correction.y_correction.shift_x_monitors,
                    [0., 0., 0.001, 0.001, 0.001, 0.001, 0., 0.], rtol=0, atol=1e-14)
    xo.assert_allclose(correction.y_correction.shift_y_monitors,
                    [0., 0., 0.002, 0.002, 0.002, 0.002, 0., 0.], rtol=0, atol=1e-14)

    # Data from previous step can be found in:
    correction.correct() # Some more steps to log the position
    xo.assert_allclose(correction.x_correction._position_before,0, rtol=0, atol=1e-9)
    xo.assert_allclose(correction.y_correction._position_before,0, rtol=0, atol=1e-9)

    xo.assert_allclose(tw_corr.rows['mq1':'corr3'].x, 1e-3, rtol=0, atol=1e-9)
    xo.assert_allclose(tw_corr.rows['mq1':'corr3'].y, 2e-3, rtol=0, atol=1e-9)
    xo.assert_allclose(tw_thread.rows['mq1':'corr3'].x, 1e-3, rtol=0, atol=1e-4)
    xo.assert_allclose(tw_thread.rows['mq1':'corr3'].y, 2e-3, rtol=0, atol=1e-4)

    xo.assert_allclose(tw_corr.x[0], 0, rtol=0, atol=1e-9)
    xo.assert_allclose(tw_corr.y[0], 0, rtol=0, atol=1e-9)
    xo.assert_allclose(tw_thread.x[0], 0, rtol=0, atol=1e-4)
    xo.assert_allclose(tw_thread.y[0], 0, rtol=0, atol=1e-4)

def test_orbit_correction_tilt_monitors():

    env = xt.Environment()
    env.particle_ref = xt.Particles(p0c=450e9, mass0=xt.PROTON_MASS_EV, q0=1)
    env.vars.default_to_zero = True

    env.new('mq', 'Quadrupole', length=0.8)
    env.new('bpm', 'Marker')
    env.new('corrector', 'Multipole', knl=[0])

    line = env.new_line(components=[

        env.new('line.start', 'Marker'),
        env.new('line.end', 'Marker', at=12.),

        env.new('mq1', 'mq', k1='kq1', at=3.),
        env.new('mq2', 'mq', k1='kq2', at=5.),
        env.new('mq3', 'mq', k1='kq3', at=7.),
        env.new('mq4', 'mq', k1='kq4', at=9.),

        env.new('bpm1', 'bpm', at='mq1@start'),
        env.new('bpm2', 'bpm', at='mq2@start'),
        env.new('bpm3', 'bpm', at='mq3@start'),
        env.new('bpm4', 'bpm', at='mq4@start'),

        env.new('corrector1', 'corrector', at='mq1@start'),
        env.new('corrector2', 'corrector', at='mq2@start'),
        env.new('corrector3', 'corrector', at='mq3@start'),
        env.new('corrector4', 'corrector', at='mq4@start'),

        env.new('bumper1', 'corrector', at=1., knl=['k0l_bumper1'], ksl=['k0sl_bumper1']),
        env.new('bumper2', 'corrector', at=2., knl=['k0l_bumper2'], ksl=['k0sl_bumper2']),
    ])

    env.set(['mq1', 'mq2', 'mq3', 'mq4'], shift_x=1e-3, shift_y=1.5e-3,
            rot_s_rad=np.deg2rad(30.))

    # Steer to enter at the center of the first quad
    line.match(
        betx=1., bety=1.,
        vary=xt.VaryList(['k0l_bumper1', 'k0l_bumper2', 'k0sl_bumper1', 'k0sl_bumper2'],
                        step=1e-6),
        targets=xt.TargetSet(x=1e-3, px=0, y=1.5e-3, py=0, at='bpm1'),
    )

    tt = line.get_table()
    tt_quad = tt.rows['mq.*']

    env['kq1'] = 0.02
    env['kq2'] = -0.02
    env['kq3'] = 0.02
    env['kq4'] = -0.02

    dx = 2e-3

    bpm_alignment ={
        'bpm1': {'shift_x': 1e-3, 'shift_y': 1.5e-3, 'rot_s_rad': np.deg2rad(30.)},
        'bpm2': {'shift_x': 1e-3, 'shift_y': 1.5e-3, 'rot_s_rad': np.deg2rad(30.)},
        'bpm3': {'shift_x': 1e-3, 'shift_y': 1.5e-3, 'rot_s_rad': np.deg2rad(30.)},
        'bpm4': {'shift_x': 1e-3, 'shift_y': 1.5e-3, 'rot_s_rad': np.deg2rad(30.)},
    }

    tw0 = line.twiss(betx=100, bety=80)

    line.steering_monitors_x = ['bpm1', 'bpm2', 'bpm3', 'bpm4']
    line.steering_monitors_y = ['bpm1', 'bpm2', 'bpm3', 'bpm4']
    line.steering_correctors_x = ['corrector1', 'corrector3', ]
    line.steering_correctors_y = ['corrector2', 'corrector4']

    tw = line.twiss(betx=1, bety=1, x=1e-3+dx*np.cos(np.deg2rad(30)), y=1.5e-3+dx*np.sin(np.deg2rad(30)))


    correction = line.correct_trajectory(twiss_table=tw0,
                                        start='line.start', end='line.end',
                                        monitor_alignment=bpm_alignment,
                                        run=False)

    correction.correct(n_iter=1)

    # Check that there is no vertical reading in the tilted bpm BPMs
    xo.assert_allclose(correction.y_correction._position_before, 0, atol=1e-15, rtol=0)
