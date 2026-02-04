import xtrack as xt
import xobjects as xo
import numpy as np

# TODO:
# - Forbid backtrack for now
# - Need to test with and without progress bar...

line = xt.load('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

# Get names of all BPMs in the line
tt = line.get_table()
tt_obs = tt.rows.match(name='bpm.*')
tt_obs.name # is ['bpmw.4r7.b1', 'bpmwe.4r7.b1', 'bpmw.5r7.b1', ...,

# Generate some particle
p0 = xt.Particles(p0c=7e12, x=1e-6*np.arange(20),
                           y=2e-6*np.arange(20),
                           delta=1e-5*np.arange(20)
)


# Track particles for 10 turns, monitoring at all BPMs
num_turns = 10
p = p0.copy()
line.track(p, num_turns=num_turns, multi_element_monitor_at=tt_obs.name)

# Get the recorded data
mon = line.record_multi_element_last_track

line_with_monitors = line.copy(shallow=True)

nn_check_1 = 'bpm.17r3.b1'
nn_check_2 = 'bpm.23l5.b1'

mon_1 = xt.ParticlesMonitor(
    start_at_turn=0,
    stop_at_turn=num_turns,
    num_particles=len(p0.x))
mon_2 = xt.ParticlesMonitor(
    start_at_turn=0,
    stop_at_turn=num_turns,
    num_particles=len(p0.x))

line_with_monitors.insert('mon_1', mon_1, at=nn_check_1+'@start')
line_with_monitors.insert('mon_2', mon_2, at=nn_check_2+'@start')
p_ref = p0.copy()
line_with_monitors.track(p_ref, num_turns=num_turns)

xo.assert_allclose(mon.get('x', obs_name=nn_check_1), mon_1.x.T, atol=1e-14)
xo.assert_allclose(mon.get('px', obs_name=nn_check_1), mon_1.px.T, atol=1e-14)
xo.assert_allclose(mon.get('y', obs_name=nn_check_1), mon_1.y.T, atol=1e-14)
xo.assert_allclose(mon.get('py', obs_name=nn_check_1), mon_1.py.T, atol=1e-14)
xo.assert_allclose(mon.get('zeta', obs_name=nn_check_1), mon_1.zeta.T, atol=1e-14)
xo.assert_allclose(mon.get('delta', obs_name=nn_check_1), mon_1.delta.T, atol=1e-14)

xo.assert_allclose(mon.get('x', obs_name=nn_check_2), mon_2.x.T, atol=1e-14)
xo.assert_allclose(mon.get('px', obs_name=nn_check_2), mon_2.px.T, atol=1e-14)
xo.assert_allclose(mon.get('y', obs_name=nn_check_2), mon_2.y.T, atol=1e-14)
xo.assert_allclose(mon.get('py', obs_name=nn_check_2), mon_2.py.T, atol=1e-14)
xo.assert_allclose(mon.get('zeta', obs_name=nn_check_2), mon_2.zeta.T, atol=1e-14)
xo.assert_allclose(mon.get('delta', obs_name=nn_check_2), mon_2.delta.T, atol=1e-14)

# Access all data for a given coordinate
assert mon.data.shape == (num_turns, len(p0.x), 6, len(tt_obs))
xo.assert_allclose(mon.get('x'), mon.data[:,:,0,:], atol=1e-14)
xo.assert_allclose(mon.get('px'), mon.data[:,:,1,:], atol=1e-14)
xo.assert_allclose(mon.get('y'), mon.data[:,:,2,:], atol=1e-14)
xo.assert_allclose(mon.get('py'), mon.data[:,:,3,:], atol=1e-14)
xo.assert_allclose(mon.get('zeta'), mon.data[:,:,4,:], atol=1e-14)
xo.assert_allclose(mon.get('delta'), mon.data[:,:,5,:], atol=1e-14)

idx = tt_obs.rows.indices['bpm.17r3.b1'][0]

# # Access data for a given coordinate and observation point
# mon.get('x', obs_name='bpmw.4r7.b1') # shape ddddis (nom_turns, n_particles)
xo.assert_allclose(mon.get('x', obs_name=nn_check_1), mon.data[:,:,0,idx], atol=1e-14)
xo.assert_allclose(mon.get('px', obs_name=nn_check_1), mon.data[:,:,1,idx], atol=1e-14)
xo.assert_allclose(mon.get('y', obs_name=nn_check_1), mon.data[:,:,2,idx], atol=1e-14)
xo.assert_allclose(mon.get('py', obs_name=nn_check_1), mon.data[:,:,3,idx], atol=1e-14)
xo.assert_allclose(mon.get('delta', obs_name=nn_check_1), mon.data[:,:,5,idx], atol=1e-14)


# # Access data for a given coordinate, observation point and particle_id
# mon.get('x', obs_name='bpmw.4r7.b1', particle_id=10) # shape is (nom_turns,)
particle_id = 10
xo.assert_allclose(mon.get('x', obs_name=nn_check_1, particle_id=particle_id),
                   mon.data[:,particle_id - mon.part_id_start,0,idx], atol=1e-14)
xo.assert_allclose(mon.get('px', obs_name=nn_check_1, particle_id=particle_id),
                   mon.data[:,particle_id - mon.part_id_start,1,idx], atol=1e-14)
xo.assert_allclose(mon.get('y', obs_name=nn_check_1, particle_id=particle_id),
                   mon.data[:,particle_id - mon.part_id_start,2,idx], atol=1e-14)
xo.assert_allclose(mon.get('py', obs_name=nn_check_1, particle_id=particle_id),
                   mon.data[:,particle_id - mon.part_id_start,3,idx], atol=1e-14)
xo.assert_allclose(mon.get('zeta', obs_name=nn_check_1, particle_id=particle_id),
                   mon.data[:,particle_id - mon.part_id_start,4,idx], atol=1e-14)
xo.assert_allclose(mon.get('delta', obs_name=nn_check_1, particle_id=particle_id),
                   mon.data[:,particle_id - mon.part_id_start,5,idx], atol=1e-14)

# # Access data for a given coordinate, observation point, particle_id and turn
# mon.get('x', obs_name='bpmw.4r7.b1', particle_id=10, turn=5) # is a scalar
turn = 5
xo.assert_allclose(mon.get('x', obs_name=nn_check_1, particle_id=particle_id, turn=turn),
                     mon.data[turn,particle_id - mon.part_id_start,0,idx], atol=1e-14)
xo.assert_allclose(mon.get('px', obs_name=nn_check_1, particle_id=particle_id, turn=turn),
                     mon.data[turn,particle_id - mon.part_id_start,1,idx], atol=1e-14)
xo.assert_allclose(mon.get('y', obs_name=nn_check_1, particle_id=particle_id, turn=turn),
                     mon.data[turn,particle_id - mon.part_id_start,2,idx], atol=1e-14)
xo.assert_allclose(mon.get('py', obs_name=nn_check_1, particle_id=particle_id, turn=turn),
                     mon.data[turn,particle_id - mon.part_id_start,3,idx], atol=1e-14)
xo.assert_allclose(mon.get('zeta', obs_name=nn_check_1, particle_id=particle_id, turn=turn),
                   mon.data[turn,particle_id - mon.part_id_start,4,idx], atol=1e-14)
xo.assert_allclose(mon.get('delta', obs_name=nn_check_1, particle_id=particle_id, turn=turn),
                     mon.data[turn,particle_id - mon.part_id_start,5,idx], atol=1e-14)

# # Get all BPMs for a given particles and turn
# mon.get('x', particle_id=10, turn=5) # shape is (n_obs_points,)
xo.assert_allclose(mon.get('x', particle_id=particle_id, turn=turn),
                   mon.data[turn,particle_id - mon.part_id_start,0,:], atol=1e-14)
xo.assert_allclose(mon.get('px', particle_id=particle_id, turn=turn),
                   mon.data[turn,particle_id - mon.part_id_start,1,:], atol=1e-14)
xo.assert_allclose(mon.get('y', particle_id=particle_id, turn=turn),
                   mon.data[turn,particle_id - mon.part_id_start,2,:], atol=1e-14)
xo.assert_allclose(mon.get('py', particle_id=particle_id, turn=turn),
                   mon.data[turn,particle_id - mon.part_id_start,3,:], atol=1e-14)
xo.assert_allclose(mon.get('zeta', particle_id=particle_id, turn=turn),
                   mon.data[turn,particle_id - mon.part_id_start,4,:], atol=1e-14)
xo.assert_allclose(mon.get('delta', particle_id=particle_id, turn=turn),
                   mon.data[turn,particle_id - mon.part_id_start,5,:], atol=1e-14)
