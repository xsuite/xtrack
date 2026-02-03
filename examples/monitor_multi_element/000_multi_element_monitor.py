import xtrack as xt
import numpy as np

# TODO:
# - Forbid backtrack for now
# - Need to test with and without progress bar...

line = xt.load('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

# Get names of all BPMs in the line
tt = line.get_table()
tt_obs = tt.rows.match(name='bpm.*')
tt_obs.name # is ['bpmw.4r7.b1', 'bpmwe.4r7.b1', 'bpmw.5r7.b1', ...,

# Generate some particles
p = xt.Particles(p0c=7e12, x=1e-6*np.arange(20),
                           delta=0)

# Track particles for 10 turns, monitoring at all BPMs
line.track(p, num_turns=10, multi_element_monitor_at=tt_obs.name)

# Get the recorded data
mon = line.record_multi_element_last_track

# Access all data for a given coordinate
mon.get('x') # shape is (nom_turns, n_particles, n_obs_points)

# Access data for a given coordinate and observation point
mon.get('x', obs_name='bpmw.4r7.b1') # shape is (nom_turns, n_particles)

# Access data for a given coordinate, observation point and particle_id
mon.get('x', obs_name='bpmw.4r7.b1', particle_id=10) # shape is (nom_turns,)

# Access data for a given coordinate, observation point, particle_id and turn
mon.get('x', obs_name='bpmw.4r7.b1', particle_id=10, turn=5) # is a scalar

# Get all BPMs for a given particles and turn
mon.get('x', particle_id=10, turn=5) # shape is (n_obs_points,)

