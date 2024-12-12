import numpy as np
import pytest

import xtrack as xt
import xpart as xp


def test_seeds():
    # Define a reference particle
    particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=4e9)

    # Create a particle object containing some particles with capacity twice the number of particles
    n_part = 10
    particles = xp.build_particles(particle_ref=particle_ref, num_particles=n_part, _capacity=2*n_part)

    # Explicitly initialize the random number generator for the particle object
    seeds = np.random.randint(0, np.iinfo(np.uint32).max, particles._capacity)
    particles._init_random_number_generator(seeds=seeds)

    # Save the seeds for the random number generator
    rng_s1_before = particles._rng_s1.copy()
    rng_s2_before = particles._rng_s2.copy()
    rng_s3_before = particles._rng_s3.copy()
    rng_s4_before = particles._rng_s4.copy()

    # Manually kill some particles
    n_part_to_kill = np.random.randint(1, n_part+1)
    idx_part_to_kill = np.random.choice(n_part, n_part_to_kill, replace=False)

    particles.state[idx_part_to_kill] = -1
    particles._num_active_particles = sum(particles.state == 1)
    particles._num_lost_particles = n_part - particles._num_lost_particles

    # Create some new particles
    n_part_new = np.random.randint(1, particles._capacity - n_part)
    new_particles = xp.build_particles(particle_ref=particle_ref, num_particles=n_part_new)

    # Add the new particles to the main particle object
    particles.add_particles(new_particles)

    # Save the seeds for the random number generator after the new particles addition
    rng_s1_after = particles._rng_s1
    rng_s2_after = particles._rng_s2
    rng_s3_after = particles._rng_s3
    rng_s4_after = particles._rng_s4

    # Verify that the seeds for the random number generator stay the same
    assert np.array_equal(rng_s1_after, rng_s1_before)
    assert np.array_equal(rng_s2_after, rng_s2_before)
    assert np.array_equal(rng_s3_after, rng_s3_before)
    assert np.array_equal(rng_s4_after, rng_s4_before)