# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xtrack as xt
import xpart as xp
import xobjects as xo

################################################################
# Definition of a beam element with an internal data recording #
################################################################

# We define a data structure to allow all elements of a new BeamElement type
# to store data in one place. Such a data structure needs to contain a field called
# `_index` of type `xtrack.RecordIndex`, which will be used internally to
# keep count of the number of records stored in the data structure. Together with
# the index, the structure can contain an arbitrary number of other fields (which
# need to be arrays) where the data will be stored.

class TestElementRecord(xo.HybridClass):
    _xofields = {
        '_index': xt.RecordIndex,
        'generated_rr': xo.Float64[:],
        'at_element': xo.Int64[:],
        'at_turn': xo.Int64[:],
        'particle_id': xo.Int64[:]
        }

# The defined data structure can be accessed in the C code of the beam element
# to log data.
# In the following example, the element applies an assigned number of random
# kicks to the horizontal momentum. The internal record is used to store the
# kicks applied together with the corresponding particle_id, turn number and
# element number.

track_method_source = r'''
/*gpufun*/
void TestElement_track_local_particle(TestElementData el, LocalParticle* part0){

    // Extract the record and record_index
    TestElementRecordData record = TestElementData_getp_internal_record(el, part0);
    RecordIndex record_index = NULL;
    if (record){
        record_index = TestElementRecordData_getp__index(record);
    }

    int64_t n_kicks = TestElementData_get_n_kicks(el);
    printf("n_kicks %d\n", (int)n_kicks);

    //start_per_particle_block (part0->part)

        for (int64_t i = 0; i < n_kicks; i++) {
            double rr = 1e-6 * LocalParticle_generate_random_double(part);
            LocalParticle_add_to_px(part, rr);

            if (record){
                // Get a slot in the record (this is thread safe)
                int64_t i_slot = RecordIndex_get_slot(record_index);
                // The returned slot id is negative if record is NULL or if record is full

                if (i_slot>=0){
                    TestElementRecordData_set_at_element(record, i_slot,
                                                LocalParticle_get_at_element(part));
                    TestElementRecordData_set_at_turn(record, i_slot,
                                                LocalParticle_get_at_turn(part));
                    TestElementRecordData_set_particle_id(record, i_slot,
                                                LocalParticle_get_particle_id(part));
                    TestElementRecordData_set_generated_rr(record, i_slot, rr);
                }
            }
        }


    //end_per_particle_block
}
'''

# To allow elements of a given type to store data in a structure of the type defined
# above we need to add in the element class an attribute called
# `_internal_record_class` to which we bind the data structure type defined above.

class TestElement(xt.BeamElement):
    _xofields={
        'n_kicks': xo.Int64,
        }

    _internal_record_class = TestElementRecord

    _extra_c_sources = [
        # The element uses the Xtrack random number generator
        xp._pkg_root.joinpath('random_number_generator/rng_src/base_rng.h'),
        xp._pkg_root.joinpath('random_number_generator/rng_src/local_particle_rng.h'),
        track_method_source]


# Once these steps are done, the TestElement and its recording feature are ready
# and can be used as follows.

###########################################################
# Usage of a beam element with an internal data recording #
###########################################################

# Line, tracker, particles can be created as usual
line=xt.Line(elements = [
    xt.Drift(length=1.), TestElement(n_kicks=10),
    xt.Drift(length=1.), TestElement(n_kicks=5)])
tracker = line.build_tracker()
tracker.line._needs_rng = True # Test elements use the random number generator
part = xp.Particles(p0c=6.5e12, x=[1e-3,2e-3,3e-3])

# The record is by default disabled and can be enabled using the following
# dedicated method of the tracker object. The argument `capacity` defines the
# number of items that can be stored in each element of the internal record
# (tha same space are shared for all the elements of the same type). The returned
# object (called `record` in the following) will be filled with the recorded
# data when the tracker is run. The recording stops when the full capacity is reached.

record = tracker.start_internal_logging_for_elements_of_type(
                                                    TestElement, capacity=10000)

# Track!
tracker.track(part, num_turns=10)

# We can now inspect `record`:
#  - `record.generated_rr` contains the random numbers generated by the elements
#  - `record.at_element` contains the element number where the recording took place
#  - `record.at_turn` contains the turn number where the recording took place
#  - `record.particle_id` contains the particle id for which the recording took place
# The number of used slots in `record` can be found in record._index.num_recorded

# The recording can be stopped with the following method:
tracker.stop_internal_logging_for_elements_of_type(TestElement)

# Track more turns (without logging information in `record`)
tracker.track(part, num_turns=10)

#!end-doc-part

# Checks

context = xo.ContextCpu()
#context = xo.ContextCupy()
#context = xo.ContextPyopencl()
n_kicks0 = 5
n_kicks1 = 3
tracker = xt.Tracker(_context=context, line=xt.Line(elements = [
    TestElement(n_kicks=n_kicks0), TestElement(n_kicks=n_kicks1)]))
tracker.line._needs_rng = True

record = tracker.start_internal_logging_for_elements_of_type(
                                                    TestElement, capacity=10000)

part = xp.Particles(_context=context, p0c=6.5e12, x=[1e-3,2e-3,3e-3])
num_turns0 = 10
num_turns1 = 3
tracker.track(part, num_turns=num_turns0)
tracker.track(part, num_turns=num_turns1)

num_recorded = record._index.num_recorded
num_turns = num_turns0 + num_turns1
num_particles = len(part.x)
part.move(_context=xo.ContextCpu())
record.move(_context=xo.ContextCpu())
assert num_recorded == (num_particles * num_turns * (n_kicks0 + n_kicks1))

assert np.sum((record.at_element[:num_recorded] == 0)) == (num_particles * num_turns
                                           * n_kicks0)
assert np.sum((record.at_element[:num_recorded] == 1)) == (num_particles * num_turns
                                           * n_kicks1)
for i_turn in range(num_turns):
    assert np.sum((record.at_turn[:num_recorded] == i_turn)) == (num_particles
                                                        * (n_kicks0 + n_kicks1))

# Check reached capacity
record = tracker.start_internal_logging_for_elements_of_type(
                                                    TestElement, capacity=20)

part = xp.Particles(_context=context, p0c=6.5e12, x=[1e-3,2e-3,3e-3])
num_turns0 = 10
num_turns1 = 3
tracker.track(part, num_turns=num_turns0)
tracker.track(part, num_turns=num_turns1)

num_recorded = record._index.num_recorded
assert num_recorded == 20


# Check stop
record = tracker.start_internal_logging_for_elements_of_type(
                                                    TestElement, capacity=10000)

part = xp.Particles(_context=context, p0c=6.5e12, x=[1e-3,2e-3,3e-3])
num_turns0 = 10
num_turns1 = 3
num_particles = len(part.x)
tracker.track(part, num_turns=num_turns0)
tracker.stop_internal_logging_for_elements_of_type(TestElement)
tracker.track(part, num_turns=num_turns1)

num_recorded = record._index.num_recorded
num_turns = num_turns0
part.move(_context=xo.ContextCpu())
record.move(_context=xo.ContextCpu())
assert np.all(part.at_turn == num_turns0 + num_turns1)
assert num_recorded == (num_particles * num_turns
                                          * (n_kicks0 + n_kicks1))

assert np.sum((record.at_element[:num_recorded] == 0)) == (num_particles * num_turns
                                           * n_kicks0)
assert np.sum((record.at_element[:num_recorded] == 1)) == (num_particles * num_turns
                                           * n_kicks1)
for i_turn in range(num_turns):
    assert np.sum((record.at_turn[:num_recorded] == i_turn)) == (num_particles
                                                        * (n_kicks0 + n_kicks1))

# Collective
n_kicks0 = 5
n_kicks1 = 3
elements = [
    TestElement(n_kicks=n_kicks0, _context=context), TestElement(n_kicks=n_kicks1)]
elements[0].iscollective = True
tracker = xt.Tracker(_context=context, line=xt.Line(elements=elements))
tracker.line._needs_rng = True

record = tracker.start_internal_logging_for_elements_of_type(
                                                    TestElement, capacity=10000)

part = xp.Particles(_context=context, p0c=6.5e12, x=[1e-3,2e-3,3e-3])
num_turns0 = 10
num_turns1 = 3
tracker.track(part, num_turns=num_turns0)
tracker.stop_internal_logging_for_elements_of_type(TestElement)
tracker.track(part, num_turns=num_turns1)

# Checks
part.move(_context=xo.ContextCpu())
record.move(_context=xo.ContextCpu())
num_recorded = record._index.num_recorded
num_turns = num_turns0
num_particles = len(part.x)
assert np.all(part.at_turn == num_turns0 + num_turns1)
assert num_recorded == (num_particles * num_turns
                                          * (n_kicks0 + n_kicks1))

assert np.sum((record.at_element[:num_recorded] == 0)) == (num_particles * num_turns
                                           * n_kicks0)
assert np.sum((record.at_element[:num_recorded] == 1)) == (num_particles * num_turns
                                           * n_kicks1)
for i_turn in range(num_turns):
    assert np.sum((record.at_turn[:num_recorded] == i_turn)) == (num_particles
                                                        * (n_kicks0 + n_kicks1))
