import numpy as np

import xtrack as xt
import xpart as xp
import xobjects as xo

# We define a data structure to allowing all elements of a new BeamElement type
# to store data in one place. Such a data structure needs to contain a files called
# `_record_index` of type `xtrack.RecordIndex`, which will be used internally to
# keep count of the number of records stored in the data structure. Furthermore,
# the structure can contain an arbitrary number of other fields where the data
# will be stored.

class TestElementRecord(xo.DressedStruct):
    _xofields = {
        '_record_index': xt.RecordIndex,
        'generated_rr': xo.Float64[:],
        'at_element': xo.Int64[:],
        'at_turn': xo.Int64[:]
        }

# To allow a elements of a given type to sore data in a structure defined above
# we need to:
# - add a in the beam element xofields a field called `_internal_record_id` of
#   type `xtrack.RecordIdentifier`, which will be used internally to reference 
#   the data structure.
# - add `internal_record_id` to the `_skip_in_to_dict` list, as the reference to
#   the data structure cannot be exported to a dictionary.
# - add an attribute called `_internal_recor_class` to which we bind the data
#   structure type defined above.

class TestElement(xt.BeamElement):
    _xofields={
        '_internal_record_id': xt.RecordIdentifier,
        'n_kicks': xo.Int64,
        }

    _skip_in_to_dict = ['_internal_record_id']

    _internal_record_class = TestElementRecord

TestElement.XoStruct.extra_sources = [
    xp._pkg_root.joinpath('random_number_generator/rng_src/base_rng.h'),
    xp._pkg_root.joinpath('random_number_generator/rng_src/local_particle_rng.h'),
    ]

TestElement.XoStruct.extra_sources.append(r'''
    /*gpufun*/
    void TestElement_track_local_particle(TestElementData el, LocalParticle* part0){

        // Check if internal record is enabled
        int record_enabled = TestElementData_get__internal_record_id_buffer_id(el) > 0;

        TestElementRecordData record = NULL;
        RecordIndex record_index = NULL;

        // Extract the record_id, record and record_index
        if (record_enabled){
            RecordIdentifier record_id = TestElementData_getp__internal_record_id(el);
            record = (TestElementRecordData) RecordIdentifier_getp_record(record_id, part0);
            if (record){
                record_index = TestElementRecordData_getp__record_index(record);
            }
        }

        int64_t n_kicks = TestElementData_get_n_kicks(el);
        printf("n_kicks %d\n", (int)n_kicks);

        //start_per_particle_block (part0->part)

            for (int64_t i = 0; i < n_kicks; i++) {
                double rr = 1e-6 * LocalParticle_generate_random_double(part);
                LocalParticle_add_to_px(part, rr);

                if (record_enabled){
                    int64_t i_slot = RecordIndex_get_slot(record_index);
                    // gives negative is record is NULL or if record is full

                    printf("Hello %d\n", (int)i_slot);
                    if (i_slot>=0){
                        TestElementRecordData_set_at_element(record, i_slot,
                                                    LocalParticle_get_at_element(part));
                        TestElementRecordData_set_at_turn(record, i_slot,
                                                    LocalParticle_get_at_turn(part));
                        TestElementRecordData_set_generated_rr(record, i_slot, rr);
                    }
                }
            }


        //end_per_particle_block
    }
    ''')



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

part = xp.Particles(_context=context, p0c=6.5e12, x=[1,2,3])
num_turns0 = 10
num_turns1 = 3
tracker.track(part, num_turns=num_turns0)
tracker.track(part, num_turns=num_turns1)

# Checks
num_recorded = record._record_index.num_recorded
num_turns = num_turns0 + num_turns1
num_particles = len(part.x)
part._move_to(_context=xo.ContextCpu())
record._move_to(_context=xo.ContextCpu())
assert num_recorded == (num_particles * num_turns * (n_kicks0 + n_kicks1))

assert np.sum((record.at_element[:num_recorded] == 0)) == (num_particles * num_turns
                                           * n_kicks0)
assert np.sum((record.at_element[:num_recorded] == 1)) == (num_particles * num_turns
                                           * n_kicks1)
for i_turn in range(num_turns):
    assert np.sum((record.at_turn[:num_recorded] == i_turn)) == (num_particles
                                                        * (n_kicks0 + n_kicks1))

# Check stop
record = tracker.start_internal_logging_for_elements_of_type(
                                                    TestElement, capacity=10000)

part = xp.Particles(_context=context, p0c=6.5e12, x=[1,2,3])
num_turns0 = 10
num_turns1 = 3
num_particles = len(part.x)
tracker.track(part, num_turns=num_turns0)
tracker.stop_internal_logging_for_elements_of_type(TestElement)
tracker.track(part, num_turns=num_turns1)

num_recorded = record._record_index.num_recorded
num_turns = num_turns0
part._move_to(_context=xo.ContextCpu())
record._move_to(_context=xo.ContextCpu())
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

part = xp.Particles(_context=context, p0c=6.5e12, x=[1,2,3])
num_turns0 = 10
num_turns1 = 3
tracker.track(part, num_turns=num_turns0)
tracker.stop_internal_logging_for_elements_of_type(TestElement)
tracker.track(part, num_turns=num_turns1)

# Checks
part._move_to(_context=xo.ContextCpu())
record._move_to(_context=xo.ContextCpu())
num_recorded = record._record_index.num_recorded
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
