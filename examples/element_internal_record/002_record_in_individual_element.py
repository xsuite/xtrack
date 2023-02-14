# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xtrack as xt
import xpart as xp
import xobjects as xo

# In this case the element internal record is made of two separate tables each
# with its own record index.

class Table1(xo.HybridClass):
    _xofields = {
        '_index': xt.RecordIndex,
        'particle_x': xo.Float64[:],
        'particle_px': xo.Float64[:],
        'at_element': xo.Int64[:],
        'at_turn': xo.Int64[:],
        'particle_id': xo.Int64[:]
        }

class Table2(xo.HybridClass):
    _xofields = {
        '_index': xt.RecordIndex,
        'generated_rr': xo.Float64[:],
        'at_element': xo.Int64[:],
        'at_turn': xo.Int64[:],
        'particle_id': xo.Int64[:]
        }

class TestElementRecord(xo.HybridClass):
    _xofields = {
        'table1': Table1,
        'table2': Table2
        }

# The two tables in the internal record can be accessed independently in the C
# code of the beam element.

TestElement_track_method_source = r'''
    /*gpufun*/
    void TestElement_track_local_particle(TestElementData el, LocalParticle* part0){

        // Extract the record and record_index
        TestElementRecordData record = TestElementData_getp_internal_record(el, part0);
        Table1Data table1 = NULL;
        Table2Data table2 = NULL;
        RecordIndex table1_index = NULL;
        RecordIndex table2_index = NULL;
        if (record){
            table1 = TestElementRecordData_getp_table1(record);
            table2 = TestElementRecordData_getp_table2(record);
            table1_index = Table1Data_getp__index(table1);
            table2_index = Table2Data_getp__index(table2);
        }

        int64_t n_kicks = TestElementData_get_n_kicks(el);
        printf("n_kicks %d\n", (int)n_kicks);

        //start_per_particle_block (part0->part)

            // Record in table1 info about the ingoing particle
            if (record){
                // Get a slot in table1
                int64_t i_slot = RecordIndex_get_slot(table1_index);
                // The returned slot id is negative if record is NULL or if record is full
                printf("i_slot %d\n", (int)i_slot);

                if (i_slot>=0){
                        Table1Data_set_particle_x(table1, i_slot,
                                                    LocalParticle_get_x(part));
                        Table1Data_set_particle_px(table1, i_slot,
                                                    LocalParticle_get_px(part));
                        Table1Data_set_at_element(table1, i_slot,
                                                    LocalParticle_get_at_element(part));
                        Table1Data_set_at_turn(table1, i_slot,
                                                    LocalParticle_get_at_turn(part));
                        Table1Data_set_particle_id(table1, i_slot,
                                                    LocalParticle_get_particle_id(part));
                }
            }

            for (int64_t i = 0; i < n_kicks; i++) {
                double rr = 1e-6 * RandomUniform_generate(part);
                LocalParticle_add_to_px(part, rr);

                // Record in table2 info about the generated kicks
                if (record){
                    // Get a slot in table2
                    int64_t i_slot = RecordIndex_get_slot(table2_index);
                    // The returned slot id is negative if record is NULL or if record is full
                    printf("i_slot %d\n", (int)i_slot);

                    if (i_slot>=0){
                            Table2Data_set_generated_rr(table2, i_slot, rr);
                            Table2Data_set_at_element(table2, i_slot,
                                                        LocalParticle_get_at_element(part));
                            Table2Data_set_at_turn(table2, i_slot,
                                                        LocalParticle_get_at_turn(part));
                            Table2Data_set_particle_id(table2, i_slot,
                                                        LocalParticle_get_particle_id(part));
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

    _depends_on = [xt.RandomUniform]

    _extra_c_sources = [TestElement_track_method_source]

# Once these steps are done, the TestElement and its recording feature are ready
# and can be used as follows.

# Beam elements can be created as usual
test_element1 = TestElement(n_kicks=10)
test_element2 = TestElement(n_kicks=5)

# The recording is by default disabled and can be enabled using the following
# dedicated function. The capacity allocated for the two tables needs to be
# provided in a dictionary:

io_buffer = xt.new_io_buffer()
xt.start_internal_logging(elements=[test_element1, test_element2],
                          io_buffer=io_buffer,
                          capacity={'table1':1000, 'table2':500})

# Make some particles
part = xp.Particles(p0c=6.5e12, x=[1e-3,2e-3,3e-3])
part._init_random_number_generator()

# Track through the first element
test_element1.track(part)
# Track through the second element
test_element2.track(part)

# We can now inspect the two tables in the `record`, e.g `record.table1.particle_x`,
# `record.table2.generated_rr`. The number of used slots in each can be found in
# the corresponding index object e.g. record.table1._index.num_recorded.

# The recording can be stopped with the following method:
xt.stop_internal_logging(elements=[test_element1, test_element2])

# Track more times (without logging information in `record`)
test_element1.track(part)
test_element2.track(part)

# The record can be accessed from test_element1.record and test_element2.record

#!end-doc-part

# Checks

context = xo.ContextCpu()
#context = xo.ContextCupy()
#context = xo.ContextPyopencl()
n_kicks0 = 5
n_kicks1 = 3
elements = [
    TestElement(_context=context, n_kicks=n_kicks0),
    TestElement(_context=context, n_kicks=n_kicks1)]

io_buffer = xt.new_io_buffer(_context=context)
record = xt.start_internal_logging(elements=elements, io_buffer=io_buffer,
                          capacity={'table1': 10000, 'table2': 10000})

part = xp.Particles(_context=context, p0c=6.5e12, x=[1e-3,2e-3,3e-3])
num_turns0 = 10
num_turns1 = 3

for i_turn in range(num_turns0 + num_turns1):
    for ee in elements:
        ee.track(part, increment_at_element=True)
    part.at_element[:] = 0
    part.at_turn += 1

part.move(_context=xo.ContextCpu())
record.move(_context=xo.ContextCpu())

num_turns = num_turns0 + num_turns1
num_particles = len(part.x)

table1 = record.table1
table2 = record.table2
num_recorded_tab1 = table1._index.num_recorded
num_recorded_tab2 = table2._index.num_recorded

assert num_recorded_tab1 == 2 * (num_particles * num_turns)
assert num_recorded_tab2 == (num_particles * num_turns * (n_kicks0 + n_kicks1))

assert np.sum((table1.at_element[:num_recorded_tab1] == 0)) == (num_particles * num_turns)
assert np.sum((table1.at_element[:num_recorded_tab1] == 1)) == (num_particles * num_turns)
assert np.sum((table2.at_element[:num_recorded_tab2] == 0)) == (num_particles * num_turns
                                           * n_kicks0)
assert np.sum((table2.at_element[:num_recorded_tab2] == 1)) == (num_particles * num_turns
                                           * n_kicks1)
for i_turn in range(num_turns):
    assert np.sum((table1.at_turn[:num_recorded_tab1] == i_turn)) == 2 * num_particles
    assert np.sum((table2.at_turn[:num_recorded_tab2] == i_turn)) == (num_particles
                                                        * (n_kicks0 + n_kicks1))

# Check reached capacity
io_buffer = xt.new_io_buffer(_context=context)
record = xt.start_internal_logging(elements=elements, io_buffer=io_buffer,
                          capacity={'table1': 20, 'table2': 15})

part = xp.Particles(_context=context, p0c=6.5e12, x=[1e-3,2e-3,3e-3])
num_turns0 = 10
num_turns1 = 3
for i_turn in range(num_turns0 + num_turns1):
    for ee in elements:
        ee.track(part, increment_at_element=True)
    part.at_element[:] = 0
    part.at_turn += 1

table1 = record.table1
table2 = record.table2
num_recorded_tab1 = table1._index.num_recorded
num_recorded_tab2 = table2._index.num_recorded

assert num_recorded_tab1 == 20
assert num_recorded_tab2 == 15


# Check stop
io_buffer = xt.new_io_buffer(_context=context)
record = xt.start_internal_logging(elements=elements, io_buffer=io_buffer,
                          capacity={'table1': 10000, 'table2': 10000})

part = xp.Particles(_context=context, p0c=6.5e12, x=[1e-3,2e-3,3e-3])
num_turns0 = 10
num_turns1 = 3
num_particles = len(part.x)


for i_turn in range(num_turns0 + num_turns1):
    if i_turn == num_turns0:
        xt.stop_internal_logging(elements=elements)
    for ee in elements:
        ee.track(part, increment_at_element=True)
    part.at_element[:] = 0
    part.at_turn += 1


part.move(_context=xo.ContextCpu())
record.move(_context=xo.ContextCpu())

num_turns = num_turns0

table1 = record.table1
table2 = record.table2
num_recorded_tab1 = table1._index.num_recorded
num_recorded_tab2 = table2._index.num_recorded

assert num_recorded_tab1 == 2 * (num_particles * num_turns)
assert num_recorded_tab2 == (num_particles * num_turns * (n_kicks0 + n_kicks1))

assert np.sum((table1.at_element[:num_recorded_tab1] == 0)) == (num_particles * num_turns)
assert np.sum((table1.at_element[:num_recorded_tab1] == 1)) == (num_particles * num_turns)
assert np.sum((table2.at_element[:num_recorded_tab2] == 0)) == (num_particles * num_turns
                                           * n_kicks0)
assert np.sum((table2.at_element[:num_recorded_tab2] == 1)) == (num_particles * num_turns
                                           * n_kicks1)
for i_turn in range(num_turns):
    assert np.sum((table1.at_turn[:num_recorded_tab1] == i_turn)) == 2 * num_particles
    assert np.sum((table2.at_turn[:num_recorded_tab2] == i_turn)) == (num_particles
                                                        * (n_kicks0 + n_kicks1))

# Separate buffers
io_buffer0 = xt.new_io_buffer(_context=context)
record0 = xt.start_internal_logging(elements=elements[0], io_buffer=io_buffer0,
                          capacity={'table1': 10000, 'table2': 10000})
io_buffer1 = xt.new_io_buffer(_context=context)
record1 = xt.start_internal_logging(elements=elements[1], io_buffer=io_buffer1,
                          capacity={'table1': 10000, 'table2': 10000})

part = xp.Particles(_context=context, p0c=6.5e12, x=[1e-3,2e-3,3e-3])
num_turns0 = 10
num_turns1 = 3

for i_turn in range(num_turns0 + num_turns1):
    for ee in elements:
        ee.track(part, increment_at_element=True)
    part.at_element[:] = 0
    part.at_turn += 1

part.move(_context=xo.ContextCpu())
record0.move(_context=xo.ContextCpu())
record1.move(_context=xo.ContextCpu())

num_turns = num_turns0 + num_turns1
num_particles = len(part.x)

table01 = record0.table1
table02 = record0.table2
num_recorded_tab01 = table01._index.num_recorded
num_recorded_tab02 = table02._index.num_recorded

assert num_recorded_tab01 == (num_particles * num_turns)
assert num_recorded_tab02 == (num_particles * num_turns * (n_kicks0))

assert np.sum((table01.at_element[:num_recorded_tab01] == 0)) == (num_particles * num_turns)
assert np.sum((table01.at_element[:num_recorded_tab01] == 1)) == 0
assert np.sum((table02.at_element[:num_recorded_tab02] == 0)) == (num_particles * num_turns
                                           * n_kicks0)
assert np.sum((table02.at_element[:num_recorded_tab02] == 1)) == 0

for i_turn in range(num_turns):
    assert np.sum((table01.at_turn[:num_recorded_tab01] == i_turn)) == num_particles
    assert np.sum((table02.at_turn[:num_recorded_tab02] == i_turn)) == (num_particles
                                                        * (n_kicks0))

table11 = record1.table1
table12 = record1.table2
num_recorded_tab11 = table11._index.num_recorded
num_recorded_tab12 = table12._index.num_recorded

assert num_recorded_tab11 == (num_particles * num_turns)
assert num_recorded_tab12 == (num_particles * num_turns * (n_kicks1))

assert np.sum((table11.at_element[:num_recorded_tab11] == 0)) == 0
assert np.sum((table11.at_element[:num_recorded_tab11] == 1)) == (num_particles * num_turns)
assert np.sum((table12.at_element[:num_recorded_tab12] == 0)) == 0
assert np.sum((table12.at_element[:num_recorded_tab12] == 1)) == (num_particles * num_turns
                                           * n_kicks1)

for i_turn in range(num_turns):
    assert np.sum((table11.at_turn[:num_recorded_tab11] == i_turn)) == num_particles
    assert np.sum((table12.at_turn[:num_recorded_tab12] == i_turn)) == (num_particles
                                                        * (n_kicks1))
