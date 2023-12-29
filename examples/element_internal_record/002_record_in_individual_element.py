# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xtrack as xt
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
        // printf("n_kicks %d\n", (int)n_kicks);

        //start_per_particle_block (part0->part)

            // Record in table1 info about the ingoing particle
            if (record){
                // Get a slot in table1
                int64_t i_slot = RecordIndex_get_slot(table1_index);
                // The returned slot id is negative if record is NULL or if record is full
                //printf("i_slot %d\n", (int)i_slot);

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
                    //printf("i_slot %d\n", (int)i_slot);

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
part = xt.Particles(p0c=6.5e12, x=[1e-3,2e-3,3e-3])
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
