# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xtrack as xt
import xpart as xp
import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_record_single_table(test_context):
    class TestElementRecord(xo.HybridClass):
        _xofields = {
            '_index': xt.RecordIndex,
            'generated_rr': xo.Float64[:],
            'at_element': xo.Int64[:],
            'at_turn': xo.Int64[:],
            'particle_id': xo.Int64[:]
            }

    extra_src = []

    extra_src.append(r'''
        /*gpufun*/
        void TestElement_track_local_particle(TestElementData el, LocalParticle* part0){

            // Extract the record and record_index
            TestElementRecordData record = TestElementData_getp_internal_record(el, part0);
            RecordIndex record_index = NULL;
            if (record){
                record_index = TestElementRecordData_getp__index(record);
            }

            int64_t n_kicks = TestElementData_get_n_kicks(el);
            //printf("n_kicks %d\n", (int)n_kicks);

            //start_per_particle_block (part0->part)

                for (int64_t i = 0; i < n_kicks; i++) {
                    double rr = 1e-6 * RandomUniform_generate(part);
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
        ''')

    class TestElement(xt.BeamElement):
        _xofields={
            'n_kicks': xo.Int64,
            }

        _internal_record_class = TestElementRecord

        _depends_on = [xt.RandomUniform]

        _extra_c_sources = extra_src

    n_kicks0 = 5
    n_kicks1 = 3
    line=xt.Line(elements = [
        TestElement(n_kicks=n_kicks0), TestElement(n_kicks=n_kicks1)])
    line._needs_rng = True
    line.build_tracker(_context=test_context)

    record = line.start_internal_logging_for_elements_of_type(
                                                        TestElement, capacity=10000)

    part = xp.Particles(_context=test_context, p0c=6.5e12, x=[1e-3,2e-3,3e-3])
    num_turns0 = 10
    num_turns1 = 3
    line.track(part, num_turns=num_turns0)
    line.track(part, num_turns=num_turns1)

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
    record = line.start_internal_logging_for_elements_of_type(
                                                        TestElement, capacity=20)

    part = xp.Particles(_context=test_context, p0c=6.5e12, x=[1e-3,2e-3,3e-3])
    num_turns0 = 10
    num_turns1 = 3
    line.track(part, num_turns=num_turns0)
    line.track(part, num_turns=num_turns1)

    num_recorded = record._index.num_recorded
    assert num_recorded == 20


    # Check stop
    record = line.start_internal_logging_for_elements_of_type(
                                                        TestElement, capacity=10000)

    part = xp.Particles(_context=test_context, p0c=6.5e12, x=[1e-3,2e-3,3e-3])
    num_turns0 = 10
    num_turns1 = 3
    num_particles = len(part.x)
    line.track(part, num_turns=num_turns0)
    line.stop_internal_logging_for_elements_of_type(TestElement)
    line.track(part, num_turns=num_turns1)

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
        TestElement(n_kicks=n_kicks0, _context=test_context), TestElement(n_kicks=n_kicks1)]
    elements[0].iscollective = True
    line=xt.Line(elements=elements)
    line.build_tracker(_context=test_context)
    line._needs_rng = True

    record = line.start_internal_logging_for_elements_of_type(
                                                        TestElement, capacity=10000)

    part = xp.Particles(_context=test_context, p0c=6.5e12, x=[1e-3,2e-3,3e-3])
    num_turns0 = 10
    num_turns1 = 3
    line.track(part, num_turns=num_turns0)
    line.stop_internal_logging_for_elements_of_type(TestElement)
    line.track(part, num_turns=num_turns1)

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


@for_all_test_contexts
def test_record_multiple_tables(test_context):
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

    extra_src = []

    extra_src.append(r'''
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
                    // printf("i_slot %d\n", (int)i_slot);

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
                        // printf("i_slot %d\n", (int)i_slot);

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
        ''')

    class TestElement(xt.BeamElement):
        _xofields={
            'n_kicks': xo.Int64,
            }
        _internal_record_class = TestElementRecord

        _depends_on = [xt.RandomUniform]

        _extra_c_sources = extra_src


    # Checks
    n_kicks0 = 5
    n_kicks1 = 3
    line=xt.Line(elements = [
        TestElement(n_kicks=n_kicks0), TestElement(n_kicks=n_kicks1)])
    line._needs_rng = True
    line.build_tracker(_context=test_context)

    record = line.start_internal_logging_for_elements_of_type(TestElement,
                                capacity={'table1': 10000, 'table2': 10000})

    part = xp.Particles(_context=test_context, p0c=6.5e12, x=[1e-3,2e-3,3e-3])
    num_turns0 = 10
    num_turns1 = 3
    line.track(part, num_turns=num_turns0)
    line.track(part, num_turns=num_turns1)

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
    record = line.start_internal_logging_for_elements_of_type(
                                                        TestElement,
                                        capacity={'table1': 20, 'table2': 15})

    part = xp.Particles(_context=test_context, p0c=6.5e12, x=[1e-3,2e-3,3e-3])
    num_turns0 = 10
    num_turns1 = 3
    line.track(part, num_turns=num_turns0)
    line.track(part, num_turns=num_turns1)

    table1 = record.table1
    table2 = record.table2
    num_recorded_tab1 = table1._index.num_recorded
    num_recorded_tab2 = table2._index.num_recorded

    assert num_recorded_tab1 == 20
    assert num_recorded_tab2 == 15


    # Check stop
    record = line.start_internal_logging_for_elements_of_type(
                                        TestElement,
                                        capacity={'table1': 1000, 'table2': 1000})

    part = xp.Particles(_context=test_context, p0c=6.5e12, x=[1e-3,2e-3,3e-3])
    num_turns0 = 10
    num_turns1 = 3
    num_particles = len(part.x)

    line.track(part, num_turns=num_turns0)
    line.stop_internal_logging_for_elements_of_type(TestElement)
    line.track(part, num_turns=num_turns1)

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

    # Collective
    n_kicks0 = 5
    n_kicks1 = 3
    elements = [
        TestElement(n_kicks=n_kicks0, _context=test_context), TestElement(n_kicks=n_kicks1)]
    elements[0].iscollective = True
    line = xt.Line(elements=elements)
    line.build_tracker(_context=test_context)
    line.line._needs_rng = True

    record = line.start_internal_logging_for_elements_of_type(
                                        TestElement,
                                        capacity={'table1': 1000, 'table2': 1000})

    part = xp.Particles(_context=test_context, p0c=6.5e12, x=[1e-3,2e-3,3e-3])
    num_turns0 = 10
    num_turns1 = 3
    line.track(part, num_turns=num_turns0)
    line.stop_internal_logging_for_elements_of_type(TestElement)
    line.track(part, num_turns=num_turns1)

    # Checks
    part.move(_context=xo.ContextCpu())
    record.move(_context=xo.ContextCpu())
    num_turns = num_turns0
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


@for_all_test_contexts
def test_record_standalone_mode(test_context):

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

    extra_src = []

    extra_src.append(r'''
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
        ''')

    class TestElement(xt.BeamElement):
        _xofields={
            'n_kicks': xo.Int64,
            }
        _internal_record_class = TestElementRecord

        _depends_on = [xt.RandomUniform]

        _extra_c_sources = extra_src


    # Checks
    n_kicks0 = 5
    n_kicks1 = 3
    elements = [
        TestElement(_context=test_context, n_kicks=n_kicks0),
        TestElement(_context=test_context, n_kicks=n_kicks1)]

    io_buffer = xt.new_io_buffer(_context=test_context)
    record = xt.start_internal_logging(elements=elements, io_buffer=io_buffer,
                            capacity={'table1': 10000, 'table2': 10000})

    part = xp.Particles(_context=test_context, p0c=6.5e12, x=[1e-3,2e-3,3e-3])
    part._init_random_number_generator()
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
    io_buffer = xt.new_io_buffer(_context=test_context)
    record = xt.start_internal_logging(elements=elements, io_buffer=io_buffer,
                            capacity={'table1': 20, 'table2': 15})

    part = xp.Particles(_context=test_context, p0c=6.5e12, x=[1e-3,2e-3,3e-3])
    part._init_random_number_generator()
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
    io_buffer = xt.new_io_buffer(_context=test_context)
    record = xt.start_internal_logging(elements=elements, io_buffer=io_buffer,
                            capacity={'table1': 10000, 'table2': 10000})

    part = xp.Particles(_context=test_context, p0c=6.5e12, x=[1e-3,2e-3,3e-3])
    part._init_random_number_generator()
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
    io_buffer0 = xt.new_io_buffer(_context=test_context)
    record0 = xt.start_internal_logging(elements=elements[0], io_buffer=io_buffer0,
                            capacity={'table1': 10000, 'table2': 10000})
    io_buffer1 = xt.new_io_buffer(_context=test_context)
    record1 = xt.start_internal_logging(elements=elements[1], io_buffer=io_buffer1,
                            capacity={'table1': 10000, 'table2': 10000})

    part = xp.Particles(_context=test_context, p0c=6.5e12, x=[1e-3,2e-3,3e-3])
    part._init_random_number_generator()
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
